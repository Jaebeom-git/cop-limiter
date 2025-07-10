import argparse
import torch
import os
import numpy as np
from scipy.spatial.transform import Rotation
from data.AddBiomechanicsDataset import AddBiomechanicsDataset, InputDataKeys, OutputDataKeys
from cli.config_manager import ConfigManager
from torch.cuda.amp import autocast
from cli.abstract_command import AbstractCommand
from models.model_selector import select_model
from scipy import signal
import pandas as pd
from scipy.ndimage import gaussian_filter1d


class InferenceCommand(AbstractCommand):
    def __init__(self):
        super().__init__()

    def register_subcommand(self, subparsers: argparse._SubParsersAction):
        subparser = subparsers.add_parser('inference', help='Run inference on dataset.')

        subparser.add_argument('--dataset-home', type=str, required=True, help='The path to the dataset.')
        subparser.add_argument('--geometry-folder', type=str, default=None, help='Path to the Geometry folder with bone mesh data.')
        subparser.add_argument('--config-path', type=str, required=True, help='Path to the configuration file for model and settings.')
        subparser.add_argument('--result-dir', type=str, required=True, help='Directory to save the .mot result files.')
        subparser.add_argument('--save-opt', type=str, choices=['grf', 'ik', 'id', 'all'], default='all', help='Choose to save "grf", "id", or "all".')
        subparser.add_argument('--model-selection', type=str, default='dev', help='Select the model to use. Options: dev, train.')

        subparser.add_argument('--sample-rate', type=int, default=100, help='Sampling rate for the data.')
        subparser.add_argument('--lowpass', type=bool, default=False, help='Apply lowpass filtering if True.')
        subparser.add_argument('--cutoff-frequency', type=float, default=15, help='Cutoff frequency for the lowpass filter.')
        subparser.add_argument('--gaussian-edge-filter', type=bool, default=False, help='Apply Gaussian edge filter if True.')
        subparser.add_argument('--sliding-window-inference', type=bool, default=False, help='Use sliding window inference if True.')

    def run(self, args: argparse.Namespace):
        if args.command != 'inference':
            return False

        self.CutOffFrequency = args.cutoff_frequency
        self.SAMPLE_RATE = args.sample_rate
        self.LOWPASS = args.lowpass
        self.GAUSSIAN_EDGE_FILTER = args.gaussian_edge_filter
        self.gaussian_edge_frame = 5
        self.sliding_window_inference_flag = args.sliding_window_inference

        self.config_manager = ConfigManager(args.config_path)
        self.config = self.config_manager.config
        self.device = torch.device(self.config.device)
        if not self.sliding_window_inference_flag:
            self.sliding_window_inference_flag = False if 'mamba' in self.config.model_params.model.lower() else True
        
        checkpoint_dir = os.path.join(os.path.abspath(self.config.checkpoint_dir), self.config.model_name)
        # Create an instance of the model
        self.model = select_model(self.config.model_params, self.device)
        self.load_best_model(model=self.model, checkpoint_dir=checkpoint_dir, opt='model', model_type=args.model_selection)
        self.model.eval()

        geometry = self.ensure_geometry(args.geometry_folder)
        self.load_dataset(args.dataset_home, geometry, short=False)
        
        self.run_inference(args)

        return True
        
    def load_dataset(self, dataset_home, geometry_folder=None, short=False):
        """
        Method to load the dataset.
        """
        data_params = self.config.data_params if hasattr(self.config, 'data_params') else ValueError("No data_params found in config.")
        self.dataset = AddBiomechanicsDataset(
            dataset_home,
            data_params.history_len,
            device=self.device,
            geometry_folder=geometry_folder,
            stride=data_params.stride,
            testing_with_short_dataset=short,
            mode='infer'
        )

    def run_inference(self, args):
        """
        Run inference on all trials and save the results to result_dir.
        """
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Please load the dataset using `load_dataset` method.")
        
        for i in range(len(self.dataset)):
            self.run_inference_on_trial(i, args)

    def run_inference_on_trial(self, trial_idx, args):
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Please load the dataset using `load_dataset` method.")
        
        # Load the specified trial data from the dataset
        inputs, _, subject_index, trial_idx = self.dataset[trial_idx]
        inputs = {key: value.unsqueeze(0).to(self.device) for key, value in inputs.items()}
        
        # Perform inference
        with torch.no_grad():
            if self.sliding_window_inference_flag:
                outputs = self.sliding_window_inference(self.model, inputs, self.config.data_params.history_len, self.config.amp, self.config.device)
            else:
                if self.config.amp:
                    with torch.amp.autocast(self.config.device, dtype=torch.bfloat16):
                        outputs = self.model(inputs)
                else:
                    outputs = self.model(inputs)

        # Get the trial name
        trial_name = self.dataset.subjects[subject_index].getTrialName(trial_idx)
        if '_segment' in trial_name:
            trial_name = trial_name.split('_segment')[0]

        os.makedirs(os.path.join(args.result_dir, trial_name), exist_ok=True)

        if args.save_opt in ['grf', 'all']:
            output_file_name = os.path.join(args.result_dir, trial_name, f'{trial_name}_forceplate.mot')
            # Get force, cop, and torque converted to the world coordinate system
            left_forces_world, right_forces_world, left_cops_world, right_cops_world, left_torques_world, right_torques_world = self.get_world_frame_data(outputs, inputs, subject_index, lowPassFilter=self.LOWPASS)
            # Save as .mot file
            self.save_mot_file(left_forces_world, right_forces_world, left_cops_world, right_cops_world, left_torques_world, right_torques_world, output_file_name, trial_name)
            
        if args.save_opt in ['id', 'all']:
            os.makedirs(os.path.join(args.result_dir, trial_name, 'Result'), exist_ok=True)
            output_file_name = os.path.join(args.result_dir, trial_name, 'Result', f'Result_ID.sto')
            # Calculate torque
            tau = self.calculate_tau(outputs, inputs, subject_index, lowPassFilter=self.LOWPASS)
            # Save as .sto file
            self.save_tau_file(tau, output_file_name, trial_name)

        if args.save_opt in ['ik', 'all']:
            os.makedirs(os.path.join(args.result_dir, trial_name, 'Result'), exist_ok=True)
            output_file_name = os.path.join(args.result_dir, trial_name, 'Result', f'Result_IK.mot')
            # Save IK data
            self.save_ik_file(inputs, output_file_name, trial_name, lowPassFilter=self.LOWPASS)

    def gaussian_filter_edge_frames(self, data, n_frames, sigma=None, edges='both'):
        """
        Apply a Gaussian filter to the full signal, but only replace the
        first/last/both n_frames segments with the filtered values.

        Args:
            data (pd.DataFrame or np.ndarray): Input data to filter.
            n_frames (int): Number of frames to overwrite at each selected edge.
            edges (str): One of 'first', 'last', or 'both' indicating which edge(s) to apply.
            sigma (float, optional): Standard deviation of the Gaussian kernel.
                                    If None, sigma is computed based on data length and n_frames.

        Returns:
            pd.DataFrame or np.ndarray: Data with filtered edge segments applied.
        """
        if sigma is None:
            # Calculate sigma value proportional to the length of the data
            sigma = max(1, min(5, len(data) // (n_frames * 2)))
        
        apply_first = edges in ("first", "both")
        apply_last  = edges in ("last",  "both")

        if isinstance(data, pd.DataFrame):
            filtered_data = data.copy()
            columns_to_filter = [col for col in filtered_data.columns if col not in ['Time', 'Frame#', 'time', '']]
            
            for column in columns_to_filter:
                # Apply Gaussian filter to the entire data
                total_filtered = gaussian_filter1d(filtered_data[column].values, sigma=sigma)
                
                # Apply the filtered values only to the first n and last n frames
                if apply_first:
                    filtered_data.iloc[:n_frames, filtered_data.columns.get_loc(column)] = total_filtered[:n_frames]
                if apply_last:
                    filtered_data.iloc[-n_frames:, filtered_data.columns.get_loc(column)] = total_filtered[-n_frames:]

        elif isinstance(data, np.ndarray):
            filtered_data = data.copy()
            # For 1D array
            if data.ndim == 1:
                total_filtered = gaussian_filter1d(data, sigma=sigma)
                if apply_first:
                    filtered_data[:n_frames] = total_filtered[:n_frames]
                if apply_last:
                    filtered_data[-n_frames:] = total_filtered[-n_frames:]
            # For 2D array (apply filter to each column)
            else:
                for i in range(data.shape[1]):
                    total_filtered = gaussian_filter1d(data[:, i], sigma=sigma)
                    if apply_first:
                        filtered_data[:n_frames, i] = total_filtered[:n_frames]
                    if apply_last:
                        filtered_data[-n_frames:, i] = total_filtered[-n_frames:]

        else:
            raise TypeError("Input data must be a pandas DataFrame or a numpy ndarray.")
        
        return filtered_data

    def low_pass_filter(self, data, sampling_rate, cutoff_frequency, filter_order):
        nyquist_frequency = 0.5 * sampling_rate
        wn = cutoff_frequency / nyquist_frequency   # Calculate normalized cutoff frequency
        sos = signal.butter(filter_order // 2, wn, btype='low', output='sos')   # Design Butterworth filter

        if isinstance(data, pd.DataFrame):
            filtered_data = data.copy()
            for column in filtered_data.columns:
                if column not in ['Time', 'Frame#', 'time', '']:
                    filtered_data[column] = signal.sosfiltfilt(sos, filtered_data[column], axis=0)
        elif isinstance(data, np.ndarray):
            if data.ndim == 1:  # Handle 1D array
                filtered_data = signal.sosfiltfilt(sos, data, axis=0)
            else:
                filtered_data = data.copy()
                for i in range(data.shape[1]):
                    filtered_data[:, i] = signal.sosfiltfilt(sos, data[:, i], axis=0)
        else:
            raise TypeError("Input data must be a pandas DataFrame or a numpy ndarray.")
        
        return filtered_data

    def get_world_frame_data(self, outputs, inputs, subject_index, lowPassFilter=False):
        """
        Convert force, cop, and torque data from the root frame to the world frame and return.
        Force and torque are multiplied by the skeleton's mass.
        """
        # Load skeleton
        skel = self.dataset.skeletons[subject_index]

        # Get mass
        mass = skel.getMass()

        # Get the world transform of the root body for each frame and convert the data
        pos = inputs[InputDataKeys.POS][0].cpu().numpy()  # [seq, num_dofs]
        pos = self.low_pass_filter(pos, sampling_rate=self.SAMPLE_RATE, cutoff_frequency=self.CutOffFrequency, filter_order=4) if lowPassFilter else pos
        num_frames = pos.shape[0]

        # Root body
        root_body_name = 'pelvis'

        # Extract force, COP, and torque data
        ground_forces_root = outputs[OutputDataKeys.GROUND_CONTACT_FORCES_IN_ROOT_FRAME][0].float().cpu().numpy()  # [seq, 6]
        ground_forces_root = self.gaussian_filter_edge_frames(ground_forces_root, n_frames=self.gaussian_edge_frame) if self.GAUSSIAN_EDGE_FILTER else ground_forces_root
        ground_forces_root = self.low_pass_filter(ground_forces_root, sampling_rate=self.SAMPLE_RATE, cutoff_frequency=self.CutOffFrequency, filter_order=4) if lowPassFilter else ground_forces_root
        cops_root = outputs[OutputDataKeys.GROUND_CONTACT_COPS_IN_ROOT_FRAME][0].float().cpu().numpy()             # [seq, 6]
        cops_root = self.gaussian_filter_edge_frames(cops_root, n_frames=self.gaussian_edge_frame) if self.GAUSSIAN_EDGE_FILTER else cops_root
        cops_root = self.low_pass_filter(cops_root, sampling_rate=self.SAMPLE_RATE, cutoff_frequency=self.CutOffFrequency, filter_order=4) if lowPassFilter else cops_root
        torques_root = outputs[OutputDataKeys.GROUND_CONTACT_TORQUES_IN_ROOT_FRAME][0].float().cpu().numpy()       # [seq, 6]
        torques_root = self.gaussian_filter_edge_frames(torques_root, n_frames=self.gaussian_edge_frame) if self.GAUSSIAN_EDGE_FILTER else torques_root
        torques_root = self.low_pass_filter(torques_root, sampling_rate=self.SAMPLE_RATE, cutoff_frequency=self.CutOffFrequency, filter_order=4) if lowPassFilter else torques_root

        # Separate left and right data
        left_forces_root = ground_forces_root[:, :3]     # [seq, 3]
        right_forces_root = ground_forces_root[:, 3:6]   # [seq, 3]
        left_cops_root = cops_root[:, :3]                # [seq, 3]
        right_cops_root = cops_root[:, 3:6]              # [seq, 3]
        left_torques_root = torques_root[:, :3]          # [seq, 3]
        right_torques_root = torques_root[:, 3:6]        # [seq, 3]

        # Initialize arrays to store the conversion results
        left_forces_world = np.zeros_like(left_forces_root)
        right_forces_world = np.zeros_like(right_forces_root)
        left_cops_world = np.zeros_like(left_cops_root)
        right_cops_world = np.zeros_like(right_cops_root)
        left_torques_world = np.zeros_like(left_torques_root)
        right_torques_world = np.zeros_like(right_torques_root)

        # Perform conversion for all frames
        for i in range(num_frames):
            # Update skeleton position
            skel.setPositions(pos[i])

            # Get the world transform of the root body
            root_body = skel.getBodyNode(root_body_name)
            root_transform = root_body.getWorldTransform()
            root_rotation = root_transform.rotation()
            root_translation = root_transform.translation()

            # Convert from root frame to world frame
            # Apply only rotation transform to force and torque, then multiply by mass
            left_forces_world[i] = mass * (root_rotation @ left_forces_root[i])
            right_forces_world[i] = mass * (root_rotation @ right_forces_root[i])
            left_torques_world[i] = mass * (root_rotation @ left_torques_root[i])
            right_torques_world[i] = mass * (root_rotation @ right_torques_root[i])

            # Apply rotation and then translation to COP (do not multiply by mass)
            left_cops_world[i] = root_rotation @ left_cops_root[i] + root_translation
            right_cops_world[i] = root_rotation @ right_cops_root[i] + root_translation

        return (
            # self.low_pass_filter(self.gaussian_filter_edge_frames(self.gaussian_filter_edge_frames(self.gaussian_filter_edge_frames(left_forces_world, n_frames=5), n_frames=25, sigma=50), n_frames=15, sigma=100), sampling_rate=self.SAMPLE_RATE, cutoff_frequency=self.CutOffFrequency, filter_order=4), 
            # self.low_pass_filter(self.gaussian_filter_edge_frames(self.gaussian_filter_edge_frames(self.gaussian_filter_edge_frames(right_forces_world, n_frames=5), n_frames=25, sigma=50), n_frames=15, sigma=100), sampling_rate=self.SAMPLE_RATE, cutoff_frequency=self.CutOffFrequency, filter_order=4),
            left_forces_world, right_forces_world,
            left_cops_world, right_cops_world,
            left_torques_world, right_torques_world
        )
    
    def save_mot_file(self, left_forces_world, right_forces_world, left_cops_world, right_cops_world, left_torques_world, right_torques_world, output_file_name, trial_name):
        """
        Create a .mot file based on forces_world, cops_world, and torques_world data.
        """
        header = [
            trial_name,
            "version=1",
            f"nRows={left_forces_world.shape[0]}",
            "nColumns=19",
            "inDegrees=yes",
            "endheader"
        ]
        column_names = [
            "time",
            "l_ground_force_vx", "l_ground_force_vy", "l_ground_force_vz",
            "l_ground_force_px", "l_ground_force_py", "l_ground_force_pz",
            "r_ground_force_vx", "r_ground_force_vy", "r_ground_force_vz",
            "r_ground_force_px", "r_ground_force_py", "r_ground_force_pz",
            "l_ground_torque_x", "l_ground_torque_y", "l_ground_torque_z",
            "r_ground_torque_x", "r_ground_torque_y", "r_ground_torque_z"
        ]

        with open(output_file_name, 'w') as f:
            f.write('\n'.join(header) + '\n')
            f.write('\t'.join(column_names) + '\n')

            # Record time, force, cop, and torque data to the file
            for i in range(left_forces_world.shape[0]):
                time = i/self.SAMPLE_RATE
                row_data = [time] + list(left_forces_world[i].flatten()) + list(left_cops_world[i].flatten()) + list(right_forces_world[i].flatten()) + list(right_cops_world[i].flatten()) + list(left_torques_world[i].flatten()) + list(right_torques_world[i].flatten())
                f.write('\t'.join(map(str, row_data)) + '\n')

        print(f"Successfully saved .mot file to {output_file_name}")

    def calculate_tau(self, outputs, inputs, subject_index, lowPassFilter=False):
        """
        Calculate joint torques (tau) using inverse dynamics for each frame.
        """
        skel = self.dataset.skeletons[subject_index]
        mass = skel.getMass()

        # Get input data
        pos = inputs[InputDataKeys.POS][0].cpu().numpy()
        pos = self.low_pass_filter(pos, sampling_rate=self.SAMPLE_RATE, cutoff_frequency=self.CutOffFrequency, filter_order=4) if lowPassFilter else pos
        vel = inputs[InputDataKeys.VEL][0].cpu().numpy()
        vel = self.low_pass_filter(vel, sampling_rate=self.SAMPLE_RATE, cutoff_frequency=self.CutOffFrequency, filter_order=4) if lowPassFilter else vel
        acc = inputs[InputDataKeys.ACC][0].cpu().numpy()
        acc = self.gaussian_filter_edge_frames(acc, n_frames=self.gaussian_edge_frame) if self.GAUSSIAN_EDGE_FILTER else acc
        acc = self.low_pass_filter(acc, sampling_rate=self.SAMPLE_RATE, cutoff_frequency=self.CutOffFrequency, filter_order=4) if lowPassFilter else acc
        num_frames = pos.shape[0]

        # Prepare contact bodies as BodyNode objects
        contact_body_nodes = self.dataset.skeletons_contact_bodies[subject_index]

        # Extract forces and torques from outputs and multiply by mass
        ground_forces_root = outputs[OutputDataKeys.GROUND_CONTACT_FORCES_IN_ROOT_FRAME][0].float().cpu().numpy() * mass    # [seq_len, 6]
        ground_forces_root = self.gaussian_filter_edge_frames(ground_forces_root, n_frames=self.gaussian_edge_frame) if self.GAUSSIAN_EDGE_FILTER else ground_forces_root
        ground_forces_root = self.low_pass_filter(ground_forces_root, sampling_rate=self.SAMPLE_RATE, cutoff_frequency=self.CutOffFrequency, filter_order=4) if lowPassFilter else ground_forces_root
        torques_root = outputs[OutputDataKeys.GROUND_CONTACT_TORQUES_IN_ROOT_FRAME][0].float().cpu().numpy() * mass # [seq_len, 6]
        torques_root = self.gaussian_filter_edge_frames(torques_root, n_frames=self.gaussian_edge_frame) if self.GAUSSIAN_EDGE_FILTER else torques_root
        torques_root = self.low_pass_filter(torques_root, sampling_rate=self.SAMPLE_RATE, cutoff_frequency=self.CutOffFrequency, filter_order=4) if lowPassFilter else torques_root
        cop_root = outputs[OutputDataKeys.GROUND_CONTACT_COPS_IN_ROOT_FRAME][0].float().cpu().numpy()
        cop_root = self.gaussian_filter_edge_frames(cop_root, n_frames=self.gaussian_edge_frame) if self.GAUSSIAN_EDGE_FILTER else cop_root
        cop_root = self.low_pass_filter(cop_root, sampling_rate=self.SAMPLE_RATE, cutoff_frequency=self.CutOffFrequency, filter_order=4) if lowPassFilter else cop_root
        
        # Split left and right data
        left_forces_root = ground_forces_root[:, :3]     # [seq_len, 3]
        right_forces_root = ground_forces_root[:, 3:6]   # [seq_len, 3]
        left_torques_root = torques_root[:, :3]          # [seq_len, 3]
        right_torques_root = torques_root[:, 3:6]        # [seq_len, 3]
        left_cop = cop_root[:, :3]
        right_cop = cop_root[:, 3:6]
        left_rxF = np.cross(left_cop, left_forces_root)
        right_rxF = np.cross(right_cop, right_forces_root)

        # Initialize array to store tau for all frames
        tau_all_frames = np.zeros((num_frames, skel.getNumDofs()))

        # For each frame, compute tau
        for i in range(num_frames):
            # Update skeleton state
            skel.setPositions(pos[i])
            skel.setVelocities(vel[i])
            accelerations = acc[i]

            # Left foot
            left_wrench = np.hstack([left_torques_root[i]+left_rxF[i], left_forces_root[i]])
            # Right foot
            right_wrench = np.hstack([right_torques_root[i]+right_rxF[i], right_forces_root[i]])

            # Assuming contact bodies are ordered as [left_foot, right_foot]
            contact_wrench_list = [left_wrench.reshape(6, 1), right_wrench.reshape(6, 1)]

            # Root residuals (assumed zero)
            root_residuals = np.zeros((6, 1))

            # Call getInverseDynamicsFromPredictions per frame
            tau = skel.getInverseDynamicsFromPredictions(
                accelerations.reshape(-1, 1),
                contact_body_nodes,
                contact_wrench_list,
                root_residuals
            ).flatten()  # Convert to 1D array

            tau_all_frames[i] = tau

        tau_all_frames = self.low_pass_filter(tau_all_frames, sampling_rate=self.SAMPLE_RATE, cutoff_frequency=self.CutOffFrequency, filter_order=4) if lowPassFilter else tau_all_frames
        return tau_all_frames  # [seq_len, num_dofs]

    def save_tau_file(self, tau, output_file_name, trial_name):
        """
        Create a .sto file based on tau data.
        """
        num_rows = tau.shape[0]
        num_dofs = tau.shape[1]
        moment_list = ['pelvis_tilt_moment', 'pelvis_list_moment', 'pelvis_rotation_moment', 'pelvis_tx_force', 'pelvis_ty_force', 'pelvis_tz_force',
                       'hip_flexion_r_moment', 'hip_adduction_r_moment', 'hip_rotation_r_moment', 'knee_angle_r_moment', 'ankle_angle_r_moment', 'subtalar_angle_r_moment', 'mtp_angle_r_moment', 
                       'hip_flexion_l_moment', 'hip_adduction_l_moment', 'hip_rotation_l_moment', 'knee_angle_l_moment', 'ankle_angle_l_moment', 'subtalar_angle_l_moment', 'mtp_angle_l_moment', 
                       'lumbar_extension_moment', 'lumbar_bending_moment', 'lumbar_rotation_moment']
        column_names = ["time"] + moment_list

        header = [
            "DataType=Double",
            f"DataRate={self.SAMPLE_RATE}",
            f"DataRows={num_rows}",
            f"DataColumns={len(column_names)}",
            "Units=SI",
            "FileType=TimeSeries",
            f"Name={trial_name}",
            "endheader"
        ]

        with open(output_file_name, 'w') as f:
            f.write('\n'.join(header) + '\n')
            f.write('\t'.join(column_names) + '\n')

            for i in range(num_rows):
                time = i/self.SAMPLE_RATE
                row_data = [time] + list(tau[i])
                f.write('\t'.join(map(str, row_data)) + '\n')

        print(f"Successfully saved .sto file: {output_file_name}")

    def save_ik_file(self, inputs, output_file_name, trial_name, lowPassFilter=False):
        """
        Create a .mot file based on IK data.
        """
        pos = inputs[InputDataKeys.POS][0].cpu().numpy()  # [seq_len, num_dofs]
        pos = self.low_pass_filter(pos, sampling_rate=self.SAMPLE_RATE, cutoff_frequency=self.CutOffFrequency, filter_order=4) if lowPassFilter else pos
        num_rows, num_dofs = pos.shape

        # Get joint names
        skel = self.dataset.skeletons[0]  # Assume all skeletons have the same DOFs
        dof_names = ['pelvis_tilt', 'pelvis_list', 'pelvis_rotation', 'pelvis_tx', 'pelvis_ty', 'pelvis_tz',
                     'hip_flexion_r', 'hip_adduction_r', 'hip_rotation_r', 'knee_angle_r', 'ankle_angle_r', 'subtalar_angle_r', 'mtp_angle_r',
                     'hip_flexion_l', 'hip_adduction_l', 'hip_rotation_l', 'knee_angle_l', 'ankle_angle_l', 'subtalar_angle_l', 'mtp_angle_l',
                     'lumbar_extension', 'lumbar_bending', 'lumbar_rotation']
        column_names = ["time"] + dof_names

        header = [
            trial_name,
            "version=1",
            f"nRows={num_rows}",
            f"nColumns={len(column_names)}",
            "inDegrees=yes",
            "endheader"
        ]

        with open(output_file_name, 'w') as f:
            f.write('\n'.join(header) + '\n')
            f.write('\t'.join(column_names) + '\n')

            for i in range(num_rows):
                time = i/self.SAMPLE_RATE
                # Convert radians to degrees
                angles_in_degrees = np.degrees(pos[i])
                row_data = [time] + list(angles_in_degrees)
                f.write('\t'.join(map(str, row_data)) + '\n')

        print(f"Successfully saved .mot file: {output_file_name}")
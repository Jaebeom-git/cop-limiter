import argparse

import torch
from data.AddBiomechanicsDataset import AddBiomechanicsDataset, InputDataKeys, OutputDataKeys
from loss.RegressionLossEvaluator import RegressionLossEvaluator
from typing import Dict, List
from cli.abstract_command import AbstractCommand
import os
import nimblephysics as nimble
from nimblephysics import NimbleGUI
import numpy as np
from cli.config_manager import ConfigManager
from torch.cuda.amp import autocast
from scipy import signal
import pandas as pd
from models.model_selector import select_model
from megablocks.layers.moe import get_load_balancing_loss, clear_load_balancing_loss
from models.CoPLimiter import CoPLimiter

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
class VisualizeCommand(AbstractCommand):
    def __init__(self):
        super().__init__()

    def register_subcommand(self, subparsers: argparse._SubParsersAction):
        subparser = subparsers.add_parser('visualize', help='Visualize the performance of a model on dataset.')

        subparser.add_argument('--dataset-home', type=str, default='../data', help='The path to the AddBiomechanics dataset.')
        subparser.add_argument('--geometry-folder', type=str, default=None, help='Path to the Geometry folder with bone mesh data.')
        subparser.add_argument('--short', action='store_true', help='Use very short datasets to test without loading a bunch of data.')
        subparser.add_argument('--config-path', type=str, required=True, help='Path to the configuration file for model and training settings.')
        subparser.add_argument('--model-selection', type=str, default='dev', help='Select the model to use. Options: dev, train.')

        subparser.add_argument('--sample-rate', type=int, default=100, help='Sampling rate for the data.')
        subparser.add_argument('--lowpass', type=bool, default=True, help='Apply lowpass filtering if True.')
        subparser.add_argument('--cutoff-frequency', type=float, default=15, help='Cutoff frequency for the lowpass filter.')

        subparser.add_argument('--predict-grf-components', type=int, nargs='+', default=[i for i in range(6)],
                               help='Which grf components to train.')
        subparser.add_argument('--predict-cop-components', type=int, nargs='+', default=[i for i in range(6)],
                               help='Which cop components to train.')
        subparser.add_argument('--predict-moment-components', type=int, nargs='+', default=[i for i in range(6)],
                               help='Which moment components to train.')
        subparser.add_argument('--predict-wrench-components', type=int, nargs='+', default=[i for i in range(12)],
                               help='Which wrench components to train.')

        subparser.add_argument('--mode', type=str, default='world_frame', choices=['world_frame', 'root_frame'], help='Visualization mode: "world_frame" or "root_frame".')
        subparser.add_argument('--viz-origin', type=str2bool, nargs='?', const=True, default=False, help='Enable visualization of origin axes (coordinate system).')
        subparser.add_argument('--viz-com', type=str2bool, nargs='?', const=True, default=False, help='Enable visualization of the center of mass (CoM).')
        subparser.add_argument('--viz-footbox', type=str2bool, nargs='?', const=True, default=False, help='Enable visualization of foot boxes.')
        subparser.add_argument('--sliding-window-inference', type=str2bool, nargs='?', const=True, default=False, help='Sliding Window Inference.')

    def run(self, args: argparse.Namespace):
        if 'command' in args and args.command != 'visualize':
            return False
        
        mode = args.mode
        viz_origin = args.viz_origin
        viz_com = args.viz_com
        viz_footbox = args.viz_footbox
        sliding_window_inference_flag = args.sliding_window_inference

        self.CutOffFrequency = args.cutoff_frequency
        self.SAMPLE_RATE = args.sample_rate
        self.LOWPASS = args.lowpass

        # Load settings
        config_manager = ConfigManager(args.config_path)
        config = config_manager.config
        if not sliding_window_inference_flag:
            sliding_window_inference_flag = False if 'mamba' in config.model_params.model.lower() else True

        amp = config.amp
        dataset_home = args.dataset_home
        device = torch.device(config.device)
        model_type = config.model_name
        checkpoint_dir = os.path.join(os.path.abspath(config.checkpoint_dir), model_type)
        short: bool = args.short

        geometry = self.ensure_geometry(args.geometry_folder)

        cached_inputs, cached_labels, cached_outputs = None, None, None

        self.cop_limiter_activation_function = config.cop_limiter_activation_function if hasattr(config, 'cop_limiter_activation_function') else 'tanh'
        self.cop_limiter = CoPLimiter(length_ratio=2.0, height_ratio=2.5, width_ratio=0.75, activation_function=self.cop_limiter_activation_function).to(device)

        print('## Loading Data set:')
        # data_path = os.path.abspath(os.path.join(dataset_home, 'dev'))
        data_path = os.path.abspath(dataset_home)
        # data_path = '/home/awear-omen/Ws/OpenSim/OpenSim_Analysis/Experiments/Subject000/InferBiomechanics/'

        data_params = config.data_params if hasattr(config, 'data_params') else ValueError("No data_params found in config.")
        dev_dataset = AddBiomechanicsDataset(
            data_path,
            data_params.history_len,
            device=device,
            geometry_folder=geometry,
            testing_with_short_dataset=short,
            stride=data_params.stride,
            mode='infer',
            # window_stride=data_params.history_len
            # trial_filter='pick'
        )

        # Create an instance of the model
        if args.model_selection.strip():
            model = select_model(config.model_params, device)
            self.load_best_model(model=model, checkpoint_dir=checkpoint_dir, opt='model', model_type=args.model_selection)
            model.eval()
            self.print_model_summary(model)
        else:
            model = None
            print("No model selected. Only visualizing the dataset.")

        # Prepare for visualization
        loss_evaluator = RegressionLossEvaluator(dataset=dev_dataset, split='test')

        world = nimble.simulation.World()
        world.setGravity([0, -9.81, 0])

        gui = NimbleGUI(world)
        gui.serve(8080)

        # Add XYZ axes
        if viz_origin:
            axis_length = 0.5  # Set axis length
            gui.nativeAPI().createLine(
                'X_axis',
                [[0, 0, 0], [axis_length, 0, 0]],
                [1, 0, 0, 1]  # RGBA color: red
            )
            gui.nativeAPI().createLine(
                'Y_axis',
                [[0, 0, 0], [0, axis_length, 0]],
                [0, 1, 0, 1]  # RGBA color: green
            )
            gui.nativeAPI().createLine(
                'Z_axis',
                [[0, 0, 0], [0, 0, axis_length]],
                [0, 0, 1, 1]  # RGBA color: blue
            )

        ticker: nimble.realtime.Ticker = nimble.realtime.Ticker(1/self.SAMPLE_RATE)
        trial_index = 0  # Load the 0th trial at the start
        frame_index = 0
        playing: bool = False
        num_trials = len(dev_dataset)
        total_frames_in_trial = 0

        # Initialize variables to store current trial data
        inputs, labels, outputs, batch_subject_indices, batch_trial_indices, skel, contact_bodies, plot_info, timesteps = None, None, None, None, None, None, None, {}, None

        def calculate_tau(outputs, inputs, subject_index, lowPassFilter=False):
            skel = dev_dataset.skeletons[subject_index]
            dof_names = []
            for i in range(skel.getNumDofs()):
                dof_names.append(skel.getDofByIndex(i).getName())
                
            mass = skel.getMass()

            pos = inputs[InputDataKeys.POS][0].cpu().numpy()
            pos = low_pass_filter(pos, sampling_rate=self.SAMPLE_RATE, cutoff_frequency=self.CutOffFrequency, filter_order=4) if lowPassFilter else pos
            vel = inputs[InputDataKeys.VEL][0].cpu().numpy()
            vel = low_pass_filter(vel, sampling_rate=self.SAMPLE_RATE, cutoff_frequency=self.CutOffFrequency, filter_order=4) if lowPassFilter else vel
            acc = inputs[InputDataKeys.ACC][0].cpu().numpy()
            acc = low_pass_filter(acc, sampling_rate=self.SAMPLE_RATE, cutoff_frequency=self.CutOffFrequency, filter_order=4) if lowPassFilter else acc
            num_frames = pos.shape[0]

            contact_body_nodes = dev_dataset.skeletons_contact_bodies[subject_index]

            ground_forces_root = outputs[OutputDataKeys.GROUND_CONTACT_FORCES_IN_ROOT_FRAME][0].float().cpu().numpy() * mass
            ground_forces_root = low_pass_filter(ground_forces_root, sampling_rate=self.SAMPLE_RATE, cutoff_frequency=self.CutOffFrequency, filter_order=4) if lowPassFilter else ground_forces_root
            torques_root = outputs[OutputDataKeys.GROUND_CONTACT_TORQUES_IN_ROOT_FRAME][0].float().cpu().numpy() * mass
            torques_root = low_pass_filter(torques_root, sampling_rate=self.SAMPLE_RATE, cutoff_frequency=self.CutOffFrequency, filter_order=4) if lowPassFilter else torques_root
            cop_root = outputs[OutputDataKeys.GROUND_CONTACT_COPS_IN_ROOT_FRAME][0].float().cpu().numpy()
            cop_root = low_pass_filter(cop_root, sampling_rate=self.SAMPLE_RATE, cutoff_frequency=self.CutOffFrequency, filter_order=4) if lowPassFilter else cop_root

            left_forces_root = ground_forces_root[:, :3]
            right_forces_root = ground_forces_root[:, 3:6]
            left_torques_root = torques_root[:, :3]
            right_torques_root = torques_root[:, 3:6]
            left_cop = cop_root[:, :3]
            right_cop = cop_root[:, 3:6]
            left_rxF = np.cross(left_cop, left_forces_root)
            right_rxF = np.cross(right_cop, right_forces_root)

            tau_all_frames = np.zeros((num_frames, skel.getNumDofs()))

            for i in range(num_frames):
                skel.setPositions(pos[i])
                skel.setVelocities(vel[i])
                accelerations = acc[i]

                left_wrench = np.hstack([left_torques_root[i]+left_rxF[i], left_forces_root[i]])
                right_wrench = np.hstack([right_torques_root[i]+right_rxF[i], right_forces_root[i]])

                contact_wrench_list = [left_wrench.reshape(6, 1), right_wrench.reshape(6, 1)]
                root_residuals = np.zeros((6, 1))

                tau = skel.getInverseDynamicsFromPredictions(
                    accelerations.reshape(-1, 1),
                    contact_body_nodes,
                    contact_wrench_list,
                    root_residuals
                ).flatten()

                tau_all_frames[i] = tau

            tau_all_frames = low_pass_filter(tau_all_frames, sampling_rate=self.SAMPLE_RATE, cutoff_frequency=self.CutOffFrequency, filter_order=4)
            return tau_all_frames
        
        def low_pass_filter(data, sampling_rate, cutoff_frequency, filter_order):
            nyquist_frequency = 0.5 * sampling_rate
            wn = cutoff_frequency / nyquist_frequency   # Calculate normalized cutoff frequency
            sos = signal.butter(filter_order // 2, wn, btype='low', output='sos')   # Design Butterworth filter

            if isinstance(data, pd.DataFrame):
                df = data.copy()
                for column in df.columns:
                    if column not in ['Time', 'Frame#', 'time', '']:
                        df[column] = signal.sosfiltfilt(sos, df[column], axis=0)
                return df
            
            if not isinstance(data, np.ndarray):
                raise TypeError("Input must be a numpy.ndarray or pandas.DataFrame")
            arr = np.asarray(data)
            if arr.ndim == 1:                              # (T,)
                return signal.sosfiltfilt(sos, arr, axis=0)

            if arr.ndim == 2:                              # (T, C)
                return signal.sosfiltfilt(sos, arr, axis=0)

            if arr.ndim == 3:                              # (B, T, C)
                out = np.empty_like(arr)
                for b in range(arr.shape[0]):              # batch loop, channel vectorised
                    out[b] = signal.sosfiltfilt(sos, arr[b], axis=0)
                return out

        def plot_tau_values(gui, timesteps, tau_values):
            plot_info = {}

            # Indices for plotting ankle, knee, and hip flexion moments
            ankle_indices = [10, 17]  # right, left ankle flexion moments
            knee_indices = [9, 16]    # right, left knee flexion moments
            hip_indices = [6, 13]     # right, left hip flexion moments

            # Plot settings
            x_size = 400
            y_size = 170
            init_pos_x = 150
            init_pos_y = 80

            # Plot ankle moments
            y_min = np.min(tau_values[:, ankle_indices])
            y_max = np.max(tau_values[:, ankle_indices])
            gui.nativeAPI().createRichPlot('ankle_moment_plot', [init_pos_x, init_pos_y], [x_size, y_size], 0, timesteps[-1], 
                                          y_min, y_max,
                                          f'Ankle Moments', 'Time (s)', 'Moment (Nm)')
            gui.nativeAPI().setRichPlotData('ankle_moment_plot', 'Right', 'blue', 'line', timesteps, tau_values[:, ankle_indices[0]])
            gui.nativeAPI().setRichPlotData('ankle_moment_plot', 'Left', 'red', 'line', timesteps, tau_values[:, ankle_indices[1]])
            plot_info['ankle_moment_plot'] = {'y_min': y_min, 'y_max': y_max}

            # Plot knee moments
            y_min = np.min(tau_values[:, knee_indices])
            y_max = np.max(tau_values[:, knee_indices])
            gui.nativeAPI().createRichPlot('knee_moment_plot', [init_pos_x, init_pos_y+(y_size+100)], [x_size, y_size], 0, timesteps[-1], 
                                          y_min, y_max,
                                          f'Knee Moments', 'Time (s)', 'Moment (Nm)')
            gui.nativeAPI().setRichPlotData('knee_moment_plot', 'Right', 'blue', 'line', timesteps, tau_values[:, knee_indices[0]])
            gui.nativeAPI().setRichPlotData('knee_moment_plot', 'Left', 'red', 'line', timesteps, tau_values[:, knee_indices[1]])
            plot_info['knee_moment_plot'] = {'y_min': y_min, 'y_max': y_max}

            # Plot hip moments
            y_min = np.min(tau_values[:, hip_indices])
            y_max = np.max(tau_values[:, hip_indices])
            gui.nativeAPI().createRichPlot('hip_moment_plot', [init_pos_x, init_pos_y+(y_size+100)*2], [x_size, y_size], 0, timesteps[-1], 
                                          y_min, y_max,
                                          f'Hip Moments', 'Time (s)', 'Moment (Nm)')
            gui.nativeAPI().setRichPlotData('hip_moment_plot', 'Right', 'blue', 'line', timesteps, tau_values[:, hip_indices[0]])
            gui.nativeAPI().setRichPlotData('hip_moment_plot', 'Left', 'red', 'line', timesteps, tau_values[:, hip_indices[1]])
            plot_info['hip_moment_plot'] = {'y_min': y_min, 'y_max': y_max}

            return plot_info

        # Load the 0th trial
        def load_trial(trial_idx):
            nonlocal playing
            nonlocal cached_inputs, cached_labels, cached_outputs
            nonlocal inputs, labels, batch_subject_indices, batch_trial_indices
            nonlocal total_frames_in_trial, outputs
            nonlocal skel, contact_bodies, frame_index
            nonlocal loss_evaluator
            nonlocal plot_info, timesteps
            nonlocal model
            nonlocal amp
            nonlocal device
            nonlocal sliding_window_inference_flag

            playing = False

            # Load data for the specified trial
            inputs, labels, batch_subject_index, trial_idx = dev_dataset[trial_idx]
            inputs = {key: value.to(device) for key, value in inputs.items()}
            labels = {key: value.to(device) for key, value in labels.items()}
            batch_subject_indices = [batch_subject_index]
            batch_trial_indices = [trial_idx]

            cached_inputs = { key: value.cpu().numpy() for key, value in inputs.items() }
            cached_labels = { key: value.cpu().numpy() for key, value in labels.items() }

            # Print the current trial name
            subject_path = dev_dataset.subject_paths[batch_subject_indices[0]]
            trial_name = dev_dataset.subjects[batch_subject_indices[0]].getTrialName(batch_trial_indices[0])
            print(f'#################### {trial_name} ####################')
            print(f'Subject: {subject_path}')

            # Add a batch dimension
            for key in inputs:
                inputs[key] = inputs[key].unsqueeze(0)
            for key in labels:
                labels[key] = labels[key].unsqueeze(0)

            total_frames_in_trial = inputs[InputDataKeys.POS].shape[1]
            frame_index = 0

            if model is not None:
                # Perform inference on the trial
                if sliding_window_inference_flag:
                    outputs = self.sliding_window_inference(model, inputs, data_params.history_len, amp, config.device)
                else:
                    with torch.no_grad():
                        if amp:
                            with torch.amp.autocast(config.device, dtype=torch.bfloat16):
                                outputs = model(inputs)
                        else:
                            outputs = model(inputs)

                if self.LOWPASS:
                    for key in outputs:
                        if isinstance(outputs[key], torch.Tensor):
                            temp = low_pass_filter(outputs[key].to(torch.float32).cpu().numpy(), sampling_rate=self.SAMPLE_RATE, cutoff_frequency=self.CutOffFrequency, filter_order=4)
                            outputs[key] = torch.as_tensor(temp, device=device, dtype=outputs[key].dtype)

                if hasattr(model, 'moe_args'):
                    if model.moe_args is not None:
                        tokens_per_expert, expert_scores = zip(*get_load_balancing_loss())
                        plot_load_balancing(gui, tokens_per_expert, expert_scores, model.moe_args)
                        clear_load_balancing_loss()

                # Evaluate performance on the current trial and print the report
                loss_evaluator(inputs, outputs, labels, batch_subject_indices, batch_trial_indices, args, compute_report=True)
                loss_evaluator.print_report(args)

                if amp:
                    cached_outputs = { key: value.squeeze(0).float().cpu().numpy() for key, value in outputs.items() }
                else:
                    cached_outputs = { key: value.squeeze(0).cpu().numpy() for key, value in outputs.items() }
            else:
                outputs = labels

            skel = dev_dataset.skeletons[batch_subject_indices[0]]
            contact_bodies = dev_dataset.skeletons_contact_bodies[batch_subject_indices[0]]

            # Calculate and store tau values for plotting
            tau_values = calculate_tau(outputs, inputs, batch_subject_indices[0], lowPassFilter=self.LOWPASS)
            timesteps = np.arange(tau_values.shape[0]) * 1/self.SAMPLE_RATE
            plot_info = plot_tau_values(gui, timesteps, tau_values)

        # Load and start the 0th trial
        load_trial(trial_index)
        
        def onKeyPress(key):
            nonlocal playing
            nonlocal frame_index
            nonlocal trial_index
            nonlocal num_trials

            if key == ' ':
                playing = not playing
            elif key == 'e':
                frame_index += 1
                if frame_index >= total_frames_in_trial:
                    frame_index = 0
            elif key == 'a':
                frame_index -= 1
                if frame_index < 0:
                    frame_index = total_frames_in_trial - 1
            elif key == 'r':
                print()
                # Move to the next trial
                trial_index += 1
                if trial_index >= num_trials:
                    trial_index = 0
                # Load new trial data
                load_trial(trial_index)
            elif key == 't':
                print()
                trial_index -= 1
                if trial_index >= num_trials:
                    trial_index = 0
                load_trial(trial_index)

        gui.nativeAPI().registerKeydownListener(onKeyPress)

        def compute_foot_ranges(joint_centers):
            """
            Use CoPLimiter to compute foot ranges.
            
            Args:
                joint_centers: numpy array of shape (num_joints * 3,)
            
            Returns:
                centers: dict - {'left': numpy array (3,), 'right': numpy array (3,)}
                sizes: dict - {'left': numpy array (3,), 'right': numpy array (3,)}
                rotation_matrices: dict - {'left': numpy array (3, 3), 'right': numpy array (3, 3)}
            """
            # Convert to tensor and compute feet parameters
            joint_centers_tensor = torch.from_numpy(joint_centers).float().to(device).unsqueeze(0).unsqueeze(0)
            
            with torch.no_grad():
                centers_t, sizes_t, rot_matrices_t = self.cop_limiter.compute_feet(joint_centers_tensor)
            
            # Convert to numpy and squeeze dimensions
            centers_np = centers_t.squeeze().cpu().numpy()
            sizes_np = sizes_t.squeeze().cpu().numpy()
            rot_matrices_np = rot_matrices_t.squeeze().cpu().numpy()
            
            # Return as dictionaries for compatibility
            return (
                {'left': centers_np[0], 'right': centers_np[1]},
                {'left': sizes_np[0], 'right': sizes_np[1]},
                {'left': rot_matrices_np[0], 'right': rot_matrices_np[1]}
            )
        
        def calculate_box_corners(center, size, rotation_matrix):
            """
            center: numpy array of shape (3,)
            size: numpy array of shape (3,) - [width (x), height (y), depth (z)]
            rotation_matrix: numpy array of shape (3, 3)
            Returns:
                corners: list of numpy arrays, each of shape (3,)
            """
            half_size = size / 2
            # Define the 8 corners relative to the center in local coordinates
            relative_corners = [
                np.array([dx, dy, dz])
                for dx in [-half_size[0], half_size[0]]
                for dy in [-half_size[1], half_size[1]]
                for dz in [-half_size[2], half_size[2]]
            ]
            # Apply rotation and translate to the center
            rotated_corners = [rotation_matrix @ corner + center for corner in relative_corners]
            return rotated_corners

        def plot_foot_boxes(gui, centers, sizes, rotation_matrices, root_rotation_matrix, root_translation):
            """
            gui: NimbleGUI instance
            centers: dict - {'left': numpy array (3,), 'right': numpy array (3,)}
            sizes: dict - {'left': numpy array (3,), 'right': numpy array (3,)}
            rotation_matrices: dict - {'left': numpy array (3, 3), 'right': numpy array (3, 3)}
            mode: str - 'world_frame' or 'root_frame'
            root_rotation_matrix: numpy array of shape (3, 3)
            root_translation: numpy array of shape (3,)
            """
            # Left foot
            left_center = centers['left']
            left_size = sizes['left']
            left_rotation_matrix = rotation_matrices['left']
            left_corners = calculate_box_corners(left_center, left_size, left_rotation_matrix)

            # Right foot
            right_center = centers['right']
            right_size = sizes['right']
            right_rotation_matrix = rotation_matrices['right']
            right_corners = calculate_box_corners(right_center, right_size, right_rotation_matrix)

            # Transform corners to world frame if necessary
            if mode == 'root_frame':
                pass  # Already in root frame, no transformation needed
            else:
                left_corners = [root_rotation_matrix @ corner + root_translation for corner in left_corners]
                right_corners = [root_rotation_matrix @ corner + root_translation for corner in right_corners]

            # Visualize boxes
            visualize_box(gui, 'left_box', left_corners, color=[0, 0.5, 0, 0.5])
            visualize_box(gui, 'right_box', right_corners, color=[0, 0.5, 0, 0.5])

        def visualize_box(gui, box_id, corners, color=[0, 1, 0, 1]):
            """
            gui: NimbleGUI instance
            box_id: unique identifier for the box
            corners: list of 8 numpy arrays, each of shape (3,)
            color: list of 4 floats - RGBA
            """
            # Define the 12 edges of the box by connecting corners
            edges = [
                (0, 1), (0, 2), (0, 4),
                (1, 3), (1, 5),
                (2, 3), (2, 6),
                (3, 7),
                (4, 5), (4, 6),
                (5, 7),
                (6, 7)
            ]
            
            for idx, (start, end) in enumerate(edges):
                line_name = f'{box_id}_edge_{idx}'
                gui.nativeAPI().createLine(
                    line_name,
                    [corners[start], corners[end]],
                    color  # RGBA color
                )

        def onTick(now):
            with torch.no_grad():
                nonlocal cached_inputs, cached_labels, cached_outputs
                nonlocal frame_index
                nonlocal playing
                nonlocal inputs
                nonlocal labels
                nonlocal outputs
                nonlocal skel
                nonlocal contact_bodies
                nonlocal total_frames_in_trial
                nonlocal plot_info
                nonlocal timesteps
                nonlocal trial_index
                nonlocal num_trials
                nonlocal model

                if playing:
                    frame_index += 1
                    if frame_index >= total_frames_in_trial:
                        # Move to the next trial
                        trial_index += 1
                        if trial_index >= num_trials:
                            trial_index = 0
                        load_trial(trial_index)

                        frame_index = 0

                # Get current frame data
                current_inputs = { key: cached_inputs[key][frame_index] for key in cached_inputs }
                current_labels = { key: cached_labels[key][frame_index] for key in cached_labels }

                # Visualization code
                # Set joint positions
                pos = current_inputs[InputDataKeys.POS]

                # Set CoM position
                if mode == 'root_frame':
                    com = current_inputs[InputDataKeys.comPosInRootFrame]
                    pos[0:6] = 0

                    joint_centers = current_inputs[InputDataKeys.JOINT_CENTERS_IN_ROOT_FRAME]
                    num_joints = int(len(joint_centers) / 3)
                    for j in range(num_joints):
                        gui.nativeAPI().createSphere('joint_' + str(j), [0.01, 0.01, 0.01], joint_centers[j * 3:(j + 1) * 3], [0.5, 0, 0, 0.5])

                    root_pos_history = current_inputs[InputDataKeys.ROOT_POS_HISTORY_IN_ROOT_FRAME]
                    num_history = int(len(root_pos_history) / 3)
                    for h in range(num_history):
                        gui.nativeAPI().createSphere('root_pos_history_' + str(h), [0.01, 0.01, 0.01], root_pos_history[h * 3:(h + 1) * 3], [0, 0.5, 0, 0.5])

                else:
                    # com_world = skel.getCOM()
                    com = current_inputs[InputDataKeys.comPOS]

                if viz_com:
                    gui.nativeAPI().createSphere('com', [0.025, 0.025, 0.025], com, [0, 0, 0, 0.5])

                skel.setPositions(pos)
                gui.nativeAPI().renderSkeleton(skel)

                # Get the world transform of the root body
                root_body = skel.getBodyNode("pelvis")  # Check if the root body name is 'pelvis'
                root_transform = root_body.getWorldTransform()
                root_rotation_matrix = root_transform.rotation()
                root_translation = root_transform.translation()


                # Visualize foot boxes
                if viz_footbox:
                    joint_centers = current_inputs[InputDataKeys.JOINT_CENTERS_IN_ROOT_FRAME]
                    centers, sizes, rotation_matrices = compute_foot_ranges(joint_centers)
                    plot_foot_boxes(gui, centers, sizes, rotation_matrices, root_rotation_matrix, root_translation)

                if model is not None:
                    current_outputs = { key: cached_outputs[key][frame_index] for key in cached_outputs }

                    # Convert CoP and forces from root frame to world frame
                    ground_forces_root = current_outputs[OutputDataKeys.GROUND_CONTACT_FORCES_IN_ROOT_FRAME]
                    cops_root = current_outputs[OutputDataKeys.GROUND_CONTACT_COPS_IN_ROOT_FRAME]

                    # Convert CoP and forces to world frame
                    ground_forces_world = []
                    cops_world = []

                    num_contacts = int(len(ground_forces_root) / 3)
                    for f in range(num_contacts):
                        if contact_bodies[f] == 'pelvis':
                            continue
                        cop_root = cops_root[f * 3:(f + 1) * 3]
                        force_root = ground_forces_root[f * 3:(f + 1) * 3]

                        # Convert from root frame to world frame
                        cop_world = root_rotation_matrix @ cop_root + root_translation
                        force_world = root_rotation_matrix @ force_root

                        cops_world.append(cop_world)
                        ground_forces_world.append(force_world)

                    # Visualize ground reaction forces and CoP (predicted values)
                    for idx, (cop_world, force_world) in enumerate(zip(cops_world, ground_forces_world)):
                        gui.nativeAPI().createLine('predicted_force_' + str(idx),
                                                [cop_world, cop_world + force_world/10],
                                                [0, 0, 1, 1])

                # Visualize actual ground reaction forces
                gt_forces_root = current_labels[OutputDataKeys.GROUND_CONTACT_FORCES_IN_ROOT_FRAME]
                gt_cops_root = current_labels[OutputDataKeys.GROUND_CONTACT_COPS_IN_ROOT_FRAME]

                gt_forces_world = []
                gt_cops_world = []
                
                num_contacts = int(len(gt_forces_root) / 3)
                for f in range(num_contacts):
                    if contact_bodies[f] == 'pelvis':
                        continue
                    cop_root = gt_cops_root[f * 3:(f + 1) * 3]
                    force_root = gt_forces_root[f * 3:(f + 1) * 3]

                    # Convert from root frame to world frame
                    cop_world = root_rotation_matrix @ cop_root + root_translation
                    force_world = root_rotation_matrix @ force_root

                    gt_cops_world.append(cop_world)
                    if np.linalg.norm(force_world) >= 1.0:
                        gt_forces_world.append(force_world)
                    else:
                        gt_forces_world.append(np.zeros(3))

                for idx, (cop_world, force_world) in enumerate(zip(gt_cops_world, gt_forces_world)):
                    gui.nativeAPI().createLine('force_' + str(idx),
                                               [cop_world, cop_world + force_world/10],
                                               [1, 0, 0, 1])
                
                # Update plots with current time indicator
                current_time = timesteps[frame_index]
                for plot_key, info in plot_info.items():
                    y_min = info['y_min']
                    y_max = info['y_max']
                    gui.nativeAPI().setRichPlotData(plot_key, 'Time', 'black', 'line', [current_time, current_time], [y_min, y_max])
                    
        ticker.registerTickListener(onTick)
        ticker.start()
        # Do not exit while the server is running
        gui.blockWhileServing()
        return True
    

def plot_load_balancing(gui, tokens_per_expert, expert_scores, args):
    """
    Function to visualize the number of tokens and expert scores for each expert per layer on a single plot using Nimble GUI.

    Args:
        gui (nimble.NimbleGUI): Nimble GUI instance.
        tokens_per_expert (list of torch.Tensor): List of token counts per expert for each layer.
        expert_scores (list of torch.Tensor): List of expert scores for each layer.
        args (Arguments): Arguments object (containing moe_num_experts, etc.).
    """
    num_layers = args.num_layers
    num_experts = args.moe_num_experts
    experts = list(range(num_experts))

    # Define color palette for layers
    color_palette = ['blue', 'green', 'red', 'purple', 'orange', 'cyan', 'magenta', 'yellow']
    
    # Position the plot in the top-right corner
    plot_width = 600
    plot_height = 400
    margin_x = 1920 - plot_width - 50
    margin_y = 1080 - plot_height - 50
    # Adjust the values below if you know the total size of the GUI.
    fromTopLeft = np.array([margin_x, margin_y], dtype=np.int32)  
    
    # Create a single RichPlot
    gui.nativeAPI().createRichPlot(
        key='load_balancing_plot',
        fromTopLeft=fromTopLeft,
        size=np.array([plot_width, plot_height], dtype=np.int32),
        minX=0,
        maxX=num_experts - 1,
        minY=0,
        maxY=1,  # Normalize data
        title='Load Balancing Across Layers',
        xAxisLabel='Expert',
        yAxisLabel='Normalized Value',
        layer='default'
    )

    # Add data for each layer
    for layer_idx in range(num_layers):
        color = color_palette[layer_idx % len(color_palette)]
        layer_number = layer_idx + 1

        # Tokens per Expert
        tokens = tokens_per_expert[layer_idx].float().cpu().numpy()  # Shape: (num_experts,)
        tokens_normalized = tokens / np.max(tokens) if np.max(tokens) > 0 else tokens

        gui.nativeAPI().setRichPlotData(
            key='load_balancing_plot',
            name=f'Layer {layer_number} Tokens',
            color=color,
            plotType='bar',
            xs=list(experts),
            ys=list(tokens_normalized)
        )

        # Expert Scores
        scores = expert_scores[layer_idx].float()
        scores_mean = scores.mean(dim=0).cpu().numpy() if scores.shape[0] != 0 else scores.sum(dim=0).cpu().numpy()
        scores_normalized = scores_mean / np.max(scores_mean) if np.max(scores_mean) > 0 else scores_mean

        gui.nativeAPI().setRichPlotData(
            key='load_balancing_plot',
            name=f'Layer {layer_number} Scores',
            color=color,
            plotType='line',
            xs=list(experts),
            ys=list(scores_normalized)
        )
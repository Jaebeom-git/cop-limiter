import nimblephysics as nimble
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Tuple, Optional
import os
import numpy as np
from collections import defaultdict
from models.CoPLimiter import CoPLimiter
from scipy.spatial.transform import Rotation

class InputDataKeys:
    # timeSteps = 'getTrialTimestep'

    comPOS = 'comPos'
    comVEL = 'comVel'
    comACC = 'comAcc'
    comPosInRootFrame = 'comPosInRootFrame'
    comVelInRootFrame = 'comVelInRootFrame'
    comAccInRootFrame = 'comAccInRootFrame'
    # COM_ACC_IN_ROOT_FRAME = 'comAccInRootFrame'   # same as comAccInRootFrame

    # These are the joint quantities for the joints that we are observing
    POS = 'pos'
    VEL = 'vel'
    ACC = 'acc'

    # The location of the joint centers, in the root frame
    JOINT_CENTERS_IN_ROOT_FRAME = 'jointCentersInRootFrame'

    # Root velocity and acceleration, in the root frame
    ROOT_LINEAR_VEL_IN_ROOT_FRAME = 'rootLinearVelInRootFrame'
    ROOT_ANGULAR_VEL_IN_ROOT_FRAME = 'rootAngularVelInRootFrame'
    ROOT_LINEAR_ACC_IN_ROOT_FRAME = 'rootLinearAccInRootFrame'
    ROOT_ANGULAR_ACC_IN_ROOT_FRAME = 'rootAngularAccInRootFrame'

    # Recent history of the root position and orientation, in the root frame
    ROOT_POS_HISTORY_IN_ROOT_FRAME = 'rootPosHistoryInRootFrame'
    ROOT_EULER_HISTORY_IN_ROOT_FRAME = 'rootEulerHistoryInRootFrame'


class OutputDataKeys:
    TAU = 'tau'

    # These are enough to compute ID
    
    # GROUND_CONTACT_WRENCHES_IN_ROOT_FRAME = 'groundContactWrenchesInRootFrame'
    # RESIDUAL_WRENCH_IN_ROOT_FRAME = 'residualWrenchInRootFrame'

    # These are various other things we might want to predict
    CONTACT = 'contact'
    # COM_ACC_IN_ROOT_FRAME = 'comAccInRootFrame'
    GROUND_CONTACT_COPS_IN_ROOT_FRAME = 'groundContactCenterOfPressureInRootFrame'
    GROUND_CONTACT_TORQUES_IN_ROOT_FRAME = 'groundContactTorqueInRootFrame'
    GROUND_CONTACT_FORCES_IN_ROOT_FRAME = 'groundContactForceInRootFrame'


class AddBiomechanicsDataset(Dataset):
    stride: int
    data_path: str
    window_size: int
    geometry_folder: str
    device: torch.device
    dtype: torch.dtype
    subject_paths: List[str]
    subjects: List[nimble.biomechanics.SubjectOnDisk]
    windows: List[Tuple[int, int, int]]  # Subject, trial, start_frame
    num_dofs: int
    num_joints: int
    contact_bodies: List[str]
    # For each subject, we store the skeleton and the contact bodies in memory, so they're ready to use with Nimble
    skeletons: List[nimble.dynamics.Skeleton]
    skeletons_contact_bodies: List[List[nimble.dynamics.BodyNode]]
    subject_indices: Dict[str, int]

    def __init__(self,
                 data_path: str,
                 window_size: int,
                 geometry_folder: str,
                 device: torch.device = torch.device('cpu'),
                 dtype: torch.dtype = torch.float32,
                 testing_with_short_dataset: bool = False,
                 stride: int = 1,
                #  output_data_format: str = 'last_frame',
                 skip_loading_skeletons: bool = False,
                 window_stride: int = 1,
                 unbalanced_stride: bool = False,
                 mode: str = 'train',
                 window_missing_threshold: float = 0.1,
                 trial_filter: Optional[str] = None,
                 ):
        self.stride = stride
        # self.output_data_format = output_data_format
        self.subject_paths = []
        self.subjects = []
        self.window_size = window_size
        self.geometry_folder = geometry_folder
        self.device = device
        self.dtype = dtype
        self.windows = []
        self.motion_classes = []
        self.contact_bodies = []
        self.skeletons = []
        self.skeletons_contact_bodies = []
        self.window_stride = window_stride
        self.skip_loading_skeletons = skip_loading_skeletons
        self.mode = mode
        self.window_missing_threshold = window_missing_threshold
        self.trial_filter = trial_filter

        # if os.path.isdir(data_path):
        #     for root, dirs, files in os.walk(data_path):
        #         for file in files:
        #             if file.endswith(".b3d") and "vander" not in file.lower():
        #                 self.subject_paths.append(os.path.join(root, file))
        # else:
        #     assert data_path.endswith(".b3d")
        #     # self.subject_paths.append(data_path)

        self.cop_limiter = CoPLimiter(
            length_ratio=2.0, 
            height_ratio=2.5, 
            width_ratio=0.75, 
            activation_function='clip' 
        )
        self.force_norm_threshold = 1.0  # Check CoP distance difference only when force is above 1N
        self.cop_distance_threshold = 0.01  # Invalid if CoP distance is above 0.01m

        if os.path.isdir(data_path):
            for root, dirs, files in os.walk(data_path):
                for file in files:
                    if file.endswith(".b3d") and "vander" not in file.lower():
                        self.subject_paths.append(os.path.join(root, file))
        else:
            # Default handling: when the path is a file
            assert data_path.endswith(".b3d")
            self.subject_paths.append(data_path)

        if testing_with_short_dataset:
            self.subject_paths = self.subject_paths[11:12]
        self.subject_indices = {subject_path: i for i, subject_path in enumerate(self.subject_paths)}

        # Walk the folder path, and check for any with the ".b3d" extension (indicating that they are
        # AddBiomechanics binary data files)
        if len(self.subject_paths) > 0:
            # Create a subject object for each file. This will load just the header from this file, and keep that
            # around in memory
            valid_subject_paths = []
            for i, subject_path in enumerate(self.subject_paths):
                try:
                    subject = nimble.biomechanics.SubjectOnDisk(subject_path)
                    valid_subject_paths.append(subject_path)
                except Exception as e:
                    print(f"Error reading {subject_path}: Skipping.")
                    continue
            self.subject_paths = valid_subject_paths

            subject = nimble.biomechanics.SubjectOnDisk(
                self.subject_paths[0])
            # Get the number of degrees of freedom for this subject
            self.num_dofs = subject.getNumDofs()
            # Get the number of joints for this subject
            self.num_joints = subject.getNumJoints()
            DEFAULT_NUM_JOINTS = 12
            if self.num_joints == 0:
                self.num_joints = DEFAULT_NUM_JOINTS
                print(f"No joints found. Using default value of {self.num_joints} joints.")
                
            # Get the contact bodies for this subject, and put them into a consistent order for the dataset
            contact_bodies = subject.getGroundForceBodies()
            for body in contact_bodies:
                if body == 'pelvis':
                    continue
                if body not in self.contact_bodies:
                    self.contact_bodies.append(body)
        
        # Sort the contact bodies so that they are always in the same order
        # calcn_l, calcn_r
        self.contact_bodies.sort()

        if unbalanced_stride and self.mode != 'infer' and self.mode != 'test':
            motion_counts = defaultdict(int)             

            for i, subject_path in enumerate(self.subject_paths):
                # Add the skeleton to the list of skeletons
                subject = nimble.biomechanics.SubjectOnDisk(subject_path)
                if not self.skip_loading_skeletons:
                    print('Loading skeleton ' + str(i + 1) + '/' + str(
                        len(self.subject_paths)) + f' for subject {subject_path}')
                # Prepare the list of windows we can use for training
                for trial_index in range(subject.getNumTrials()):
                    trial_length = subject.getTrialLength(trial_index)
                    trial_name = subject.getTrialName(trial_index)
                    probably_missing: List[bool] = [reason != nimble.biomechanics.MissingGRFReason.notMissingGRF for reason
                                                    in subject.getMissingGRF(trial_index)]
                    
                    contact_indices: List[int] = [
                        subject.getGroundForceBodies().index(body) if body in subject.getGroundForceBodies() else -1 
                        for body in self.contact_bodies
                    ]
                    cop_invalid = self.get_invalid_cop_flags_by_foot_bounds(subject, trial_index, contact_indices)

                    for window_start in range(0, max(trial_length - self.window_size - 1, 0), self.window_stride):
                        window_missing = probably_missing[window_start:window_start + self.window_size:self.stride]
                        window_cop_invalid = cop_invalid[window_start:window_start + self.window_size:self.stride]
                            
                        window_length = len(window_missing)
                        # if (sum(window_missing) + sum(window_cop_invalid)) < window_length * self.window_missing_threshold:
                        if sum(window_missing) < window_length * self.window_missing_threshold and not any(window_cop_invalid):
                        # if not any(window_missing) and not any(window_cop_invalid):
                            motion_class = classify_motion(subject_path, trial_name)
                            if motion_class == "bad":
                                continue
                            motion_counts[motion_class] += 1

            total_trials = sum(motion_counts.values())
            motion_ratios = {k: v / total_trials for k, v in motion_counts.items()}

            def scale_stride(ratio, min_r, max_r, window_size):
                # window size ratio
                min_ratio = 0.1 # 10% of window size
                max_ratio = 0.5 # 50% of window size
                scaled = ((ratio - min_r) / (max_r - min_r)) * (max_ratio - min_ratio) * window_size + min_ratio * window_size
                return int(scaled)
            
            unbalance_window_strides = {
                k: scale_stride(v, min(motion_ratios.values()), max(motion_ratios.values()), self.window_size) 
                for k, v in motion_ratios.items()
            }
        else:
            unbalance_window_strides = None

        for i, subject_path in enumerate(self.subject_paths):
            # Add the skeleton to the list of skeletons
            subject = nimble.biomechanics.SubjectOnDisk(subject_path)
            if not self.skip_loading_skeletons:
                print('Loading skeleton ' + str(i + 1) + '/' + str(
                    len(self.subject_paths)) + f' for subject {subject_path}')
                skeleton = subject.readSkel(subject.getNumProcessingPasses() - 1, geometry_folder)
                self.skeletons.append(skeleton)
                self.skeletons_contact_bodies.append([skeleton.getBodyNode(body) for body in self.contact_bodies])
            self.subjects.append(subject)
            # Prepare the list of windows we can use for training
            for trial_index in range(subject.getNumTrials()):
                trial_length = subject.getTrialLength(trial_index)
                trial_name = subject.getTrialName(trial_index)

                # Filter by trial name if specified
                if self.trial_filter is not None and self.trial_filter not in trial_name.lower():
                    continue

                if self.mode == 'infer':  # Use all frames
                    self.windows.append((i, trial_index, 0))
                
                # For AWEAR dataset
                elif self.mode == 'test':
                    if "walk" in trial_name.lower() or "tug" in trial_name.lower() or "static" in trial_name.lower():
                        continue
                    if "000" in subject_path.lower() and "step" in trial_name.lower():
                        continue
                    if "007" in subject_path.lower() and "ftsts" in trial_name.lower():
                        continue
                    if "010" in subject_path.lower() and "ftsts" in trial_name.lower():
                        continue
                    if "028" in subject_path.lower() and "pick" in trial_name.lower():
                        continue
                    contact_indices: List[int] = [
                        subject.getGroundForceBodies().index(body) if body in subject.getGroundForceBodies() else -1 
                        for body in self.contact_bodies
                    ]
                    for window_start in range(0, max(trial_length - self.window_size - 1, 0), self.window_stride):
                        try:
                            assert window_start + self.window_size < trial_length
                            self.windows.append((i, trial_index, window_start))
                        except AssertionError:
                            print(f"Window start {window_start} + window size {self.window_size} exceeds trial length {trial_length} for Subject {subject_path}, Trial {trial_index}. Skipping window.")
                            continue

                else:
                    probably_missing: List[bool] = [reason != nimble.biomechanics.MissingGRFReason.notMissingGRF for reason
                                                    in subject.getMissingGRF(trial_index)]
                    ######################################################################

                    contact_indices: List[int] = [
                        subject.getGroundForceBodies().index(body) if body in subject.getGroundForceBodies() else -1 
                        for body in self.contact_bodies
                    ]
                    
                    ## Unbalance window stride
                    motion_class = classify_motion(subject_path, trial_name)
                    if motion_class == "bad":
                        continue
                    if unbalance_window_strides is not None:
                        unbalance_window_stride = unbalance_window_strides.get(motion_class, int(self.window_size * 0.1))
                    else:
                        unbalance_window_stride = self.window_stride

                    # Pre-calculate CoP validity
                    cop_invalid = self.get_invalid_cop_flags_by_foot_bounds(subject, trial_index, contact_indices)

                    for window_start in range(0, max(trial_length - self.window_size - 1, 0), unbalance_window_stride):
                        window_missing = probably_missing[window_start:window_start + self.window_size:self.stride]
                        window_cop_invalid = cop_invalid[window_start:window_start + self.window_size:self.stride]
                            
                        window_length = len(window_missing)
                        # if (sum(window_missing) + sum(window_cop_invalid)) < window_length * self.window_missing_threshold:
                        if sum(window_missing) < window_length * self.window_missing_threshold and not any(window_cop_invalid):
                        # if not any(window_missing) and not any(window_cop_invalid):
                            try:
                                assert window_start + self.window_size < trial_length
                                self.windows.append((i, trial_index, window_start))
                                self.motion_classes.append(motion_class)
                                # # for test
                                # if len(self.windows) > 100:
                                #     return
                            except AssertionError:
                                print(f"Window start {window_start} + window size {self.window_size} exceeds trial length {trial_length} for Subject {subject_path}, Trial {trial_index}. Skipping window.")
                                continue

                    ######################################################################
                    # for window_start in range(0, max(trial_length - self.window_size - 1, 0), self.window_stride):      # stride = window_size//4: 4x overlap
                    #     if not any(probably_missing[window_start:window_start + self.window_size:self.stride]):
                    #         assert window_start + self.window_size < trial_length
                    #         self.windows.append((i, trial_index, window_start))

    def get_invalid_cop_flags_by_foot_bounds(self, subject: nimble.biomechanics.SubjectOnDisk, trial_index: int, contact_indices: List[int]) -> List[bool]:
        """
        Determines whether the Center of Pressure (CoP) for each frame in a specific trial 
        is within the anatomical boundaries of the subject's feet.
        
        Additional Conditions:
            - Only evaluates CoP distance differences when the force norm is above (self.force_norm_threshold).
            - CoP distance differences exceeding (self.cop_distance_threshold)m are considered invalid.
            - Validates only the active CoP based on `contact_indices`.
            - Frames with foot length (`sizes[:, :, :, 0]`) <= 0.3m are marked invalid.

        Args:
            subject (nimble.biomechanics.SubjectOnDisk): The subject being analyzed.
            trial_index (int): The index of the trial to analyze.
            contact_indices (List[int]): Indices indicating active CoP for each contact body. 
                                        A value of -1 denotes an inactive contact body.

        Returns:
            List[bool]: A list where each element corresponds to a frame. `True` indicates 
                        the frame is invalid, and `False` otherwise.
        """
        trial_length = subject.getTrialLength(trial_index)
        frames = subject.readFrames(trial_index, 0, trial_length, includeSensorData=False, includeProcessingPasses=True)
        
        frame_passes: List[nimble.biomechanics.FramePass] = [frame.processingPasses[0] for frame in frames]

        joint_centers = torch.stack([
            torch.tensor(p.jointCentersInRootFrame, dtype=self.dtype) for p in frame_passes
        ]).unsqueeze(0)  # Shape: [1, T, joints, 3]
        
        cop = torch.stack([
            torch.tensor(p.groundContactCenterOfPressureInRootFrame, dtype=self.dtype) for p in frame_passes
        ])  # Shape: [T, 3 * contact_bodies]
        
        ground_contact_forces = torch.stack([
            torch.tensor(p.groundContactForceInRootFrame, dtype=self.dtype) for p in frame_passes
        ])  # Shape: [T, 3 * contact_bodies]
        
        B, T, _ = joint_centers.shape
        num_contact_bodies = len(self.contact_bodies)
        
        # Initialize on CPU
        cop_temp = torch.zeros([T, 3 * num_contact_bodies], dtype=self.dtype) 
        ground_contact_forces_temp = torch.zeros([T, 3 * num_contact_bodies], dtype=self.dtype) 

        for i in range(num_contact_bodies):
            if contact_indices[i] >= 0:
                cop_temp[:, 3 * i:3 * i + 3] = cop[:, 3 * contact_indices[i]:3 * contact_indices[i] + 3]
                ground_contact_forces_temp[:, 3 * i:3 * i + 3] = ground_contact_forces[:, 3 * contact_indices[i]:3 * contact_indices[i] + 3]

        # Add batch dimension (B=1)
        cop_temp = cop_temp.unsqueeze(0)  # Shape: [1, T, 3 * contact_bodies]
        ground_contact_forces_temp = ground_contact_forces_temp.unsqueeze(0)  # Shape: [1, T, 3 * contact_bodies]

        # Reshape for two feet (assuming 2 contact points)
        cop_temp = cop_temp.view(B, T, 2, 3)  # Shape: [1, T, 2, 3]
        ground_contact_forces_temp = ground_contact_forces_temp.view(B, T, 2, 3)  # Shape: [1, T, 2, 3]

        # Compute centers, sizes, rotation matrices for both feet
        centers, sizes, rotation_matrices = self.cop_limiter.compute_feet(joint_centers)  # [1, T, 2, 3], [1, T, 2, 3], [1, T, 2, 3, 3]
        
        # Check if sizes are valid (length > 0.3)
        size_length_invalid = sizes[:, :, :, 0] <= 0.3  # Shape: [1, T, 2]

        # Limit CoP
        limited_cop = self.cop_limiter.limit_cop(cop_temp, centers, sizes, rotation_matrices)  # [1, T, 2, 3]
        
        # Compute distance between original and limited CoP
        distance = torch.norm(cop_temp - limited_cop, dim=-1)  # [1, T, 2]
        
        # Compute force norms
        force_norms = torch.norm(ground_contact_forces_temp, dim=-1)  # [1, T, 2]
        
        # Determine validity: if force_norm > threshold and distance > threshold, then invalid
        invalid = (force_norms > self.force_norm_threshold) & (distance > self.cop_distance_threshold)  # [1, T, 2]
        invalid = invalid | size_length_invalid
        
        # For each frame, if any foot is invalid, mark the frame as invalid
        frame_invalid = torch.any(invalid, dim=-1)  # Shape: [1, T]
        
        # Convert to list of bools and return
        invalid_flags = frame_invalid.squeeze(0).cpu().tolist()  # [T]
        
        return invalid_flags
    

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index: int) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], int, int]:
        subject_index, trial, window_start = self.windows[index]

        # Read the frames from disk
        subject = self.subjects[subject_index]
        if self.mode == 'infer':
            frames: nimble.biomechanics.FrameList = subject.readFrames(trial, 
                                                                    0, 
                                                                    subject.getTrialLength(trial),
                                                                    includeSensorData=False,
                                                                    includeProcessingPasses=True
                                                                    )
        else:
            frames: nimble.biomechanics.FrameList = subject.readFrames(trial,
                                                                    window_start,
                                                                    self.window_size // self.stride,
                                                                    stride=self.stride,
                                                                    includeSensorData=False,
                                                                    includeProcessingPasses=True)
            assert (len(frames) == self.window_size // self.stride)

        first_passes: List[nimble.biomechanics.FramePass] = [frame.processingPasses[0] for frame in frames]
        output_passes: List[nimble.biomechanics.FramePass] = [frame.processingPasses[-1] for frame in frames]

        input_dict: Dict[str, torch.Tensor] = {}
        label_dict: Dict[str, torch.Tensor] = {}

        with torch.no_grad():
            input_dict[InputDataKeys.comPOS] = torch.row_stack([
                torch.tensor(p.comPos, dtype=self.dtype).detach() for p in first_passes
            ])
            input_dict[InputDataKeys.comVEL] = torch.row_stack([
                torch.tensor(p.comVel, dtype=self.dtype).detach() for p in first_passes
            ])
            input_dict[InputDataKeys.comACC] = torch.row_stack([
                torch.tensor(p.comAcc, dtype=self.dtype).detach() for p in first_passes
            ])
            input_dict[InputDataKeys.POS] = torch.row_stack([
                torch.tensor(p.pos, dtype=self.dtype).detach() for p in first_passes
            ])
            input_dict[InputDataKeys.VEL] = torch.row_stack([
                torch.tensor(p.vel, dtype=self.dtype).detach() for p in first_passes
            ])
            input_dict[InputDataKeys.ACC] = torch.row_stack([
                torch.tensor(p.acc, dtype=self.dtype).detach() for p in first_passes
            ])
            input_dict[InputDataKeys.JOINT_CENTERS_IN_ROOT_FRAME] = torch.row_stack([
                torch.tensor(p.jointCentersInRootFrame, dtype=self.dtype).detach() for p in first_passes
            ])
            input_dict[InputDataKeys.ROOT_LINEAR_VEL_IN_ROOT_FRAME] = torch.row_stack([
                torch.tensor(p.rootLinearVelInRootFrame, dtype=self.dtype).detach() for p in first_passes
            ])
            input_dict[InputDataKeys.ROOT_LINEAR_ACC_IN_ROOT_FRAME] = torch.row_stack([
                torch.tensor(p.rootLinearAccInRootFrame, dtype=self.dtype).detach() for p in first_passes
            ])
            input_dict[InputDataKeys.ROOT_ANGULAR_VEL_IN_ROOT_FRAME] = torch.row_stack([
                torch.tensor(p.rootAngularVelInRootFrame, dtype=self.dtype).detach() for p in first_passes
            ])
            input_dict[InputDataKeys.ROOT_ANGULAR_ACC_IN_ROOT_FRAME] = torch.row_stack([
                torch.tensor(p.rootAngularAccInRootFrame, dtype=self.dtype).detach() for p in first_passes
            ])
            input_dict[InputDataKeys.ROOT_POS_HISTORY_IN_ROOT_FRAME] = torch.row_stack([
                torch.tensor(p.rootPosHistoryInRootFrame, dtype=self.dtype).detach() for p in first_passes
            ])
            input_dict[InputDataKeys.ROOT_EULER_HISTORY_IN_ROOT_FRAME] = torch.row_stack([
                torch.tensor(p.rootEulerHistoryInRootFrame, dtype=self.dtype).detach() for p in first_passes
            ])
            input_dict[InputDataKeys.comAccInRootFrame] = torch.row_stack([
                torch.tensor(p.comAccInRootFrame, dtype=self.dtype).detach() for p in first_passes
            ])

            # Convert CoM position, velocity, and acceleration to the root coordinate system
            R_rw = R_world2root(input_dict[InputDataKeys.POS])
            input_dict[InputDataKeys.comPosInRootFrame] = get_comPosInRootFrame(
                R_rw, 
                input_dict[InputDataKeys.POS][:, 3:6], 
                input_dict[InputDataKeys.comPOS]
                )
            
            input_dict[InputDataKeys.comVelInRootFrame] = get_comVelInRootFrame(
                R_rw,
                input_dict[InputDataKeys.ROOT_LINEAR_VEL_IN_ROOT_FRAME],
                input_dict[InputDataKeys.comVEL],
                input_dict[InputDataKeys.ROOT_ANGULAR_VEL_IN_ROOT_FRAME],
                input_dict[InputDataKeys.comPosInRootFrame]
            )

            # input_dict[InputDataKeys.comAccInRootFrame] = get_comAccInRootFrame(
            #     R_rw,
            #     input_dict[InputDataKeys.ROOT_LINEAR_ACC_IN_ROOT_FRAME],
            #     input_dict[InputDataKeys.comACC],
            #     input_dict[InputDataKeys.ROOT_ANGULAR_VEL_IN_ROOT_FRAME],
            #     input_dict[InputDataKeys.ROOT_ANGULAR_ACC_IN_ROOT_FRAME],
            #     input_dict[InputDataKeys.comPosInRootFrame],
            #     input_dict[InputDataKeys.comVelInRootFrame]
            # )

            # The output dictionary contains a single frame, the last frame in the window if output_data_format is 2d
            # else it contains outputs for all the frames in first_passes
            mass = subject.getMassKg()
            # start_index = 0 if self.output_data_format == 'all_frames' else -1
            start_index = 0
            label_dict[OutputDataKeys.TAU] = torch.row_stack([
                torch.tensor(p.tau, dtype=self.dtype).detach() for p in output_passes[start_index:]
            ])
            # label_dict[OutputDataKeys.RESIDUAL_WRENCH_IN_ROOT_FRAME] = torch.row_stack([torch.tensor(p.residualWrenchInRootFrame, dtype=self.dtype).detach() for p in output_passes[start_index:]])
            # label_dict[OutputDataKeys.COM_ACC_IN_ROOT_FRAME] = torch.row_stack([torch.tensor(p.comAccInRootFrame, dtype=self.dtype).detach() for p in output_passes[start_index:]])
            # label_dict[OutputDataKeys.GROUND_CONTACT_WRENCHES_IN_ROOT_FRAME] = torch.zeros((len(output_passes) if start_index != -1 else 1, 6 * len(self.contact_bodies)), dtype=self.dtype)
            label_dict[OutputDataKeys.GROUND_CONTACT_COPS_IN_ROOT_FRAME] = torch.zeros((len(output_passes) if start_index != -1 else 1, 3 * len(self.contact_bodies)), dtype=self.dtype)
            label_dict[OutputDataKeys.GROUND_CONTACT_TORQUES_IN_ROOT_FRAME] = torch.zeros((len(output_passes) if start_index != -1 else 1, 3 * len(self.contact_bodies)), dtype=self.dtype)
            label_dict[OutputDataKeys.GROUND_CONTACT_FORCES_IN_ROOT_FRAME] = torch.zeros((len(output_passes) if start_index != -1 else 1, 3 * len(self.contact_bodies)), dtype=self.dtype)
            contact_indices: List[int] = [
                subject.getGroundForceBodies().index(body) if body in subject.getGroundForceBodies() else -1 for
                body in self.contact_bodies]
            # ground_contact_wrenches_in_root_frame: torch.Tensor = torch.row_stack([torch.tensor(p.groundContactWrenchesInRootFrame, dtype=self.dtype) for p in first_passes[start_index:]])
            ground_contact_forces_in_root_frame: torch.Tensor = torch.row_stack([torch.tensor(p.groundContactForceInRootFrame, dtype=self.dtype) for p in first_passes[start_index:]])
            ground_contact_cop_in_root_frame: torch.Tensor = torch.row_stack([torch.tensor(p.groundContactCenterOfPressureInRootFrame, dtype=self.dtype) for p in first_passes[start_index:]])
            ground_contact_torque_in_root_frame: torch.Tensor = torch.row_stack([torch.tensor(p.groundContactTorqueInRootFrame, dtype=self.dtype) for p in first_passes[start_index:]])
            for i in range(len(self.contact_bodies)):
                if contact_indices[i] >= 0:
                    # label_dict[OutputDataKeys.GROUND_CONTACT_WRENCHES_IN_ROOT_FRAME][:, 6 * i:6 * i + 6] = ground_contact_wrenches_in_root_frame[:, 6 * contact_indices[i]:6 * contact_indices[i] + 6] / mass
                    label_dict[OutputDataKeys.GROUND_CONTACT_COPS_IN_ROOT_FRAME][:, 3 * i:3 * i + 3] = ground_contact_cop_in_root_frame[:, 3 * contact_indices[i]:3 * contact_indices[i] + 3]
                    label_dict[OutputDataKeys.GROUND_CONTACT_TORQUES_IN_ROOT_FRAME][:, 3 * i:3 * i + 3] = ground_contact_torque_in_root_frame[:, 3 * contact_indices[i]:3 * contact_indices[i] + 3] / mass
                    label_dict[OutputDataKeys.GROUND_CONTACT_FORCES_IN_ROOT_FRAME][:,3 * i:3 * i + 3] = ground_contact_forces_in_root_frame[:, 3 * contact_indices[i]:3 * contact_indices[i] + 3] / mass

        # trial_timestep = round(subject.getTrialTimestep(trial), 8)  # in seconds
        # time_steps = torch.full((input_dict[InputDataKeys.POS].shape[0], 1), trial_timestep, dtype=self.dtype)
        # input_dict[InputDataKeys.timeSteps] = time_steps

        # Convert the frames to a dictionary of matrices, where columns are timesteps and rows are degrees of freedom / dimensions
        # (the DataLoader will then convert this to a batched tensor)

        # print(f"{numpy_output_dict[OutputDataKeys.CONTACT_FORCES]=}")
        # ###################################################
        # # Plotting
        # import matplotlib.pyplot as plt
        # x = np.arange(self.window_size)
        # # plotting each row
        # for i in range(len(self.input_dofs)):
        #     # plt.plot(x, numpy_input_dict[InputDataKeys.POS][i, :], label='pos_'+self.input_dofs[i])
        #     plt.plot(x, numpy_input_dict[InputDataKeys.VEL][i, :], label='vel_' + self.input_dofs[i])
        #     plt.plot(x, numpy_input_dict[InputDataKeys.ACC][i, :], label='acc_' + self.input_dofs[i])
        # for i in range(3):
        #     plt.plot(x, numpy_input_dict[InputDataKeys.COM_ACC][i, :], label='com_acc_' + str(i))
        # # Add the legend outside the plot
        # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        # plt.show()
        # ###################################################

        # Return the input and output dictionaries at this timestep, as well as the skeleton pointer

        return input_dict, label_dict, subject_index, trial

    def __getstate__(self):
        state = self.__dict__.copy()
        # Remove the unpicklable entries.
        del state['subjects']
        del state['skeletons']
        del state['skeletons_contact_bodies']
        return state

    def __setstate__(self, state):
        # Restore instance attributes.
        self.__dict__.update(state)
        self.subjects = []
        print('Unpickling AddBiomechanicsDataset copy in reader worker thread')
        # Create the non picklable SubjectOnDisk objects. Skip loading the skeletons and contact bodies, since these
        # are not used in the reader worker threads.
        for i, subject_path in enumerate(self.subject_paths):
            self.subjects.append(nimble.biomechanics.SubjectOnDisk(subject_path))


def classify_motion(subj_path: str, trial_name: str) -> str:
    name_lower = trial_name.lower()

    if "Fregly2012" in subj_path:
        if "walk" in name_lower:
            return "gait"
        else:
            return "standing"

    if "Moore2015" in subj_path:
        if "segment_1" in name_lower:
            return "standing"
        else:
            return "gait"
    
    if "Santos2017" in subj_path:
        return "standing"
    
    if "Tan2021" in subj_path:
        if "hip_calibration" in trial_name:
            return "other"
        # elif "nike_SR_24" in trial_name and "s3" in subj_path:  # It might be Swap Left/Right Acromion
        #     return "bad"
        # elif ("mini_SR_28" in trial_name or "mini_baseline_24" in trial_name) and "s4" in subj_path:    # It might be Swap Left/Right Acromion
        #     return "bad"
        return "gait"
    if "Tan2022" in subj_path:
        return "gait"
    
    if "Tiziana2019" in subj_path:
        return "gait"
    
    if "vanderZee2022" in subj_path:
        if "standing" in name_lower:
            return "standing"
        else:
            return "gait"
    
    if any(keyword in name_lower for keyword in ["gait", "walk", "run", "treadmill", "ground_"]):
        return "gait"
    elif any(keyword in name_lower for keyword in ["idling", "static", "stand"]):
        return "standing"
    elif any(keyword in name_lower for keyword in ["ramp"]):
        return "ramp"
    # elif any(keyword in name_lower for keyword in ["stair"]):
    #     return "stairs"
    # elif any(motion in trial_name for motion in ["sts"]):
    #     return "sts"
    # elif any(motion in trial_name for motion in ["chair", "_squat_"]):
    #     return "squat"
    else:
        return "other"

def R_world2root(pos: torch.Tensor) -> torch.Tensor:
    """
    Build the rotation matrix [world->root].
    We assume pos[:,0:3] are Euler angles (Z, X, Y) for 'root->world' rotation.
    Hence we compute R_wr via 'ZXY', then transpose to get R_rw (world->root).
    
    :param pos: shape [N, 6] 
                pos[:, :3] => [Euler_Z, Euler_X, Euler_Y]
                pos[:, 3:6] => root position in world frame
    :return: R_rw (shape [N, 3, 3]) which transforms a vector from world frame to root frame
    """
    # 1) Extract Euler angles Z, X, Y
    root_euler_zxy = pos[:, 0:3].cpu().numpy()  # shape [N, 3]

    # 2) R_wr: 'root->world'
    R_wr = Rotation.from_euler('ZXY', root_euler_zxy, degrees=False).as_matrix()  # [N, 3, 3]

    # 3) R_rw: 'world->root' by transpose
    R_rw = R_wr.transpose((0, 2, 1))
    R_rw_torch = torch.tensor(R_rw, dtype=pos.dtype, device=pos.device)
    return R_rw_torch.detach()

def get_comPosInRootFrame(
    R_rw: torch.Tensor,
    rootPosInWorld: torch.Tensor,
    comPosInWorld: torch.Tensor
) -> torch.Tensor:
    """
    rᵢⱼ = rᵢ + Aᵢsᵢⱼ
    Thus, sᵢⱼ = Aᵀᵢ(rᵢⱼ - rᵢ)

    Aᵀᵢ :param R_rw:           [N, 3, 3], rotation matrix world->root 
    rᵢ  :param rootPosInWorld: [N, 3], root position in world
    rᵢⱼ :param comPosInWorld:  [N, 3], CoM position in world
    :return:                   [N, 3], CoM position in root frame (sᵢⱼ)
    """
    # Transform the relative position (comPosInWorld - rootPosInWorld) to root frame
    comPosRoot = torch.einsum('bij,bj->bi', R_rw, comPosInWorld - rootPosInWorld)
    return comPosRoot.detach()

def get_comVelInRootFrame(
    R_rw: torch.Tensor,
    rootLinearVelInRootFrame: torch.Tensor,
    comVelInWorld: torch.Tensor,
    rootAngularVelInRootFrame: torch.Tensor,
    comPosInRootFrame: torch.Tensor
) -> torch.Tensor:
    """
    ṙᵢⱼ = ṙᵢ + Aᵢ(ωᵢ × sᵢⱼ) + Aᵢṡᵢⱼ
    Thus, ṡᵢⱼ = Aᵀᵢ(ṙᵢⱼ - ṙᵢ) - ωᵢ × sᵢⱼ = Aᵀᵢṙᵢⱼ - Aᵀᵢṙᵢ - ωᵢ × sᵢⱼ

    Aᵀᵢ  :param R_rw:                      [N, 3, 3], rotation matrix world->root 
    Aᵀᵢṙᵢ:param rootLinearVelInRootFrame:  [N, 3], root linear velocity in root frame
    ṙᵢⱼ  :param comVelInWorld:             [N, 3], CoM velocity in world
    ωᵢ   :param rootAngularVelInRootFrame: [N, 3], root angular velocity in root frame
    sᵢⱼ  :param comPosInRootFrame:         [N, 3], CoM position in root frame
    :return:                               [N, 3], CoM velocity in root frame (ṡᵢⱼ)
    """  
    # 1. Transform CoM velocity from world to root frame: Aᵀᵢṙᵢⱼ
    comVelInRoot = torch.einsum('bij,bj->bi', R_rw, comVelInWorld)
    
    # 2. Subtract root velocity in root frame: Aᵀᵢṙᵢⱼ - Aᵀᵢṙᵢ
    relVelInRoot = comVelInRoot - rootLinearVelInRootFrame
    
    # 3. Subtract angular velocity component: - ωᵢ × sᵢⱼ
    angVelComponent = torch.cross(rootAngularVelInRootFrame, comPosInRootFrame, dim=1)
    
    # 4. Calculate final velocity in root frame
    comVelRoot = relVelInRoot - angVelComponent
    
    return comVelRoot.detach()

def get_comAccInRootFrame(
    R_rw: torch.Tensor,
    rootLinearAccInRootFrame: torch.Tensor,
    comAccInWorld: torch.Tensor,
    rootAngularVelInRootFrame: torch.Tensor,
    rootAngularAccInRootFrame: torch.Tensor,
    comPosInRootFrame: torch.Tensor,
    comVelInRootFrame: torch.Tensor
) -> torch.Tensor:
    """
    r̈ᵢⱼ = r̈ᵢ + Aᵢ[ωᵢ × (ωᵢ × sᵢⱼ)] + Aᵢ(ω̇ᵢ × sᵢⱼ) + Aᵢ(2ωᵢ × ṡᵢⱼ) + Aᵢs̈ᵢⱼ
    Thus, s̈ᵢⱼ = Aᵀᵢ(r̈ᵢⱼ - r̈ᵢ) - ωᵢ × (ωᵢ × sᵢⱼ) - ω̇ᵢ × sᵢⱼ - 2ωᵢ × ṡᵢⱼ 
               = Aᵀᵢ(r̈ᵢⱼ) - Aᵀᵢ(r̈ᵢ) - ωᵢ × (ωᵢ × sᵢⱼ) - ω̇ᵢ × sᵢⱼ - 2ωᵢ × ṡᵢⱼ

    Aᵀᵢ    :param R_rw:                      [N, 3, 3], rotation matrix world->root
    AᵢT(r̈ᵢ):param rootLinearAccInRootFrame:  [N, 3], root linear acceleration in root frame
    r̈ᵢⱼ    :param comAccInWorld:             [N, 3], CoM acceleration in world
    ωᵢ     :param rootAngularVelInRootFrame: [N, 3], root angular velocity in root frame
    ω̇ᵢ     :param rootAngularAccInRootFrame: [N, 3], root angular acceleration in root frame
    sᵢⱼ    :param comPosInRootFrame:         [N, 3], CoM position in root frame
    ṡᵢⱼ    :param comVelInRootFrame:         [N, 3], CoM velocity in root frame
    :return:                                 [N, 3], CoM acceleration in root frame (s̈ᵢⱼ)
    """ 
    # 1. Transform CoM acceleration from world to root frame: AᵢT(r̈ᵢⱼ)
    comAccInRoot = torch.einsum('bij,bj->bi', R_rw, comAccInWorld)
    
    # 2. Subtract root acceleration in root frame: AᵢT(r̈ᵢⱼ) - AᵢT(r̈ᵢ)
    relAccInRoot = comAccInRoot - rootLinearAccInRootFrame
    
    # 3. Calculate centripetal acceleration: ωᵢ × (ωᵢ × sᵢⱼ)
    centripetalAcc = torch.cross(
        rootAngularVelInRootFrame, 
        torch.cross(rootAngularVelInRootFrame, comPosInRootFrame, dim=1),
        dim=1
    )
    
    # 4. Calculate angular acceleration component: ω̇ᵢ × sᵢⱼ
    angAccComponent = torch.cross(rootAngularAccInRootFrame, comPosInRootFrame, dim=1)
    
    # 5. Calculate Coriolis acceleration: 2ωᵢ × ṡᵢⱼ
    coriolisAcc = 2 * torch.cross(rootAngularVelInRootFrame, comVelInRootFrame, dim=1)
    
    # 6. Calculate final acceleration in root frame
    comAccRoot = relAccInRoot - centripetalAcc - angAccComponent - coriolisAcc
    
    return comAccRoot.detach()
import torch
import torch.nn as nn

class CoPLimiter(nn.Module):
    """
    CoPRangeLimiter constrains the Center of Pressure (CoP) within predefined foot boundaries
    based on the foot's local coordinate system. It calculates foot dimensions and rotation
    matrices to transform and limit the CoP predictions appropriately.
    """
    def __init__(self, length_ratio=2.0, height_ratio=2.5, width_ratio=0.75, activation_function='tanh'):
        """
        Initializes the CoPRangeLimiter module.

        Args:
            length_ratio (float): Scaling factor for the foot length.
            height_ratio (float): Scaling factor for the foot height.
            width_ratio (float): Scaling factor for the foot width.
            activation_function (str): Activation function to use ('tanh', 'sigmoid', 'clamp', 'softsign').
        """
        super(CoPLimiter, self).__init__()
        self.length_ratio = length_ratio
        self.height_ratio = height_ratio
        self.width_ratio = width_ratio
        self.ratio_foot_center = 0.75  # Ratio to determine the foot center between subtalar and MTP joints
        self.eps = 1e-8  # Small epsilon to prevent division by zero

        # Joint indices for left and right feet in the input tensor
        self.ankle_r_idx = 3
        self.subtalar_r_idx = 4
        self.mtp_r_idx = 5
        self.ankle_l_idx = 8
        self.subtalar_l_idx = 9
        self.mtp_l_idx = 10

        # Activation function for constraining CoP values
        self.activation_function = activation_function.lower()
        if self.activation_function == 'tanh':
            self.activation = torch.tanh
        elif self.activation_function == 'sigmoid':
            self.activation = lambda x: 2 * torch.sigmoid(x) - 1
        elif self.activation_function == 'clamp':
            self.activation = lambda x: torch.clamp(x, -1, 1)
        elif self.activation_function == 'softsign':
            self.activation = lambda x: x / (1 + torch.abs(x))
        elif self.activation_function == 'clip':
            self.activation = None
        else:
            raise ValueError("Invalid activation_function. Choose from 'tanh', 'sigmoid', 'clamp', 'softsign', 'clip'.")

    def compute_feet(self, joint_centers):
        """
        Computes the centers, sizes, and rotation matrices for both feet based on joint positions.

        Args:
            joint_centers (torch.Tensor): Tensor containing joint positions with shape [B, T, F],
                                         where B is batch size, T is sequence length,
                                         and F is the number of features (3 * number of joints).

        Returns:
            centers (torch.Tensor): Centers of both feet with shape [B, T, 2, 3].
            sizes (torch.Tensor): Dimensions (length, height, width) of both feet with shape [B, T, 2, 3].
            rotation_matrices (torch.Tensor): Rotation matrices for both feet with shape [B, T, 2, 3, 3].
        """
        batch_size, seq_len, num_features = joint_centers.shape
        num_joints = num_features // 3  # Each joint has 3 coordinates (x, y, z)

        # Reshape joint_centers to separate joints and their coordinates
        joint_centers = joint_centers.view(batch_size, seq_len, num_joints, 3)

        # Extract specific joint positions for the left foot
        ankle_l = joint_centers[:, :, self.ankle_l_idx]         # Left ankle
        subtalar_l = joint_centers[:, :, self.subtalar_l_idx] # Left subtalar joint
        mtp_l = joint_centers[:, :, self.mtp_l_idx]           # Left MTP joint

        # Extract specific joint positions for the right foot
        ankle_r = joint_centers[:, :, self.ankle_r_idx]         # Right ankle
        subtalar_r = joint_centers[:, :, self.subtalar_r_idx] # Right subtalar joint
        mtp_r = joint_centers[:, :, self.mtp_r_idx]           # Right MTP joint

        # Compute rotation matrices and sizes for both feet
        left_rot, left_sizes = self.compute_foot(ankle_l, subtalar_l, mtp_l)
        right_rot, right_sizes = self.compute_foot(ankle_r, subtalar_r, mtp_r)

        # Stack sizes for both feet
        sizes = torch.stack([left_sizes, right_sizes], dim=2)  # [B, T, 2, 3]

        # Concatenate rotation matrices for both feet
        rotation_matrices = torch.cat([left_rot.unsqueeze(2), right_rot.unsqueeze(2)], dim=2)   # [B, T, 2, 3, 3]

        # Calculate the center position of each foot using a weighted average of subtalar and MTP joints
        left_center = subtalar_l * (1 - self.ratio_foot_center) + mtp_l * self.ratio_foot_center
        right_center = subtalar_r * (1 - self.ratio_foot_center) + mtp_r * self.ratio_foot_center

        # Stack centers for both feet
        centers = torch.stack([left_center, right_center], dim=2)  # [B, T, 2, 3]

        return centers, sizes, rotation_matrices

    def compute_foot(self, ankle, subtalar, mtp):
        """
        Computes the rotation matrix and sizes for a single foot based on joint positions.
        This method consolidates the functionalities of compute_foot_axes and compute_foot_dimensions
        to eliminate redundant calculations.

        Args:
            ankle (torch.Tensor): Ankle joint positions with shape [B, T, 3].
            subtalar (torch.Tensor): Subtalar joint positions with shape [B, T, 3].
            mtp (torch.Tensor): MTP joint positions with shape [B, T, 3].

        Returns:
            rot (torch.Tensor): Rotation matrices with shape [B, T, 3, 3].
            sizes (torch.Tensor): Foot dimensions (length, height, width) with shape [B, T, 3].
        """
        # Compute the forward vector from subtalar to MTP joint
        forward = mtp - subtalar  # [B, T, 3]
        forward_norm = torch.norm(forward, dim=2, keepdim=True).clamp(min=self.eps)  # [B, T, 1]
        x_axis = forward / forward_norm  # Normalized forward direction vector [B, T, 3]

        # Compute the up vector by projecting the ankle position onto the forward axis and finding the perpendicular component
        projection = subtalar + torch.sum((ankle - subtalar) * x_axis, dim=2, keepdim=True) * x_axis  # [B, T, 3]
        up = ankle - projection  # Up vector from projection to ankle [B, T, 3]
        up_norm = torch.norm(up, dim=2, keepdim=True).clamp(min=self.eps)  # [B, T, 1]
        y_axis = up / up_norm  # Normalized upward direction vector [B, T, 3]

        # Compute the right vector as the cross product of x_axis and y_axis
        z_axis = torch.cross(x_axis, y_axis, dim=2)  # [B, T, 3]
        z_axis_norm = torch.norm(z_axis, dim=2, keepdim=True).clamp(min=self.eps)  # [B, T, 1]
        z_axis = z_axis / z_axis_norm  # Normalized right direction vector [B, T, 3]

        # Assemble the rotation matrix from the local axes
        rot = torch.stack([x_axis, y_axis, z_axis], dim=3)  # [B, T, 3, 3]

        # Calculate the foot dimensions by scaling the forward and up vectors
        length = forward_norm.squeeze(2) * self.length_ratio  # Scaled foot length [B, T]
        height = up_norm.squeeze(2) * self.height_ratio        # Scaled foot height [B, T]
        width = forward_norm.squeeze(2) * self.width_ratio    # Scaled foot width [B, T]

        # Stack the dimensions into a single tensor
        sizes = torch.stack([length, height, width], dim=2)  # [B, T, 3]

        return rot, sizes

    def limit_cop(self, output_cop, centers, sizes, rotation_matrices):
        """
        Limits the Center of Pressure (CoP) predictions within the foot box boundaries in the foot's local coordinate frame.

        Depending on the activation function, it either applies an activation function to constrain CoP values between -1 and 1,
        or performs clipping in the local coordinate frame.

        Args:
            output_cop (torch.Tensor): Predicted CoP values with shape [B, T, 2, 3].
            centers (torch.Tensor): Centers of both feet with shape [B, T, 2, 3].
            sizes (torch.Tensor): Dimensions (length, height, width) of both feet with shape [B, T, 2, 3].
            rotation_matrices (torch.Tensor): Rotation matrices for both feet with shape [B, T, 2, 3, 3].

        Returns:
            output_cop_limited (torch.Tensor): Limited CoP values within foot boundaries in global coordinates
                                               with shape [B, T, 2, 3].
        """
        if self.activation_function == 'clip':
            # Limit CoP using clipping
            # Step 1: Transform the predicted CoP to the foot's local coordinate frame
            cop_relative = output_cop - centers  # [B, T, 2, 3]

            # Transform CoP to foot's local frame using inverse rotation matrices
            cop_relative = cop_relative.unsqueeze(-1)  # [B, T, 2, 3, 1]
            rotation_matrices_inv = rotation_matrices.transpose(-1, -2)  # [B, T, 2, 3, 3]
            cop_local = torch.matmul(rotation_matrices_inv, cop_relative).squeeze(-1)  # [B, T, 2, 3]

            # Step 2: Compute allowed ranges in the local coordinate frame
            half_sizes = sizes / 2.0  # [B, T, 2, 3]
            min_values = -half_sizes  # [B, T, 2, 3]
            max_values = half_sizes   # [B, T, 2, 3]

            # Step 3: Clip the CoP values in the local coordinate frame
            cop_local_limited = torch.max(torch.min(cop_local, max_values), min_values)  # [B, T, 2, 3]

            # Step 4: Transform the clipped CoP back to global coordinates
            cop_local_limited = cop_local_limited.unsqueeze(-1)  # [B, T, 2, 3, 1]
            cop_global = torch.matmul(rotation_matrices, cop_local_limited).squeeze(-1)  # [B, T, 2, 3]
            output_cop_limited = cop_global + centers  # [B, T, 2, 3]

        else:
            # Limit CoP using activation function
            # Step 0: Apply activation function to constrain CoP values between -1 and 1
            output_cop_activated = self.activation(output_cop)  # [B, T, 2, 3]

            # Step 1: Scale the activated output to the foot's local coordinate limits
            half_sizes = sizes / 2.0  # [B, T, 2, 3]
            cop_local_limited = output_cop_activated * half_sizes  # [B, T, 2, 3]

            # Step 2: Transform the limited CoP back to the global (root) coordinate frame
            cop_local_limited = cop_local_limited.unsqueeze(-1)  # [B, T, 2, 3, 1]
            cop_global = torch.matmul(rotation_matrices, cop_local_limited).squeeze(-1)  # [B, T, 2, 3]
            output_cop_limited = cop_global + centers  # [B, T, 2, 3]

        return output_cop_limited
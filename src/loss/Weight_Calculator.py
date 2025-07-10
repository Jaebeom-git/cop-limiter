import torch
import torch.nn as nn
from typing import Optional
import matplotlib.pyplot as plt

def smoothstep(x: torch.Tensor, edge0: float, edge1: float) -> torch.Tensor:
    """
    Implements the smoothstep interpolation function.

    Args:
        x (Tensor): Input tensor.
        edge0 (float): Lower edge of the transition.
        edge1 (float): Upper edge of the transition.

    Returns:
        Tensor: Smoothly interpolated values between 0 and 1.
    """
    x = (x - edge0) / (edge1 - edge0 + 1e-8)  # Normalize to [0, 1] with epsilon to prevent division by zero
    x = x.clamp(0.0, 1.0)  # Clamp to [0, 1]
    return x * x * (3 - 2 * x)

class ForceWeightCalculator:
    def __init__(self, scale_param=None, max_param=None, contact_param=None):
        """
        Computes weights for 3D forces for two feet.
        Provides three functionalities:
        
        0. Component-wise scaling: If 'scale_param' is provided, scales the x, y, and z 
           components of each foot's force by the specified factors (e.g., [scale_x, scale_y, scale_z]).
        1. Maximum force weight: If 'max_param' is provided, computes a dynamic threshold based on 
           the maximum force of each foot and adjusts the weight accordingly 
           (e.g., {"max_force_ratio": 0.9, "fixed_weight_max": 2.0}).
        2. No-contact weight: If 'contact_param' is provided, applies a fixed weight when the force magnitude
           is below a specified threshold (e.g., {"no_contact_force_threshold": 1.0, "fixed_weight_no_contact": 2.0}).
        
        Only functionalities corresponding to provided parameters are applied.
        
        Args:
            scale_param (iterable of 3 floats, optional): Scaling factors for the x, y, z components.
            max_param (dict, optional): Parameters for maximum force weight adjustment.
            contact_param (dict, optional): Parameters for no-contact weight adjustment.
        """
        self.scale_param = scale_param
        self.max_param = max_param
        self.contact_param = contact_param

        self.compute_scale = scale_param is not None
        self.compute_max = max_param is not None
        self.compute_contact = contact_param is not None

    def compute_weight(self, force: torch.Tensor) -> torch.Tensor:
        """
        Computes weights for each force vector corresponding to two feet. The input tensor is expected
        to have shape [B, T, 6], where 6 corresponds to two feet with 3 force components each.
        
        The final weight is computed as the product of:
         - no-contact weight,
         - maximum force weight, and 
         - component-wise scaling.
        Only the functionalities corresponding to the provided parameters are applied.
        
        Args:
            force (torch.Tensor): Tensor of shape [B, T, 6].
        
        Returns:
            torch.Tensor: Tensor of shape [B, T, 6] containing the computed weights.
        """
        B, T, F = force.shape
        assert F == 6, "Force tensor must have shape [B, T, 6]"

        # Reshape force to [B, T, 2, 3]: two feet, each with 3 components.
        force_reshaped = force.view(B, T, 2, 3)
        # Compute the norm (magnitude) of the force for each foot, shape: [B, T, 2].
        force_norm = force_reshaped.norm(dim=-1)

        # Initialize weight for each foot as ones, shape: [B, T, 2].
        weight = torch.ones(B, T, 2, device=force.device)

        # (2) Apply no-contact weight if 'contact_param' is provided.
        if self.compute_contact:
            # Retrieve threshold and fixed weight from contact_param.
            no_contact_threshold = torch.tensor(self.contact_param.no_contact_force_threshold, device=force.device)
            fixed_weight_no_contact = self.contact_param.fixed_weight_no_contact
            # Compute a smooth transition from 0 to 1 between 0 and no_contact_threshold.
            s_contact = smoothstep(force_norm,
                                   torch.tensor(0.0, device=force.device),
                                   no_contact_threshold)
            # Transition the weight from fixed_weight_no_contact to 1.0.
            weight_contact = fixed_weight_no_contact + (1.0 - fixed_weight_no_contact) * s_contact
            weight = weight * weight_contact

        # (1) Apply maximum force weight if 'max_param' is provided.
        if self.compute_max:
            max_force_ratio = self.max_param.max_force_ratio
            fixed_weight_max = self.max_param.fixed_weight_max

            # Compute the maximum force for each foot over the time dimension, shape: [B, 2].
            max_force_per_foot = force_norm.amax(dim=1)
            # Compute dynamic threshold: maximum force multiplied by max_force_ratio, shape: [B, 2].
            dynamic_force_threshold = max_force_per_foot * max_force_ratio

            # For computing the mean force, use the no-contact threshold if available; otherwise, use 0.0.
            threshold_for_mean = self.contact_param.no_contact_force_threshold if self.compute_contact else 0.0
            non_zero_mask = force_norm > threshold_for_mean
            sum_force = torch.sum(force_norm * non_zero_mask, dim=1)
            count_non_zero = non_zero_mask.sum(dim=1).clamp(min=1)
            mean_force = sum_force / count_non_zero  # shape: [B, 2]

            # Determine whether to apply the maximum force weight for each foot.
            apply_max = (dynamic_force_threshold >= mean_force).unsqueeze(1)  # shape: [B, 1, 2]
            # Compute a smooth transition between the dynamic threshold and the maximum force.
            s_max = smoothstep(force_norm,
                               dynamic_force_threshold.unsqueeze(1),
                               max_force_per_foot.unsqueeze(1))
            # Weight transitions from 1.0 to fixed_weight_max.
            weight_max = 1.0 + (fixed_weight_max - 1.0) * s_max  # shape: [B, T, 2]
            # Only apply the maximum force weight if the condition is met.
            weight_max = torch.where(apply_max, weight_max, torch.ones_like(weight_max))
            weight = weight * weight_max

        # (0) Apply component-wise scaling if 'scale_param' is provided.
        # The current weight has shape [B, T, 2] (one value per foot). To apply scaling to each component,
        # we expand it to shape [B, T, 2, 1] and multiply by a scale tensor of shape [1, 1, 1, 3].
        if self.compute_scale:
            scale_tensor = torch.tensor(self.scale_param, device=force.device).view(1, 1, 1, 3)
            weight = weight.unsqueeze(-1) * scale_tensor  # shape becomes [B, T, 2, 3]
        else:
            # If no scale_param is provided, replicate the weight across the 3 components.
            weight = weight.unsqueeze(-1).repeat(1, 1, 1, 3)  # shape: [B, T, 2, 3]

        # Reshape the weight from [B, T, 2, 3] to [B, T, 6] (flattening the foot and component dimensions).
        weight = weight.view(B, T, 6)
        return weight
    
class ForceMaskCalculator:
    def __init__(self, threshold=5.0):
        """
        Class to calculate masks based on force thresholds.

        Args:
            threshold (float): Threshold to apply the mask.
        """
        self.threshold = threshold

    def get_mask_by_threes(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Generates a mask for the tensor based on 3-dimensional vectors.

        Args:
            tensor (Tensor): Input tensor of shape [B, T, C].

        Returns:
            Tensor: Mask tensor of shape [B, T, C].
        """
        with torch.no_grad():
            if tensor.dim() != 3:
                raise ValueError('Mask tensor must be 3-dimensional')
            if tensor.numel() == 0:
                raise ValueError('Mask tensor must not be empty')
            if tensor.shape[-1] % 3 != 0:
                raise ValueError('Mask tensor must have a final dimension divisible by 3')

            # Reshape the tensor to split the last dimension into chunks of 3
            reshaped_tensor = tensor.view(tensor.shape[0], tensor.shape[1], -1, 3)

            # Compute the norm over the last dimension
            norms = torch.norm(reshaped_tensor, dim=-1)  # [B, T, N]

            # Create a mask where the norm is greater than the threshold
            mask = (norms > self.threshold).float()  # [B, T, N]

            # Expand the mask to cover the original last dimension size
            expanded_mask = mask.unsqueeze(3).expand(-1, -1, -1, 3)  # [B, T, N, 3]

            # Reshape the expanded mask back to the original tensor shape
            return expanded_mask.reshape(tensor.shape)  # [B, T, C]


class CoPWeightCalculator:
    def __init__(self, scale_param=None, dist_param=None):
        """
        Computes weights for Center of Pressure (CoP) based on distance thresholds and applies 
        component-wise scaling to the x, y, and z axes for each foot.

        Args:
            scale_param (iterable of 3 floats, optional): Scaling factors for the x, y, z components.
                If provided, these are duplicated for both feet to form a 6-element scaling vector.
            dist_param (object, optional): Object with attributes:
                - threshold_min (float): Minimum threshold for distance.
                - threshold_max (float): Maximum threshold for distance.
                - fixed_weight_below_threshold (float): Weight when distance is below threshold_min.
                - distance_max_weight (float): Weight when distance is above threshold_max.
                If provided, distance-based weighting is enabled.
        """
        self.compute_dist_weight = dist_param is not None
        if self.compute_dist_weight:
            self.threshold_min = dist_param.threshold_min
            self.threshold_max = dist_param.threshold_max
            self.fixed_weight_below_threshold = dist_param.fixed_weight_below_threshold
            self.distance_max_weight = dist_param.distance_max_weight

        # If scale_param is provided, ensure it has exactly 3 elements and duplicate them for both feet.
        self.compute_scale = scale_param is not None
        if self.compute_scale:
            if len(scale_param) != 3:
                raise ValueError("scale_param must have exactly 3 elements for x, y, z scaling")
            # Create a 6-element scale tensor: [x, y, z, x, y, z]
            self.scale_tensor = torch.tensor(list(scale_param) * 2).view(1, 1, 6)

    def compute_weight_based_on_distance(self, dist: torch.Tensor) -> torch.Tensor:
        """
        Computes weights using smoothstep interpolation based on distance.

        Args:
            dist (torch.Tensor): Distance tensor. Can be of shape [B, T] or [B, T, C].

        Returns:
            torch.Tensor: Interpolated weights between fixed_weight_below_threshold and distance_max_weight.
        """
        s = smoothstep(dist, self.threshold_min, self.threshold_max)
        weight = self.fixed_weight_below_threshold + (self.distance_max_weight - self.fixed_weight_below_threshold) * s
        return weight

    def compute_weight(self, predicted_cop: torch.Tensor, true_cop: torch.Tensor) -> torch.Tensor:
        """
        Calculates weights based on the element-wise absolute difference between predicted and true CoP.
        Then, if a scale_param was provided at initialization, applies component-wise scaling using the
        precomputed 6-element scale tensor.

        Args:
            predicted_cop (torch.Tensor): Predicted CoP tensor of shape [B, T, 6].
            true_cop (torch.Tensor): True CoP tensor of shape [B, T, 6].

        Returns:
            torch.Tensor: Weight tensor of shape [B, T, 6].
        """
        if self.compute_dist_weight:
            # Compute element-wise absolute difference.
            diff_GT = torch.abs(predicted_cop - true_cop)  # [B, T, 6]
            # Compute weights based on the absolute differences.
            cop_weight_tensor = self.compute_weight_based_on_distance(diff_GT)  # [B, T, 6]
        else:
            cop_weight_tensor = torch.ones_like(predicted_cop)

        # If scaling is enabled, multiply directly with the precomputed scale tensor.
        if self.compute_scale:
            cop_weight_tensor = cop_weight_tensor * self.scale_tensor.to(cop_weight_tensor.device)
        
        return cop_weight_tensor

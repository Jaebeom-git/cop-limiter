import torch
import torch.nn as nn
from typing import Optional
import matplotlib.pyplot as plt
from denseweight import DenseWeight

def calculate_diff(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return pred - target

def se_loss(diff: torch.Tensor, reduction: str = 'mean') -> torch.Tensor:
    """
    Computes the Squared Error (SE) loss using the provided diff.

    Args:
        diff (torch.Tensor): Difference between predictions and targets.
        reduction (str, optional): Reduction method: 'mean', 'sum', or 'none'.

    Returns:
        torch.Tensor: MSE loss.
    """
    squared_diff = diff ** 2

    if reduction == 'mean':
        se = torch.mean(squared_diff)  # Scalar
    elif reduction == 'sum':
        se = torch.sum(squared_diff)  # Scalar
    elif reduction == 'none':
        se = squared_diff  # Shape: [B, T, C]
    else:
        raise ValueError(f"Invalid reduction '{reduction}'")

    return se


def mean_norm_error(diff: torch.Tensor, vec_size: int = 3, std: bool = False) -> torch.Tensor:
    """
    Computes the Mean Norm Error using the provided diff by calculating the norm over vec_size dimensions.

    Args:
        diff (torch.Tensor): Difference between predictions and targets.
        vec_size (int, optional): Number of dimensions per vector. Defaults to 3.

    Returns:
        torch.Tensor: Scalar representing the mean norm error.
    """
    reshaped = diff.view(diff.shape[0], diff.shape[1], -1, vec_size)  # Shape: [B, T, N, vec_size]
    norms = torch.norm(reshaped, dim=3)  # Shape: [B, T, N]
    mean_norm = torch.mean(norms)  # Scalar
    if std:
        std_norm = torch.std(norms)
        return mean_norm, std_norm
    
    return mean_norm

def TCLoss(pred: torch.Tensor) -> torch.Tensor:
    """
    Calculates the Temporal Consistency Loss (TCLoss) between the predicted values.
    """
    diff = pred[:, 1:, :] - pred[:, :-1, :]
    return torch.mean(diff ** 2)


def MPJVE(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Calculates the Mean Per Joint Velocity Error (MPJVE) between the predicted and target values.
    """
    pred_vel = pred[:, 1:, :] - pred[:, :-1, :]
    target_vel = target[:, 1:, :] - target[:, :-1, :]
    diff = pred_vel - target_vel
    return torch.mean(diff ** 2)


class GHMR_mean(nn.Module):
    """GHM Regression Loss with optimized diff usage and momentum scheduling.

    Args:
        bins (int): Number of unit regions for gradient distribution calculation.
        momentum (float): The final momentum value for moving average in gradient histogram.
        loss_weight (float): The overall weight of the GHMR loss.
        k (float): Scaling constant for diff_mean.
        momentum_increment (float, optional): Amount to increase momentum each step. 
            Defaults to 0.0. If set to 0.0, momentum is set to target_momentum immediately.
    """
    def __init__(
            self,
            mu: float = 0.02,
            bins: int = 10,
            momentum: float = 0.99,
            loss_weight: float = 1.0,
            k: float = 2.0,
            momentum_increment: float = 0.0,
            debug: bool = False):
        super(GHMR, self).__init__()
        self.mu = mu
        self.bins = bins
        self.register_buffer('edges', torch.linspace(0, 1, bins + 1))
        self.edges[-1] = float('inf')  # Set the last edge to a large number
        self.loss_weight = loss_weight
        self.k = k
        self.debug = debug

        # Initialize moving average parameters
        self.register_buffer('diff_mean', torch.tensor(0.0))
        if momentum > 0:
            self.register_buffer('acc_sum', torch.zeros(bins))
            # Momentum scheduling parameters
            self.momentum_increment = momentum_increment
            if self.momentum_increment > 0:
                # Gradually increase momentum
                self.register_buffer('init_state', torch.tensor(1))  # 1 for True, 0 for False
                self.register_buffer('step', torch.tensor(0))
                self.register_buffer('current_momentum', torch.tensor(0.0))
                self.target_momentum = momentum
                # Automatically determine max_steps based on momentum and increment
                self.max_steps = int(self.target_momentum / self.momentum_increment)
            else:
                # Directly set momentum to the target if no increment is specified
                self.register_buffer('init_state', torch.tensor(0))  # 0 for False
                self.register_buffer('current_momentum', torch.tensor(momentum))
        else:
            self.acc_sum = None
            self.register_buffer('init_state', torch.tensor(0))  # 0 for False
            self.register_buffer('current_momentum', torch.tensor(momentum))

    def plot_histogram(self, grad_norm):
        """Plots a clean histogram of the gradient norms.

        Args:
            grad_norm (Tensor): Gradient norm tensor.
        """
        grad_norm_np = grad_norm.detach().cpu().numpy().flatten()

        plt.figure(figsize=(8, 6))
        plt.hist(grad_norm_np, bins=self.bins, range=(0, 1), color='skyblue', edgecolor='black', alpha=0.7)
        plt.title("Histogram of Gradient Norm", fontsize=16)
        plt.xlabel("Gradient Norm", fontsize=14)
        plt.ylabel("Frequency", fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.show()

    def update_momentum(self):
        """Gradually increase momentum from 0 to target_momentum over max_steps."""
        self.step += 1
        new_momentum = self.current_momentum + self.momentum_increment
        clamped_momentum = torch.clamp(new_momentum, max=self.target_momentum)
        self.current_momentum.copy_(clamped_momentum)
        if self.step.item() >= self.max_steps:
            self.init_state.fill_(0)  # False

    def forward(self, loss: torch.Tensor, diff: torch.Tensor, weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Calculate the GHMR loss using MSE loss and diff.

        Args:
            loss (Tensor): MSE loss per sample of shape [B, T, C].
            diff (Tensor): Difference between predictions and targets of shape [B, T, C].
            weight (Tensor, optional): Weights tensor of shape [B, T, C].

        Returns:
            Tensor: The gradient harmonized regression loss.
        """
        # Step 0: Update momentum if in initialization phase
        if self.init_state.item() == 1:
            self.update_momentum()

        # ASL1 loss
        ASL1_loss_temp = torch.sqrt(loss + self.mu * self.mu)
        loss = ASL1_loss_temp - self.mu

        # Apply weights to the loss if provided
        if weight is not None:
            loss = loss * weight

        # Update diff_mean with moving average without tracking gradients
        epsilon = 1e-8
        current_diff_mean = diff.abs().mean().detach()

        with torch.no_grad():
            if self.diff_mean.item() == 0.0:
                self.diff_mean.copy_(current_diff_mean)
            else:
                self.diff_mean.copy_(self.current_momentum * self.diff_mean + (1 - self.current_momentum) * current_diff_mean)

        # Calculate gradient magnitude
        grad_norm = diff.abs() / (self.k * self.diff_mean + epsilon)

        # Initialize weights tensor
        weights = torch.zeros_like(grad_norm)

        # Total number of samples
        total_samples = max(grad_norm.numel(), 1.0)

        # Number of bins that have at least one sample
        num_valid_bins = 0

        # Compute weights based on gradient distribution
        for i in range(self.bins):
            inds = (grad_norm >= self.edges[i]) & (grad_norm < self.edges[i + 1])
            num_in_bin = inds.sum().item()
            if num_in_bin > 0:
                num_valid_bins += 1
                if self.current_momentum > 0 and self.acc_sum is not None:
                    with torch.no_grad():
                        self.acc_sum[i].copy_(self.current_momentum * self.acc_sum[i] + (1 - self.current_momentum) * num_in_bin)
                    weights[inds] = total_samples / (self.acc_sum[i] + epsilon)
                else:
                    weights[inds] = total_samples / num_in_bin

        if num_valid_bins > 0:
            weights = weights / num_valid_bins

        # Apply weights to the loss
        loss = loss * weights
        loss = loss.sum() / total_samples

        if self.debug:
            self.plot_histogram(grad_norm)

        return loss * self.loss_weight


class GHMR(nn.Module):
    """GHM Regression Loss with optimized diff usage and momentum scheduling.

    Args:
        mu (float): The parameter for the Authentic Smooth L1 loss.
        bins (int): Number of unit regions for gradient distribution calculation.
        momentum (float): The final momentum value for moving average in gradient histogram.
        loss_weight (float): The overall weight of the GHMR loss.
        momentum_increment (float, optional): Amount to increase momentum each step.
            Defaults to 0.0. If set to 0.0, momentum is set to target_momentum immediately.
        debug (bool, optional): If True, plots the histogram of gradient norms. Defaults to False.
        device (str, optional): Device to run the GHMR loss on ('cpu' or 'cuda'). 
            If None, defaults to current device of the model's parameters.
    """
    def __init__(
            self,
            mu: float = 0.02,
            bins: int = 10,
            momentum: float = 0.99,
            loss_weight: float = 1.0,
            momentum_increment: float = 0.0,
            debug: bool = False,
            device: Optional[str] = None):
        super(GHMR, self).__init__()
        self.mu = mu
        self.bins = bins
        self.loss_weight = loss_weight
        self.debug = debug
        self.epsilon = 1e-8  # Small value added for numerical stability
        self.device = torch.device(device) if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Register buffer for bin edges on the specified device
        edges = torch.linspace(0, self.bin_max, bins + 1, device=self.device)
        edges[-1] = float('inf')  # Set the last edge to a large number
        self.register_buffer('edges', edges)

        # Initialize buffer for moving average of histogram bins
        self.register_buffer('acc_sum', torch.zeros(bins, device=self.device))
        
        # Momentum scheduling parameters
        self.momentum_increment = momentum_increment
        if self.momentum_increment > 0:
            # Gradually increase momentum
            self.register_buffer('init_state', torch.tensor(1, dtype=torch.long, device=self.device))  # 1 indicates initial state
            self.register_buffer('step', torch.tensor(0, dtype=torch.long, device=self.device))  # Step count
            self.register_buffer('current_momentum', torch.tensor(0.0, device=self.device))
            self.target_momentum = momentum
            # Calculate max_steps based on momentum and increment to reach target momentum
            self.max_steps = int(self.target_momentum / self.momentum_increment)
        else:
            # If no momentum increment, set momentum to target immediately
            self.register_buffer('init_state', torch.tensor(0, dtype=torch.long, device=self.device))  # 0 indicates momentum is set
            self.register_buffer('current_momentum', torch.tensor(momentum, device=self.device))

    def plot_histogram(self, grad_norm):
        """Plots a clean histogram of the gradient norms.

        Args:
            grad_norm (Tensor): Gradient norm tensor.
        """
        grad_norm_np = grad_norm.detach().cpu().numpy().flatten()

        plt.figure(figsize=(8, 6))
        plt.hist(grad_norm_np, bins=self.bins, range=(0, 1), color='skyblue', edgecolor='black', alpha=0.7)
        plt.title("Histogram of Gradient Norm", fontsize=16)
        plt.xlabel("Gradient Norm", fontsize=14)
        plt.ylabel("Frequency", fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.show()

    def update_momentum(self):
        """Gradually increases momentum from 0 to target_momentum."""
        if self.init_state.item() == 1:
            with torch.no_grad():
                self.step += 1
                new_momentum = self.current_momentum + self.momentum_increment
                clamped_momentum = torch.clamp(new_momentum, max=self.target_momentum)
                self.current_momentum.copy_(clamped_momentum)
                if self.step.item() >= self.max_steps:
                    self.init_state.fill_(0)  # Change initial state to completed

    def forward(self, loss: torch.Tensor, diff: torch.Tensor, weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Calculates the GHMR loss.

        Args:
            loss (Tensor): SE loss (Mean Squared Error) per sample tensor [B, T, C].
            diff (Tensor): Difference between predictions and targets tensor [B, T, C].
            weight (Tensor, optional): Weights tensor [B, T, C]. Defaults to None.

        Returns:
            Tensor: Gradient Harmonized Regression loss.
        """
        # Update momentum if in the initial state
        if self.init_state.item() == 1:
            self.update_momentum()

        # Calculate ASL1 loss
        ASL1_loss_temp = torch.sqrt(loss + self.mu * self.mu)
        loss = ASL1_loss_temp - self.mu

        # Calculate gradient magnitude |dL/dPred|
        grad_norm = torch.abs(diff / ASL1_loss_temp.detach())
        valid = torch.ones_like(grad_norm, dtype=torch.bool, device=self.device)

        # Calculate total number of valid samples
        tot = max(valid.float().sum().item(), 1.0)

        # Initialize weights tensor
        weights = torch.zeros_like(grad_norm, device=self.device)

        # Count of valid bins
        num_valid_bins = 0

        with torch.no_grad():
            # Compute weights based on gradient distribution
            for i in range(self.bins):
                inds = (grad_norm >= self.edges[i]) & (grad_norm < self.edges[i + 1]) & valid
                num_in_bin = inds.sum().item()
                if num_in_bin > 0:
                    num_valid_bins += 1
                    if self.current_momentum > 0:
                        # Update acc_sum with momentum (in-place)
                        updated_acc_sum = self.current_momentum * self.acc_sum[i] + (1 - self.current_momentum) * num_in_bin
                        self.acc_sum[i].copy_(updated_acc_sum)
                        weights[inds] = tot / (self.acc_sum[i] + self.epsilon)
                    else:
                        # Compute weights without momentum
                        weights[inds] = tot / (num_in_bin + self.epsilon)

            # Average the weights across valid bins
            if num_valid_bins > 0:
                weights = weights / num_valid_bins

        # Apply weights to the loss
        if weight is not None:
            weights = weights * weight
        loss = loss * weights

        # Compute the average loss
        loss = loss.sum() / tot

        # Plot histogram if debug mode is enabled
        if self.debug:
            self.plot_histogram(grad_norm)

        return loss * self.loss_weight


class DenseLoss(nn.Module):
    """
    DenseLoss calculates weighted loss for imbalanced regression tasks.
    """
    def __init__(self, alpha: float = 1.0, bandwidth: Optional[float] = None, eps: float = 1e-6, loss_weight: float = 1.0):
        super(DenseLoss, self).__init__()
        self.dense_weight = DenseWeight(alpha=alpha, bandwidth=bandwidth, eps=eps)
        self.loss_weight = loss_weight

    def forward(self, loss: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for DenseLoss.

        Args:
            loss (torch.Tensor): Loss tensor.
            target (torch.Tensor): Target values.

        Returns:
            torch.Tensor: Weighted loss value.
        """
        target_np = target.detach().cpu().numpy().flatten()
        weights = self.dense_weight.fit(target_np)
        weights_tensor = torch.tensor(weights, dtype=loss.dtype, device=loss.device).view(loss.shape)

        # Calculate Mean Squared Error with weights
        loss = weights_tensor * loss
        return torch.mean(loss) * self.loss_weight

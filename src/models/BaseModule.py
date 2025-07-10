import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import torch
import torch.nn as nn
from data.AddBiomechanicsDataset import InputDataKeys
from models.CoPLimiter import CoPLimiter
    
# RMSNorm for normalization
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

# FFNSwiGLU: MLP with Swish GLU activation
class FFNSwiGLU(nn.Module):
    def __init__(self, input_dim, hidden_dim=None, output_dim=None, dropout=0.0):
        super(FFNSwiGLU, self).__init__()
        hidden_dim = hidden_dim or int(4 * input_dim)
        output_dim = output_dim or input_dim
        self.W = nn.Linear(input_dim, hidden_dim, bias=False)
        self.V = nn.Linear(input_dim, hidden_dim, bias=False)
        self.W2 = nn.Linear(hidden_dim, output_dim, bias=False)
        self.beta = nn.Parameter(torch.tensor(1.0))  # Initialize beta as a learnable parameter
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x_w = self.W(x)  # Linear transformation with W
        x_v = self.V(x)  # Linear transformation with V
        swish_beta = x_w * torch.sigmoid(self.beta * x_w)  # Swish activation with learnable beta
        intermediate_output = swish_beta * x_v  # Element-wise multiplication with xV
        intermediate_output = self.dropout(intermediate_output)
        output = self.W2(intermediate_output)  # Final linear transformation with W2
        return output

# DropPath for stochastic depth regularization
class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = torch.bernoulli(torch.full(shape, keep_prob, device=x.device))
        if self.scale_by_keep and keep_prob > 0:
            random_tensor.div_(keep_prob)
        return x * random_tensor

# ProjectionLayer: Projects input data into a higher-dimensional space
class ProjectionLayer(nn.Module):
    def __init__(self, num_channels, embed_dim, bias=False):
        super(ProjectionLayer, self).__init__()
        self.proj = nn.Linear(num_channels, embed_dim, bias=bias)
        # Orthogonal initialization
        nn.init.orthogonal_(self.proj.weight)
        if self.proj.bias is not None:
            nn.init.constant_(self.proj.bias, 0.0)

    def forward(self, x):
        x = self.proj(x)
        return x

# Head: Single linear layer without non-linearity for final predictions
class Head(nn.Module):
    def __init__(self, embed_dim, output_dim):
        super(Head, self).__init__()
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.fc = nn.Linear(embed_dim, output_dim)

    def forward(self, x):
        x = self.norm(x)
        x = self.fc(x)  # No non-linearity
        return x

class GroundBase(nn.Module):
    def __init__(self, config, device):
        super().__init__()

        # Assign configurations from config
        self.device = device
        self.config = config
        self.num_dofs = config.num_dofs
        self.num_joints = config.num_joints
        self.root_history_len = config.root_history_len

        # CoPLimiter
        self.use_cop_limiter = config.use_cop_limiter if hasattr(config, 'use_cop_limiter') else False
        self.cop_limiter_activation_function = config.cop_limiter_activation_function if hasattr(config, 'cop_limiter_activation_function') else 'tanh'
        self.cop_limiter = CoPLimiter(length_ratio=2.0, height_ratio=2.5, width_ratio=0.75, activation_function=self.cop_limiter_activation_function).to(self.device)

        # Define input sizes
        self.com_input_size = 9   # 3 (Vel) + 3 (Acc) + 3 (Pos)
        self.joint_input_size = self.num_dofs * 3  # Pos, Vel, Acc
        self.root_input_size = self.root_history_len * 6 + 12  # 10 * 6 (root_history_len*6 (x, y, z, roll, pitch, yaw)) + 3 (Lin Vel) + 3 (Ang Vel) + 3 (Lin Acc) + 3 (Ang Acc)
        self.joint_center_input_size = self.num_joints * 3
        self.spatial_input_size = self.joint_center_input_size + 3  # Joint centers + CoM
        self.input_size = self.com_input_size + self.joint_input_size + self.root_input_size + self.joint_center_input_size
        
        # Define output sizes
        # self.num_output_frames = (self.history_len // stride) if self.output_data_format == 'all_frames' else 1
        self.output_force_size = 6
        self.output_moment_size = 6
        self.output_cop_size = 6
        self.output_size = self.output_force_size + self.output_moment_size + self.output_cop_size  # Predict only CoP, Forces, and Moments (6 each, for 3 sets of outputs)

    def input_parsing(self, input):
        # 1. Extract and check input shapes
        assert input[InputDataKeys.comPosInRootFrame].shape[-1] == 3
        assert input[InputDataKeys.comVelInRootFrame].shape[-1] == 3
        assert input[InputDataKeys.comAccInRootFrame].shape[-1] == 3
        assert input[InputDataKeys.POS].shape[-1] == self.num_dofs
        assert input[InputDataKeys.VEL].shape[-1] == self.num_dofs
        assert input[InputDataKeys.ACC].shape[-1] == self.num_dofs
        assert input[InputDataKeys.ROOT_POS_HISTORY_IN_ROOT_FRAME].shape[-1] == self.root_history_len * 3
        assert input[InputDataKeys.ROOT_EULER_HISTORY_IN_ROOT_FRAME].shape[-1] == self.root_history_len * 3
        assert input[InputDataKeys.ROOT_LINEAR_VEL_IN_ROOT_FRAME].shape[-1] == 3
        assert input[InputDataKeys.ROOT_LINEAR_ACC_IN_ROOT_FRAME].shape[-1] == 3
        assert input[InputDataKeys.ROOT_ANGULAR_VEL_IN_ROOT_FRAME].shape[-1] == 3
        assert input[InputDataKeys.ROOT_ANGULAR_ACC_IN_ROOT_FRAME].shape[-1] == 3
        assert input[InputDataKeys.JOINT_CENTERS_IN_ROOT_FRAME].shape[-1] == self.num_joints * 3

        # 2. Prepare each component input
        # Component 1: CoM inputs & Root inputs & Joint inputs
        temporal_inputs = torch.concat([
            input[InputDataKeys.comVelInRootFrame].to(self.device),
            input[InputDataKeys.comAccInRootFrame].to(self.device),
            input[InputDataKeys.ROOT_POS_HISTORY_IN_ROOT_FRAME].to(self.device),
            input[InputDataKeys.ROOT_EULER_HISTORY_IN_ROOT_FRAME].to(self.device),
            input[InputDataKeys.ROOT_LINEAR_VEL_IN_ROOT_FRAME].to(self.device),
            input[InputDataKeys.ROOT_ANGULAR_VEL_IN_ROOT_FRAME].to(self.device),
            input[InputDataKeys.ROOT_LINEAR_ACC_IN_ROOT_FRAME].to(self.device),
            input[InputDataKeys.ROOT_ANGULAR_ACC_IN_ROOT_FRAME].to(self.device),
            input[InputDataKeys.POS].to(self.device),
            input[InputDataKeys.VEL].to(self.device),
            input[InputDataKeys.ACC].to(self.device)
        ], dim=-1)  # Shape: [batch_size, seq_len, 6 + 12 + num_dofs * 3]

        # Component 2: Joint centers
        joint_center_inputs = input[InputDataKeys.JOINT_CENTERS_IN_ROOT_FRAME].to(self.device)  # Shape: [batch_size, seq_len, num_joints * 3]

        # Component 3: Spatial inputs
        spatial_inputs = torch.concat([
            input[InputDataKeys.JOINT_CENTERS_IN_ROOT_FRAME].to(self.device),
            input[InputDataKeys.comPosInRootFrame].to(self.device)
        ], dim=-1)  # Shape: [batch_size, seq_len, (num_joints + 1) * 3]

        return spatial_inputs, temporal_inputs, joint_center_inputs

    def CoPLimiter(self, output_cop, joint_centers):
        """
        Limits the CoP predictions using CoPLimiter.
        """
        batch_size, seq_len, _ = joint_centers.shape
        output_cop = output_cop.view(batch_size, seq_len, 2, 3)  # [B, T, 2, 3]

        # Compute foot boxes
        centers, sizes, rotation_matrices = self.cop_limiter.compute_feet(joint_centers)  # [B, T, 2, 3], [B, T, 2, 3], [B, T, 2, 3, 3]

        # Limit CoP
        output_cop_limited = self.cop_limiter.limit_cop(output_cop, centers, sizes, rotation_matrices)  # [B, T, 2, 3]
        output_cop = output_cop_limited.view(batch_size, seq_len, 6)  # [B, T, 6]
        return output_cop
    

def config_MoE(MoE_params, embed_dim, mlp_ratio, num_of_blocks, device):
    from megablocks.layers.arguments import Arguments
    """
    Configures the MoE (Mixture of Experts) parameters if specified in config.
    
    Args:
        config: Configuration object containing MoE settings.
        embed_dim (int): Embedding dimension.
        mlp_ratio (float): Ratio for hidden dimension.
        device: Torch device.
    
    Returns:
        moe_args (Arguments or None): Configured MoE arguments or None if not specified.
    """
    if MoE_params is None:
        return None
    
    moe_args = Arguments()
    moe_args.hidden_size = embed_dim
    moe_args.ffn_hidden_size = int(embed_dim * mlp_ratio)
    moe_args.device = device
    moe_args.mlp_type = MoE_params.mlp_type  # Choose from ['mlp', 'glu', 'swiglu']
    moe_args.mlp_impl = 'sparse'
    if moe_args.mlp_type == 'mlp':
        moe_args.memory_optimized_mlp=True   # Enable memory-optimized MLP ('mlp' only)
    moe_args.moe_num_experts = MoE_params.num_experts   # Set the number of experts in the Mixture of Experts (MoE) model
    moe_args.moe_top_k = MoE_params.top_k   # Set the number of top experts to select for each input (Top-K)
    moe_args.moe_loss_weight = MoE_params.loss_weight   # Set the weight for the load-balancing loss to encourage balanced usage of experts
    moe_args.moe_zloss_weight = MoE_params.zloss_weight # Set the weight for the router Z-loss to stabilize the gating network
    # moe_args.moe_capacity_factor = MoE_params.capacity_factor   # Set the capacity factor to control the maximum number of tokens each expert can handle
    moe_args.moe_jitter_eps = MoE_params.jitter_eps
    moe_args.bf16 = False
    moe_args.fp16 = False
    moe_args.return_bias = False
    moe_args.bias = False
    moe_args.shared_expert = MoE_params.shared_expert
    moe_args.shared_expert_weighted_sum = MoE_params.shared_expert_weighted_sum
    moe_args.moe_dropout = MoE_params.moe_dropout
    moe_args.layer_freq = MoE_params.mode
    if moe_args.layer_freq == 'even':
        moe_args.num_layers = (num_of_blocks + 1) // 2
    elif moe_args.layer_freq == 'odd':
        moe_args.num_layers = num_of_blocks // 2 
    elif moe_args.layer_freq == 'all':
        moe_args.num_layers = num_of_blocks
    return moe_args

import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import torch
import torch.nn as nn
import math
from typing import Dict
from data.AddBiomechanicsDataset import InputDataKeys, OutputDataKeys
from megablocks.layers.dmoe import dMoE
from megablocks.layers.arguments import Arguments
from flash_attn import flash_attn_func
from models.BaseModule import RMSNorm, FFNSwiGLU, DropPath, ProjectionLayer, Head, GroundBase, config_MoE

##############################
# RoPE (Rotary Positional Encoding)
# https://github.com/meta-llama/llama3/blob/main/llama/model.py
##############################

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)

def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor):
    """
    Apply rotary positional embeddings to query and key.
    
    Args:
        xq: Query tensor of shape [B, T, num_heads, head_dim]
        xk: Key tensor of shape [B, T, num_heads, head_dim]
        freqs_cis: Precomputed complex frequencies of shape [T, head_dim]
    
    Returns:
        Tuple of (xq, xk) with RoPE applied.
    """
    # Reshape xq, xk to complex numbers: assume head_dim is even.
    B, T, num_heads, head_dim = xq.shape
    # Convert last dimension into pairs for complex numbers.
    xq_ = torch.view_as_complex(xq.reshape(B, T, num_heads, head_dim // 2, 2).float())
    xk_ = torch.view_as_complex(xk.reshape(B, T, num_heads, head_dim // 2, 2).float())
    
    # Reshape freqs_cis for broadcast: [T, head_dim] -> [1, T, 1, head_dim]
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_.view(B, T, num_heads, -1))
    # Multiply in complex domain
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(-2)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(-2)
    
    return xq_out.type_as(xq), xk_out.type_as(xk)

##############################

##############################
# PositionalEncoding
# https://github.com/codertimo/BERT-pytorch/blob/master/bert_pytorch/model/embedding/position.py#L6
##############################

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]

##############################


# TransformerBlock: Transformer block with Flash Attention, causal masking, and optional MoE MLP
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads=8, dropout=0.0, drop_path_rate=0.0, mlp_ratio=4.0, moe_args=None, causal=True):
        """
        Args:
            embed_dim (int): Embedding dimension.
            num_heads (int): Number of attention heads.
            dropout (float): Dropout rate.
            mlp_ratio (float): Ratio of hidden dimension to embedding dimension.
            moe_args (Arguments): Arguments for the Mixture of Experts (MoE) block.
            causal (bool): Whether to use causal attention (mask future tokens).
        """
        super(TransformerBlock, self).__init__()
        self.pre_norm = RMSNorm(embed_dim, eps=1e-6)
        self.mlp_norm = RMSNorm(embed_dim, eps=1e-6)
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # Linear projection to obtain Q, K, and V in one step.
        self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.flash_attn_dropout = dropout
        self.causal = causal
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()

        # Use MoE for the MLP if moe_args is provided, otherwise use standard FFNSwiGLU.
        if moe_args is not None:
            self.mlp = dMoE(moe_args)
        else:
            self.mlp = FFNSwiGLU(input_dim=embed_dim, hidden_dim=int(mlp_ratio * embed_dim), dropout=dropout)

    def forward(self, x, freqs_cis):
        """
        Args:
            x: Input tensor of shape [B, T, embed_dim].
            freqs_cis: Precomputed rotary frequencies for current sequence length, shape [T, head_dim].
        """
        # Self-Attention branch
        residual = x
        x_norm = self.pre_norm(x)
        B, T, C = x_norm.shape
        # Compute Q, K, V: shape becomes [B, T, 3, num_heads, head_dim]
        qkv = self.qkv_proj(x_norm).view(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, num_heads, T, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]   # Each: [B, num_heads, T, head_dim]
        # Transpose to [B, T, num_heads, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        # Apply RoPE: rotary positional encoding
        if freqs_cis is not None:
            q, k = apply_rotary_emb(q, k, freqs_cis)
        # Compute Flash Attention
        attn_out = flash_attn_func(q, k, v, dropout_p=self.flash_attn_dropout if self.training else 0.0, causal=self.causal)
        # Reshape back to [B, T, embed_dim]
        attn_out = attn_out.transpose(1, 2).reshape(B, T, C)
        attn_out = self.out_proj(attn_out)
        x = residual + self.drop_path(attn_out)
        
        # MLP branch
        x = x + self.drop_path(self.mlp(self.mlp_norm(x)))
        return x


class GroundTransformer(GroundBase):
    def __init__(self, config, device):
        super().__init__(config, device)

        # MoE
        self.MoE_params = getattr(config, 'MoE_params', None)
        self.moe_args = config_MoE(self.MoE_params, config.embed_dim, config.mlp_ratio, config.num_temporal_blocks, device)

        # Projection layer
        self.projection_layers = ProjectionLayer(self.input_size, config.embed_dim).to(self.device)

        # RoPE (Rotary Positional Encoding)
        self.use_rope = getattr(config, 'use_RoPE', True)
        self.rope_theta = getattr(config, 'rope_theta', 10000.0)    # 500000 in LLAMA3
        self.max_seq_len = config.max_seq_length
        self.head_dim = config.embed_dim // config.num_heads
        if self.use_rope:
            self.freqs_cis = precompute_freqs_cis(self.head_dim, self.max_seq_len, theta=self.rope_theta).to(self.device)
        else:
            # Traditional positional encoding
            # # learnable parameter of shape [max_seq_len, embed_dim]
            # self.pos_enc = nn.Parameter(torch.zeros(self.max_seq_len, config.embed_dim))
            # nn.init.normal_(self.pos_enc, std=0.02)
            # sinusoidal positional encoding
            self.pos_enc = PositionalEncoding(config.embed_dim, self.max_seq_len)
            self.freqs_cis = None
            
        # Transformer blocks (with MoE applied conditionally based on configuration)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim=config.embed_dim, num_heads=config.num_heads, dropout=config.drop_rate, drop_path_rate=config.drop_path_rate, mlp_ratio=config.mlp_ratio, causal=config.causal, 
            moe_args=(
                self.moe_args if (
                    ((self.moe_args.layer_freq == 'even' if self.moe_args is not None else False) and idx % 2 == 0) or
                    ((self.moe_args.layer_freq == 'odd' if self.moe_args is not None else False) and idx % 2 != 0) or
                    (self.moe_args.layer_freq == 'all' if self.moe_args is not None else False)
                ) else None
            ))
            for idx in range(config.num_temporal_blocks)
        ]).to(self.device)

        # MLP Head for final output
        self.head = Head(embed_dim=config.embed_dim, output_dim=self.output_size).to(self.device)

    def forward(self, input: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # 1. Parse input data
        # 1. Parse input data
        spatial_inputs, temporal_inputs, joint_center_inputs = self.input_parsing(input)

        # 2. Apply embedding layers
        combined_inputs = torch.concat([spatial_inputs, temporal_inputs], dim=-1)  # Shape: [batch_size, seq_len, input_size]
        fused_embeddings = self.projection_layers(combined_inputs)  # Shape: [batch_size, seq_len, embed_dim]

        seq_length = fused_embeddings.shape[1]
        if self.use_rope:
            # Use RoPE: slice freqs_cis to current sequence length.
            freqs_cis = self.freqs_cis[:seq_length]  # shape: [seq_length, head_dim//2]
        else:
            # Traditional positional encoding: add learned pos_enc.
            # fused_embeddings = fused_embeddings + self.pos_enc[:seq_length].unsqueeze(0)
            fused_embeddings = fused_embeddings + self.pos_enc(fused_embeddings)
            freqs_cis = None
        
        # 3. Apply Mamba blocks
        x = fused_embeddings
        for block in self.transformer_blocks:
            x = block(x, freqs_cis)

        # 4. Head MLP for final predictions
        outputs = self.head(x)
        output_cop = outputs[:, :, :6]
        output_force = outputs[:, :, 6:12]
        output_torque = outputs[:, :, 12:18]

        # 5. Apply CoP limiter
        if self.use_cop_limiter:
            output_cop = self.CoPLimiter(output_cop, joint_center_inputs)

        return {
            OutputDataKeys.GROUND_CONTACT_COPS_IN_ROOT_FRAME: output_cop,
            OutputDataKeys.GROUND_CONTACT_FORCES_IN_ROOT_FRAME: output_force,
            OutputDataKeys.GROUND_CONTACT_TORQUES_IN_ROOT_FRAME: output_torque,
        }
    

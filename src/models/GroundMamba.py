import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import torch
import torch.nn as nn
from mamba_ssm import Mamba2, Mamba
from typing import Dict
from data.AddBiomechanicsDataset import OutputDataKeys
from megablocks.layers.dmoe import dMoE
from megablocks.layers.arguments import Arguments
from models.BaseModule import RMSNorm, FFNSwiGLU, DropPath, ProjectionLayer, Head, GroundBase, config_MoE

# MambaBlock: A block using Mamba for modeling long-term dependencies in time series data.
class MambaBlock(nn.Module):
    def __init__(self, mamba_class, mamba_params, drop_path_rate=0.0, mlp_ratio=4.0, dropout=0.0, moe_args=None):
        super(MambaBlock, self).__init__()
        embed_dim = mamba_params['d_model']
        self.pre_norm = RMSNorm(embed_dim, eps=1e-6)
        self.mamba = mamba_class(**mamba_params)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        self.mlp_norm = RMSNorm(embed_dim, eps=1e-6)
        if moe_args is not None:
            self.mlp = dMoE(moe_args)
        else:
            self.mlp = FFNSwiGLU(input_dim=embed_dim, hidden_dim=int(mlp_ratio*embed_dim), dropout=dropout)

    def forward(self, x):
        x = self.drop_path(self.mamba(self.pre_norm(x))) + x
        x = self.drop_path(self.mlp(self.mlp_norm(x))) + x
        return x
    

# BMambaBlock: A bidirectional block using Mamba for modeling long-term dependencies in time series data.
class BMambaBlock(nn.Module):
    def __init__(self, mamba_class, mamba_params, drop_path_rate=0.0, mlp_ratio=4.0, dropout=0.0, bidirectional_mode='parallel', moe_args=None):
        super(BMambaBlock, self).__init__()
        self.bidirectional_mode = bidirectional_mode.lower()
        assert self.bidirectional_mode in ['parallel', 'serial'], "bidirectional_mode must be either 'parallel' or 'serial'"
        embed_dim = mamba_params['d_model']

        # Forward and backward Mamba layers
        self.mamba_forward = mamba_class(**mamba_params)
        self.mamba_backward = mamba_class(**mamba_params)

        # DropPath 
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()

        # RMSNorm layers
        self.pre_norm = RMSNorm(embed_dim, eps=1e-6)
        self.forward_norm = RMSNorm(embed_dim, eps=1e-6)
        self.backward_norm = RMSNorm(embed_dim, eps=1e-6)

        # MLP
        if moe_args is not None:
            self.mlp = dMoE(moe_args)
        else:
            self.mlp = FFNSwiGLU(input_dim=embed_dim, hidden_dim=int(mlp_ratio*embed_dim), dropout=dropout)

    def reverse_sequence(self, x):
        return torch.flip(x, dims=[1])

    def forward_parallel(self, x):
        # Forward processing
        out_forward = self.mamba_forward(x)
        out_forward = self.forward_norm(self.drop_path(out_forward))
        
        # Backward processing (flip the sequence along the time axis)
        out_backward = self.reverse_sequence(x)
        out_backward = self.mamba_backward(out_backward)
        out_backward = self.reverse_sequence(out_backward)
        out_backward = self.backward_norm(self.drop_path(out_backward))
        
        # Combine forward and backward outputs
        out_combined = out_forward + out_backward
        
        return out_combined

    def forward_serial(self, x):
        # Forward processing
        out_forward = self.mamba_forward(x)
        
        # Combine with residual
        out_forward = x + self.forward_norm(self.drop_path(out_forward))
        
        # Backward processing (flip the sequence along the time axis)
        out_backward = self.reverse_sequence(out_forward)
        out_backward = self.mamba_backward(out_backward)
        out_backward = self.reverse_sequence(out_backward)
        
        # Combine forward and backward outputs
        out_combined = out_forward + self.backward_norm(self.drop_path(out_backward))
        
        return out_combined

    def forward(self, x):
        res = x
        x = self.pre_norm(x)
        if self.bidirectional_mode == 'parallel':
            out_combined = self.forward_parallel(x)
        elif self.bidirectional_mode == 'serial':
            out_combined = self.forward_serial(x)
        else:
            raise ValueError("Invalid bidirectional_mode. Choose 'parallel' or 'serial'.")
        
        # Residual connection with DropPath and MLP
        x = self.mlp(out_combined)
        x = res + x
        return x
    

# GroundMamba: A model for predicting ground reaction forces and moments using Mamba.
class GroundMamba(GroundBase):
    def __init__(self, config, device):
        super().__init__(config, device)

        # Assign configurations from config
        self.temporal_mamba_params = {'d_model': config.temporal_embed_dim, 'd_state': config.temporal_d_state, 'd_conv': config.temporal_d_conv, 'expand': config.temporal_expand}
        self.mamba_ver = getattr(config, 'mamba_ver', 'Mamba2')
        if self.mamba_ver == 'Mamba2':
            mamba_class = Mamba2
            min_headdim = int(self.temporal_mamba_params['d_model']*self.temporal_mamba_params['expand']/8)
            self.temporal_mamba_params['headdim'] = min_headdim if min_headdim < 64 else 64                
        elif self.mamba_ver == 'Mamba':
            mamba_class = Mamba
        else:
            raise ValueError(f"Invalid mamba_ver: {self.mamba_ver}. Choose 'Mamba' or 'Mamba2'.")
        
        # MoE
        self.MoE_params = getattr(config, 'MoE_params', None)
        self.moe_args = config_MoE(self.MoE_params, self.temporal_mamba_params['d_model'], config.mlp_ratio, config.num_temporal_blocks, device)

        # Projection layer
        self.projection_layers = ProjectionLayer(self.input_size, self.temporal_mamba_params['d_model'], bias=True).to(self.device)

        # Mamba blocks repeated `num_mamba_blocks` times
        # MoE block is used for every even-indexed block
        temporal_bidirectional_mode = getattr(config, 'temporal_bidirectional_mode', 'single')
        if temporal_bidirectional_mode == 'single':
            self.temporal_mamba_blocks = nn.ModuleList([
                MambaBlock(mamba_class=mamba_class, mamba_params=self.temporal_mamba_params, drop_path_rate=config.drop_path_rate, mlp_ratio=config.mlp_ratio, dropout=config.drop_rate, 
                moe_args=(
                    self.moe_args if (
                        ((self.moe_args.layer_freq == 'even' if self.moe_args is not None else False) and idx % 2 == 0) or
                        ((self.moe_args.layer_freq == 'odd' if self.moe_args is not None else False) and idx % 2 != 0) or
                        (self.moe_args.layer_freq == 'all' if self.moe_args is not None else False)
                    ) else None
                ))
                for idx in range(config.num_temporal_blocks)
            ]).to(self.device)
        elif temporal_bidirectional_mode == 'parallel' or temporal_bidirectional_mode == 'serial':
            self.temporal_mamba_blocks = nn.ModuleList([
                BMambaBlock(mamba_class=mamba_class, mamba_params=self.temporal_mamba_params, drop_path_rate=config.drop_path_rate, mlp_ratio=config.mlp_ratio, dropout=config.drop_rate, bidirectional_mode=temporal_bidirectional_mode, 
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
        self.head = Head(embed_dim=self.temporal_mamba_params['d_model'], output_dim=self.output_size).to(self.device)

    def forward(self, input: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # 1. Parse input data
        spatial_inputs, temporal_inputs, joint_center_inputs = self.input_parsing(input)

        # 2. Apply embedding layers
        combined_inputs = torch.concat([spatial_inputs, temporal_inputs], dim=-1)  # Shape: [batch_size, seq_len, input_size]
        x = self.projection_layers(combined_inputs)  # Shape: [batch_size, seq_len, embed_dim]

        # 3. Apply Mamba blocks
        for block in self.temporal_mamba_blocks:
            x = block(x)

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
    
    
class GroundSTMamba(GroundMamba):
    def __init__(self, config, device):
        super().__init__(config, device)
        self.__scan_order__()

        # Assign configurations from config
        self.spatial_mamba_params = {'d_model': config.spatial_embed_dim, 'd_state': config.spatial_d_state, 'd_conv': config.spatial_d_conv, 'expand': config.spatial_expand}
        if self.mamba_ver == 'Mamba2':
            mamba_class = Mamba2
            spatial_min_headdim = int(self.spatial_mamba_params['d_model']*self.spatial_mamba_params['expand']/8)
            self.spatial_mamba_params['headdim'] = spatial_min_headdim if spatial_min_headdim < 64 else 64           
        elif self.mamba_ver == 'Mamba':
            mamba_class = Mamba
        else:
            raise ValueError(f"Invalid mamba_ver: {self.mamba_ver}. Choose 'Mamba' or 'Mamba2'.")
        self.chunk_size = getattr(config, 'chunk_size', None)

        # PELVIS(0)~COM(12)
        self.node_embedding = nn.Embedding(self.num_unique_nodes, self.spatial_mamba_params['d_model'])
        self.node_ids = torch.tensor(self.scan_order, dtype=torch.long, device=self.device)

        # Spatial projection layer
        self.spatial_projection_layer = ProjectionLayer(3, self.spatial_mamba_params['d_model'])  # 3 (x, y, z)

        # Spatial Mamba blocks
        spatial_bidirectional_mode = getattr(config, 'spatial_bidirectional_mode', 'single')
        if spatial_bidirectional_mode == 'single':
            self.spatial_mamba_blocks = nn.ModuleList([
                MambaBlock(mamba_class=mamba_class, mamba_params=self.spatial_mamba_params, drop_path_rate=config.drop_path_rate, mlp_ratio=config.mlp_ratio, dropout=config.drop_rate)
                for _ in range(config.num_spatial_blocks)
            ]).to(self.device)
        elif spatial_bidirectional_mode == 'parallel' or spatial_bidirectional_mode == 'serial':
            self.spatial_mamba_blocks = nn.ModuleList([
                BMambaBlock(mamba_class=mamba_class, mamba_params=self.spatial_mamba_params, drop_path_rate=config.drop_path_rate, mlp_ratio=config.mlp_ratio, dropout=config.drop_rate, bidirectional_mode=spatial_bidirectional_mode)
                for _ in range(config.num_spatial_blocks)
            ]).to(self.device)

        # Spatial to Temporal
        projection_dim_unit = int((self.temporal_mamba_params['d_model'] - (self.num_nodes - len(self.pelvis_idx)) * self.spatial_mamba_params['d_model']))
        if projection_dim_unit < 0 or projection_dim_unit % 2 == 1:
            raise ValueError("(temporal_embed_dim > num_nodes*spatial_embed_dim) must be True.")

        # Temporal projection layers
        self.temporal_projection_layers = ProjectionLayer(self.input_size - self.spatial_input_size, int(projection_dim_unit), bias=True).to(self.device)

    def __scan_order__(self):
        # Define joint indices
        PELVIS = 0
        HIP_R, KNEE_R, ANKLE_R, SUBTALAR_R, MTP_R = 1, 2, 3, 4, 5
        HIP_L, KNEE_L, ANKLE_L, SUBTALAR_L, MTP_L = 6, 7, 8, 9, 10
        BACK = 11 # Fixed Point in Root Body
        COM = 12
        self.num_unique_nodes = 13
        
        # Define scan orders
        # leg_right = [HIP_R, KNEE_R, ANKLE_R, SUBTALAR_R, MTP_R]
        # leg_left = [HIP_L, KNEE_L, ANKLE_L, SUBTALAR_L, MTP_L]
        # self.scan_order = [PELVIS] + leg_right + [PELVIS, COM, PELVIS] + leg_left + [PELVIS]
        # self.scan_order = [PELVIS] + leg_left + [PELVIS, COM, PELVIS] + leg_right + [PELVIS]
        self.scan_order = [
            PELVIS, 
            HIP_R, KNEE_R, ANKLE_R, SUBTALAR_R, MTP_R,
            PELVIS, 
            COM,
            PELVIS, 
            HIP_L, KNEE_L, ANKLE_L, SUBTALAR_L, MTP_L,
            PELVIS
            ]
        # self.scan_order = [
        #     HIP_R, KNEE_R, ANKLE_R, SUBTALAR_R, MTP_R,
        #     COM,
        #     HIP_L, KNEE_L, ANKLE_L, SUBTALAR_L, MTP_L
        #     ]
        self.pelvis_idx = [i for i, node in enumerate(self.scan_order) if node == 0]
        self.num_nodes = len(self.scan_order)

    def remove_pelvis_nodes(self, x):
        """
        Removes the pelvis nodes from the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape [B, T, N, D].

        Returns:
            torch.Tensor: Tensor with pelvis nodes removed, shape [B, T, N', D].
        """
        B, T, N, D = x.shape
        mask = torch.ones(N, dtype=torch.bool, device=x.device)
        mask[self.pelvis_idx] = False
        return x[:, :, mask, :]

    def Spatial_Embedding(self, spatial_inputs: torch.Tensor) -> torch.Tensor:
        """
        Performs spatial embedding on the input positions.

        Args:
            spatial_inputs (torch.Tensor): [Batch, Seq, (Num_Joints + 1) * 3]

        Returns:
            torch.Tensor: [Batch, Seq, Num_Nodes, Spatial_Embed_Dim]
        """
        batch_size, seq_len, _ = spatial_inputs.shape
        spatial_pos = spatial_inputs.view(batch_size, seq_len, self.num_unique_nodes, 3)  # [Batch, Seq, (Num_Joints + 1), 3]
        spatial_embeddings = self.spatial_projection_layer(spatial_pos)  # [Batch, Seq, (Num_Joints + 1), Spatial_Embed_Dim]
        spatial_embeddings_scan = spatial_embeddings[:, :, self.scan_order, :]  # [Batch, Seq, Num_Nodes, Spatial_Embed_Dim]

        # Adding node embeddings
        node_pos_embed = self.node_embedding(self.node_ids)
        node_pos_embed = node_pos_embed.unsqueeze(0).unsqueeze(0)   # [1, 1, Num_Nodes, Spatial_Embed_Dim]
        node_pos_embed = node_pos_embed.expand(batch_size, seq_len, self.num_nodes, self.spatial_mamba_params['d_model']) # [B, S, Num_Nodes, Spatial_Embed_DimD]

        return spatial_embeddings_scan + node_pos_embed
    
    def Spatial_Learning(self, spatial_embeddings: torch.Tensor, block: nn.Module, chunk_size: int = None) -> torch.Tensor:
        """
        Optimized spatial learning using chunking.

        Args:
            spatial_embeddings (torch.Tensor): [Batch, Seq, Num_Nodes, Spatial_Embed_Dim]
            block (nn.Module): Spatial Mamba block to apply
            chunk_size (int, optional): Number of sequence steps to process in each chunk. Default is None, which means all at once.

        Returns:
            torch.Tensor: [Batch, Seq, Num_Nodes, Spatial_Embed_Dim]
        """
        batch_size, seq_len, num_nodes, embed_dim = spatial_embeddings.shape

        # Default to full parallel processing if chunk_size is None
        if chunk_size is None or chunk_size >= seq_len:
            x = spatial_embeddings.view(batch_size * seq_len, num_nodes, embed_dim)  # [Batch * Seq, Num_Nodes, Spatial_Embed_Dim]
            spatial_outputs = block(x).view(batch_size, seq_len, num_nodes, embed_dim)  # Reshape back
        else:
            spatial_outputs = torch.empty_like(spatial_embeddings)  # Initialize with empty_like for memory efficiency
            for start in range(0, seq_len, chunk_size):
                end = min(start + chunk_size, seq_len)
                x_chunk = spatial_embeddings[:, start:end, :, :]  # Removed .contiguous() for optimization
                x_chunk = x_chunk.view(-1, num_nodes, embed_dim)  # [Batch * Chunk_Size, Num_Nodes, Spatial_Embed_Dim]
                x_chunk = block(x_chunk)  # Process block on each chunk
                spatial_outputs[:, start:end, :, :] = x_chunk.view(batch_size, end - start, num_nodes, embed_dim)  # Reshape back

        return spatial_outputs

    def forward(self, input: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # Step 1: Input Parsing
        spatial_inputs, temporal_inputs, joint_center_inputs = self.input_parsing(input)

        # Step 2: Spatial Embedding
        spatial_embeddings = self.Spatial_Embedding(spatial_inputs)  # [Batch, Seq, Num_Nodes, Spatial_Embed_Dim]

        # Step 3: Spatial Learning
        x_spatial = spatial_embeddings
        for block in self.spatial_mamba_blocks:
            x_spatial = self.Spatial_Learning(x_spatial, block, chunk_size=self.chunk_size)

        # Step 4: Spatial to Temporal Conversion
        x_spatial = self.remove_pelvis_nodes(x_spatial)  # Remove pelvis nodes
        spatial_temporal_embedding = x_spatial.view(x_spatial.size(0), x_spatial.size(1), -1)   # [Batch, Seq, Num_Nodes(pelvis removed) * Spatial_Embed_Dim]

        # Step 5: Temporal Embedding
        temporal_embeddings = self.temporal_projection_layers(temporal_inputs)
        fused_embeddings = torch.concat([spatial_temporal_embedding, temporal_embeddings], dim=-1)  # [Batch, Seq, Temporal_Embed_Dim]
        
        # Step 6: Temporal Learning
        x_temporal = fused_embeddings
        for block in self.temporal_mamba_blocks:
            x_temporal = block(x_temporal)

        # Step 7: Final Predictions
        outputs = self.head(x_temporal)  # [Batch, Seq, Output_Size]

        # Step 8: Extract Outputs
        output_cop = outputs[:, :, :6]
        output_force = outputs[:, :, 6:12]
        output_torque = outputs[:, :, 12:18]

        # Step 9: Apply CoP Limiter
        if self.use_cop_limiter:
            output_cop = self.CoPLimiter(output_cop, joint_center_inputs)

        return {
            OutputDataKeys.GROUND_CONTACT_COPS_IN_ROOT_FRAME: output_cop,
            OutputDataKeys.GROUND_CONTACT_FORCES_IN_ROOT_FRAME: output_force,
            OutputDataKeys.GROUND_CONTACT_TORQUES_IN_ROOT_FRAME: output_torque,
        }
import torch
import torch.nn as nn
from typing import Dict, List
from data.AddBiomechanicsDataset import OutputDataKeys
from models.BaseModule import RMSNorm, ProjectionLayer, Head, GroundBase

class GroundFFN(GroundBase):
    def __init__(self, config, device):
        super().__init__(config, device)
        
        hidden_dims = getattr(config, 'hidden_dims', [512, 512])

        self.projection = ProjectionLayer(self.input_size, hidden_dims[0], bias=True).to(self.device)
        
        layers = []
        for i in range(len(hidden_dims) - 1):
            layers.append(RMSNorm(hidden_dims[i], eps=1e-6))
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1], bias=True))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(config.drop_rate))
        self.ffn = nn.Sequential(*layers).to(self.device)
        
        self.head = Head(embed_dim=hidden_dims[-1], output_dim=self.output_size).to(self.device)
        
    def forward(self, input: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        spatial_inputs, temporal_inputs, joint_center_inputs = self.input_parsing(input)
        x = torch.concat([spatial_inputs, temporal_inputs], dim=-1)  # shape: [B, T, input_size]

        x = self.projection(x)  # shape: [B, T, embed_dim]
        x = self.ffn(x)
        outputs = self.head(x)  # shape: [B, T, output_size]
        
        output_cop = outputs[:, :, :6]
        output_force = outputs[:, :, 6:12]
        output_torque = outputs[:, :, 12:18]
        
        if self.use_cop_limiter:
            output_cop = self.CoPLimiter(output_cop, joint_center_inputs)

        return {
            OutputDataKeys.GROUND_CONTACT_COPS_IN_ROOT_FRAME: output_cop,
            OutputDataKeys.GROUND_CONTACT_FORCES_IN_ROOT_FRAME: output_force,
            OutputDataKeys.GROUND_CONTACT_TORQUES_IN_ROOT_FRAME: output_torque,
        }
    
"""
GroundLink-style network.
- keep the GroundLink core layers: Dropout -> Conv1d -> ELU stacks, then Dropout -> Linear -> ELU stacks.

Original GroundLink code reference:
https://github.com/hanxingjian/GroundLink/blob/main/UnderPressure/models.py

@inproceedings{
  han2023groundlink,
  title = {GroundLink: A Dataset Unifying Human Body Movement and Ground Reaction Dynamics},
  author={Han, Xingjian and Senderling, Benjamin and To, Stanley and Kumar, Deepak and Whiting, Emily and Saito, Jun},
  booktitle={ACM SIGGRAPH Asia 2023 Conference Proceedings},
  year = {2023},
  pages = {1--10},
}
"""

import torch
import torch.nn as nn
from typing import Dict

from data.AddBiomechanicsDataset import OutputDataKeys
from models.BaseModule import GroundBase


class Transpose(nn.Module):
    """Same helper module as in the original GroundLink implementation."""
    def __init__(self, dim1: int, dim2: int):
        super().__init__()
        self._dim1, self._dim2 = dim1, dim2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.transpose(self._dim1, self._dim2)


class GroundLink(GroundBase):
    """
    GroundLink backbone with CoPLimiter.
    """
    def __init__(self, config, device):
        super().__init__(config, device)

        # Hyperparameters (defaults match the original GroundLink file)
        self.cnn_kernel: int = getattr(config, "cnn_kernel", 7)
        self.cnn_dropout: float = getattr(config, "cnn_dropout", 0.0)
        self.fc_depth: int = getattr(config, "fc_depth", 3)
        self.fc_dropout: float = getattr(config, "fc_dropout", 0.2)

        # Input is [B, T, C=input_size]
        self.to_channels_first = Transpose(-2, -1)  # [B, T, C] -> [B, C, T]
        self.to_time_first = Transpose(-2, -1)      # [B, C, T] -> [B, T, C]

        # Convolutional part
        cnn_channels = getattr(config, 'cnn_channels', [128, 128, 256, 256])
        cnn_features = [self.input_size] + cnn_channels

        def conv(c_in: int, c_out: int) -> nn.Module:
            return nn.Conv1d(
                c_in,
                c_out,
                kernel_size=self.cnn_kernel,
                padding=self.cnn_kernel // 2,
                padding_mode="replicate",
            )

        cnn_layers = []
        for c_in, c_out in zip(cnn_features[:-1], cnn_features[1:]):
            cnn_layers += [
                nn.Dropout(p=self.cnn_dropout),
                conv(c_in, c_out),
                nn.ELU(),
            ]
        self.cnn = nn.Sequential(*cnn_layers).to(self.device)

        # Fully connected part
        hidden_dim = cnn_features[-1]  # 256
        fc_layers = []
        for _ in range(max(self.fc_depth - 1, 0)):
            fc_layers += [
                nn.Dropout(p=self.fc_dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ELU(),
            ]
        fc_layers += [
            nn.Dropout(p=self.fc_dropout),
            nn.Linear(hidden_dim, self.output_size, bias=False),
        ]
        self.fc = nn.Sequential(*fc_layers).to(self.device)

        # Initialization
        if getattr(config, "groundlink_init", True):
            self._initialize_layer(self.cnn)
            self._initialize_layer(self.fc)

    @staticmethod
    def _initialize_layer(net: nn.Sequential) -> None:
        """Match the original GroundLink initialization logic (ELU uses ReLU gain)."""
        GAINS = {
            nn.Sigmoid:   nn.init.calculate_gain("sigmoid"),
            nn.ReLU:      nn.init.calculate_gain("relu"),
            nn.LeakyReLU: nn.init.calculate_gain("leaky_relu"),
            nn.ELU:       nn.init.calculate_gain("relu"),
            nn.Softplus:  nn.init.calculate_gain("relu"),
        }
        layers = list(net)
        for layer, activation in zip(layers[:-1], layers[1:]):
            if len(list(layer.parameters())) > 0 and type(activation) in GAINS:
                gain = GAINS[type(activation)]
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_normal_(layer.weight, gain)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
                elif isinstance(layer, nn.Conv1d):
                    nn.init.xavier_normal_(layer.weight, gain)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

    def forward(self, input: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        spatial_inputs, temporal_inputs, joint_center_inputs = self.input_parsing(input)

        # [B, T, input_size]
        x = torch.concat([spatial_inputs, temporal_inputs], dim=-1)

        # [B, C, T] -> CNN -> [B, 256, T] -> [B, T, 256]
        x = self.to_channels_first(x)
        x = self.cnn(x)
        x = self.to_time_first(x)

        outputs = self.fc(x)  # [B, T, output_size]

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


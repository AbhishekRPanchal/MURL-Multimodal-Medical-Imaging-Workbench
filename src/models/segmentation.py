"""
Ultrasound Segmentation Models
Leverages MONAI's official medical imaging U-Net architecture (monai.networks.nets.UNet).
"""

import torch
import torch.nn as nn
from monai.networks.nets import UNet

class UltrasoundUNet(nn.Module):
    """
    Ultrasound Lesion Segmentation network based on MONAI's 2D UNet implementation.
    Configured with residual units for enhanced gradient propagation across
    ultrasound acoustic boundaries and deep feature maps.
    """
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        channels: tuple = (16, 32, 64, 128),
        strides: tuple = (2, 2, 2),
        num_res_units: int = 1,
        dropout: float = 0.2
    ):
        super().__init__()
        self.unet = UNet(
            spatial_dims=2,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=channels,
            strides=strides,
            num_res_units=num_res_units,
            dropout=dropout
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Returns raw logits for loss calculation; apply sigmoid for probabilities
        return self.unet(x)

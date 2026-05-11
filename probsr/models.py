from typing import *
"""
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_wavelets import DWTForward, DWTInverse


class WaveletModule(nn.Module):
    """
    Wavelet transformation module for multi-resolution analysis
    """

    def __init__(self, wave='sym4', J=4, device='cuda'):
        super().__init__()
        # 重命名属性避免与方法名冲突
        self.dwt_transform = DWTForward(J=J, wave=wave, mode='symmetric').to(device)
        self.idwt_transform = DWTInverse(wave=wave, mode='symmetric').to(device)

    def dwt(self, x):
        """Convenience method for forward wavelet transform"""
        return self.analysis(x)

    def idwt(self, coeffs):
        """Convenience method for inverse wavelet transform - ADDED THIS METHOD"""
        return self.synthesis(coeffs)

    def analysis(self, x):
        """
        Forward wavelet transform (analysis)
        x: (B, C, H, W)
        Returns: tuple (Yl, Yh) where Yl is low-frequency, Yh is high-frequency coefficients
        """
        Yl, Yh = self.dwt_transform(x)  # 使用重命名后的属性
        return (Yl, Yh)

    def synthesis(self, coeffs):
        """Inverse wavelet transform (synthesis)"""
        return self.idwt_transform(coeffs)


class DownscaleResNet(nn.Module):
    """
    Supports both single- and multi-channel input/output
    Automatically matches output channels = input channels
    """

    def __init__(self, in_ch=1, out_size=(40, 40), n_res_blocks=4):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = in_ch  # output same number of channels as input
        self.out_size = out_size

        # Initial convolution
        self.conv1 = nn.Conv2d(in_ch, 64, 3, padding=1)

        # Residual blocks with proper residual connections
        res_layers = []
        for i in range(n_res_blocks):
            res_layers.append(self._make_residual_block(64, 64))
        self.res_blocks = nn.Sequential(*res_layers)

        # Output head
        self.head = nn.Conv2d(64, self.out_ch, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(out_size)

        # Xavier initialization for stability
        self._initialize_weights()

    def _make_residual_block(self, in_channels, out_channels):
        """Create a residual block with proper skip connection"""
        return ResidualBlock(in_channels, out_channels)

    def _initialize_weights(self):
        """Xavier initialization for all convolutional layers"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        """
        Forward pass
        x: (B, C, H, W)
        Returns: (B, C, out_size[0], out_size[1])
        """
        # Initial feature extraction
        z = F.relu(self.conv1(x))

        # Residual blocks with skip connections
        residual = z
        z = self.res_blocks(z)
        z = z + residual  # Residual connection

        # Output projection and pooling
        z = self.head(z)
        return self.pool(z)


class ResidualBlock(nn.Module):
    """
    Standard residual block implementation following ResNet architecture
    Uses proper batch normalization and residual connections
    """

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

        # Downsample if dimensions change
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x

        # Main path
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # Skip connection
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


# Utility functions for creating specific model instances
def create_wavelet_model(wave='sym4', J=3, device='cuda'):
    """Convenience function to create a WaveletModule instance"""
    return WaveletModule(wave=wave, J=J, device=device)


def create_downscale_resnet(in_channels=1, out_size=(40, 40), blocks=4):
    """Convenience function to create a DownscaleResNet instance"""
    return DownscaleResNet(in_ch=in_channels, out_size=out_size, n_res_blocks=blocks)


if __name__ == "__main__":
    # Test the WaveletModule
    print("Testing Fixed WaveletModule...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wavelet_model = WaveletModule(device=device)

    test_input = torch.randn(2, 3, 128, 128).to(device)

    Yl, Yh = wavelet_model.dwt(test_input)
    print(f"dwt method works: Input shape {test_input.shape}, Output Yl shape {Yl.shape}")

    reconstructed = wavelet_model.idwt((Yl, Yh))
    print(f"idwt method works: Reconstructed shape {reconstructed.shape}")

    Yl2, Yh2 = wavelet_model.analysis(test_input)
    reconstructed2 = wavelet_model.synthesis((Yl2, Yh2))
    print(f"analysis/synthesis methods work: Reconstructed shape {reconstructed2.shape}")
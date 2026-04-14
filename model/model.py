"""
Masked Autoencoder — Enhanced UNet-style Convolutional Autoencoder.

Architecture
────────────
The model follows an encoder–bottleneck–decoder pattern with skip
connections (like UNet) and residual blocks for better reconstruction.

    Input  (3 × 64 × 64)  — masked image (some patches zeroed)
    ├── Encoder 1 → 64 ch,  64×64
    │   └── MaxPool → 32×32
    ├── Encoder 2 → 128 ch, 32×32
    │   └── MaxPool → 16×16
    ├── Encoder 3 → 256 ch, 16×16
    │   └── MaxPool → 8×8
    ├── Encoder 4 → 512 ch, 8×8
    │   └── MaxPool → 4×4
    ├── Bottleneck → 1024 ch, 4×4 (with Dropout + Residual)
    ├── Decoder 4 → 512 ch, 8×8  (+ skip from Enc 4)
    ├── Decoder 3 → 256 ch, 16×16 (+ skip from Enc 3)
    ├── Decoder 2 → 128 ch, 32×32 (+ skip from Enc 2)
    ├── Decoder 1 →  64 ch, 64×64 (+ skip from Enc 1)
    └── Output   →   3 ch, 64×64 (Sigmoid → [0, 1])

Key improvements:
  • Deeper encoder (4 levels instead of 3)
  • Larger bottleneck (1024 channels)
  • Residual connections in decoder blocks
  • Dropout in bottleneck for regularization
  • Better output head with intermediate layer
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Double convolution with residual connection: Conv3×3 → BN → ReLU  ×2."""

    def __init__(self, in_channels: int, out_channels: int, use_residual: bool = False):
        super().__init__()
        self.use_residual = use_residual and (in_channels == out_channels)
        
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        out = self.block(x)
        if self.use_residual:
            out = out + x
        return out


class MaskedAutoencoder(nn.Module):
    """
    UNet-style autoencoder trained in a self-supervised manner.

    The model receives a *masked* image (patches set to 0) and learns to
    reconstruct the full, unmasked image.  Skip connections between
    corresponding encoder and decoder stages help preserve spatial detail.
    """

    def __init__(self, in_channels: int = 3):
        super().__init__()

        # ── Encoder ──────────────────────────────────────────────────────
        self.enc1 = ConvBlock(in_channels, 64, use_residual=False)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = ConvBlock(64, 128, use_residual=False)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = ConvBlock(128, 256, use_residual=False)
        self.pool3 = nn.MaxPool2d(2)

        self.enc4 = ConvBlock(256, 512, use_residual=False)
        self.pool4 = nn.MaxPool2d(2)

        # ── Bottleneck ───────────────────────────────────────────────────
        # Increased capacity to 1024 for better feature representation
        self.bottleneck = nn.Sequential(
            ConvBlock(512, 1024, use_residual=False),
            nn.Dropout2d(0.2),  # Add dropout for regularization
            ConvBlock(1024, 1024, use_residual=True),
        )

        # ── Decoder (with skip connections) ──────────────────────────────
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = ConvBlock(1024, 512, use_residual=True)  # 512 (up) + 512 (skip)

        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(512, 256, use_residual=True)   # 256 (up) + 256 (skip)

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(256, 128, use_residual=True)   # 128 (up) + 128 (skip)

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(128, 64, use_residual=True)    # 64  (up) + 64  (skip)

        # ── Output head ─────────────────────────────────────────────────
        self.output_conv = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, in_channels, kernel_size=1),
            nn.Sigmoid(),  # pixel values in [0, 1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (B, C, H, W) masked input image.
        Returns:
            Reconstructed image of the same shape.
        """
        # Encoder
        e1 = self.enc1(x)                       # (B,  64, 64, 64)
        e2 = self.enc2(self.pool1(e1))          # (B, 128, 32, 32)
        e3 = self.enc3(self.pool2(e2))          # (B, 256, 16, 16)
        e4 = self.enc4(self.pool3(e3))          # (B, 512,  8,  8)

        # Bottleneck
        b = self.bottleneck(self.pool4(e4))     # (B, 1024, 4,  4)

        # Decoder
        d4 = self.up4(b)                        # (B, 512,  8,  8)
        d4 = self.dec4(torch.cat([d4, e4], 1)) # (B, 512,  8,  8)

        d3 = self.up3(d4)                       # (B, 256, 16, 16)
        d3 = self.dec3(torch.cat([d3, e3], 1)) # (B, 256, 16, 16)

        d2 = self.up2(d3)                       # (B, 128, 32, 32)
        d2 = self.dec2(torch.cat([d2, e2], 1)) # (B, 128, 32, 32)

        d1 = self.up1(d2)                       # (B,  64, 64, 64)
        d1 = self.dec1(torch.cat([d1, e1], 1)) # (B,  64, 64, 64)

        return self.output_conv(d1)             # (B,   3, 64, 64)

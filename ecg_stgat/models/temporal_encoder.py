"""
Per-Lead 1D-ResNet and 1D-CNN Temporal Feature Extractor.
Processes 12 leads independently via shared weights: (B, 12, T) -> (B, 12, D).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResNetBlock1D(nn.Module):
    """Residual Convolutional Block with 1D convolutions."""
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=7, stride=stride, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.GELU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.relu(out + res)
        return out


class TemporalResNetEncoder(nn.Module):
    """
    Shared 1D-ResNet encoder applied per lead.
    Maps (B, 12, T) -> (B, 12, embed_dim).
    """
    def __init__(self, in_channels: int = 1, embed_dim: int = 128):
        super().__init__()
        self.embed_dim = embed_dim
        # Initial stem
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )

        # Residual stages
        self.layer1 = ResNetBlock1D(32, 64, stride=2)
        self.layer2 = ResNetBlock1D(64, 128, stride=2)
        self.layer3 = ResNetBlock1D(128, embed_dim, stride=2)

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (B, 12, T)
        Returns:
            H_temp: Tensor of shape (B, 12, embed_dim)
        """
        B, num_leads, T = x.shape
        # Flatten leads into batch: (B * 12, 1, T)
        x_flat = x.view(B * num_leads, 1, T)
        h = self.stem(x_flat)
        h = self.layer1(h)
        h = self.layer2(h)
        h = self.layer3(h)
        pooled = self.global_pool(h).squeeze(-1)  # (B * 12, embed_dim)
        out = self.proj(pooled)
        return out.view(B, num_leads, self.embed_dim)

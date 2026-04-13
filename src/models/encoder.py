"""Encoders for dense and object-centric world models."""

from __future__ import annotations

import torch
from torch import nn


class ConvEncoder1D(nn.Module):
    """Small encoder from binary CA state to a spatial latent map."""

    def __init__(self, in_channels: int = 1, latent_channels: int = 32, depth: int = 3) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Conv1d(in_channels, latent_channels, 5, padding=2, padding_mode="circular"), nn.GELU()]
        for _ in range(depth - 1):
            layers.extend([nn.Conv1d(latent_channels, latent_channels, 3, padding=1, padding_mode="circular"), nn.GELU()])
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ConvEncoder2D(nn.Module):
    """2D convolutional encoder."""

    def __init__(self, in_channels: int = 1, latent_channels: int = 32, depth: int = 3) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Conv2d(in_channels, latent_channels, 5, padding=2, padding_mode="circular"), nn.GELU()]
        for _ in range(depth - 1):
            layers.extend([nn.Conv2d(latent_channels, latent_channels, 3, padding=1, padding_mode="circular"), nn.GELU()])
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GlobalVectorEncoder1D(nn.Module):
    """Encode a 1D state to a global latent vector."""

    def __init__(self, latent_dim: int = 64, hidden_channels: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, hidden_channels, 5, padding=2, padding_mode="circular"),
            nn.GELU(),
            nn.Conv1d(hidden_channels, hidden_channels, 3, padding=1, padding_mode="circular"),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Linear(hidden_channels, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = self.net(x).squeeze(-1)
        return self.head(pooled)

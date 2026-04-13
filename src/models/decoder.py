"""Decoders for state reconstruction."""

from __future__ import annotations

import torch
from torch import nn


class ConvDecoder1D(nn.Module):
    def __init__(self, latent_channels: int = 32, out_channels: int = 1, depth: int = 3) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        for _ in range(depth - 1):
            layers.extend([nn.Conv1d(latent_channels, latent_channels, 3, padding=1, padding_mode="circular"), nn.GELU()])
        layers.append(nn.Conv1d(latent_channels, out_channels, 3, padding=1, padding_mode="circular"))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class ConvDecoder2D(nn.Module):
    def __init__(self, latent_channels: int = 32, out_channels: int = 1, depth: int = 3) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        for _ in range(depth - 1):
            layers.extend([nn.Conv2d(latent_channels, latent_channels, 3, padding=1, padding_mode="circular"), nn.GELU()])
        layers.append(nn.Conv2d(latent_channels, out_channels, 3, padding=1, padding_mode="circular"))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class VectorDecoder1D(nn.Module):
    """Decode a global vector back to 1D logits."""

    def __init__(self, latent_dim: int, length: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.length = length
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, length),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z).unsqueeze(1)

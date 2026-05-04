"""Decoders for state reconstruction."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ConvDecoder1D(nn.Module):
    def __init__(
        self,
        latent_channels: int = 32,
        out_channels: int = 1,
        depth: int = 3,
        output_length: int | None = None,
    ) -> None:
        super().__init__()
        self.output_length = int(output_length) if output_length is not None else None
        layers: list[nn.Module] = []
        for _ in range(depth - 1):
            layers.extend([nn.Conv1d(latent_channels, latent_channels, 3, padding=1, padding_mode="circular"), nn.GELU()])
        layers.append(nn.Conv1d(latent_channels, out_channels, 3, padding=1, padding_mode="circular"))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if self.output_length is not None and z.shape[-1] != self.output_length:
            z = F.interpolate(z, size=self.output_length, mode="linear", align_corners=False)
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


class LearnedBottleneckDecoder1D(nn.Module):
    """Decode a fixed-size [B, C, S] bottleneck to position-wise 1D logits."""

    def __init__(
        self,
        latent_channels: int,
        latent_length: int,
        output_length: int,
        hidden_dim: int = 256,
        position_dim: int = 32,
    ) -> None:
        super().__init__()
        self.output_length = int(output_length)
        latent_dim = int(latent_channels) * int(latent_length)
        self.context = nn.Sequential(
            nn.Flatten(),
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
        )
        self.position_embedding = nn.Parameter(torch.randn(self.output_length, position_dim) * 0.02)
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim + position_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        context = self.context(z)
        context = context[:, None, :].expand(-1, self.output_length, -1)
        position = self.position_embedding[None, :, :].expand(z.shape[0], -1, -1)
        logits = self.readout(torch.cat([context, position], dim=-1))
        return logits.transpose(1, 2)


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

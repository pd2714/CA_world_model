"""Latent transition models."""

from __future__ import annotations

import torch
from torch import nn


class SpatialLatentDynamics1D(nn.Module):
    def __init__(
        self,
        channels: int = 32,
        depth: int = 3,
        step_size: float = 1.0,
        use_post_norm: bool = False,
        clamp_delta: float = 0.0,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        for _ in range(depth):
            layers.extend([nn.Conv1d(channels, channels, 3, padding=1, padding_mode="circular"), nn.GELU()])
        self.net = nn.Sequential(*layers)
        self.step_size = float(step_size)
        self.post_norm = nn.GroupNorm(1, channels) if use_post_norm else nn.Identity()
        self.clamp_delta = float(clamp_delta)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        delta = self.net(z)
        if self.clamp_delta > 0.0:
            delta = torch.clamp(delta, min=-self.clamp_delta, max=self.clamp_delta)
        return self.post_norm(z + self.step_size * delta)


class SpatialLatentDynamics2D(nn.Module):
    def __init__(
        self,
        channels: int = 32,
        depth: int = 3,
        step_size: float = 1.0,
        use_post_norm: bool = False,
        clamp_delta: float = 0.0,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        for _ in range(depth):
            layers.extend([nn.Conv2d(channels, channels, 3, padding=1, padding_mode="circular"), nn.GELU()])
        self.net = nn.Sequential(*layers)
        self.step_size = float(step_size)
        self.post_norm = nn.GroupNorm(1, channels) if use_post_norm else nn.Identity()
        self.clamp_delta = float(clamp_delta)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        delta = self.net(z)
        if self.clamp_delta > 0.0:
            delta = torch.clamp(delta, min=-self.clamp_delta, max=self.clamp_delta)
        return self.post_norm(z + self.step_size * delta)


class VectorLatentDynamics(nn.Module):
    def __init__(
        self,
        latent_dim: int = 64,
        hidden_dim: int = 128,
        step_size: float = 1.0,
        use_post_norm: bool = False,
        clamp_delta: float = 0.0,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.step_size = float(step_size)
        self.post_norm = nn.LayerNorm(latent_dim) if use_post_norm else nn.Identity()
        self.clamp_delta = float(clamp_delta)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        delta = self.net(z)
        if self.clamp_delta > 0.0:
            delta = torch.clamp(delta, min=-self.clamp_delta, max=self.clamp_delta)
        return self.post_norm(z + self.step_size * delta)

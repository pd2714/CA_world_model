"""Legacy-compatible dense world model used by the original Rule 184 checkpoint."""

from __future__ import annotations

import torch
from torch import nn

from src.models.decoder import ConvDecoder1D
from src.models.dense_world_model import DenseWorldModelOutput
from src.models.encoder import ConvEncoder1D


class LegacySpatialLatentDynamics1D(nn.Module):
    """Exact 1D latent dynamics used by the initial Rule 184 rollout-only runs."""

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


class LegacyDenseWorldModel(nn.Module):
    """Minimal dense world model matching the original Rule 184 checkpoint layout."""

    def __init__(
        self,
        channels: int = 32,
        depth: int = 3,
        step_size: float = 1.0,
        use_post_norm: bool = False,
        clamp_delta: float = 0.0,
    ) -> None:
        super().__init__()
        self.encoder = ConvEncoder1D(latent_channels=channels, depth=depth)
        self.dynamics = LegacySpatialLatentDynamics1D(
            channels=channels,
            depth=depth,
            step_size=step_size,
            use_post_norm=use_post_norm,
            clamp_delta=clamp_delta,
        )
        self.decoder = ConvDecoder1D(latent_channels=channels, depth=depth)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def step_latent(self, z: torch.Tensor) -> torch.Tensor:
        return self.dynamics(z)

    def forward(self, x: torch.Tensor) -> DenseWorldModelOutput:
        z = self.encode(x)
        x_recon_logits = self.decode(z)
        z_next = self.step_latent(z)
        x_next_logits = self.decode(z_next)
        return DenseWorldModelOutput(
            reconstruction_logits=x_recon_logits,
            prediction_logits=x_next_logits,
            latent=z,
            next_latent=z_next,
        )

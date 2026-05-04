"""Dense latent-state world models for CA dynamics."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from src.models.decoder import ConvDecoder1D, ConvDecoder2D, LearnedBottleneckDecoder1D, VectorDecoder1D
from src.models.encoder import ConvEncoder1D, ConvEncoder2D, GlobalVectorEncoder1D, LearnedBottleneckEncoder1D
from src.models.latent_dynamics import BottleneckMLPDynamics1D, SpatialLatentDynamics1D, SpatialLatentDynamics2D, VectorLatentDynamics


@dataclass
class DenseWorldModelOutput:
    reconstruction_logits: torch.Tensor
    prediction_logits: torch.Tensor
    latent: torch.Tensor
    next_latent: torch.Tensor


class DenseWorldModel(nn.Module):
    """World model with encoder, latent transition, and decoder."""

    def __init__(
        self,
        dimension: str = "1d",
        latent_type: str = "bottleneck",
        latent_channels: int = 1,
        latent_dim: int = 8,
        latent_length: int | None = 8,
        input_size: int | tuple[int, int] = 128,
        depth: int = 3,
        dynamics_depth: int | None = None,
        dynamics_kernel_size: int = 3,
        dynamics_alpha: float = 1.0,
        dynamics_norm_type: str = "none",
        dynamics_init_scale: float = 1.0,
        dynamics_step_size: float = 1.0,
        dynamics_use_post_norm: bool = False,
        dynamics_clamp_delta: float = 0.0,
        bottleneck_hidden_channels: int = 32,
        bottleneck_hidden_dim: int = 256,
        bottleneck_position_dim: int = 32,
        bottleneck_dynamics_hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.dimension = dimension
        self.latent_type = latent_type
        self.latent_length = int(latent_length) if latent_length is not None else None
        effective_dynamics_depth = int(dynamics_depth if dynamics_depth is not None else depth)

        if dimension == "1d" and latent_type == "spatial":
            self.encoder = ConvEncoder1D(latent_channels=latent_channels, depth=depth, latent_length=self.latent_length)
            self.dynamics = SpatialLatentDynamics1D(
                channels=latent_channels,
                depth=effective_dynamics_depth,
                kernel_size=dynamics_kernel_size,
                alpha=dynamics_alpha,
                norm_type=dynamics_norm_type,
                init_scale=dynamics_init_scale,
                step_size=dynamics_step_size,
                use_post_norm=dynamics_use_post_norm,
                clamp_delta=dynamics_clamp_delta,
            )
            output_length = int(input_size) if self.latent_length is not None else None
            self.decoder = ConvDecoder1D(latent_channels=latent_channels, depth=depth, output_length=output_length)
        elif dimension == "1d" and latent_type == "bottleneck":
            if self.latent_length is None:
                raise ValueError("The 1D bottleneck latent type requires latent_length.")
            input_length = int(input_size)
            self.encoder = LearnedBottleneckEncoder1D(
                input_length=input_length,
                latent_channels=latent_channels,
                latent_length=self.latent_length,
                hidden_channels=bottleneck_hidden_channels,
                hidden_dim=bottleneck_hidden_dim,
                depth=depth,
            )
            self.dynamics = BottleneckMLPDynamics1D(
                latent_channels=latent_channels,
                latent_length=self.latent_length,
                hidden_dim=bottleneck_dynamics_hidden_dim,
                depth=effective_dynamics_depth,
                step_size=dynamics_step_size,
                use_post_norm=dynamics_use_post_norm,
                clamp_delta=dynamics_clamp_delta,
            )
            self.decoder = LearnedBottleneckDecoder1D(
                latent_channels=latent_channels,
                latent_length=self.latent_length,
                output_length=input_length,
                hidden_dim=bottleneck_hidden_dim,
                position_dim=bottleneck_position_dim,
            )
        elif dimension == "1d" and latent_type == "vector":
            self.encoder = GlobalVectorEncoder1D(latent_dim=latent_dim, hidden_channels=latent_channels)
            self.dynamics = VectorLatentDynamics(
                latent_dim=latent_dim,
                hidden_dim=2 * latent_dim,
                step_size=dynamics_step_size,
                use_post_norm=dynamics_use_post_norm,
                clamp_delta=dynamics_clamp_delta,
            )
            self.decoder = VectorDecoder1D(latent_dim=latent_dim, length=int(input_size))
        elif dimension == "2d":
            self.encoder = ConvEncoder2D(latent_channels=latent_channels, depth=depth)
            self.dynamics = SpatialLatentDynamics2D(
                channels=latent_channels,
                depth=effective_dynamics_depth,
                kernel_size=dynamics_kernel_size,
                alpha=dynamics_alpha,
                norm_type=dynamics_norm_type,
                init_scale=dynamics_init_scale,
                step_size=dynamics_step_size,
                use_post_norm=dynamics_use_post_norm,
                clamp_delta=dynamics_clamp_delta,
            )
            self.decoder = ConvDecoder2D(latent_channels=latent_channels, depth=depth)
        else:
            raise ValueError(f"Unsupported dense world model configuration: {dimension=}, {latent_type=}")

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

    def rollout(
        self,
        x0: torch.Tensor,
        steps: int,
        threshold: bool = True,
        rollout_mode: str = "latent",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x0)
        logits_seq = [self.decode(z)]
        states = [x0]
        current_state = x0
        for _ in range(steps):
            if rollout_mode == "reencode":
                z = self.encode(current_state)
            elif rollout_mode != "latent":
                raise ValueError(f"Unsupported rollout mode: {rollout_mode}")
            z = self.step_latent(z)
            logits = self.decode(z)
            logits_seq.append(logits)
            state = (torch.sigmoid(logits) >= 0.5).float() if threshold else torch.sigmoid(logits)
            current_state = state
            states.append(state)
        return torch.stack(states, dim=1), torch.stack(logits_seq, dim=1)

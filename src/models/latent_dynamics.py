"""Latent transition models."""

from __future__ import annotations

import torch
from torch import nn


class LocalChannelNorm1D(nn.Module):
    """Normalize across channels independently at each spatial position."""

    def __init__(self, channels: int, norm_type: str = "none", eps: float = 1.0e-5) -> None:
        super().__init__()
        self.norm_type = str(norm_type).lower()
        self.eps = float(eps)
        if self.norm_type == "none":
            self.norm = None
        elif self.norm_type == "layer":
            self.norm = nn.LayerNorm(channels, eps=eps)
        elif self.norm_type == "rms":
            self.norm = nn.Parameter(torch.ones(channels))
        else:
            raise ValueError(f"Unsupported 1D norm type: {norm_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.norm_type == "none":
            return x
        y = x.transpose(1, 2)
        if self.norm_type == "layer":
            return self.norm(y).transpose(1, 2)
        rms = y.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return ((y / rms) * self.norm.view(1, 1, -1)).transpose(1, 2)


class LocalChannelNorm2D(nn.Module):
    """Normalize across channels independently at each spatial position."""

    def __init__(self, channels: int, norm_type: str = "none", eps: float = 1.0e-5) -> None:
        super().__init__()
        self.norm_type = str(norm_type).lower()
        self.eps = float(eps)
        if self.norm_type == "none":
            self.norm = None
        elif self.norm_type == "layer":
            self.norm = nn.LayerNorm(channels, eps=eps)
        elif self.norm_type == "rms":
            self.norm = nn.Parameter(torch.ones(channels))
        else:
            raise ValueError(f"Unsupported 2D norm type: {norm_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.norm_type == "none":
            return x
        y = x.permute(0, 2, 3, 1)
        if self.norm_type == "layer":
            return self.norm(y).permute(0, 3, 1, 2)
        rms = y.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return ((y / rms) * self.norm.view(1, 1, 1, -1)).permute(0, 3, 1, 2)


def _scaled_final_init(module: nn.Module, scale: float) -> None:
    if not hasattr(module, "weight"):
        return
    with torch.no_grad():
        module.weight.mul_(float(scale))
        if module.bias is not None:
            module.bias.mul_(float(scale))


class SpatialLatentDynamics1D(nn.Module):
    def __init__(
        self,
        channels: int = 32,
        depth: int = 3,
        kernel_size: int = 3,
        alpha: float = 1.0,
        norm_type: str = "none",
        init_scale: float = 1.0,
        step_size: float = 1.0,
        use_post_norm: bool = False,
        clamp_delta: float = 0.0,
    ) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError(f"SpatialLatentDynamics1D requires an odd kernel size, got {kernel_size}.")
        hidden_layers = max(0, int(depth) - 1)
        padding = kernel_size // 2
        layers: list[nn.Module] = []
        effective_norm = norm_type if norm_type != "none" else ("layer" if use_post_norm else "none")
        for _ in range(hidden_layers):
            layers.extend(
                [
                    LocalChannelNorm1D(channels, norm_type=effective_norm),
                    nn.Conv1d(channels, channels, kernel_size, padding=padding, padding_mode="circular"),
                    nn.GELU(),
                ]
            )
        self.net = nn.Sequential(*layers)
        self.output_norm = LocalChannelNorm1D(channels, norm_type=effective_norm)
        self.final = nn.Conv1d(channels, channels, kernel_size, padding=padding, padding_mode="circular")
        _scaled_final_init(self.final, init_scale)
        self.alpha = float(step_size if alpha == 1.0 and step_size != 1.0 else alpha)
        self.clamp_delta = float(clamp_delta)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        hidden = self.net(z)
        delta = self.final(self.output_norm(hidden if len(self.net) > 0 else z))
        if self.clamp_delta > 0.0:
            delta = torch.clamp(delta, min=-self.clamp_delta, max=self.clamp_delta)
        return z + self.alpha * delta


class SpatialLatentDynamics2D(nn.Module):
    def __init__(
        self,
        channels: int = 32,
        depth: int = 3,
        kernel_size: int = 3,
        alpha: float = 1.0,
        norm_type: str = "none",
        init_scale: float = 1.0,
        step_size: float = 1.0,
        use_post_norm: bool = False,
        clamp_delta: float = 0.0,
    ) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError(f"SpatialLatentDynamics2D requires an odd kernel size, got {kernel_size}.")
        hidden_layers = max(0, int(depth) - 1)
        padding = kernel_size // 2
        layers: list[nn.Module] = []
        effective_norm = norm_type if norm_type != "none" else ("layer" if use_post_norm else "none")
        for _ in range(hidden_layers):
            layers.extend(
                [
                    LocalChannelNorm2D(channels, norm_type=effective_norm),
                    nn.Conv2d(channels, channels, kernel_size, padding=padding, padding_mode="circular"),
                    nn.GELU(),
                ]
            )
        self.net = nn.Sequential(*layers)
        self.output_norm = LocalChannelNorm2D(channels, norm_type=effective_norm)
        self.final = nn.Conv2d(channels, channels, kernel_size, padding=padding, padding_mode="circular")
        _scaled_final_init(self.final, init_scale)
        self.alpha = float(step_size if alpha == 1.0 and step_size != 1.0 else alpha)
        self.clamp_delta = float(clamp_delta)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        hidden = self.net(z)
        delta = self.final(self.output_norm(hidden if len(self.net) > 0 else z))
        if self.clamp_delta > 0.0:
            delta = torch.clamp(delta, min=-self.clamp_delta, max=self.clamp_delta)
        return z + self.alpha * delta


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

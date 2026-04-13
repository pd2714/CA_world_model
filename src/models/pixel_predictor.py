"""Direct pixel-space CA predictors."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class ResidualBlock1D(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=padding, padding_mode="circular"),
            nn.GELU(),
            nn.Conv1d(channels, channels, kernel_size, padding=padding, padding_mode="circular"),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.net(x) + x)


class ResidualBlock2D(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size, padding=padding, padding_mode="circular"),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size, padding=padding, padding_mode="circular"),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.net(x) + x)


class PixelPredictor(nn.Module):
    """Convolutional baseline that predicts x_{t+1} from x_t."""

    def __init__(
        self,
        dimension: str = "1d",
        hidden_channels: int = 64,
        depth: int = 4,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        self.dimension = dimension
        if dimension == "1d":
            conv = nn.Conv1d
            block = ResidualBlock1D
        else:
            conv = nn.Conv2d
            block = ResidualBlock2D
        padding = kernel_size // 2
        layers = [conv(1, hidden_channels, kernel_size, padding=padding, padding_mode="circular"), nn.GELU()]
        layers.extend(block(hidden_channels, kernel_size=kernel_size) for _ in range(depth))
        layers.append(conv(hidden_channels, 1, kernel_size, padding=padding, padding_mode="circular"))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def rollout(self, x0: torch.Tensor, steps: int, threshold: bool = True) -> torch.Tensor:
        states = [x0]
        current = x0
        for _ in range(steps):
            logits = self(current)
            current = (torch.sigmoid(logits) >= 0.5).float() if threshold else torch.sigmoid(logits)
            states.append(current)
        return torch.stack(states, dim=1)

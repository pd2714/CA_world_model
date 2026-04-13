"""Small slot modules for object-style 1D models."""

from __future__ import annotations

import math

import torch
from torch import nn


class SoftSlotPooling1D(nn.Module):
    """Pool a spatial feature map into K soft slots."""

    def __init__(self, in_channels: int, num_slots: int, slot_dim: int) -> None:
        super().__init__()
        self.num_slots = num_slots
        self.query = nn.Parameter(torch.randn(num_slots, slot_dim) / math.sqrt(slot_dim))
        self.key = nn.Conv1d(in_channels, slot_dim, kernel_size=1)
        self.value = nn.Conv1d(in_channels, slot_dim, kernel_size=1)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        keys = self.key(features).transpose(1, 2)
        values = self.value(features).transpose(1, 2)
        logits = torch.einsum("kd,bld->bkl", self.query, keys) / math.sqrt(keys.shape[-1])
        attn = torch.softmax(logits, dim=-1)
        slots = torch.einsum("bkl,bld->bkd", attn, values)
        return slots, attn


class SlotDynamics(nn.Module):
    """Lightweight slot interaction module."""

    def __init__(self, slot_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(slot_dim)
        self.attn = nn.MultiheadAttention(slot_dim, num_heads=2, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(slot_dim, 2 * slot_dim),
            nn.GELU(),
            nn.Linear(2 * slot_dim, slot_dim),
        )

    def forward(self, slots: torch.Tensor) -> torch.Tensor:
        normed = self.norm(slots)
        attended, _ = self.attn(normed, normed, normed)
        return slots + attended + self.mlp(self.norm(slots + attended))


class SlotDecoder1D(nn.Module):
    """Broadcast slots to a 1D grid with soft masks."""

    def __init__(self, num_slots: int, slot_dim: int, length: int) -> None:
        super().__init__()
        self.length = length
        self.pos = nn.Parameter(torch.linspace(-1.0, 1.0, steps=length).view(1, 1, length))
        self.slot_to_params = nn.Linear(slot_dim, 3)
        self.slot_to_value = nn.Linear(slot_dim, 1)
        self.num_slots = num_slots

    def forward(self, slots: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        params = self.slot_to_params(slots)
        centers = torch.tanh(params[..., 0:1])
        widths = torch.sigmoid(params[..., 1:2]) * 0.7 + 0.05
        strengths = params[..., 2:3]
        masks = -((self.pos - centers) ** 2) / (widths**2)
        masks = torch.softmax(masks + strengths, dim=1)
        values = self.slot_to_value(slots)
        logits = (masks * values).sum(dim=1)
        return logits, masks

"""Simple object-style 1D world model with slots."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from src.models.encoder import ConvEncoder1D
from src.models.slots import SlotDecoder1D, SlotDynamics, SoftSlotPooling1D


@dataclass
class ObjectWorldModelOutput:
    reconstruction_logits: torch.Tensor
    prediction_logits: torch.Tensor
    slots: torch.Tensor
    next_slots: torch.Tensor
    attention: torch.Tensor
    masks: torch.Tensor


class ObjectWorldModel(nn.Module):
    """A modest slot-based latent model for 1D CA."""

    def __init__(self, length: int, num_slots: int = 6, slot_dim: int = 32, feature_channels: int = 32) -> None:
        super().__init__()
        self.encoder = ConvEncoder1D(latent_channels=feature_channels, depth=3)
        self.slot_pool = SoftSlotPooling1D(in_channels=feature_channels, num_slots=num_slots, slot_dim=slot_dim)
        self.dynamics = SlotDynamics(slot_dim=slot_dim)
        self.decoder = SlotDecoder1D(num_slots=num_slots, slot_dim=slot_dim, length=length)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.encoder(x)
        return self.slot_pool(features)

    def decode(self, slots: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.decoder(slots)

    def step_slots(self, slots: torch.Tensor) -> torch.Tensor:
        return self.dynamics(slots)

    def forward(self, x: torch.Tensor) -> ObjectWorldModelOutput:
        slots, attention = self.encode(x)
        reconstruction_logits, _ = self.decode(slots)
        next_slots = self.step_slots(slots)
        prediction_logits, masks = self.decode(next_slots)
        return ObjectWorldModelOutput(
            reconstruction_logits=reconstruction_logits.unsqueeze(1),
            prediction_logits=prediction_logits.unsqueeze(1),
            slots=slots,
            next_slots=next_slots,
            attention=attention,
            masks=masks,
        )

    def rollout(self, x0: torch.Tensor, steps: int, threshold: bool = True) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        slots, attention = self.encode(x0)
        states = [x0]
        masks_over_time = []
        slots_over_time = [slots]
        for _ in range(steps):
            slots = self.step_slots(slots)
            logits, masks = self.decode(slots)
            state = (torch.sigmoid(logits.unsqueeze(1)) >= 0.5).float() if threshold else torch.sigmoid(logits.unsqueeze(1))
            states.append(state)
            masks_over_time.append(masks)
            slots_over_time.append(slots)
        return torch.stack(states, dim=1), {
            "attention": attention,
            "masks": torch.stack(masks_over_time, dim=1) if masks_over_time else None,
            "slots": torch.stack(slots_over_time, dim=1),
        }

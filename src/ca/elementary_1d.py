"""Elementary one-dimensional cellular automata."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.ca.base import CellularAutomaton


@dataclass
class ElementaryCA(CellularAutomaton):
    """Binary radius-1 elementary cellular automaton."""

    rule: int

    def __post_init__(self) -> None:
        if not 0 <= self.rule <= 255:
            raise ValueError(f"Rule must be in [0, 255], got {self.rule}.")
        bits = np.array([(self.rule >> i) & 1 for i in range(8)], dtype=np.uint8)
        self.lookup = bits

    def step(self, state: np.ndarray) -> np.ndarray:
        state = np.asarray(state, dtype=np.uint8)
        left = np.roll(state, 1, axis=-1)
        center = state
        right = np.roll(state, -1, axis=-1)
        neighborhood = (left << 2) | (center << 1) | right
        return self.lookup[neighborhood].astype(np.uint8)

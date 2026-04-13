"""Life-like two-dimensional binary cellular automata."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.ca.base import CellularAutomaton


def parse_bs_rule(rule: str) -> tuple[set[int], set[int]]:
    """Parse a Life-like rule string such as B3/S23."""
    upper = rule.upper().strip()
    if "/" not in upper or "B" not in upper or "S" not in upper:
        raise ValueError(f"Invalid Life-like rule: {rule}")
    birth_str, survive_str = upper.split("/")
    birth = {int(char) for char in birth_str.replace("B", "")}
    survive = {int(char) for char in survive_str.replace("S", "")}
    return birth, survive


@dataclass
class LifeLikeCA(CellularAutomaton):
    """Life-like CA with periodic boundary conditions."""

    rule: str = "B3/S23"

    def __post_init__(self) -> None:
        self.birth, self.survive = parse_bs_rule(self.rule)

    def step(self, state: np.ndarray) -> np.ndarray:
        state = np.asarray(state, dtype=np.uint8)
        neighbors = np.zeros_like(state, dtype=np.uint8)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                neighbors += np.roll(np.roll(state, dx, axis=-2), dy, axis=-1)
        born = (state == 0) & np.isin(neighbors, list(self.birth))
        survives = (state == 1) & np.isin(neighbors, list(self.survive))
        return (born | survives).astype(np.uint8)

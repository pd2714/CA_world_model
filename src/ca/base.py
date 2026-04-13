"""Base interfaces for cellular automata."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class CellularAutomaton(ABC):
    """Abstract cellular automaton interface."""

    @abstractmethod
    def step(self, state: np.ndarray) -> np.ndarray:
        """Advance one timestep."""

    def rollout(self, state0: np.ndarray, steps: int) -> np.ndarray:
        """Roll out a trajectory including the initial state."""
        frames = [state0.astype(np.float32)]
        state = state0
        for _ in range(steps):
            state = self.step(state)
            frames.append(state.astype(np.float32))
        return np.stack(frames, axis=0)

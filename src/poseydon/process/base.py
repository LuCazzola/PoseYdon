"""Generative processes.

A process owns corruption, its prediction parameterization, and the inverse that
recovers a clean sample. It knows nothing about motion, skeletons or features.

``to_z0`` is what keeps structural losses process-agnostic: whatever the network
predicts -- x0, epsilon, v, or a velocity field -- a loss that needs rotations
always receives a clean sample. The reference instead folds its losses into the
diffusion class, so its geodesic and foot-skate terms cannot be reused by any
other process.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from poseydon.core.registry import Registry


class Process(ABC):
    """Corruption and parameterization, independent of what is being generated."""

    @property
    @abstractmethod
    def num_steps(self) -> int:
        """Number of discrete steps used when sampling."""

    @abstractmethod
    def sample_t(self, batch_size: int, device: torch.device | str = "cpu") -> torch.Tensor:
        """Draw training timesteps, shape ``(B,)``."""

    @abstractmethod
    def corrupt(self, z0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Corrupt a clean sample to time ``t``."""

    @abstractmethod
    def target(self, z0: torch.Tensor, noise: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """What the network is trained to regress."""

    @abstractmethod
    def to_z0(
        self, pred: torch.Tensor, z_t: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """Recover the clean sample from a prediction. Inverse of :meth:`target`."""


PROCESSES: Registry[Process] = Registry("process")


def expand_to(values: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    """Reshape a per-item vector so it broadcasts against ``like``."""
    return values.reshape(-1, *([1] * (like.ndim - 1)))

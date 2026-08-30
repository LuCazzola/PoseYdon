"""Rectified flow matching."""

from __future__ import annotations

import torch

from poseydon.process.base import PROCESSES, Process, expand_to


@PROCESSES.register("flow")
class FlowMatching(Process):
    """Straight-line interpolant between data and noise.

    ``z_t = (1 - t) * z0 + t * noise`` with continuous ``t`` in [0, 1], and the
    network regresses the constant velocity ``noise - z0``. Sharing the
    :class:`Process` interface means every structural loss written for diffusion
    applies here unchanged, because both expose ``to_z0``.
    """

    def __init__(self, num_steps: int = 100) -> None:
        if num_steps < 1:
            raise ValueError(f"num_steps must be positive, got {num_steps}")
        self._num_steps = num_steps

    @property
    def num_steps(self) -> int:
        return self._num_steps

    def sample_t(self, batch_size: int, device: torch.device | str = "cpu") -> torch.Tensor:
        return torch.rand(batch_size, device=device)

    def corrupt(self, z0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        time = expand_to(t.to(z0.device, z0.dtype), z0)
        return (1.0 - time) * z0 + time * noise

    def target(self, z0: torch.Tensor, noise: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return noise - z0

    def to_z0(self, pred: torch.Tensor, z_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        time = expand_to(t.to(z_t.device, z_t.dtype), z_t)
        return z_t - time * pred

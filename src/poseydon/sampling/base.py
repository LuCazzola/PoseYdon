"""Sampling: how a trajectory is solved, and what is applied along the way.

Three concepts, deliberately separate:

* a :class:`Sampler` solves the reverse trajectory;
* a :class:`Control` modifies state or conditioning at each step;
* an :class:`~poseydon.ops.base.Operation` is the user-facing verb.

The reference collapses all three. Its blending lives inside ``MoDiffAE.forward``
as a ``_mix`` method, which is why in-betweening is unreachable from that
codebase without editing the model.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

import torch

from poseydon.core.batch import Cond
from poseydon.core.registry import Registry
from poseydon.models.base import Denoiser
from poseydon.process.base import Process


class Control(ABC):
    """A hook applied inside the sampling loop."""

    def before_step(
        self, z_t: torch.Tensor, t: torch.Tensor, cond: Cond
    ) -> tuple[torch.Tensor, Cond]:
        """Adjust the state or conditioning before the network is called."""
        return z_t, cond

    def after_step(self, z_t: torch.Tensor, t: torch.Tensor, cond: Cond) -> torch.Tensor:
        """Adjust the state after a step has been taken."""
        return z_t


class Sampler(ABC):
    """Solves the reverse trajectory from noise to a clean sample."""

    @abstractmethod
    def sample(
        self,
        model: Denoiser,
        process: Process,
        shape: tuple[int, ...],
        cond: Cond,
        controls: Sequence[Control] = (),
        device: torch.device | str = "cpu",
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Draw one clean sample of ``shape``."""


CONTROLS: Registry[Control] = Registry("control")
SAMPLERS: Registry[Sampler] = Registry("sampler")


def apply_before(
    controls: Sequence[Control], z_t: torch.Tensor, t: torch.Tensor, cond: Cond
) -> tuple[torch.Tensor, Cond]:
    for control in controls:
        z_t, cond = control.before_step(z_t, t, cond)
    return z_t, cond


def apply_after(
    controls: Sequence[Control], z_t: torch.Tensor, t: torch.Tensor, cond: Cond
) -> torch.Tensor:
    for control in controls:
        z_t = control.after_step(z_t, t, cond)
    return z_t

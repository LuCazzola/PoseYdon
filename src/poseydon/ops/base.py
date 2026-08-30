"""Operations: the user-facing verbs.

An operation assembles inputs, picks the controls that express what is being
done, and runs a sampler. Adding a motion task is usually a Control plus a thin
Operation, not a new sampling loop.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

import torch

from poseydon.core.batch import Cond
from poseydon.core.registry import Registry
from poseydon.models.base import Denoiser
from poseydon.process.base import Process
from poseydon.sampling.base import Control, Sampler


class Operation(ABC):
    """What to do with a trained model."""

    @abstractmethod
    def build_controls(self, cond: Cond) -> Sequence[Control]:
        """The controls that express this operation."""

    def run(
        self,
        model: Denoiser,
        process: Process,
        sampler: Sampler,
        shape: tuple[int, ...],
        cond: Cond,
        device: torch.device | str = "cpu",
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        return sampler.sample(
            model=model,
            process=process,
            shape=shape,
            cond=cond,
            controls=self.build_controls(cond),
            device=device,
            generator=generator,
        )


OPERATIONS: Registry[Operation] = Registry("operation")

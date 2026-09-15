"""Operations: the user-facing verbs.

An operation assembles inputs, picks the controls that express what is being
done, and runs a sampler. Adding a motion task is usually a Control plus a thin
Operation, not a new sampling loop.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

import torch

from poseydon.core.batch import Cond, Masks
from poseydon.core.registry import Registry
from poseydon.models.base import Denoiser
from poseydon.process.base import Process
from poseydon.sampling.base import Control, Sampler


class Operation(ABC):
    """What to do with a trained model."""

    @abstractmethod
    def build_controls(self, cond: Cond) -> Sequence[Control]:
        """The controls that express this operation."""

    def wrap_model(self, model: Denoiser) -> Denoiser:
        """The denoiser this operation actually samples with.

        Identity for everything that samples the model as-is. Guidance needs
        two forward passes per step, which a `Control` cannot do -- it is given
        the state and the conditioning, never the model -- so it wraps here
        instead.
        """
        return model

    def run(
        self,
        model: Denoiser,
        process: Process,
        sampler: Sampler,
        shape: tuple[int, ...],
        cond: Cond,
        device: torch.device | str = "cpu",
        generator: torch.Generator | None = None,
        masks: Masks | None = None,
    ) -> torch.Tensor:
        return sampler.sample(
            model=self.wrap_model(model),
            process=process,
            shape=shape,
            cond=cond,
            controls=self.build_controls(cond),
            device=device,
            generator=generator,
            masks=masks,
        )


OPERATIONS: Registry[Operation] = Registry("operation")

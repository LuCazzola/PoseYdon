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

    #: Classifier-free guidance strength. 0 samples the model as-is; above 0
    #: wraps it so each step evaluates a conditional and an unconditional
    #: branch and extrapolates between them. Declared here rather than on each
    #: subclass so `wrap_model` has ONE implementation: `Generate` and
    #: `Retarget` differ in what they condition on, not in how guidance works.
    guidance_scale: float = 0.0

    def wrap_model(self, model: Denoiser) -> Denoiser:
        """The denoiser this operation actually samples with.

        Identity at scale 0, which is every operation's default. Guidance needs
        two forward passes per step, which a `Control` cannot do -- it is given
        the state and the conditioning, never the model -- so it wraps here
        instead.
        """
        if self.guidance_scale > 0.0:
            from poseydon.sampling.guidance import GuidedDenoiser

            return GuidedDenoiser(model, scale=self.guidance_scale)
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

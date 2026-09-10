"""Retargeting: decode one clip's semantics onto another rig."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from poseydon.core.batch import Cond, Masks
from poseydon.models.base import Denoiser
from poseydon.ops.base import OPERATIONS, Operation
from poseydon.process.base import Process
from poseydon.sampling.base import Control, Sampler
from poseydon.sampling.controls.latent_pin import LatentPin


@OPERATIONS.register("retarget")
class Retarget(Operation):
    """Carry one clip's motion onto a different skeleton.

    The semantic latent is pooled over joints, so it describes WHAT is being
    done without describing what is doing it. Encoding a 40-joint Flamingo and
    decoding on a 63-joint Scorpion is therefore shape-valid by construction;
    only the frame count has to match.
    """

    def __init__(self, content: torch.Tensor, content_cond: Cond) -> None:
        self.content = content
        self.content_cond = content_cond
        self._latent: torch.Tensor | None = None

    def build_controls(self, cond: Cond) -> Sequence[Control]:
        if self._latent is None:
            raise RuntimeError("`run` encodes the content before building controls")
        return [LatentPin(self._latent)]

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
        encode = getattr(model, "encode", None)
        if encode is None:
            raise TypeError(
                f"{type(model).__name__} has no `encode`, so there is no semantic "
                "latent to retarget. Sampling anyway would produce unconditional "
                "motion that looks like a retarget while ignoring the content clip."
            )

        # ONCE, before the trajectory: the latent describes the content, which
        # does not change as the target is denoised. Encoding per step would be
        # the same answer computed a thousand times.
        # BOTH move: `Cond` carries topology, tpose and normalization tensors,
        # and encoding a CUDA clip against CPU conditioning faults inside the
        # model. The stub model in the tests ignores `cond`, so the suite
        # cannot see this one -- it only shows up on a real GPU run.
        self._latent, _ = encode(self.content.to(device), self.content_cond.to(device))
        return super().run(
            model, process, sampler, shape, cond, device, generator, masks
        )

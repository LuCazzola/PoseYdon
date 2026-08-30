"""Blending: interpolate between two motions in the semantic latent."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

from poseydon.core.batch import Cond
from poseydon.ops.base import OPERATIONS, Operation
from poseydon.sampling.base import Control
from poseydon.sampling.controls.latent_mix import LatentMix, build_schedule


@OPERATIONS.register("blend")
class Blend(Operation):
    """Cross-fade two clips through a latent model's semantic space."""

    def __init__(
        self,
        reference: torch.Tensor,
        target: torch.Tensor,
        schedule: str = "ease",
        length: int | None = None,
        **schedule_kwargs: object,
    ) -> None:
        if reference.shape != target.shape:
            raise ValueError(
                f"reference {tuple(reference.shape)} and target {tuple(target.shape)} "
                "must have the same shape to be blended"
            )
        self.reference = reference
        self.target = target
        frames = length if length is not None else reference.shape[0]
        self.alpha: np.ndarray = build_schedule(schedule, frames, **schedule_kwargs)

    def build_controls(self, cond: Cond) -> Sequence[Control]:
        return [LatentMix(self.reference, self.target, self.alpha)]

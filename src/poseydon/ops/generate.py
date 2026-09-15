"""Unconditional and conditional synthesis."""

from __future__ import annotations

from collections.abc import Sequence

from poseydon.core.batch import Cond
from poseydon.models.base import Denoiser
from poseydon.ops.base import OPERATIONS, Operation
from poseydon.sampling.base import Control
from poseydon.sampling.guidance import GuidedDenoiser


@OPERATIONS.register("generate")
class Generate(Operation):
    """Sample motion from noise. The base case: no controls at all."""

    def __init__(self, guidance_scale: float = 0.0) -> None:
        self.guidance_scale = guidance_scale

    def build_controls(self, cond: Cond) -> Sequence[Control]:
        return []

    def wrap_model(self, model: Denoiser) -> Denoiser:
        if self.guidance_scale > 0.0:
            return GuidedDenoiser(model, scale=self.guidance_scale)
        return model

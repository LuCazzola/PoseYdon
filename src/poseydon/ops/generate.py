"""Unconditional and conditional synthesis."""

from __future__ import annotations

from collections.abc import Sequence

from poseydon.core.batch import Cond
from poseydon.ops.base import OPERATIONS, Operation
from poseydon.sampling.base import Control
from poseydon.sampling.controls.guidance import ClassifierFreeGuidance


@OPERATIONS.register("generate")
class Generate(Operation):
    """Sample motion from noise. The base case: no controls unless guided."""

    def __init__(self, guidance_scale: float = 0.0) -> None:
        self.guidance_scale = guidance_scale

    def build_controls(self, cond: Cond) -> Sequence[Control]:
        if self.guidance_scale > 0.0:
            return [ClassifierFreeGuidance(scale=self.guidance_scale)]
        return []

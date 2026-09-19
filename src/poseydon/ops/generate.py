"""Unconditional and conditional synthesis."""

from __future__ import annotations

from collections.abc import Sequence

from poseydon.core.batch import Cond
from poseydon.ops.base import OPERATIONS, Operation
from poseydon.sampling.base import Control


@OPERATIONS.register("generate")
class Generate(Operation):
    """Sample motion from noise. The base case: no controls at all."""

    def __init__(self, guidance_scale: float = 1.0) -> None:
        self.guidance_scale = guidance_scale

    def build_controls(self, cond: Cond) -> Sequence[Control]:
        return []

"""Classifier-free guidance."""

from __future__ import annotations

import torch

from poseydon.core.batch import Cond
from poseydon.sampling.base import CONTROLS, Control


@CONTROLS.register("cfg")
class ClassifierFreeGuidance(Control):
    """Push the trajectory away from the unconditional prediction.

    Implemented as a control rather than baked into a sampler, so it composes
    with in-betweening and blending instead of multiplying into a sampler per
    combination.
    """

    def __init__(self, scale: float = 2.5, drop_key: str = "uncond") -> None:
        self.scale = scale
        self.drop_key = drop_key

    def before_step(
        self, z_t: torch.Tensor, t: torch.Tensor, cond: Cond
    ) -> tuple[torch.Tensor, Cond]:
        # Signals the model to also produce an unconditional branch; models that
        # do not support it ignore the flag and guidance is a no-op.
        payloads = dict(cond.payloads)
        payloads[self.drop_key] = self.scale
        return z_t, Cond(payloads)

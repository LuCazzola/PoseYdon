"""Pinning a semantic latent for the whole trajectory."""

from __future__ import annotations

import torch

from poseydon.core.batch import Cond
from poseydon.sampling.base import CONTROLS, Control


@CONTROLS.register("latent_pin")
class LatentPin(Control):
    """Hold one semantic latent fixed across every sampling step.

    `LatentMix`'s sibling minus the schedule, and deliberately without its
    `reference.shape == target.shape` guard. That guard is what makes `Blend`
    unable to express cross-rig transfer: the semantic latent is pooled over
    joints (`SemanticEncoder.forward` returns `(T, B, 1, C)`), so a 40-joint
    encode is shape-compatible with a 63-joint decode by construction, and
    only the frame count has to match. Refusing a shape mismatch here would
    refuse retargeting itself.
    """

    def __init__(self, latent: torch.Tensor, key: str = "z_sem") -> None:
        self.latent = latent
        self.key = key

    def before_step(
        self, z_t: torch.Tensor, t: torch.Tensor, cond: Cond
    ) -> tuple[torch.Tensor, Cond]:
        # A new Cond, never an edit of the caller's: controls compose, and one
        # that mutates its input corrupts every control after it.
        payloads = dict(cond.payloads)
        payloads[self.key] = self.latent.to(z_t.device)
        return z_t, Cond(payloads)

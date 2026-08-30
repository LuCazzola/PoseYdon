"""Latent blending between two control signals."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch

from poseydon.core.batch import Cond
from poseydon.core.registry import Registry
from poseydon.sampling.base import CONTROLS, Control

Schedule = Callable[[int], np.ndarray]
SCHEDULES: Registry[object] = Registry("schedule")


def static(length: int, alpha: float = 0.5, **_: object) -> np.ndarray:
    return np.full(length, alpha, dtype=np.float64)


def linear(length: int, **_: object) -> np.ndarray:
    return np.linspace(0.0, 1.0, length, dtype=np.float64)


def ease(length: int, slope: float = 1.0, **_: object) -> np.ndarray:
    """Symmetric sine ease, sharpened or softened by ``slope``."""
    t = np.linspace(0.0, 1.0, length, dtype=np.float64)
    s = 0.5 * (1.0 + np.sin(np.pi * (t - 0.5)))
    if slope == 1.0:
        return s
    return np.where(s < 0.5, 0.5 * (2.0 * s) ** slope, 1.0 - 0.5 * (2.0 * (1.0 - s)) ** slope)


_SCHEDULES = {"static": static, "linear": linear, "ease": ease}


def build_schedule(name: str, length: int, **kwargs: object) -> np.ndarray:
    if name not in _SCHEDULES:
        raise ValueError(
            f"unknown blend schedule `{name}`. Available: {', '.join(sorted(_SCHEDULES))}"
        )
    return _SCHEDULES[name](length, **kwargs)


@CONTROLS.register("latent_mix")
class LatentMix(Control):
    """Interpolate two semantic latents along a per-frame schedule.

    In the reference this lives inside ``MoDiffAE.forward`` as ``_mix``. Moving
    it out changes no parameter -- only the signature -- so converted weights
    still load, while blending becomes reusable by any latent model.
    """

    def __init__(
        self,
        reference: torch.Tensor,
        target: torch.Tensor,
        alpha: np.ndarray,
        key: str = "z_sem",
        frame_dim: int = -2,
    ) -> None:
        if reference.shape != target.shape:
            raise ValueError(
                f"reference {tuple(reference.shape)} and target {tuple(target.shape)} differ"
            )
        self.reference = reference
        self.target = target
        self.alpha = torch.as_tensor(np.asarray(alpha), dtype=torch.float32)
        self.key = key
        # The schedule is per FRAME, so it has to be aligned to the frame axis
        # rather than broadcast from the left -- for a (B, T, C) latent, the
        # leading axis is batch, and blending along it would be meaningless.
        self.frame_dim = frame_dim
        frames = reference.shape[frame_dim]
        if self.alpha.shape[0] != frames:
            raise ValueError(
                f"schedule has {self.alpha.shape[0]} entries but the latent has "
                f"{frames} frames on axis {frame_dim}"
            )

    def before_step(
        self, z_t: torch.Tensor, t: torch.Tensor, cond: Cond
    ) -> tuple[torch.Tensor, Cond]:
        shape = [1] * self.reference.ndim
        shape[self.frame_dim] = self.alpha.shape[0]
        alpha = self.alpha.to(self.reference.device).view(shape)
        mixed = torch.lerp(self.reference, self.target, alpha)

        payloads = dict(cond.payloads)
        payloads[self.key] = mixed
        return z_t, Cond(payloads)

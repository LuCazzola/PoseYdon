"""Known-frame imputation."""

from __future__ import annotations

import torch

from poseydon.core.batch import Cond
from poseydon.process.gaussian import GaussianDiffusion
from poseydon.sampling.base import CONTROLS, Control


@CONTROLS.register("inbetween")
class Inbetween(Control):
    """Hold given frames fixed while the rest is generated.

    After every step the known frames are overwritten with the reference,
    re-noised to the current level, so the trajectory is conditioned on them
    without the model ever needing an in-betweening mode. This is the whole of
    in-betweening: a control, not a model change.
    """

    def __init__(
        self, reference: torch.Tensor, known: torch.Tensor, process: GaussianDiffusion
    ) -> None:
        if known.dtype is not torch.bool:
            raise TypeError(f"known must be a bool mask, got {known.dtype}")
        self.reference = reference
        self.known = known
        self.process = process

    def after_step(self, z_t: torch.Tensor, t: torch.Tensor, cond: Cond) -> torch.Tensor:
        reference = self.reference.to(z_t.device, z_t.dtype)
        # Re-noise the reference to this step so the two are on the same manifold.
        noised = self.process.corrupt(reference, t, torch.randn_like(reference))
        known = self.known.to(z_t.device)
        while known.ndim < z_t.ndim:
            known = known.unsqueeze(1)
        return torch.where(known.expand_as(z_t), noised, z_t)

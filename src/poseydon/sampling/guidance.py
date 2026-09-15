"""Classifier-free guidance, as a denoiser that wraps a denoiser."""

from __future__ import annotations

import torch

from poseydon.core.batch import Cond, Masks
from poseydon.models.base import COND_DROP, Denoiser, Prediction


class GuidedDenoiser(Denoiser):
    """Evaluates the inner model twice a step and extrapolates between them.

    A wrapper rather than a :class:`~poseydon.sampling.base.Control`, because a
    control is never given the model: ``before_step(z_t, t, cond)`` can add a
    scale to the conditioning but cannot run a second forward pass, which is
    why the control this replaces set a key nothing read and guided by nothing.

    A wrapper rather than logic inside ``forward``, because that is the shape of
    the mistake this codebase was written to avoid -- the reference's blending
    lived in ``MoDiffAE.forward``, which is why in-betweening was unreachable
    there. Here the model's only part in guidance is honouring ``COND_DROP``;
    the combination lives out here, once, for every model.

    Everything else delegates, so a wrapped model still has its inner model's
    ``prepare``, ``restore`` and -- where it has one -- ``encode``. A model
    with no ``encode`` still raises ``AttributeError`` through the wrapper,
    which is what `Retarget`'s check depends on.
    """

    def __init__(self, inner: Denoiser, scale: float) -> None:
        super().__init__()
        if scale < 0.0:
            raise ValueError(f"guidance scale must be non-negative, got {scale}")
        self.inner = inner
        self.scale = float(scale)
        # Instance attributes shadowing the ClassVars, so `check_conditioners`
        # asks for what the INNER model needs rather than for nothing.
        self.requires = inner.requires
        self.optional = inner.optional

    def __getattr__(self, name: str):
        """Anything this wrapper does not define comes from the inner model."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            if name == "inner":
                raise
            return getattr(self.inner, name)

    def prepare(self, batch):
        return self.inner.prepare(batch)

    def restore(self, z0: torch.Tensor, batch) -> torch.Tensor:
        return self.inner.restore(z0, batch)

    def forward(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        cond: Cond,
        masks: Masks | None = None,
    ) -> Prediction:
        conditional = self.inner(z_t, t, cond, masks)
        if self.scale == 0.0:
            # Not merely equivalent -- skipping the branch entirely is what
            # makes `guidance_scale: 0.0`, the shipped default, free.
            return conditional

        unconditional = self.inner(
            z_t, t, Cond({**cond.payloads, COND_DROP: True}), masks
        )
        guided = unconditional.out + self.scale * (conditional.out - unconditional.out)
        # The conditional branch's aux: its `z_sem` is the latent that describes
        # the sample being drawn, and a KL term reads its own moments.
        return Prediction(out=guided, aux=conditional.aux)

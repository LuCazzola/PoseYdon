"""Foot sliding penalty."""

from __future__ import annotations

from typing import Any

import torch

from poseydon.core.batch import MotionBatch
from poseydon.core.spec import Block
from poseydon.losses.base import LOSSES, LossTerm, masked_mean, raw_block


@LOSSES.register("footskate")
class FootSkateLoss(LossTerm):
    """Penalize movement of joints the target marks as in contact.

    Uses raw positions, so the penalty is a real displacement rather than a
    displacement in units of each channel's standard deviation.
    """

    name = "footskate"
    needs = (Block("ric_pos", space="raw"), Block("foot_contact"))

    def __call__(
        self,
        x0_hat: torch.Tensor,
        x0: torch.Tensor,
        batch: MotionBatch,
        aux: dict[str, Any],
    ) -> torch.Tensor:
        predicted = raw_block(batch, x0_hat, "ric_pos")
        contact = x0[:, :, batch.spec.slice("foot_contact"), :]

        # Contact is a flag on a frame; a slide is movement between two of them.
        velocity = predicted[..., 1:] - predicted[..., :-1]
        planted = (contact[..., :-1] > 0.5).to(velocity.dtype)

        frames = batch.masks.frames[:, None, None, :]
        valid = (frames[..., 1:] & frames[..., :-1]).to(velocity.dtype)
        joints = batch.masks.joints[:, :, None, None].to(velocity.dtype)

        return masked_mean(velocity.pow(2).sum(dim=2, keepdim=True), planted * valid * joints)

"""Plain reconstruction loss."""

from __future__ import annotations

from typing import Any

import torch

from poseydon.core.batch import MotionBatch
from poseydon.losses.base import LOSSES, LossTerm, element_mask, masked_mean


@LOSSES.register("simple")
class SimpleLoss(LossTerm):
    """Masked MSE over the whole feature vector."""

    name = "simple"

    def __call__(
        self,
        x0_hat: torch.Tensor,
        x0: torch.Tensor,
        batch: MotionBatch,
        aux: dict[str, Any],
    ) -> torch.Tensor:
        return masked_mean((x0_hat - x0) ** 2, element_mask(batch))

"""Plain reconstruction loss."""

from __future__ import annotations

from typing import Any

import torch

from poseydon.core.batch import MotionBatch
from poseydon.losses.base import (
    LOSSES,
    LossTerm,
    element_mask,
    masked_mean,
    masked_mean_per_sample,
)


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

    def per_sample(
        self,
        x0_hat: torch.Tensor,
        x0: torch.Tensor,
        batch: MotionBatch,
        aux: dict[str, Any],
    ) -> torch.Tensor:
        return masked_mean_per_sample((x0_hat - x0) ** 2, element_mask(batch))

    def per_sample_blocks(
        self,
        x0_hat: torch.Tensor,
        x0: torch.Tensor,
        batch: MotionBatch,
        aux: dict[str, Any],
    ) -> dict[str, torch.Tensor]:
        """The same MSE, per feature block.

        The total mixes quantities that are not comparable -- a position error
        and a 6D rotation error land in one number -- so it cannot say WHICH
        block is hard.

        These SUM to the total, they do not average to it. `element_mask` is
        `(B, J, 1, T)`, so the divisor counts joints and frames but not
        channels: `simple` is a sum over channels, meaned over the real
        (joint, frame) pairs. That makes each block's value its literal
        contribution to the total, which is the reading worth having -- a
        wide block is allowed to carry more of the loss than a narrow one, and
        the numbers say by how much.

        Channels sit on axis 2 of a `(B, J, D, T)` batch, which is why the axis
        is named rather than assumed.
        """
        squared = (x0_hat - x0) ** 2
        mask = element_mask(batch)
        return {
            name: masked_mean_per_sample(
                batch.spec.take(squared, name, axis=2), mask
            )
            for name in batch.spec.names
        }

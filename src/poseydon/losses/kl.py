"""Latent regularization."""

from __future__ import annotations

from typing import Any

import torch

from poseydon.core.batch import MotionBatch
from poseydon.losses.base import LOSSES, LossTerm


@LOSSES.register("kl")
class KLDivergence(LossTerm):
    """KL between a diagonal Gaussian posterior and a standard normal.

    Reads ``mu`` and ``logvar`` from the model's auxiliary outputs, so a model
    without a stochastic bottleneck simply cannot be paired with this term --
    and says so, rather than silently contributing zero.
    """

    name = "kl"

    def __call__(
        self,
        x0_hat: torch.Tensor,
        x0: torch.Tensor,
        batch: MotionBatch,
        aux: dict[str, Any],
    ) -> torch.Tensor:
        missing = [key for key in ("mu", "logvar") if key not in aux]
        if missing:
            raise ValueError(
                f"loss `kl` needs {', '.join(missing)} in the model's auxiliary "
                "outputs; this model has no stochastic bottleneck"
            )
        mu, logvar = aux["mu"], aux["logvar"]
        return -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())

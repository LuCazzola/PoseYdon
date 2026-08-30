"""The single training task.

One LightningModule serves every model, process and loss combination. There is
no DiffusionTask and LatentDiffusionTask to keep in step, because latent
operation lives in the model's class rather than in the task's.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from poseydon.core.batch import MotionBatch
from poseydon.losses.base import LossTerm
from poseydon.models.base import Denoiser
from poseydon.process.base import Process


class MotionTask(nn.Module):
    """Ties a denoiser, a process and weighted loss terms together.

    Deliberately a plain ``nn.Module`` for now: everything here is framework
    independent, and wrapping it in a LightningModule adds optimizer and logging
    concerns that belong with the training entry point rather than the objective.
    """

    def __init__(
        self,
        model: Denoiser,
        process: Process,
        losses: Sequence[tuple[str, float, LossTerm]],
    ) -> None:
        super().__init__()
        self.model = model
        self.process = process
        self.losses = list(losses)

    def setup_checks(self, batch: MotionBatch) -> None:
        """Validate the whole configuration against one real batch.

        Run once, before training, so a mismatch costs a second rather than an
        epoch: every loss must find the feature blocks it needs, and the model
        must find the conditioners it requires.
        """
        for _, _, term in self.losses:
            term.validate(batch.spec)
        self.model.check_conditioners(set(batch.cond.payloads))

    def compute_losses(
        self, batch: MotionBatch, noise: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        """One training step's losses, keyed by name plus a `total`."""
        z0 = self.model.prepare(batch)
        t = self.process.sample_t(len(batch), device=z0.device)
        if noise is None:
            noise = torch.randn_like(z0)

        z_t = self.process.corrupt(z0, t, noise)
        prediction = self.model(z_t, t, batch.cond)
        x0_hat = self.model.restore(
            self.process.to_z0(prediction.out, z_t, t), batch
        )

        terms: dict[str, torch.Tensor] = {}
        total = torch.zeros((), device=x0_hat.device, dtype=x0_hat.dtype)
        for name, weight, term in self.losses:
            value = term(x0_hat, batch.x, batch, prediction.aux)
            terms[name] = value
            total = total + weight * value
        terms["total"] = total
        return terms

    def forward(self, batch: MotionBatch) -> torch.Tensor:
        return self.compute_losses(batch)["total"]

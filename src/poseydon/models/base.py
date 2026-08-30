"""Denoiser interface.

Latent operation is expressed by SUBCLASSING, not by injecting an identity
autoencoder. A model that works directly on features inherits pass-through
``prepare`` and ``restore``; a latent model overrides them with its own encoder
and decoder. Configuration never mentions autoencoding -- the model's class
decides -- and the task body is identical for both.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar

import torch
from torch import nn

from poseydon.core.batch import Cond, Masks, MotionBatch
from poseydon.core.registry import Registry


@dataclass
class Prediction:
    """What a denoiser returns.

    ``aux`` carries anything a loss or a probe needs beyond the prediction
    itself -- ``mu``/``logvar`` for a KL term, layer activations for analysis.
    """

    out: torch.Tensor
    aux: dict[str, Any] = field(default_factory=dict)


class Denoiser(nn.Module, ABC):
    """Operates directly on motion features."""

    #: Conditioners this model cannot run without.
    requires: ClassVar[tuple[str, ...]] = ()
    #: Conditioners this model will use if configured, and ignore otherwise.
    optional: ClassVar[tuple[str, ...]] = ()

    def check_conditioners(self, configured: set[str]) -> None:
        """Fail at fit start, not forty minutes into training."""
        missing = sorted(set(self.requires) - configured)
        if missing:
            raise ValueError(
                f"{type(self).__name__} requires conditioner(s) {', '.join(missing)}, "
                f"but only {', '.join(sorted(configured)) or '(none)'} are configured"
            )

    def prepare(self, batch: MotionBatch) -> torch.Tensor:
        """The tensor the process corrupts. Feature space by default."""
        return batch.x

    @abstractmethod
    def forward(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        cond: Cond,
        masks: Masks | None = None,
    ) -> Prediction:
        """Predict the process target at time ``t``.

        ``masks`` says which joints and frames are real. It is a separate
        argument rather than a conditioner because padding is a property of how
        the batch was assembled, not something the user declares. ``None`` means
        everything is valid, which is the case when sampling from noise.
        """

    def restore(self, z0: torch.Tensor, batch: MotionBatch) -> torch.Tensor:
        """Map a clean sample back to feature space. Identity by default."""
        return z0


class LatentDenoiser(Denoiser, ABC):
    """Operates in a learned latent space."""

    @abstractmethod
    def encode(self, x0: torch.Tensor, cond: Cond) -> torch.Tensor: ...

    @abstractmethod
    def decode(self, z0: torch.Tensor, cond: Cond) -> torch.Tensor: ...

    def prepare(self, batch: MotionBatch) -> torch.Tensor:
        return self.encode(batch.x, batch.cond)

    def restore(self, z0: torch.Tensor, batch: MotionBatch) -> torch.Tensor:
        return self.decode(z0, batch.cond)


#: Reserved conditioner name. A model that lists this in ``requires`` is asking
#: the task for the CLEAN motion alongside the corrupted one -- an autoencoder
#: needs its own input. The task supplies it; no dataset conditioner produces it.
CLEAN_MOTION = "clean_motion"

MODELS: Registry[Denoiser] = Registry("model")

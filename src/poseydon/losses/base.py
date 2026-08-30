"""Loss terms.

Each term declares the feature blocks it needs and the space it needs them in,
and that declaration is checked against the active :class:`FeatureSpec` once at
fit start. A term asking for rotations against a representation that has none
fails in the first second with a readable message, rather than slicing whatever
happens to occupy those columns.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar

import torch

from poseydon.core.batch import MotionBatch
from poseydon.core.registry import Registry
from poseydon.core.spec import Block, FeatureSpec

# Conditioner key holding per-skeleton normalization statistics. A term that
# needs a block in raw space needs these to undo normalization.
NORM_STATS = "norm_stats"


class LossTerm(ABC):
    """One weighted contribution to the training objective."""

    name: ClassVar[str]
    needs: ClassVar[tuple[Block, ...]] = ()

    def validate(self, spec: FeatureSpec) -> None:
        """Raise unless every needed block exists in this representation."""
        for block in self.needs:
            if block.name not in spec:
                raise ValueError(
                    f"loss `{self.name}` needs feature block `{block.name}`, but the "
                    f"active representation has {', '.join(spec.names) or '(none)'}. "
                    f"Either add it to `features:` or drop this loss."
                )

    @abstractmethod
    def __call__(
        self,
        x0_hat: torch.Tensor,
        x0: torch.Tensor,
        batch: MotionBatch,
        aux: dict[str, Any],
    ) -> torch.Tensor:
        """Return a scalar loss."""


LOSSES: Registry[LossTerm] = Registry("loss")


def element_mask(batch: MotionBatch) -> torch.Tensor:
    """(B, J, 1, T) mask selecting real joints on real frames."""
    joints = batch.masks.joints[:, :, None, None]
    frames = batch.masks.frames[:, None, None, :]
    return (joints & frames).to(batch.x.dtype)


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean of ``values`` over the unmasked elements only.

    Padding must not dilute the loss: a 28-joint skeleton in a 63-joint batch
    would otherwise have most of its gradient averaged against zeros.
    """
    total = (values * mask).sum()
    count = mask.sum().clamp(min=1.0)
    return total / count


def raw_block(batch: MotionBatch, tensor: torch.Tensor, name: str) -> torch.Tensor:
    """Extract one block and undo normalization, so the values mean something.

    The reference computes its geodesic and foot-skate losses on z-normalized
    values. A per-channel divide by the standard deviation is not a
    rotation-preserving operation, so the 6D rows it re-orthonormalizes are not
    the rotations the animation contains. This helper is why that cannot happen
    here by accident.
    """
    where = batch.spec.slice(name)
    values = tensor[:, :, where, :]

    stats = batch.cond.get(NORM_STATS)
    if stats is None:
        raise ValueError(
            f"block `{name}` was requested in raw space, but no `{NORM_STATS}` "
            "conditioner is configured to undo normalization"
        )
    mean, std = stats["mean"], stats["std"]
    mean = mean[:, :, where, None].to(values.device, values.dtype)
    std = std[:, :, where, None].to(values.device, values.dtype)
    return values * std + mean

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

#: Conditioning key holding an explicit ``(B, T+1, T+1)`` frame-attention mask.
#: A model's ``temporal_window`` is the DEFAULT band; this key, when present,
#: REPLACES THE BAND -- the caller declared their own temporal locality, and
#: applying the band afterwards would silently narrow a mask they chose
#: deliberately. It does NOT replace the padding mask: see
#: :func:`apply_temporal_override`.
TEMPORAL_VALID = "temporal_valid"


def padding_pair_mask(frame_valid: torch.Tensor) -> torch.Tensor:
    """``(B, T)`` frame validity to a ``(B, T+1, T+1)`` pair mask, and nothing else.

    No band, no rest-row policy -- just which (query, key) frame pairs are real
    data, with the rest frame prepended at index 0. Padded positions are literal
    zero-fill (``poseydon.data.collate``), so this is the one part of the
    temporal mask that is not a policy anybody may opt out of.
    """
    batch, _ = frame_valid.shape
    leading = torch.ones(batch, 1, dtype=torch.bool, device=frame_valid.device)
    extended = torch.cat([leading, frame_valid], dim=1)
    return extended[:, :, None] & extended[:, None, :]


def temporal_pair_mask(
    frame_valid: torch.Tensor, window: int, rest_attends_all: bool = False
) -> torch.Tensor:
    """``(B, T)`` frame validity to a band-limited ``(B, T+1, T+1)`` pair mask.

    Index 0 of the pair mask is the REST frame, which conditions everything; the
    band applies to the real frames after it. ``rest_attends_all`` selects the
    rest row's policy, because the two models genuinely differ: MoDiffAE's rest
    token attends only itself (the default), AnyTop's attends every valid frame.
    That is a per-model choice, not something one model should inherit from the
    other by sharing this function.

    The band is ANDed with the padding mask, never ORed: it NARROWS attention
    and must never make a padded frame visible. ``window`` of 0 (or None)
    disables it; 1 leaves every real frame attending only itself.
    """
    _, frames = frame_valid.shape
    pair = padding_pair_mask(frame_valid)

    if window:
        offsets = torch.arange(frames, device=frame_valid.device)
        band = (offsets[:, None] - offsets[None, :]).abs() <= window // 2
        pair[:, 1:, 1:] &= band

    if not rest_attends_all:
        # The rest frame conditions everything but attends only to itself.
        pair[:, 0, :] = False
        pair[:, 0, 0] = True
    return pair


def apply_temporal_override(override: torch.Tensor, frame_valid: torch.Tensor) -> torch.Tensor:
    """Intersect a caller-supplied temporal mask with the padding pair.

    An override REPLACES THE BAND -- band width and rest-row policy are both the
    caller's to choose, and narrowing their mask would defeat the point. It is
    still ANDed with the PADDING pair: a padded frame is zero-fill, not data,
    and no caller can make it real. Widening attention is not on the menu.
    """
    expected = (frame_valid.shape[0], frame_valid.shape[1] + 1, frame_valid.shape[1] + 1)
    if tuple(override.shape) != expected:
        raise ValueError(
            f"`cond[{TEMPORAL_VALID!r}]` must be {expected} (T+1 for the rest frame), "
            f"got {tuple(override.shape)}"
        )
    return override.to(frame_valid.device) & padding_pair_mask(frame_valid)


MODELS: Registry[Denoiser] = Registry("model")

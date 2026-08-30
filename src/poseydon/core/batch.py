"""The dataset-to-model contract.

Only four things are mandatory: the motion tensor, its layout, the masks that
padding makes unavoidable, and where the window came from. Everything the
reference packs into a fourteen-key ``cond['y']`` dict -- T-pose, joint names,
topology, object type, normalization statistics -- is a declared conditioner
instead, so a fixed-skeleton text-conditioned corpus drops in without touching
this file.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import torch

from poseydon.core.spec import FeatureSpec


@dataclass(frozen=True)
class Masks:
    """Which frames and joints are real rather than padding.

    The reference threads four overlapping masks through its forwards and
    reshapes them inline; these are computed once and named.
    """

    frames: torch.Tensor  # (B, T) bool
    joints: torch.Tensor  # (B, J) bool

    def __post_init__(self) -> None:
        for name, tensor in (("frames", self.frames), ("joints", self.joints)):
            if tensor.dtype is not torch.bool:
                raise TypeError(f"{name} mask must be bool, got {tensor.dtype}")
            if tensor.ndim != 2:
                raise ValueError(f"{name} mask must be (B, N), got {tuple(tensor.shape)}")

    @property
    def lengths(self) -> torch.Tensor:
        """(B,) number of real frames per item."""
        return self.frames.sum(dim=-1)

    @property
    def n_joints(self) -> torch.Tensor:
        """(B,) number of real joints per item."""
        return self.joints.sum(dim=-1)

    def temporal(self) -> torch.Tensor:
        """(B, 1, T, T) pairwise frame validity, for attention."""
        pair = self.frames[:, :, None] & self.frames[:, None, :]
        return pair[:, None]

    def spatial(self) -> torch.Tensor:
        """(B, 1, J, J) pairwise joint validity, for attention."""
        pair = self.joints[:, :, None] & self.joints[:, None, :]
        return pair[:, None]

    def to(self, device: torch.device | str) -> Masks:
        return Masks(frames=self.frames.to(device), joints=self.joints.to(device))


@dataclass(frozen=True)
class WindowInfo:
    """Where each item's window sits inside its source animation.

    A model may use ``start`` to offset positional encodings. Because ingest
    never chunks, this is an offset into the real animation rather than into an
    arbitrary preprocessing slice.
    """

    start: torch.Tensor  # (B,) long
    source_length: torch.Tensor  # (B,) long

    def to(self, device: torch.device | str) -> WindowInfo:
        return WindowInfo(
            start=self.start.to(device), source_length=self.source_length.to(device)
        )


@dataclass(frozen=True)
class Cond:
    """Conditioning payloads, keyed by the conditioner that produced them."""

    payloads: dict[str, Any]

    def __contains__(self, name: object) -> bool:
        return name in self.payloads

    def __getitem__(self, name: str) -> Any:
        try:
            return self.payloads[name]
        except KeyError:
            present = ", ".join(sorted(self.payloads)) or "(none)"
            raise KeyError(
                f"conditioner `{name}` was not configured. Present: {present}"
            ) from None

    def get(self, name: str, default: Any = None) -> Any:
        return self.payloads.get(name, default)

    def to(self, device: torch.device | str) -> Cond:
        moved = {
            key: value.to(device) if hasattr(value, "to") else value
            for key, value in self.payloads.items()
        }
        return Cond(payloads=moved)


@dataclass(frozen=True)
class MotionBatch:
    x: torch.Tensor  # (B, J, D, T)
    spec: FeatureSpec
    masks: Masks
    window: WindowInfo
    cond: Cond

    def __post_init__(self) -> None:
        if self.x.ndim != 4:
            raise ValueError(f"x must be (B, J, D, T), got {tuple(self.x.shape)}")
        batch, joints, dim, frames = self.x.shape
        if dim != self.spec.dim:
            raise ValueError(f"x has width {dim} but the spec declares {self.spec.dim}")
        if tuple(self.masks.frames.shape) != (batch, frames):
            raise ValueError(
                f"frame mask {tuple(self.masks.frames.shape)} does not match "
                f"x ({batch}, {frames})"
            )
        if tuple(self.masks.joints.shape) != (batch, joints):
            raise ValueError(
                f"joint mask {tuple(self.masks.joints.shape)} does not match "
                f"x ({batch}, {joints})"
            )

    def __len__(self) -> int:
        return int(self.x.shape[0])

    @property
    def n_joints(self) -> int:
        return int(self.x.shape[1])

    @property
    def n_frames(self) -> int:
        return int(self.x.shape[3])

    def block(self, name: str) -> torch.Tensor:
        """(B, J, width, T) for one named feature block."""
        return self.x[:, :, self.spec.slice(name), :]

    def to(self, device: torch.device | str) -> MotionBatch:
        return replace(
            self,
            x=self.x.to(device),
            masks=self.masks.to(device),
            window=self.window.to(device),
            cond=self.cond.to(device),
        )

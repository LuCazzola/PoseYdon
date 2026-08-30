"""Feature layout.

A ``FeatureSpec`` names every block's slice into the last axis of a motion
tensor. It exists so a loss can ask for ``spec.slice("rot6d")`` and fail loudly
against a representation that has no rotations, instead of a magic ``[..., 3:9]``
silently reading whatever happens to sit there.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Space = Literal["raw", "normalized"]


@dataclass(frozen=True)
class Block:
    """A request for one named block, in a particular space."""

    name: str
    space: Space = "normalized"


@dataclass(frozen=True)
class FeatureSpec:
    """Ordered block layout of the per-joint feature vector."""

    blocks: tuple[tuple[str, int], ...]  # (name, width), in order

    def __post_init__(self) -> None:
        names = [name for name, _ in self.blocks]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate feature blocks: {names}")
        for name, width in self.blocks:
            if width <= 0:
                raise ValueError(f"block `{name}` has non-positive width {width}")

    @property
    def dim(self) -> int:
        return sum(width for _, width in self.blocks)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(name for name, _ in self.blocks)

    def slice(self, name: str) -> slice:
        start = 0
        for block_name, width in self.blocks:
            if block_name == name:
                return slice(start, start + width)
            start += width
        raise KeyError(
            f"feature block `{name}` is not in this representation. "
            f"Present: {', '.join(self.names) or '(none)'}"
        )

    def __contains__(self, name: object) -> bool:
        return name in self.names

    def describe(self) -> str:
        parts = []
        start = 0
        for name, width in self.blocks:
            parts.append(f"{name}[{start}:{start + width}]")
            start += width
        return f"FeatureSpec(dim={self.dim}, {' '.join(parts)})"

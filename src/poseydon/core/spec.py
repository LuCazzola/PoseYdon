"""Feature layout.

A ``FeatureSpec`` names every block's slice into the last axis of a motion
tensor. It exists so a loss can ask for ``spec.slice("rot6d")`` and fail loudly
against a representation that has no rotations, instead of a magic ``[..., 3:9]``
silently reading whatever happens to sit there.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

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

    def take(self, tensor: Any, name: str, axis: int = -1, *, drop: bool = False) -> Any:
        """The named block of ``tensor``, sliced along ``axis``.

        Works on numpy arrays and torch tensors alike -- it only builds an index.

        ``axis`` is explicit and NOT guessed, because the channel axis is not the
        same everywhere in this repo: features are ``(F, J, D)`` with channels
        last, a `MotionBatch` is ``(B, J, D, T)`` with channels at 2, and fitted
        statistics are ``(J, D)`` with channels at 1. A helper that assumed the
        last axis would silently read frames as channels on the training path.

        ``drop=True`` removes the block axis for a width-1 block, and refuses
        loudly for any other width. Callers were writing that squeeze by hand as
        a bare ``[..., 0]``, which is undocumented and wrong the moment a block
        grows a channel.
        """
        where = self.slice(name)
        index: list[Any] = [slice(None)] * tensor.ndim
        index[axis] = where
        block = tensor[tuple(index)]
        if not drop:
            return block
        width = where.stop - where.start
        if width != 1:
            raise ValueError(
                f"`drop=True` removes the block axis, but `{name}` is "
                f"{width} wide, not 1"
            )
        drop_index: list[Any] = [slice(None)] * block.ndim
        drop_index[axis] = 0
        return block[tuple(drop_index)]

    def __contains__(self, name: object) -> bool:
        return name in self.names

    def describe(self) -> str:
        parts = []
        start = 0
        for name, width in self.blocks:
            parts.append(f"{name}[{start}:{start + width}]")
            start += width
        return f"FeatureSpec(dim={self.dim}, {' '.join(parts)})"

"""Windowing policies.

Windowing is a load-time sampling concern, not a preprocessing one. The
reference cuts every animation into fixed 200-frame chunks on disk, so a dance
is split at an arbitrary boundary the model never sees across, and the two
halves can land in different splits. Here a clip stays whole and the window is
chosen when it is read, so changing window length is a config edit.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from poseydon.core.registry import Registry


class Window(ABC):
    """How many windows a clip yields, and where each one sits."""

    @abstractmethod
    def count(self, n_frames: int) -> int:
        """Number of windows this policy draws from a clip of ``n_frames``."""

    @abstractmethod
    def bounds(
        self, n_frames: int, index: int, rng: np.random.Generator
    ) -> tuple[int, int]:
        """``(start, length)`` of window ``index``. ``length`` may exceed the clip."""


WINDOWS: Registry[Window] = Registry("window")


@WINDOWS.register("full_clip")
@dataclass(frozen=True)
class FullClip(Window):
    """One window covering the whole animation."""

    def count(self, n_frames: int) -> int:
        return 1

    def bounds(self, n_frames: int, index: int, rng: np.random.Generator) -> tuple[int, int]:
        return 0, n_frames


@WINDOWS.register("random_crop")
@dataclass(frozen=True)
class RandomCrop(Window):
    """One randomly placed window per read, for training.

    A clip shorter than ``length`` yields its whole self; the collate pads and
    masks the remainder.
    """

    length: int = 40

    def __post_init__(self) -> None:
        if self.length < 1:
            raise ValueError(f"length must be positive, got {self.length}")

    def count(self, n_frames: int) -> int:
        return 1

    def bounds(self, n_frames: int, index: int, rng: np.random.Generator) -> tuple[int, int]:
        if n_frames <= self.length:
            return 0, n_frames
        return int(rng.integers(0, n_frames - self.length + 1)), self.length


@WINDOWS.register("sliding")
@dataclass(frozen=True)
class SlidingWindow(Window):
    """Deterministic overlapping coverage, for validation and evaluation."""

    length: int = 40
    stride: int = 20

    def __post_init__(self) -> None:
        if self.length < 1:
            raise ValueError(f"length must be positive, got {self.length}")
        if self.stride < 1:
            raise ValueError(f"stride must be positive, got {self.stride}")

    def count(self, n_frames: int) -> int:
        if n_frames <= self.length:
            return 1
        return 1 + (n_frames - self.length + self.stride - 1) // self.stride

    def bounds(self, n_frames: int, index: int, rng: np.random.Generator) -> tuple[int, int]:
        if n_frames <= self.length:
            return 0, n_frames
        start = min(index * self.stride, n_frames - self.length)
        return start, self.length

"""Conditioners.

A conditioner owns its whole data path -- read, collate, move to device -- and
may ship a default encoder when the conditioning needs one. How the result is
INJECTED stays the model's business, because injection is genuinely
model-specific: AnyTop concatenates the T-pose as an extra frame and adds joint
name embeddings to per-joint tokens, which no generic mechanism captures
honestly.

Only declared conditioners are loaded. A fixed-skeleton text-conditioned corpus
declares ``[text]``; an unconditional one declares nothing.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar

from poseydon.core.registry import Registry


class Conditioner(ABC):
    name: ClassVar[str]

    @abstractmethod
    def extract(self, item: Any) -> Any:
        """Per-clip payload, from a :class:`ClipView`."""

    @abstractmethod
    def collate(self, payloads: list[Any], max_joints: int) -> Any:
        """Batch per-clip payloads, padding joints to ``max_joints``."""

    def encoder(self, d_model: int):
        """Default module embedding this conditioning, or None."""
        return


CONDITIONERS: Registry[Conditioner] = Registry("conditioner")

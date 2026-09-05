"""Feature extractors.

A feature is a pure function of an aligned :class:`RigidBodyAnimation` producing one named
block of the per-joint feature vector. Extractors are registry-named so a config
can say ``features: [rot6d, local_vel]`` rather than spelling out import paths.

Adding a representation is one class plus a decorator; nothing on disk changes,
because ingest stores the canonical animation, not features.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from poseydon.core.animation import RigidBodyAnimation
from poseydon.core.registry import Registry
from poseydon.core.skeleton import ResolvedSkeleton
from poseydon.ingest.align import facing_quats


@dataclass(frozen=True)
class FeatureContext:
    """Shared work, computed once per clip rather than once per feature.

    Forward kinematics and the per-frame facing rotation are needed by three of
    the four built-in features, and FK over a 63-joint skeleton is the expensive
    part of extraction.
    """

    anim: RigidBodyAnimation
    resolved: ResolvedSkeleton
    positions: np.ndarray  # (F, J, 3) global joint positions
    root_quats: np.ndarray  # (F, 4) per-frame rotation taking forward onto +Z

    @classmethod
    def build(cls, anim: RigidBodyAnimation, resolved: ResolvedSkeleton) -> FeatureContext:
        positions = anim.global_positions()
        return cls(
            anim=anim,
            resolved=resolved,
            positions=positions,
            root_quats=facing_quats(
                positions,
                resolved.facing_indices,
                resolved.manifest.extra_yaw_deg,
            ),
        )

    @property
    def n_joints(self) -> int:
        return int(self.positions.shape[1])


class Feature(ABC):
    """One named block of the feature vector."""

    name: ClassVar[str]
    width: ClassVar[int]

    @abstractmethod
    def __call__(self, ctx: FeatureContext) -> np.ndarray:
        """Return ``(frames, joints, width)``.

        A feature built from finite differences returns one frame fewer than the
        animation; the assembler truncates every block to the shortest.
        """


FEATURES: Registry[Feature] = Registry("feature")

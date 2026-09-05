"""Assemble requested feature blocks into one tensor plus its layout."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from poseydon.core.animation import RigidBodyAnimation
from poseydon.core.skeleton import ResolvedSkeleton
from poseydon.core.spec import FeatureSpec
from poseydon.features import contact as _contact  # noqa: F401  (registers features)
from poseydon.features import kinematics as _kinematics  # noqa: F401
from poseydon.features.base import FEATURES, FeatureContext

DEFAULT_FEATURES: tuple[str, ...] = ("ric_pos", "rot6d", "local_vel", "foot_contact")


def extract_features(
    anim: RigidBodyAnimation,
    resolved: ResolvedSkeleton,
    names: Sequence[str] = DEFAULT_FEATURES,
) -> tuple[np.ndarray, FeatureSpec]:
    """Build ``(frames, joints, dim)`` plus the :class:`FeatureSpec` naming it.

    Blocks built from finite differences are one frame shorter, so every block is
    truncated to the shortest. Requesting only whole-frame features therefore
    keeps all frames; asking for a velocity costs the last one.
    """
    if not names:
        raise ValueError("at least one feature must be requested")

    ctx = FeatureContext.build(anim, resolved)
    blocks = [(name, FEATURES.get(name)()(ctx)) for name in names]

    n_frames = min(block.shape[0] for _, block in blocks)
    array = np.concatenate([block[:n_frames] for _, block in blocks], axis=-1)
    spec = FeatureSpec(tuple((name, block.shape[-1]) for name, block in blocks))

    if array.shape[-1] != spec.dim:
        raise AssertionError(f"assembled width {array.shape[-1]} != spec dim {spec.dim}")
    return array, spec

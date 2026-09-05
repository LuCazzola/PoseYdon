"""Augmentation contract and pipeline.

An augmentation is a small class with two independent, individually optional
hooks: ``apply_structural`` edits the skeleton itself (joint count, order,
rotations) BEFORE feature extraction, and ``apply_features`` edits the
already-extracted, already-normalized ``(frames, joints, dim)`` array. Most
augmentations need only one; the base class no-ops the other so a subclass
overrides exactly what it changes.

``AugmentPipeline`` composes a config-ordered list, each entry gated by its
own ``p`` -- an empty list, or every entry's trial failing, is a true no-op:
the identity ``JointEdit`` and the input array unchanged.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from poseydon.augment.joint_edit import JointEdit
from poseydon.core.registry import Registry


@dataclass
class Augmentation:
    p: float = 1.0

    def apply_structural(
        self, anim: Any, resolved: Any, rng: np.random.Generator
    ) -> tuple[Any, Any, JointEdit]:
        return anim, resolved, JointEdit.identity(anim.n_joints)

    def apply_features(
        self,
        features: np.ndarray,
        spec: Any,
        resolved: Any,
        rng: np.random.Generator,
    ) -> np.ndarray:
        return features


AUGMENTATIONS: Registry[Augmentation] = Registry("augmentation")


class AugmentPipeline:
    """Ordered, independently-gated composition of augmentations."""

    def __init__(self, augmentations: Sequence[Augmentation] = ()) -> None:
        self.augmentations = tuple(augmentations)

    def apply_structural(
        self, anim: Any, resolved: Any, rng: np.random.Generator
    ) -> tuple[Any, Any, JointEdit]:
        edit = JointEdit.identity(anim.n_joints)
        for augmentation in self.augmentations:
            if rng.random() < augmentation.p:
                anim, resolved, step_edit = augmentation.apply_structural(anim, resolved, rng)
                edit = step_edit.compose(edit)
        return anim, resolved, edit

    def apply_features(
        self,
        features: np.ndarray,
        spec: Any,
        resolved: Any,
        rng: np.random.Generator,
    ) -> np.ndarray:
        for augmentation in self.augmentations:
            if rng.random() < augmentation.p:
                features = augmentation.apply_features(features, spec, resolved, rng)
        return features

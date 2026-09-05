"""Structural augmentations: joint drop and joint duplication.

Ports the reference's per-sample topology augmentation (see design spec
section 3) as exact mutations of :class:`~poseydon.core.anim.Anim`, so every
feature block -- present or future -- is recomputed from a genuinely valid
skeleton rather than patched after extraction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from poseydon.augment.base import AUGMENTATIONS, Augmentation
from poseydon.augment.joint_edit import JointEdit
from poseydon.core.anim import Anim
from poseydon.core.rotations import QUAT_IDENTITY
from poseydon.core.skeleton import ResolvedSkeleton


def leaves(parents: np.ndarray) -> list[int]:
    """Joint indices that are nobody's parent."""
    parents = np.asarray(parents)
    has_children = np.zeros(len(parents), dtype=bool)
    valid = parents >= 0
    has_children[parents[valid]] = True
    return [j for j in range(len(parents)) if not has_children[j]]


def child_count(parents: np.ndarray) -> np.ndarray:
    """Number of children per joint, ``(J,)``."""
    parents = np.asarray(parents)
    counts = np.zeros(len(parents), dtype=np.int64)
    valid = parents >= 0
    np.add.at(counts, parents[valid], 1)
    return counts


def _excluded_joints(resolved: ResolvedSkeleton) -> set[int]:
    facing = {i for pair in resolved.facing_indices for i in pair}
    return facing | set(resolved.foot_indices)


def _reindex_resolved(
    resolved: ResolvedSkeleton, old_to_new: dict[int, int]
) -> ResolvedSkeleton:
    facing = tuple((old_to_new[r], old_to_new[l]) for r, l in resolved.facing_indices)
    feet = tuple(old_to_new[f] for f in resolved.foot_indices)
    return ResolvedSkeleton(manifest=resolved.manifest, facing_indices=facing, foot_indices=feet)


def drop_joints(
    anim: Anim, resolved: ResolvedSkeleton, joints: set[int]
) -> tuple[Anim, ResolvedSkeleton, JointEdit]:
    """Remove ``joints`` (must all be leaves) from ``anim``.

    Every surviving joint's forward-kinematics position is unchanged: a leaf
    has no children, so nothing downstream referenced it.
    """
    keep = [j for j in range(anim.n_joints) if j not in joints]
    old_to_new = {old: new for new, old in enumerate(keep)}

    new_parents = np.array(
        [-1 if anim.parents[j] == -1 else old_to_new[int(anim.parents[j])] for j in keep],
        dtype=np.int32,
    )
    new_anim = Anim(
        rotations=anim.rotations[:, keep, :].copy(),
        root_pos=anim.root_pos.copy(),
        offsets=anim.offsets[keep].copy(),
        parents=new_parents,
        names=tuple(anim.names[j] for j in keep),
        fps=anim.fps,
    )
    new_resolved = _reindex_resolved(resolved, old_to_new)
    return new_anim, new_resolved, JointEdit(source_of=tuple(keep))


@AUGMENTATIONS.register("drop_end_effector")
@dataclass
class DropEndEffector(Augmentation):
    """Randomly drop a fraction of non-foot, non-facing end-effector joints.

    Ports the reference's ``remove_joints_augmentation``. A rate is drawn
    uniformly from ``rate``; the number of joints dropped is
    ``floor(len(candidates) * rate)``, so a small rate on a small candidate
    set can legally drop zero joints (identity edit).
    """

    rate: tuple[float, ...] = field(default_factory=lambda: (0.1, 0.2, 0.3))

    def apply_structural(self, anim: Anim, resolved: ResolvedSkeleton, rng: np.random.Generator):
        excluded = _excluded_joints(resolved)
        candidates = sorted(j for j in leaves(anim.parents) if j not in excluded and j != 0)
        if not candidates:
            return anim, resolved, JointEdit.identity(anim.n_joints)

        rate = float(rng.choice(self.rate))
        n_remove = math.floor(len(candidates) * rate)
        if n_remove == 0:
            return anim, resolved, JointEdit.identity(anim.n_joints)

        drop = set(rng.choice(candidates, size=n_remove, replace=False).tolist())
        return drop_joints(anim, resolved, drop)


def duplicate_joint(
    anim: Anim, resolved: ResolvedSkeleton, joint: int
) -> tuple[Anim, ResolvedSkeleton, JointEdit]:
    """Insert a midpoint joint between ``joint`` and its parent.

    The new joint takes slot ``joint``; every original joint at or after
    ``joint`` shifts up by one. The new joint gets an identity rotation and
    half of ``joint``'s offset; ``joint`` itself (now at ``joint + 1``) keeps
    its own rotation and the other half-offset -- so every world position,
    ``joint``'s and everything below it, is unchanged (design spec section 4).
    """
    n = anim.n_joints
    n_frames = anim.n_frames
    old_parent = int(anim.parents[joint])

    def shift(old_index: int) -> int:
        return old_index if old_index < joint else old_index + 1

    new_parents = np.empty(n + 1, dtype=np.int32)
    new_offsets = np.empty((n + 1, 3), dtype=anim.offsets.dtype)
    new_rotations = np.empty((n_frames, n + 1, 4), dtype=anim.rotations.dtype)
    new_names: list[str] = []
    source_of: list[int] = []

    for old_index in range(n):
        new_index = shift(old_index)
        new_offsets[new_index] = anim.offsets[old_index]
        new_rotations[:, new_index] = anim.rotations[:, old_index]
        parent = anim.parents[old_index]
        new_parents[new_index] = -1 if parent == -1 else shift(int(parent))
        new_names.append(anim.names[old_index])
        source_of.append(old_index)

    new_names.insert(joint, f"{anim.names[joint]}__mid")
    source_of.insert(joint, joint)
    new_offsets[joint] = anim.offsets[joint] / 2
    new_offsets[joint + 1] = anim.offsets[joint] / 2
    new_rotations[:, joint] = QUAT_IDENTITY
    new_parents[joint] = -1 if old_parent == -1 else shift(old_parent)
    new_parents[joint + 1] = joint

    new_anim = Anim(
        rotations=new_rotations,
        root_pos=anim.root_pos.copy(),
        offsets=new_offsets,
        parents=new_parents,
        names=tuple(new_names),
        fps=anim.fps,
    )
    old_to_new = {old: shift(old) for old in range(n)}
    new_resolved = _reindex_resolved(resolved, old_to_new)
    return new_anim, new_resolved, JointEdit(source_of=tuple(source_of))


@AUGMENTATIONS.register("duplicate_joint")
@dataclass
class DuplicateJoint(Augmentation):
    """Randomly duplicate one single-child, non-root-adjacent joint.

    Ports the reference's ``add_joint_augmentation``, but the split is exact
    under forward kinematics rather than an interpolation of features (see
    design spec section 4).
    """

    def apply_structural(self, anim: Anim, resolved: ResolvedSkeleton, rng: np.random.Generator):
        excluded = _excluded_joints(resolved)
        counts = child_count(anim.parents)
        candidates = [
            j
            for j in range(1, anim.n_joints)
            if counts[j] == 1 and anim.parents[j] != 0 and j not in excluded
        ]
        if not candidates:
            return anim, resolved, JointEdit.identity(anim.n_joints)

        joint = int(rng.choice(candidates))
        return duplicate_joint(anim, resolved, joint)

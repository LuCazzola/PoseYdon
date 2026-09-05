"""Forward kinematics over a joint hierarchy."""

from __future__ import annotations

import numpy as np

from poseydon.core.rotations import quat_apply, quat_mul


def check_topological_order(parents: np.ndarray) -> None:
    """Every joint must appear after its parent, and joint 0 must be the root.

    BVH files are written depth-first, so this always holds for parsed files.
    Asserting it lets forward kinematics run as a single forward sweep.
    """
    parents = np.asarray(parents)
    if parents.ndim != 1 or parents.size == 0:
        raise ValueError(f"parents must be a non-empty 1-D array, got shape {parents.shape}")
    if parents[0] != -1:
        raise ValueError(f"joint 0 must be the root with parent -1, got {parents[0]}")
    if np.any(parents[1:] < 0):
        raise ValueError("only joint 0 may have parent -1; found another negative parent")
    bad = np.nonzero(parents[1:] >= np.arange(1, parents.size))[0]
    if bad.size:
        joint = int(bad[0]) + 1
        raise ValueError(
            f"parents must be in topological order: joint {joint} has parent "
            f"{parents[joint]}, which does not precede it"
        )


def forward_kinematics(
    rotations: np.ndarray,
    translations: np.ndarray,
    parents: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Local rotations and translations to global joint positions and rotations.

    Each joint's ``(rotation, translation)`` pair is read as its own local
    transform, which is the general case: a joint that translates is
    carried correctly rather than being pinned to a fixed bone length.
    Rigid motion is the special case where ``translations`` repeats each
    joint's rest offset on every frame -- the same sweep then computes
    exactly the rigid result, so there is one definition of forward
    kinematics rather than a rigid one and a raw one.

    Args:
        rotations:    ``(F, J, 4)`` local rotations, scalar-last quaternions.
        translations: ``(F, J, 3)`` local translation of each joint from its
            parent. Joint 0 has no parent, so its entry is the root's GLOBAL
            position.
        parents:      ``(J,)`` parent index per joint, ``-1`` for the root.

    Returns:
        ``(positions (F, J, 3), global_rotations (F, J, 4))``.
    """
    rotations = np.asarray(rotations, dtype=np.float64)
    translations = np.asarray(translations, dtype=np.float64)
    parents = np.asarray(parents, dtype=np.int32)

    check_topological_order(parents)
    n_frames, n_joints = rotations.shape[:2]
    if translations.shape != (n_frames, n_joints, 3):
        raise ValueError(
            f"translations must be ({n_frames}, {n_joints}, 3), got {translations.shape}"
        )

    positions = np.empty((n_frames, n_joints, 3), dtype=np.float64)
    global_rot = np.empty((n_frames, n_joints, 4), dtype=np.float64)

    positions[:, 0] = translations[:, 0]
    global_rot[:, 0] = rotations[:, 0]

    for joint in range(1, n_joints):
        parent = parents[joint]
        positions[:, joint] = positions[:, parent] + quat_apply(
            global_rot[:, parent], translations[:, joint]
        )
        global_rot[:, joint] = quat_mul(global_rot[:, parent], rotations[:, joint])

    return positions, global_rot

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
    root_pos: np.ndarray,
    offsets: np.ndarray,
    parents: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Local rotations to global joint positions and rotations.

    Args:
        rotations: ``(F, J, 4)`` local rotations, scalar-last quaternions.
        root_pos:  ``(F, 3)`` global translation of joint 0.
        offsets:   ``(J, 3)`` rest-pose offset of each joint from its parent.
        parents:   ``(J,)`` parent index per joint, ``-1`` for the root.

    Returns:
        ``(positions (F, J, 3), global_rotations (F, J, 4))``.
    """
    rotations = np.asarray(rotations, dtype=np.float64)
    root_pos = np.asarray(root_pos, dtype=np.float64)
    offsets = np.asarray(offsets, dtype=np.float64)
    parents = np.asarray(parents, dtype=np.int32)

    check_topological_order(parents)
    n_frames, n_joints = rotations.shape[:2]
    if offsets.shape != (n_joints, 3):
        raise ValueError(f"offsets must be ({n_joints}, 3), got {offsets.shape}")
    if root_pos.shape != (n_frames, 3):
        raise ValueError(f"root_pos must be ({n_frames}, 3), got {root_pos.shape}")

    positions = np.empty((n_frames, n_joints, 3), dtype=np.float64)
    global_rot = np.empty((n_frames, n_joints, 4), dtype=np.float64)

    positions[:, 0] = root_pos
    global_rot[:, 0] = rotations[:, 0]

    for joint in range(1, n_joints):
        parent = parents[joint]
        positions[:, joint] = positions[:, parent] + quat_apply(
            global_rot[:, parent], offsets[joint]
        )
        global_rot[:, joint] = quat_mul(global_rot[:, parent], rotations[:, joint])

    return positions, global_rot

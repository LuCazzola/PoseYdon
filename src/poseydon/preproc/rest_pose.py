"""T-pose geometry recovery and bind-pose removal for raw rig exports.

Raw Biped rigs bake an arbitrary, per-joint "bind" rotation into every clip
-- an exporter axis-convention artifact, not real animation. This module
removes it, and separately recovers the true rest-pose bone geometry a raw
T-pose file's position channels carry (see
docs/superpowers/specs/2026-09-05-preproc-package-design.md section 1).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from poseydon.core.anim import Anim
from poseydon.core.kinematics import forward_kinematics
from poseydon.core.rotations import matrix_to_quat, quat_apply, quat_inverse, quat_mul
from poseydon.preproc.raw_bvh import _parse_and_merge
from poseydon.solvers import IK_TERMS, GradientIK, SolverSkeleton


def remove_bind_pose(anim: Anim, rest_anim: Anim) -> Anim:
    """Re-express ``anim`` so zero rotation on every joint reproduces the rest pose.

    Ports the reference's ``compute_rots_from_tpos``, verified term-by-term
    against the real source and the real ``Quaternions.__mul__``/``__neg__``
    (design spec section 1). Matched by joint NAME -- ``rest_anim`` need not
    cover every joint ``anim`` does. A joint missing from ``rest_anim``
    falls back to ``anim``'s own frame-0 rotation as its bind (exact for a
    childless End Site, an approximation elsewhere).
    """
    rest_index = {name: i for i, name in enumerate(rest_anim.names)}

    # `local_bind[j]` is joint j's own LOCAL rest rotation. `bind[j]` is the
    # GLOBAL rest rotation -- needed for the PARENT side of the formula
    # below. Both are built in one pass since parents always precede
    # children.
    local_bind = np.empty((anim.n_joints, 4))
    bind = np.empty((anim.n_joints, 4))
    for joint, name in enumerate(anim.names):
        local_bind[joint] = (
            rest_anim.rotations[0, rest_index[name]] if name in rest_index else anim.rotations[0, joint]
        )
        bind[joint] = (
            local_bind[joint]
            if joint == 0
            else quat_mul(bind[anim.parents[joint]], local_bind[joint])
        )

    new_offsets = anim.offsets.copy()
    for joint, name in enumerate(anim.names):
        if name in rest_index:
            new_offsets[joint] = rest_anim.offsets[rest_index[name]]
    new_rotations = anim.rotations.copy()

    new_rotations[:, 0] = quat_mul(anim.rotations[:, 0], quat_inverse(local_bind[0]))

    for joint in range(1, anim.n_joints):
        parent_bind = bind[anim.parents[joint]]

        # Sandwiched between its own LOCAL bind (undone) and its parent's
        # GLOBAL bind (removed, then reapplied) -- the reference's
        # compute_rots_from_tpos. Offsets are untouched: they stay the
        # skeleton-level constant from the rest pose, reused for every
        # frame and every clip.
        new_rotations[:, joint] = quat_mul(
            quat_mul(quat_mul(parent_bind, anim.rotations[:, joint]), quat_inverse(local_bind[joint])),
            quat_inverse(parent_bind),
        )

    return Anim(
        rotations=new_rotations,
        root_pos=anim.root_pos,
        offsets=new_offsets,
        parents=anim.parents,
        names=anim.names,
        fps=anim.fps,
    )


def recover_raw_global_pose(
    rotations: np.ndarray,
    positions: np.ndarray,
    parents: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Global positions/rotations treating every joint's raw (rotation,
    position) pair as its own local transform.

    This is how a raw Biped export's per-joint position channel has to be
    read to recover a T-pose file's TRUE joint positions: PoseYdon's own
    rigid-bone-length FK (``poseydon.core.kinematics.forward_kinematics``)
    assumes only the root translates, which is exactly the assumption a raw
    T-pose file violates.
    """
    n_frames, n_joints = rotations.shape[:2]
    global_pos = np.empty((n_frames, n_joints, 3))
    global_rot = np.empty((n_frames, n_joints, 4))
    global_pos[:, 0] = positions[:, 0]
    global_rot[:, 0] = rotations[:, 0]
    for joint in range(1, n_joints):
        parent = parents[joint]
        global_pos[:, joint] = global_pos[:, parent] + quat_apply(
            global_rot[:, parent], positions[:, joint]
        )
        global_rot[:, joint] = quat_mul(global_rot[:, parent], rotations[:, joint])
    return global_pos, global_rot


def establish_rest_pose(path: str | Path, iterations: int = 1000) -> Anim:
    """Recover a raw T-pose file's true geometry and fit clean rotations to it.

    Raw per-joint translation genuinely carries geometric information for
    these files (see :func:`recover_raw_global_pose`'s docstring) -- simply
    parsing the declared rotation channels, as
    ``poseydon.preproc.raw_bvh.load_raw_biped_bvh`` does for ordinary clips,
    would use the wrong (arbitrarily rotated) offsets for a REST pose
    specifically. This recovers frame 0's TRUE global positions using both
    raw channels together, then fits PoseYdon's rotation-only,
    rigid-bone-length representation to those positions using
    ``poseydon.solvers.gradient_ik.GradientIK`` with the registered
    "position" ``IKTerm``, holding the raw (structurally merged) offsets
    fixed. Final offsets are the fitted pose's own bone vectors -- same
    LENGTH as the raw declared offset (rotation cannot change that), only
    the DIRECTION is corrected; see this function's use in the
    implementation plan (Task 5) for why that's the real scope of what this
    buys.
    """
    names, parents, offsets, rotations, positions, fps = _parse_and_merge(path)
    global_pos, _ = recover_raw_global_pose(rotations[:1], positions[:1], parents)

    skeleton = SolverSkeleton(
        parents=torch.from_numpy(parents.astype(np.int64)),
        offsets=torch.from_numpy(offsets).float(),
    )
    targets = torch.from_numpy(global_pos).float()
    solver = GradientIK(terms=[(1.0, IK_TERMS.get("position")())], iterations=iterations)
    fitted_matrices = solver.solve(targets, skeleton)
    fitted_rotations = matrix_to_quat(fitted_matrices.numpy())

    fitted_positions, fitted_global_rot = forward_kinematics(
        fitted_rotations, global_pos[:, 0], offsets, parents
    )

    # FK expects an offset expressed in the PARENT's own local frame, not a
    # bare world-space displacement, so this un-rotates by the parent's
    # global rotation rather than subtracting positions directly.
    final_offsets = np.zeros_like(offsets)
    parent_of = parents[1:]
    final_offsets[1:] = quat_apply(
        quat_inverse(fitted_global_rot[0, parent_of]),
        fitted_positions[0, 1:] - fitted_positions[0, parent_of],
    )

    return Anim(
        rotations=fitted_rotations,
        root_pos=global_pos[:, 0],
        offsets=final_offsets,
        parents=parents,
        names=names,
        fps=fps,
    )

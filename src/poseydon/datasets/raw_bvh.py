"""Raw BVH hygiene, shared across dataset preprocessing scripts.

3ds Max Biped rigs export every joint with 6 channels (position + rotation),
not just the root, and wrap the true root in a redundant zero-offset child
joint. Neither quirk is Truebones-specific -- any future dataset sourced
from the same export lineage (Mixamo, other Biped-rigged BVH dumps) hits it
too, which is why this lives here rather than under a Truebones-named
module. See docs/superpowers/specs/2026-09-05-truebones-preprocessing-design.md.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from poseydon.core.anim import Anim
from poseydon.core.rotations import quat_apply, quat_inverse, quat_mul
from poseydon.io.bvh import _channels_to_arrays, _parse_hierarchy, _parse_motion

_ZERO_OFFSET_ATOL = 1e-6


def merge_redundant_root(
    names: tuple[str, ...],
    parents: np.ndarray,
    offsets: np.ndarray,
    rotations: np.ndarray,
    positions: np.ndarray,
) -> tuple[tuple[str, ...], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Merge a zero-offset single child of the root into the root.

    Ports the one branch of the reference's (external/neural_motion_blending
    vendored BVH.py) redundant-root handling that applies to Truebones data:
    the new root's offset is the old root's, its rotation is the old root's
    composed with the old child's (root applied first, matching
    poseydon.core.kinematics.forward_kinematics's own chaining order), and
    its translation is the two old translations summed. The old root is
    dropped entirely, including its name.

    Raises ``ValueError`` if the shape doesn't match -- callers should treat
    that as "nothing to merge", not call this function.
    """
    if parents.size < 2 or np.count_nonzero(parents == 0) != 1:
        raise ValueError(
            "root does not have exactly one child; nothing to merge"
        )
    if not np.allclose(offsets[1], 0.0, atol=_ZERO_OFFSET_ATOL):
        raise ValueError(
            f"joint 1's offset {offsets[1].tolist()} is not within "
            f"{_ZERO_OFFSET_ATOL} of zero; nothing to merge"
        )

    new_offsets = offsets.copy()
    new_offsets[1] = offsets[0]

    new_rotations = rotations.copy()
    new_rotations[:, 1] = quat_mul(rotations[:, 0], rotations[:, 1])

    new_positions = positions.copy()
    new_positions[:, 1] = positions[:, 0] + positions[:, 1]

    new_parents = parents[1:] - 1
    new_names = names[1:]

    return new_names, new_parents, new_offsets[1:], new_rotations[:, 1:], new_positions[:, 1:]


def freeze_non_root_translation(positions: np.ndarray) -> np.ndarray:
    """Discard every non-root joint's animated translation.

    PoseYdon's ``Anim`` has no field to carry per-joint translation (bone
    lengths are fixed; only the root moves), so this drops it rather than
    averaging or sampling it. Returns the root's own trajectory unchanged.
    """
    return positions[:, 0].copy()


def remove_bind_pose(anim: Anim, rest_anim: Anim) -> Anim:
    """Re-express ``anim`` so zero rotation on every joint reproduces the rest pose.

    Raw Biped rigs bake an arbitrary, per-joint "bind" rotation into every
    clip -- e.g. one joint reading a constant ~90 degree rotation in every
    clip of a skeleton, not because it moves 90 degrees, but because that's
    the exporter's zero point for that joint. This removes it, using
    ``rest_anim`` (typically the skeleton's T-pose, or a fallback frame,
    cleaned the same way as ``anim`` via :func:`load_raw_biped_bvh`) as the
    reference for what "zero" should mean.

    Offsets come from ``rest_anim`` unchanged (or ``anim``'s own, for a
    joint missing from the rest reference -- see below; skeleton-level
    offsets are the same file-declared constant either way) and are never
    rotated per clip: this matches the reference's own per-clip step
    (`motion_process.py::compute_rots_from_tpos`, ported here verbatim
    modulo quaternion storage convention) exactly, which reuses one
    canonical offset array for every clip of a skeleton rather than
    re-deriving it per clip.

    ``rest_anim``'s joints need not be every one of ``anim``'s -- matched by
    NAME. Raw T-pose files sometimes omit whole sub-chains that action clips
    still declare (confirmed on real data: Crab's T-pose lacks 10 small
    limb-tip bones its action clips have, each with its own End Site child).
    A joint missing from ``rest_anim`` falls back to ``anim``'s OWN frame-0
    rotation as its bind -- the same idea as this module's skeleton-level
    T-pose fallback (a clip's own first frame standing in for a missing rest
    reference), applied per joint. This is exact for a childless End Site
    (never has rotation channels, so frame 0 is already Identity) and a
    reasonable approximation for anything else missing from the T-pose,
    which in practice has turned out to be a bone that barely moves at all
    (verified: the Crab limb-tip bones read ~1e-7-magnitude rotation, i.e.
    already-identity noise, in every clip checked).
    """
    rest_index = {name: i for i, name in enumerate(rest_anim.names)}

    # `local_bind[j]` is joint j's own LOCAL rest rotation: straight from
    # rest_anim when present, else falls back to anim's own frame-0 value
    # (see docstring). `bind[j]` is the GLOBAL rest rotation -- the usual FK
    # accumulation, parent's global composed with this joint's own local --
    # needed for the PARENT side of the formula below. Both are built in one
    # pass since parents always precede children.
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
        # GLOBAL bind (removed, then reapplied) -- ported verbatim from the
        # reference's compute_rots_from_tpos. Offsets are untouched: they
        # stay the skeleton-level constant from the rest pose, reused for
        # every frame and every clip, exactly as the reference does.
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


def _should_merge(parents: np.ndarray, offsets: np.ndarray) -> bool:
    return (
        parents.size >= 2
        and np.count_nonzero(parents == 0) == 1
        and np.allclose(offsets[1], 0.0, atol=_ZERO_OFFSET_ATOL)
    )


def _parse_and_merge(path: str | Path):
    """Hierarchy/motion parse plus the conditional redundant-root merge.

    Returns ``(names, parents, offsets, rotations, positions, fps)`` with
    ``positions`` still carrying every joint's raw per-frame translation --
    shared by :func:`load_raw_biped_bvh` (which freezes it) and
    :func:`establish_rest_pose` (which needs it intact to recover the true
    T-pose geometry).
    """
    text = Path(path).read_text()
    head, marker, motion = text.partition("MOTION")
    if not marker:
        raise ValueError(f"{path}: no MOTION block found")

    names, parents, offsets, channels = _parse_hierarchy(head)
    n_channels = sum(len(spec) for spec in channels)
    values, frame_time = _parse_motion(motion, n_channels)
    rotations, positions = _channels_to_arrays(values, channels, len(names))

    if _should_merge(parents, offsets):
        names, parents, offsets, rotations, positions = merge_redundant_root(
            names, parents, offsets, rotations, positions
        )

    return names, parents, offsets, rotations, positions, 1.0 / frame_time


def load_raw_biped_bvh(path: str | Path) -> Anim:
    """Read a raw multi-channel Biped BVH export into a valid ``Anim``.

    Merges a redundant zero-offset root when the raw hierarchy has one, then
    freezes every remaining non-root joint's translation to its declared
    offset. Reuses ``poseydon.io.bvh``'s hierarchy/motion grammar parsing --
    the "only the root may translate" restriction in ``load_bvh`` is the
    only thing this skips.
    """
    names, parents, offsets, rotations, positions, fps = _parse_and_merge(path)
    root_pos = freeze_non_root_translation(positions)

    return Anim(
        rotations=rotations,
        root_pos=root_pos,
        offsets=offsets,
        parents=parents,
        names=names,
        fps=fps,
    )


def _damaged_global(rotations: np.ndarray, positions: np.ndarray, parents: np.ndarray):
    """Global positions/rotations treating every joint's raw (rotation,
    position) pair as its own local transform.

    This is how the reference's vendored Animation model reads these raw
    files (``Animation.transforms_local``: the translation slot is the raw
    per-joint position, not just the root's) -- "damaged" relative to
    PoseYdon's own rigid-bone-length model, but it is the only way to
    recover a raw T-pose file's TRUE joint positions, since these files
    really do encode geometry across both channels together.
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


def _kabsch_init(
    offsets: np.ndarray, targets: np.ndarray, parents: np.ndarray
) -> np.ndarray:
    """Per-joint local rotations giving gradient descent a real starting point.

    ``GradientIK`` always starts from Identity for every joint, which turns
    out to be a genuine local minimum for this recovery problem (verified:
    thousands of extra iterations don't move it). This instead solves each
    joint's GLOBAL rotation, top-down, as the best-fit (Kabsch/Wahba) rotation
    aligning that joint's CHILDREN's offset vectors onto their target
    directions simultaneously -- exact for a single child, a genuine
    least-squares compromise for a fork (e.g. a hip with two legs). Gradient
    descent from here reaches the same fixed point as from Identity (verified
    numerically), so this isn't escaping a bad local minimum -- it just gets
    there in far fewer iterations, letting a single confirmed-plateaued
    configuration (see :func:`establish_rest_pose`) run quickly.
    """
    from scipy.spatial.transform import Rotation

    n_joints = offsets.shape[0]
    children: list[list[int]] = [[] for _ in range(n_joints)]
    for joint in range(1, n_joints):
        children[parents[joint]].append(joint)

    global_rot = np.zeros((n_joints, 4))
    global_rot[:, 3] = 1.0
    global_pos = np.zeros((n_joints, 3))
    global_pos[0] = targets[0]

    for parent in range(n_joints):
        kids = [j for j in children[parent] if np.linalg.norm(offsets[j]) > 1e-9]
        if kids:
            src = np.array([offsets[j] for j in kids])
            dst = np.array([targets[j] - global_pos[parent] for j in kids])
            src = src / np.linalg.norm(src, axis=1, keepdims=True)
            dst = dst / np.linalg.norm(dst, axis=1, keepdims=True).clip(1e-9)
            fit, _ = Rotation.align_vectors(dst, src)
            global_rot[parent] = fit.as_quat()
        for j in children[parent]:
            global_pos[j] = global_pos[parent] + quat_apply(global_rot[parent], offsets[j])

    local_rot = global_rot.copy()
    for joint in range(1, n_joints):
        local_rot[joint] = quat_mul(quat_inverse(global_rot[parents[joint]]), global_rot[joint])
    return local_rot


def establish_rest_pose(path: str | Path) -> Anim:
    """Recover a raw T-pose file's true geometry and fit clean rotations to it.

    Ports the reference's ``get_common_features_from_T_pose`` (minus its
    final alignment step, which the caller applies per clip anyway via the
    existing ``poseydon.ingest.align`` pipeline -- alignment normalizes
    scale/facing/ground regardless of what scale this function's offsets
    start at, so duplicating it here would be redundant).

    Raw per-joint translation genuinely carries geometric information for
    these files (see :func:`freeze_non_root_translation`'s docstring) --
    simply parsing the declared rotation channels alone, as
    :func:`load_raw_biped_bvh` does for ordinary clips, would use the wrong
    (arbitrarily rotated) offsets for a REST pose specifically, where
    getting the bone directions right actually matters (this is the offset
    -- not rotation -- half of what makes a "clean" BVH). This recovers the
    TRUE global positions using both channels together (:func:`_damaged_global`),
    then fits this repo's rotation-only, rigid-bone-length representation to
    those positions by gradient descent (from a :func:`_kabsch_init` starting
    point, for speed -- see its docstring), using the raw (structurally
    merged, not yet bind-corrected) offsets as the fixed reference rotated
    around. Final offsets are simply the fitted pose's own bone vectors.

    The fit does not reach zero residual, and cannot: verified on real data
    (BrownBear) that the raw T-pose recording's own declared bone lengths
    are internally inconsistent by construction (``Bip01_R_Thigh`` and
    ``Bip01_L_Thigh`` differ from their own recorded positions by ~1.4 units
    each, in opposite directions) -- the same per-frame noise
    :func:`freeze_non_root_translation` discusses affects supposedly-static
    T-pose recordings too. 5000 iterations at ``learning_rate=0.1`` (chosen
    empirically: error plateaus by ~5000 iterations and does not improve
    with more; a larger rate -- tuned for a plain-Identity start, which is
    far from the answer -- overshoots badly from this already-close Kabsch
    start, so the rate has to match the init, not just the joint count)
    lands at mean/max position error of roughly 1.3%/14% of mean bone
    length for BrownBear's T-pose -- the practical floor given the data,
    not a solver shortfall.
    """
    import torch

    from poseydon.core.kinematics import forward_kinematics as kinematics_forward
    from poseydon.core.rotations import matrix_to_quat, matrix_to_rot6d, quat_to_matrix
    from poseydon.core.torch_kinematics import forward_kinematics, rot6d_to_matrix

    names, parents, offsets, rotations, positions, fps = _parse_and_merge(path)
    global_pos, _ = _damaged_global(rotations[:1], positions[:1], parents)

    init_local = _kabsch_init(offsets, global_pos[0], parents)
    init_matrices = matrix_to_rot6d(quat_to_matrix(init_local))

    offsets_t = torch.from_numpy(offsets)
    parents_t = torch.from_numpy(parents.astype(np.int64))
    targets_t = torch.from_numpy(global_pos)
    root_pos_t = torch.from_numpy(global_pos[:, 0])
    params = torch.from_numpy(init_matrices[None]).clone().requires_grad_(True)
    optimizer = torch.optim.Adam([params], lr=0.1)
    for _ in range(5000):
        optimizer.zero_grad()
        fitted = forward_kinematics(rot6d_to_matrix(params), root_pos_t, offsets_t, parents_t)
        loss = (fitted - targets_t).pow(2).sum(-1).mean()
        loss.backward()
        optimizer.step()

    fitted_rotations = matrix_to_quat(rot6d_to_matrix(params.detach()).numpy())

    # Final offsets are the fitted pose's own bone vectors -- but FK expects
    # an offset expressed in the PARENT's own local frame, not a bare
    # world-space displacement, so this un-rotates by the parent's global
    # rotation rather than subtracting positions directly.
    fitted_positions, fitted_global_rot = kinematics_forward(
        fitted_rotations, global_pos[:, 0], offsets, parents
    )
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

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
from poseydon.core.rotations import quat_inverse, quat_mul
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


def load_raw_biped_bvh(path: str | Path) -> Anim:
    """Read a raw multi-channel Biped BVH export into a valid ``Anim``.

    Merges a redundant zero-offset root when the raw hierarchy has one
    (``_should_merge``), then freezes every remaining non-root joint's
    translation to its declared offset. Reuses ``poseydon.io.bvh``'s
    hierarchy/motion grammar parsing -- the "only the root may translate"
    restriction in ``load_bvh`` is the only thing this skips.
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

    root_pos = freeze_non_root_translation(positions)

    return Anim(
        rotations=rotations,
        root_pos=root_pos,
        offsets=offsets,
        parents=parents,
        names=names,
        fps=1.0 / frame_time,
    )

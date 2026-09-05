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
from poseydon.core.kinematics import forward_kinematics
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

    Global joint POSITIONS are preserved exactly, at every frame -- only the
    rotation/offset convention changes, not the physically observed motion.
    Offsets are recomputed from the rest pose's own global geometry (the
    bone vectors implied by its joint positions), since keeping the raw,
    arbitrarily-oriented offsets while changing what "zero rotation" means
    would visually distort the skeleton's shape.

    Raises ``ValueError`` if ``anim`` and ``rest_anim`` are not the same
    skeleton (same names and parents) -- they must come from cleaning the
    same species' raw files, so their bone lengths necessarily agree too.
    """
    if anim.names != rest_anim.names or not np.array_equal(anim.parents, rest_anim.parents):
        raise ValueError("anim and rest_anim must be the same skeleton")

    rest_positions, rest_global_rot = forward_kinematics(
        rest_anim.rotations[:1], rest_anim.root_pos[:1], rest_anim.offsets, rest_anim.parents
    )
    rest_pos = rest_positions[0]  # (J, 3)
    bind = rest_global_rot[0]  # (J, 4) -- each joint's global rest rotation

    new_offsets = np.zeros_like(anim.offsets)
    parent_of = anim.parents[1:]
    new_offsets[1:] = rest_pos[1:] - rest_pos[parent_of]

    # A joint's raw local rotation, sandwiched between its own bind (undone on
    # the right) and its parent's bind (reapplied on the left) -- the two-sided
    # change of basis that makes the rest frame read as identity while new_offsets
    # (the parent-bind-rotated raw offset) keeps every frame's global positions
    # exactly matching the original animation.
    new_rotations = np.empty_like(anim.rotations)
    new_rotations[:, 0] = quat_mul(anim.rotations[:, 0], quat_inverse(bind[0]))
    new_rotations[:, 1:] = quat_mul(
        bind[parent_of], quat_mul(anim.rotations[:, 1:], quat_inverse(bind[1:]))
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

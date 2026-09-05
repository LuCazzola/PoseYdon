"""Raw Biped-rig BVH hygiene, shared across dataset preprocessing scripts.

3ds Max Biped rigs export every joint with 6 channels (position + rotation),
not just the root, and sometimes wrap the true root in a redundant
zero-offset child joint. Neither quirk is Truebones-specific -- any future
dataset sourced from the same export lineage hits it too, which is why this
lives here rather than under a Truebones-named module. Verified against the
real reference BVH loader; see
docs/superpowers/specs/2026-09-05-preproc-package-design.md section 1.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from poseydon.core.anim import Anim
from poseydon.core.rotations import quat_mul
from poseydon.io.bvh import _POSITION_CHANNELS, _channels_to_arrays, _parse_hierarchy, _parse_motion

_ZERO_OFFSET_ATOL = 1e-6


def merge_redundant_root(
    names: tuple[str, ...],
    parents: np.ndarray,
    offsets: np.ndarray,
    rotations: np.ndarray,
    positions: np.ndarray,
) -> tuple[tuple[str, ...], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Merge a zero-offset single child of the root into the root.

    Matches the one branch of the real reference BVH loader's redundant-root
    handling that occurs in Truebones data (root has exactly one child, that
    child's offset is ~zero): the new root's offset is the old root's, its
    rotation is root-then-child composed (same order
    ``poseydon.core.kinematics.forward_kinematics`` chains parent and
    child), its translation is the two summed. The old root is dropped.

    Raises ``ValueError`` if the shape doesn't match -- callers should treat
    that as "nothing to merge", not call this function.
    """
    if parents.size < 2 or np.count_nonzero(parents == 0) != 1:
        raise ValueError("root does not have exactly one child; nothing to merge")
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
    averaging or sampling it. Returns the root's own (F, 3) trajectory
    unchanged.
    """
    return positions[:, 0].copy()


def _should_merge(parents: np.ndarray, offsets: np.ndarray) -> bool:
    return (
        parents.size >= 2
        and np.count_nonzero(parents == 0) == 1
        and np.allclose(offsets[1], 0.0, atol=_ZERO_OFFSET_ATOL)
    )


def _parse_and_merge(path: str | Path, merge: bool = True):
    """Hierarchy/motion parse plus the conditional redundant-root merge.

    ``merge=False`` skips the redundant-root merge entirely, preserving
    every joint the raw file declares -- for callers that want the
    original Truebones hierarchy kept 1:1 (see
    :func:`load_raw_biped_bvh`'s ``merge`` parameter).

    Returns ``(names, parents, offsets, rotations, positions, fps)`` with
    ``positions`` still carrying every joint's raw per-frame translation --
    shared by :func:`load_raw_biped_bvh` (which freezes it) and
    :mod:`poseydon.preproc.rest_pose` (which needs it intact).
    """
    text = Path(path).read_text()
    head, marker, motion = text.partition("MOTION")
    if not marker:
        raise ValueError(f"{path}: no MOTION block found")

    names, parents, offsets, channels = _parse_hierarchy(head)
    n_channels = sum(len(spec) for spec in channels)
    values, frame_time = _parse_motion(motion, n_channels)
    rotations, positions = _channels_to_arrays(values, channels, len(names))

    # A joint with no position channels at all (every End Site, by BVH
    # convention) has no raw per-frame translation to read -- _channels_to_arrays
    # defaults it to zero, which reads as "coincident with its parent" once
    # interpreted as a local transform (poseydon.preproc.rest_pose.recover_raw_global_pose),
    # even though its offset is not zero. The real reference avoids this by
    # pre-filling every joint's position with its own OFFSET before overwriting
    # from parsed data; this does the same, for the same reason.
    for joint, spec in enumerate(channels):
        if not any(name in spec for name in _POSITION_CHANNELS):
            positions[:, joint] = offsets[joint]

    if merge and _should_merge(parents, offsets):
        names, parents, offsets, rotations, positions = merge_redundant_root(
            names, parents, offsets, rotations, positions
        )

    return names, parents, offsets, rotations, positions, 1.0 / frame_time


def load_raw_biped_bvh(path: str | Path, merge: bool = True) -> Anim:
    """Read a raw multi-channel Biped BVH export into a valid ``Anim``.

    When ``merge`` (the default), merges a redundant zero-offset root when
    the raw hierarchy has one. Pass ``merge=False`` to keep every joint the
    raw file declares, unchanged in count -- for callers that need the
    original Truebones hierarchy preserved 1:1 rather than cleaned up.
    Either way, every remaining non-root joint's translation is frozen to
    its declared offset (that part isn't optional: PoseYdon's ``Anim`` has
    no field to carry per-joint translation at all).
    """
    names, parents, offsets, rotations, positions, fps = _parse_and_merge(path, merge=merge)
    root_pos = freeze_non_root_translation(positions)

    return Anim(
        rotations=rotations,
        root_pos=root_pos,
        offsets=offsets,
        parents=parents,
        names=names,
        fps=fps,
    )

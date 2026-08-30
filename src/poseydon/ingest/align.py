"""Canonical alignment: orientation, centring, scale, ground.

The order below is the reference's ``process_anim`` order and is part of the
contract: rotate, centre XZ, scale, ground. Reordering changes the result.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from poseydon.core.anim import Anim
from poseydon.core.rotations import euler_to_quat, quat_apply, quat_between, quat_mul
from poseydon.core.skeleton import HML_MEAN_BONE_LENGTH, ResolvedSkeleton

_WORLD_UP = np.array([0.0, 1.0, 0.0])
_TARGET_FORWARD = np.array([0.0, 0.0, 1.0])


@dataclass(frozen=True)
class AlignmentParams:
    """Skeleton-level alignment constants.

    These are derived ONCE per skeleton -- from its T-pose where the manifest
    names one -- and reused for every clip of that character. The reference does
    the same, and it matters: computing them per clip would ground a flying
    creature onto the floor and destroy the height relationship between a
    crouch and a stand. Only the facing rotation is genuinely per-clip.
    """

    root_xz: np.ndarray
    scale_factor: float
    ground_height: float


def facing_quat(
    positions: np.ndarray,
    facing_indices: tuple[tuple[int, int], ...],
    extra_yaw_deg: float = 0.0,
) -> np.ndarray:
    """Rotation taking the character's forward axis onto +Z, from frame 0.

    ``across`` is the normalized sum of ``(right - left)`` over the facing pairs;
    ``forward = cross(world_up, across)``.
    """
    first = positions[0]
    across = np.zeros(3)
    for right, left in facing_indices:
        across = across + (first[right] - first[left])

    norm = np.linalg.norm(across)
    if norm < 1e-8:
        raise ValueError(
            "facing joints are coincident in the first frame, so the forward "
            "direction is undefined; check the manifest's facing pairs"
        )
    across = across / norm

    forward = np.cross(_WORLD_UP, across)
    forward_norm = np.linalg.norm(forward)
    if forward_norm < 1e-8:
        raise ValueError(
            "the across-body axis is parallel to world up, so the forward "
            "direction is undefined; check the manifest's facing pairs"
        )
    forward = forward / forward_norm

    rotation = quat_between(forward, _TARGET_FORWARD)
    if extra_yaw_deg:
        yaw = euler_to_quat(np.array([0.0, extra_yaw_deg, 0.0]), "ZYX")
        rotation = quat_mul(yaw, rotation)
    return rotation


def rotate_to_face_z(anim: Anim, rotation: np.ndarray) -> Anim:
    """Apply a global rotation by composing it onto the root only."""
    rotations = anim.rotations.copy()
    rotations[:, 0] = quat_mul(np.broadcast_to(rotation, rotations[:, 0].shape), rotations[:, 0])
    return Anim(
        rotations=rotations,
        root_pos=quat_apply(rotation, anim.root_pos),
        offsets=anim.offsets,
        parents=anim.parents,
        names=anim.names,
        fps=anim.fps,
    )


def move_xz_to_origin(anim: Anim) -> tuple[Anim, np.ndarray]:
    """Translate so the root sits at XZ origin on the first frame."""
    root_xz = anim.root_pos[0] * np.array([1.0, 0.0, 1.0])
    return (
        Anim(
            rotations=anim.rotations,
            root_pos=anim.root_pos - root_xz,
            offsets=anim.offsets,
            parents=anim.parents,
            names=anim.names,
            fps=anim.fps,
        ),
        root_xz,
    )


def scale_to_mean_bone_length(anim: Anim, target: float) -> tuple[Anim, float]:
    """Uniformly scale so the MEAN bone length equals ``target``."""
    lengths = np.linalg.norm(anim.offsets[1:], axis=-1)
    mean_length = float(lengths.mean())
    if mean_length < 1e-12:
        raise ValueError("skeleton has zero mean bone length; cannot scale")
    factor = target / mean_length
    return (
        Anim(
            rotations=anim.rotations,
            root_pos=anim.root_pos * factor,
            offsets=anim.offsets * factor,
            parents=anim.parents,
            names=anim.names,
            fps=anim.fps,
        ),
        factor,
    )


def put_on_ground(anim: Anim) -> tuple[Anim, float]:
    """Translate in Y so the lowest joint over the whole clip sits at y = 0."""
    ground_height = float(anim.global_positions()[..., 1].min())
    shift = np.array([0.0, ground_height, 0.0])
    return (
        Anim(
            rotations=anim.rotations,
            root_pos=anim.root_pos - shift,
            offsets=anim.offsets,
            parents=anim.parents,
            names=anim.names,
            fps=anim.fps,
        ),
        ground_height,
    )


def compute_alignment_params(
    reference: Anim,
    resolved: ResolvedSkeleton,
    target_bone_length: float = HML_MEAN_BONE_LENGTH,
) -> AlignmentParams:
    """Derive the skeleton-level constants from a reference clip (the T-pose)."""
    rotation = facing_quat(
        reference.global_positions(),
        resolved.facing_indices,
        resolved.manifest.extra_yaw_deg,
    )
    rotated = rotate_to_face_z(reference, rotation)
    _centred, root_xz = move_xz_to_origin(rotated)
    scaled, factor = scale_to_mean_bone_length(rotated, target_bone_length)
    _grounded, ground_height = put_on_ground(
        Anim(
            rotations=scaled.rotations,
            root_pos=scaled.root_pos - root_xz * factor,
            offsets=scaled.offsets,
            parents=scaled.parents,
            names=scaled.names,
            fps=scaled.fps,
        )
    )
    return AlignmentParams(
        root_xz=root_xz, scale_factor=factor, ground_height=ground_height
    )


def align(
    anim: Anim,
    resolved: ResolvedSkeleton,
    params: AlignmentParams,
) -> Anim:
    """Rotate to face +Z, then apply the skeleton-level constants.

    Order is contractual: rotate, centre XZ, scale, ground. The rotation is
    recomputed for this clip; everything else comes from ``params``.
    """
    rotation = facing_quat(
        anim.global_positions(),
        resolved.facing_indices,
        resolved.manifest.extra_yaw_deg,
    )
    out = rotate_to_face_z(anim, rotation)
    out = Anim(
        rotations=out.rotations,
        root_pos=(out.root_pos - params.root_xz) * params.scale_factor
        - np.array([0.0, params.ground_height, 0.0]),
        offsets=out.offsets * params.scale_factor,
        parents=out.parents,
        names=out.names,
        fps=out.fps,
    )
    return out

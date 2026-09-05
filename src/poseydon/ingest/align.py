"""Canonical alignment: orientation, centring, scale, ground.

The order below is the reference's ``process_anim`` order and is part of the
contract: rotate, centre XZ, scale, ground. Reordering changes the result.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from poseydon.core.animation import Animation, RigidBodyAnimation
from poseydon.core.rotations import (
    euler_to_quat,
    quat_apply,
    quat_between,
    quat_inverse,
    quat_mul,
)
from poseydon.core.skeleton import HML_MEAN_BONE_LENGTH, ResolvedSkeleton

_WORLD_UP = np.array([0.0, 1.0, 0.0])
_TARGET_FORWARD = np.array([0.0, 0.0, 1.0])
_AXES = {
    "X": np.array([1.0, 0.0, 0.0]),
    "Y": np.array([0.0, 1.0, 0.0]),
    "Z": np.array([0.0, 0.0, 1.0]),
}


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


def facing_quats(
    positions: np.ndarray,
    facing_indices: tuple[tuple[int, int], ...],
    extra_yaw_deg: float = 0.0,
    target: np.ndarray = _TARGET_FORWARD,
) -> np.ndarray:
    """Per-frame rotation taking the character's forward axis onto ``target``.

    ``across`` is the normalized sum of ``(right - left)`` over the facing pairs;
    ``forward = cross(world_up, across)``. Returns ``(F, 4)``.

    ``target`` defaults to +Z, the canonical forward for this codebase; it is
    a parameter only so :func:`rotate_to_face_axis` can aim at any signed axis
    without restating what "forward" means.

    Alignment uses only frame 0 (see :func:`facing_quat`); the feature layer
    needs every frame, because the root-invariant position and local-velocity
    blocks are expressed in each frame's own root frame.
    """
    positions = np.asarray(positions, dtype=np.float64)
    across = np.zeros_like(positions[:, 0])
    for right, left in facing_indices:
        across = across + (positions[:, right] - positions[:, left])

    norms = np.linalg.norm(across, axis=-1, keepdims=True)
    if np.any(norms < 1e-8):
        raise ValueError(
            "facing joints are coincident in at least one frame, so the forward "
            "direction is undefined; check the manifest's facing pairs"
        )
    across = across / norms

    forward = np.cross(np.broadcast_to(_WORLD_UP, across.shape), across)
    forward_norms = np.linalg.norm(forward, axis=-1, keepdims=True)
    if np.any(forward_norms < 1e-8):
        raise ValueError(
            "the across-body axis is parallel to world up in at least one frame, "
            "so the forward direction is undefined; check the manifest's facing pairs"
        )
    forward = forward / forward_norms

    target = np.asarray(target, dtype=np.float64)
    rotations = quat_between(forward, np.broadcast_to(target, forward.shape))
    if extra_yaw_deg:
        yaw = euler_to_quat(np.array([0.0, extra_yaw_deg, 0.0]), "ZYX")
        rotations = quat_mul(np.broadcast_to(yaw, rotations.shape), rotations)
    return rotations


def facing_quat(
    positions: np.ndarray,
    facing_indices: tuple[tuple[int, int], ...],
    extra_yaw_deg: float = 0.0,
) -> np.ndarray:
    """Facing rotation from frame 0 only, as ``(4,)``. Used by alignment."""
    return facing_quats(positions[:1], facing_indices, extra_yaw_deg)[0]


def rotate_to_face_axis(
    anim: Animation,
    facing_indices: tuple[tuple[int, int], ...],
    axis: str = "+Z",
    extra_yaw_deg: float = 0.0,
) -> Animation:
    """Turn a whole rig so its frame-0 facing direction points along ``axis``.

    One call does the lot: derive the character's forward direction from
    ``facing_indices``, build the rotation onto ``axis``, and apply it to the
    animation. ``axis`` is a signed axis name -- ``"+Z"``, ``"Z"``, ``"-X"``,
    ``"+y"`` -- case-insensitive, with a bare name meaning positive.

    Unlike :func:`rotate_to_face_z`, which composes the rotation onto the
    root's rotation channel and leaves ``offsets`` untouched, this turns the
    SKELETON as well as the motion:

    * ``offsets' = R . offsets`` -- the rest geometry points at ``axis`` too.
      The root's own offset turns with the rest, so a reader that adds it to
      the position channels (Blender does) stays consistent.
    * ``rotations' = R . rotations . R^-1`` -- conjugated, not merely
      pre-multiplied, because every joint's local frame has itself turned.
    * ``root_pos' = R . root_pos``.

    Both forms give identical world-space motion: telescoping the conjugation
    gives ``global_rot'[j] = R . global_rot[j] . R^-1``, and
    ``global_rot'[parent]`` applied to ``R . offset[j]`` is ``R`` applied to
    ``global_rot[parent] . offset[j]``, so every global position is just the
    old one turned by ``R``. They differ in where the rotation is STORED, and
    that matters the moment an animation is written back out as a BVH: under
    the root-only form the ``OFFSET`` block still describes the ORIGINAL
    orientation, so a DCC tool draws the rest/bind skeleton (Blender's Edit
    Mode, i.e. Tab) facing one way and the animation facing another. Reach
    for this function when the rig itself must be reoriented, and for
    :func:`rotate_to_face_z` when only the feature-level pose should turn.

    It also preserves a T-pose-relative rotation representation: wherever a
    local rotation is identity, ``R . identity . R^-1`` is identity still --
    with the rest pose those offsets describe now pointing at ``axis``.

    Facing comes from frame 0 alone, matching :func:`facing_quat`, and the
    single resulting rotation is applied to every frame.

    ``"+Y"``/``"-Y"`` are accepted and behave correctly -- forward is derived
    as ``cross(world_up, across)`` and so is horizontal, and aiming it at
    world up simply tips the whole rig onto its face or back. That is rarely
    what a caller wants, but it is not an error, so it is not rejected.
    """
    spec = str(axis).strip().upper()
    name = spec[1:] if spec[:1] in "+-" else spec
    if name not in _AXES:
        raise ValueError(
            f"target axis must be X, Y or Z with an optional sign, got `{axis}`"
        )
    target = (-1.0 if spec.startswith("-") else 1.0) * _AXES[name]

    rotation = facing_quats(
        anim.global_positions()[:1], facing_indices, extra_yaw_deg, target
    )[0]
    inverse = quat_inverse(rotation)

    # Every local translation turns with the frame it is expressed in, the
    # root's included (its entry IS the root's global position). Rebuilding
    # from a root trajectory instead would pin each joint to its rest offset
    # and throw away any real per-joint translation, so this carries the
    # translations through -- and a rigid animation stays rigid, since
    # rotating `translations` and `offsets` by the same R preserves their
    # equality. `type(anim)` keeps a RigidBodyAnimation rigid on the way out.
    return type(anim)(
        rotations=quat_mul(quat_mul(rotation, anim.rotations), inverse),
        translations=quat_apply(rotation, anim.translations),
        offsets=quat_apply(rotation, anim.offsets),
        parents=anim.parents,
        names=anim.names,
        fps=anim.fps,
    )


def rotate_to_face_z(anim: RigidBodyAnimation, rotation: np.ndarray) -> RigidBodyAnimation:
    """Apply a global rotation by composing it onto the root only.

    Leaves ``offsets`` -- the rest geometry -- in their original orientation;
    see :func:`rotate_to_face_axis` for the whole-rig form and when each is
    the right one.
    """
    rotations = anim.rotations.copy()
    rotations[:, 0] = quat_mul(np.broadcast_to(rotation, rotations[:, 0].shape), rotations[:, 0])
    return RigidBodyAnimation.from_root_motion(
        rotations=rotations,
        root_pos=quat_apply(rotation, anim.root_pos),
        offsets=anim.offsets,
        parents=anim.parents,
        names=anim.names,
        fps=anim.fps,
    )


def move_xz_to_origin(anim: RigidBodyAnimation) -> tuple[RigidBodyAnimation, np.ndarray]:
    """Translate so the root sits at XZ origin on the first frame."""
    root_xz = anim.root_pos[0] * np.array([1.0, 0.0, 1.0])
    return (
        RigidBodyAnimation.from_root_motion(
            rotations=anim.rotations,
            root_pos=anim.root_pos - root_xz,
            offsets=anim.offsets,
            parents=anim.parents,
            names=anim.names,
            fps=anim.fps,
        ),
        root_xz,
    )


def scale_to_mean_bone_length(anim: RigidBodyAnimation, target: float) -> tuple[RigidBodyAnimation, float]:
    """Uniformly scale so the MEAN bone length equals ``target``."""
    lengths = np.linalg.norm(anim.offsets[1:], axis=-1)
    mean_length = float(lengths.mean())
    if mean_length < 1e-12:
        raise ValueError("skeleton has zero mean bone length; cannot scale")
    factor = target / mean_length
    return (
        RigidBodyAnimation.from_root_motion(
            rotations=anim.rotations,
            root_pos=anim.root_pos * factor,
            offsets=anim.offsets * factor,
            parents=anim.parents,
            names=anim.names,
            fps=anim.fps,
        ),
        factor,
    )


def put_on_ground(anim: RigidBodyAnimation) -> tuple[RigidBodyAnimation, float]:
    """Translate in Y so the lowest joint over the whole clip sits at y = 0."""
    ground_height = float(anim.global_positions()[..., 1].min())
    shift = np.array([0.0, ground_height, 0.0])
    return (
        RigidBodyAnimation.from_root_motion(
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
    reference: RigidBodyAnimation,
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
        RigidBodyAnimation.from_root_motion(
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
    anim: RigidBodyAnimation,
    resolved: ResolvedSkeleton,
    params: AlignmentParams,
) -> RigidBodyAnimation:
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
    out = RigidBodyAnimation.from_root_motion(
        rotations=out.rotations,
        root_pos=(out.root_pos - params.root_xz) * params.scale_factor
        - np.array([0.0, params.ground_height, 0.0]),
        offsets=out.offsets * params.scale_factor,
        parents=out.parents,
        names=out.names,
        fps=out.fps,
    )
    return out

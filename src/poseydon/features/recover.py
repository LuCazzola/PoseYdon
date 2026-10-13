"""Features back to an animation.

Recovery is exact, not fitted. The rotations are present in the representation,
so there is no need to solve for them from positions -- which is what the
reference does at export time, running 150 iterations of Jacobian IK per clip and
discarding the rotations it already had.

An IK solver is still useful for the case that genuinely needs one: retargeting
or editing that produces positions without matching rotations.
"""

from __future__ import annotations

import numpy as np

from poseydon.core.animation import RigidBodyAnimation
from poseydon.core.rotations import (
    QUAT_IDENTITY,
    matrix_to_quat,
    quat_apply,
    quat_inverse,
    quat_mul,
    rot6d_to_matrix,
)
from poseydon.core.spec import FeatureSpec

_REFERENCE_6D_LAYOUT = "columns"


class RecoveryError(Exception):
    """Raised when a representation cannot be turned back into an animation."""


def _require(spec: FeatureSpec, *names: str) -> None:
    missing = [name for name in names if name not in spec]
    if missing:
        raise RecoveryError(
            f"cannot rebuild an animation without {', '.join(missing)}; "
            f"this representation has {', '.join(spec.names) or '(none)'}"
        )


def heading(features: np.ndarray, spec: FeatureSpec) -> np.ndarray:
    """The per-frame heading ``(F, 4)`` the features are expressed in.

    From the `heading` block if present, else from `rot6d_bone`'s root slot,
    which is where that layout keeps it.
    """
    if "heading" in spec:
        cos_sin = spec.take(features[:, 0], "heading")
        half = 0.5 * np.arctan2(cos_sin[:, 1], cos_sin[:, 0])
        out = np.zeros((features.shape[0], 4))
        out[:, 1], out[:, 3] = np.sin(half), np.cos(half)
        return out
    if "rot6d_bone" in spec:
        return matrix_to_quat(
            rot6d_to_matrix(spec.take(features[:, 0], "rot6d_bone"), layout=_REFERENCE_6D_LAYOUT)
        )
    raise RecoveryError(
        "cannot place the clip without its heading: add the `heading` block "
        f"(or use `rot6d_bone`); this representation has {', '.join(spec.names)}"
    )


def local_rotations(features: np.ndarray, spec: FeatureSpec, parents: np.ndarray) -> np.ndarray:
    """Every joint's ``(F, J, 4)`` local rotation, from either layout.

    `rot6d_joint` stores them as they are, the root's in the heading frame.
    `rot6d_bone` stores each joint's PARENT's rotation, so a joint's own is
    read from any of its children; a leaf appears in no slot and comes back
    as the identity.
    """
    name = spec.rotation_block()
    quats = matrix_to_quat(
        rot6d_to_matrix(spec.take(features, name), layout=_REFERENCE_6D_LAYOUT)
    )
    if name == "rot6d_joint":
        quats[:, 0] = quat_mul(quat_inverse(heading(features, spec)), quats[:, 0])
        return quats

    n_frames, n_joints = quats.shape[:2]
    out = np.broadcast_to(QUAT_IDENTITY, (n_frames, n_joints, 4)).copy()
    for joint in range(1, n_joints):
        out[:, parents[joint]] = quats[:, joint]
    return out


def root_trajectory(
    features: np.ndarray, spec: FeatureSpec, facing: np.ndarray
) -> np.ndarray:
    """Integrate root displacement back into a world-space ``(F, 3)`` path.

    The representation is deliberately root-invariant: absolute horizontal
    position is removed so the same motion reads identically wherever it
    happens. That information is therefore NOT recoverable -- the trajectory
    comes back starting at the XZ origin, and only its shape is meaningful.
    Height is stored per frame and does come back exactly.
    """
    _require(spec, "ric_pos", "local_vel")
    height = spec.take(features[:, 0], "ric_pos")[:, 1]
    velocity = spec.take(features[:, 0], "local_vel")

    n_frames = features.shape[0]
    positions = np.zeros((n_frames, 3), dtype=np.float64)
    positions[0, 1] = height[0]

    for frame in range(1, n_frames):
        # The velocity block rotates each displacement by the facing of the frame
        # it arrives at, not the one it leaves, so the inverse must use the same
        # one. Getting this backwards costs three orders of magnitude of accuracy.
        inverse = facing[frame] * np.array([-1.0, -1.0, -1.0, 1.0])
        step = quat_apply(inverse, velocity[frame - 1])
        positions[frame] = positions[frame - 1] + step
        positions[frame, 1] = height[frame]

    return positions


#: A predicted contact channel is a continuous value; this is where it becomes
#: a flag. The channel is trained against {0, 1} targets, so the midpoint is the
#: natural cut and matches what `losses.footskate` reads.
CONTACT_THRESHOLD = 0.5


def contact_flags_from(
    features: np.ndarray, spec: FeatureSpec, threshold: float = CONTACT_THRESHOLD
) -> np.ndarray | None:
    """The model's own foot-contact flags, ``(frames, joints)`` boolean.

    ``None`` when the feature set carries no ``foot_contact`` block, so a caller
    can pass the result straight through without first asking whether the run
    declared it -- a recipe may legitimately omit the block.

    This is the model's CLAIM about which feet are down, not contact re-derived
    from the output geometry (`training.metrics.contact_flags` does that). The
    difference matters wherever the two are compared: measuring the claim
    against a re-derivation of the same positions grades the output against
    itself.
    """
    if "foot_contact" not in {name for name, _ in spec.blocks}:
        return None
    return spec.take(np.asarray(features), "foot_contact", drop=True) > threshold


def positions_from_features(
    features: np.ndarray, spec: FeatureSpec
) -> np.ndarray:
    """Global joint positions taken from the POSITION channels.

    The representation is redundant: it carries both rotations and positions,
    and a model can predict the two inconsistently. This path reads the
    positions; :func:`features_to_anim` reads the rotations. Comparing them
    measures how self-consistent a generated clip actually is.

    The reference renders from this path and calls the rotation path "less
    visually appealing" -- but it needs 150 iterations of IK to get back to a
    BVH, because a BVH stores rotations.
    """
    _require(spec, "ric_pos", "local_vel")
    ric = spec.take(features, "ric_pos")

    facing = heading(features, spec)
    root = root_trajectory(features, spec, facing)

    # Undo the per-frame de-rotation, then put the trajectory back underneath.
    inverse = facing * np.array([-1.0, -1.0, -1.0, 1.0])
    positions = quat_apply(inverse[:, None, :], ric)
    positions[..., 0] += root[:, None, 0]
    positions[..., 2] += root[:, None, 2]
    positions[:, 0] = root
    return positions


def features_to_anim(
    features: np.ndarray, spec: FeatureSpec, template: RigidBodyAnimation
) -> RigidBodyAnimation:
    """Rebuild an :class:`RigidBodyAnimation` from a feature tensor.

    ``template`` supplies what the representation deliberately does not carry:
    the skeleton itself -- parents, offsets, joint names and frame rate.
    """
    if features.ndim != 3:
        raise RecoveryError(f"expected (frames, joints, dim), got {features.shape}")
    if features.shape[1] != template.n_joints:
        raise RecoveryError(
            f"features describe {features.shape[1]} joints but the skeleton has "
            f"{template.n_joints}"
        )
    if features.shape[-1] != spec.dim:
        raise RecoveryError(f"features are {features.shape[-1]} wide, spec says {spec.dim}")

    rotations = local_rotations(features, spec, template.parents)

    root_pos = root_trajectory(features, spec, heading(features, spec))

    return RigidBodyAnimation.from_root_motion(
        rotations=rotations,
        root_pos=root_pos,
        offsets=template.offsets,
        parents=template.parents,
        names=template.names,
        fps=template.fps,
    )

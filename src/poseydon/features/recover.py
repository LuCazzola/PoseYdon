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
from poseydon.core.rotations import QUAT_IDENTITY, matrix_to_quat, quat_apply, rot6d_to_matrix
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


def local_rotations(features: np.ndarray, spec: FeatureSpec, parents: np.ndarray) -> np.ndarray:
    """Undo the parent re-slotting, returning ``(F, J, 4)`` local rotations.

    The rot6d block stores each joint's PARENT's rotation, so a joint's own
    rotation is read from any of its children. Leaves appear in no slot, which
    is harmless: they are End Sites and carry no rotation.
    """
    _require(spec, "rot6d")
    block = features[..., spec.slice("rot6d")]
    matrices = rot6d_to_matrix(block, layout=_REFERENCE_6D_LAYOUT)
    quats = matrix_to_quat(matrices)

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
    height = features[:, 0, spec.slice("ric_pos")][:, 1]
    velocity = features[:, 0, spec.slice("local_vel")]

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
    return np.asarray(features)[..., spec.slice("foot_contact")][..., 0] > threshold


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
    _require(spec, "ric_pos", "local_vel", "rot6d")
    ric = features[..., spec.slice("ric_pos")]

    facing = matrix_to_quat(
        rot6d_to_matrix(features[:, 0, spec.slice("rot6d")], layout=_REFERENCE_6D_LAYOUT)
    )
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

    facing = matrix_to_quat(
        rot6d_to_matrix(features[:, 0, spec.slice("rot6d")], layout=_REFERENCE_6D_LAYOUT)
    )
    root_pos = root_trajectory(features, spec, facing)

    return RigidBodyAnimation.from_root_motion(
        rotations=rotations,
        root_pos=root_pos,
        offsets=template.offsets,
        parents=template.parents,
        names=template.names,
        fps=template.fps,
    )

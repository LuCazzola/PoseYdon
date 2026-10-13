"""Position, rotation and velocity features."""

from __future__ import annotations

import numpy as np

from poseydon.core.rotations import matrix_to_rot6d, quat_apply, quat_mul, quat_to_matrix
from poseydon.features.base import FEATURES, Feature, FeatureContext

# Holden's Quaternions.rotation_matrix(cont6d=True) takes the first two COLUMNS,
# the transpose of PyTorch3D's row convention. The reference's stored features
# use the column layout; using rows here is off by up to 2.0 against them.
_REFERENCE_6D_LAYOUT = "columns"


@FEATURES.register("ric_pos")
class RootInvariantPositions(Feature):
    """Joint positions in the root's frame: XZ removed, then de-rotated.

    Every frame is expressed as if the character stood at the XZ origin facing
    +Z, so the representation is invariant to where and which way it walked.
    """

    name = "ric_pos"
    width = 3

    def __call__(self, ctx: FeatureContext) -> np.ndarray:
        positions = ctx.positions.copy()
        positions[..., 0] -= ctx.positions[:, 0:1, 0]
        positions[..., 2] -= ctx.positions[:, 0:1, 2]
        return quat_apply(ctx.root_quats[:, None, :], positions)


@FEATURES.register("rot6d_bone")
class BoneRotation6D(Feature):
    """Per-bone rotation as a 6D vector, at the bone's child: the HumanML3D layout.

    Slot ``j`` holds ``parent(j)``'s local rotation, the one orienting the
    bone that ends at ``j``, so each slot pairs a bone's orientation with its
    endpoint's position. Slot 0 holds the root's heading. A leaf's own rotation
    is in no slot: harmless while every leaf is a zero-length End Site, lost
    when the reduction drops End Sites and their parents become leaves (an
    ear, a tail tip). Kept for parity with the reference; `rot6d_joint` is the
    default.
    """

    name = "rot6d_bone"
    width = 6

    def __call__(self, ctx: FeatureContext) -> np.ndarray:
        joint_6d = matrix_to_rot6d(
            quat_to_matrix(ctx.anim.rotations), layout=_REFERENCE_6D_LAYOUT
        )
        out = np.empty_like(joint_6d)
        out[:, 1:] = joint_6d[:, ctx.anim.parents[1:]]
        out[:, 0] = matrix_to_rot6d(
            quat_to_matrix(ctx.root_quats), layout=_REFERENCE_6D_LAYOUT
        )
        return out


@FEATURES.register("rot6d_joint")
class JointRotation6D(Feature):
    """Each joint's own local rotation as a 6D vector.

    Every joint keeps its rotation, leaves included, so nothing is lost when
    the reduction makes a joint a leaf. The root's rotation is expressed in
    the heading frame (as `ric_pos` and `local_vel` are), so the block is
    invariant to which way the character faces; the heading itself is the
    `heading` block, which recovery needs alongside this one.
    """

    name = "rot6d_joint"
    width = 6

    def __call__(self, ctx: FeatureContext) -> np.ndarray:
        rotations = ctx.anim.rotations.copy()
        rotations[:, 0] = quat_mul(ctx.root_quats, rotations[:, 0])
        return matrix_to_rot6d(quat_to_matrix(rotations), layout=_REFERENCE_6D_LAYOUT)


@FEATURES.register("heading")
class Heading(Feature):
    """The per-frame heading, ``(cos, sin)`` of its yaw, on the root; zero elsewhere.

    The heading is the rotation `ric_pos`, `local_vel` and `rot6d_joint` are
    expressed in (`FeatureContext.root_quats`, a rotation about +Y). An angle
    as (cos, sin) has no wrap-around for a model to learn across.
    """

    name = "heading"
    width = 2

    def __call__(self, ctx: FeatureContext) -> np.ndarray:
        q = ctx.root_quats
        yaw = 2.0 * np.arctan2(q[:, 1], q[:, 3])
        out = np.zeros((q.shape[0], ctx.n_joints, 2))
        out[:, 0, 0] = np.cos(yaw)
        out[:, 0, 1] = np.sin(yaw)
        return out


@FEATURES.register("local_vel")
class LocalVelocity(Feature):
    """Per-joint displacement between frames, in the root's frame.

    One frame shorter than the animation.
    """

    name = "local_vel"
    width = 3

    def __call__(self, ctx: FeatureContext) -> np.ndarray:
        delta = ctx.positions[1:] - ctx.positions[:-1]
        return quat_apply(ctx.root_quats[1:, None, :], delta)

"""Position, rotation and velocity features."""

from __future__ import annotations

import numpy as np

from poseydon.core.rotations import matrix_to_rot6d, quat_apply, quat_to_matrix
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


@FEATURES.register("rot6d")
class Rotation6D(Feature):
    """Per-joint rotation as a 6D vector, re-slotted so each joint owns its own.

    BVH stores the rotation that moves a joint's CHILDREN, so joint ``j`` holds
    what is conceptually its parent's rotation. This block reorders that -- slot
    ``j`` takes ``parent(j)``'s rotation -- and puts the root's facing rotation
    in slot 0, matching HumanML3D-style layouts and the FK model.
    """

    name = "rot6d"
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

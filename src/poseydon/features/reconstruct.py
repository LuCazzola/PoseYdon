"""How to turn generated features back into a skeleton.

The 13-dim representation is redundant: it carries rotations AND positions, and
a generated clip need not keep them consistent. On real data they agree to 1e-9;
on model output they can differ by more than three bone lengths. So "the
skeleton" is not one thing, and which reconstruction you pick is a real choice:

``fk``            forward kinematics from the rotation channels. Rigid by
                  construction, exact, needs no solver, and is the only path
                  that costs nothing.
``positions``     the position channels as they are. Tracks the intended pose
                  most closely -- it IS the prediction -- but does not respect
                  bone lengths, so limbs stretch. Cannot produce a BVH, because
                  a BVH stores rotations.
``positions_ik``  the position channels projected back onto a rigid skeleton by
                  inverse kinematics. Follows the prediction closely while
                  staying rigid, at the cost of a solve. This is what the
                  reference writes to BVH.

The solve is approximate by nature, not by implementation: rotation about a
bone's own axis does not move its children, so twist is invisible to a solver
fitting joint positions. It converges to a plateau -- on the Truebones clips the
mean error settles near 1% of a bone length with the worst joint around 15% --
rather than to zero.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

import numpy as np

from poseydon.core.animation import RigidBodyAnimation
from poseydon.core.registry import Registry
from poseydon.core.spec import FeatureSpec
from poseydon.features.recover import features_to_anim, positions_from_features


class Reconstructor(ABC):
    """Features plus a skeleton to joint positions, and where possible an RigidBodyAnimation."""

    name: ClassVar[str]
    #: Whether this path can produce rotations, and therefore a BVH.
    produces_rotations: ClassVar[bool] = True

    @abstractmethod
    def positions(self, features: np.ndarray, spec: FeatureSpec, template: RigidBodyAnimation) -> np.ndarray:
        """``(F, J, 3)`` global joint positions."""

    def anim(self, features: np.ndarray, spec: FeatureSpec, template: RigidBodyAnimation) -> RigidBodyAnimation:
        raise NotImplementedError(
            f"`{self.name}` does not produce rotations, so it cannot be written as BVH. "
            "Use `fk` or `positions_ik` if you need one."
        )


RECONSTRUCTORS: Registry[Reconstructor] = Registry("reconstructor")


@RECONSTRUCTORS.register("fk")
class ForwardKinematics(Reconstructor):
    """Rotations only. Rigid, exact, free."""

    name = "fk"

    def positions(self, features, spec, template) -> np.ndarray:
        return self.anim(features, spec, template).global_positions()

    def anim(self, features, spec, template) -> RigidBodyAnimation:
        return features_to_anim(features, spec, template)


@RECONSTRUCTORS.register("positions")
class RawPositions(Reconstructor):
    """Predicted positions, unmodified. Bone lengths are not enforced."""

    name = "positions"
    produces_rotations = False

    def positions(self, features, spec, template) -> np.ndarray:
        return positions_from_features(features, spec)


@RECONSTRUCTORS.register("positions_ik")
class SolvedPositions(Reconstructor):
    """Predicted positions, fitted back onto a rigid skeleton."""

    name = "positions_ik"

    def __init__(
        self,
        iterations: int = 150,
        learning_rate: float = 0.1,
        smoothness: float = 0.05,
    ) -> None:
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.smoothness = smoothness

    def _solve(self, features, spec, template):
        import torch

        from poseydon.core.rotations import matrix_to_quat
        from poseydon.core.torch_kinematics import forward_kinematics
        from poseydon.solvers import IK_TERMS, GradientIK, SolverSkeleton

        targets = torch.tensor(positions_from_features(features, spec), dtype=torch.float32)
        skeleton = SolverSkeleton(
            parents=torch.from_numpy(template.parents.astype("int64")),
            offsets=torch.from_numpy(template.offsets).float(),
        )
        terms = [(1.0, IK_TERMS.get("position")())]
        if self.smoothness > 0:
            terms.append((self.smoothness, IK_TERMS.get("smoothness")()))

        rotations = GradientIK(
            terms=terms, iterations=self.iterations, learning_rate=self.learning_rate
        ).solve(targets, skeleton)
        solved = forward_kinematics(
            rotations, targets[:, 0], skeleton.offsets, skeleton.parents
        )
        return matrix_to_quat(rotations.numpy()), solved.numpy(), targets[:, 0].numpy()

    def positions(self, features, spec, template) -> np.ndarray:
        return self._solve(features, spec, template)[1]

    def anim(self, features, spec, template) -> RigidBodyAnimation:
        quaternions, _, root = self._solve(features, spec, template)
        return RigidBodyAnimation.from_root_motion(
            rotations=quaternions,
            root_pos=root.astype(np.float64),
            offsets=template.offsets,
            parents=template.parents,
            names=template.names,
            fps=template.fps,
        )


def reconstruct(
    name: str, features: np.ndarray, spec: FeatureSpec, template: RigidBodyAnimation, **kwargs
) -> tuple[np.ndarray, RigidBodyAnimation | None]:
    """Reconstruct by name, returning positions and an RigidBodyAnimation where one exists."""
    reconstructor = RECONSTRUCTORS.get(name)(**kwargs)
    positions = reconstructor.positions(features, spec, template)
    anim = reconstructor.anim(features, spec, template) if reconstructor.produces_rotations else None
    return positions, anim

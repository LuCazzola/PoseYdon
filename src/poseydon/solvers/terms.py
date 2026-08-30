"""Objective terms for the IK solve."""

from __future__ import annotations

import math

import torch

from poseydon.solvers.base import IK_TERMS, IKTerm


@IK_TERMS.register("position")
class PositionTerm(IKTerm):
    """Match the target joint positions. The basic objective."""

    name = "position"

    def __init__(self, weights: torch.Tensor | None = None) -> None:
        self.weights = weights

    def __call__(self, positions, rotations, targets, skeleton) -> torch.Tensor:
        error = (positions - targets).pow(2).sum(-1)
        if self.weights is not None:
            error = error * self.weights.to(error.device, error.dtype)
        return error.mean()


@IK_TERMS.register("smoothness")
class SmoothnessTerm(IKTerm):
    """Penalize frame-to-frame change, which is what kills Jacobian IK jitter.

    Applied to rotations rather than positions, so it damps the actual degrees
    of freedom being solved rather than their downstream effect.
    """

    name = "smoothness"

    def __call__(self, positions, rotations, targets, skeleton) -> torch.Tensor:
        if rotations.shape[0] < 2:
            return rotations.new_zeros(())
        return (rotations[1:] - rotations[:-1]).pow(2).mean()


@IK_TERMS.register("joint_limits")
class JointLimitTerm(IKTerm):
    """Penalize rotations that bend a joint further than declared.

    The limit is a swing cone: the angle between the bone's rest direction and
    where the rotation puts it. Bones bend further than a real one can precisely
    because nothing stops them, which is the artifact this exists to remove.

    The angle comes from ``atan2`` rather than ``acos``. A joint sitting at rest
    has a cosine of exactly 1, where ``acos`` has an infinite derivative -- so
    the obvious formulation produces NaN gradients on the very configuration the
    solver starts from.
    """

    name = "joint_limits"

    def __init__(self, max_swing_deg: dict[int, float] | None = None, default_deg: float = 180.0):
        self.max_swing_deg = max_swing_deg or {}
        self.default_deg = default_deg

    def __call__(self, positions, rotations, targets, skeleton) -> torch.Tensor:
        if not self.max_swing_deg:
            return rotations.new_zeros(())

        penalty = rotations.new_zeros(())
        for joint, limit_deg in self.max_swing_deg.items():
            rest = skeleton.offsets[joint].to(rotations.device, rotations.dtype)
            if torch.linalg.vector_norm(rest) < 1e-8:
                continue
            direction = rest / torch.linalg.vector_norm(rest)
            rotated = torch.einsum("fij,j->fi", rotations[:, joint], direction)
            expanded = direction.expand_as(rotated)
            sine = torch.linalg.vector_norm(torch.cross(expanded, rotated, dim=-1), dim=-1)
            cosine = (rotated * expanded).sum(-1)
            swing = torch.atan2(sine, cosine)
            limit = math.radians(limit_deg)
            penalty = penalty + (swing - limit).clamp(min=0.0).pow(2).mean()
        return penalty


@IK_TERMS.register("contact_pin")
class ContactPinTerm(IKTerm):
    """Hold a joint still on frames where it is marked in contact.

    The model already predicts a foot-contact channel, and the training loss
    spends capacity encouraging consistency with it -- but the reference's
    exporter throws that signal away. Feeding it to the solver enforces at
    export what the loss only encourages during training.
    """

    name = "contact_pin"

    def __init__(self, contact: torch.Tensor) -> None:
        self.contact = contact  # (F, J) in {0, 1}

    def __call__(self, positions, rotations, targets, skeleton) -> torch.Tensor:
        if positions.shape[0] < 2:
            return positions.new_zeros(())
        planted = self.contact.to(positions.device, positions.dtype)[:-1]
        velocity = positions[1:] - positions[:-1]
        return (velocity.pow(2).sum(-1) * planted).mean()

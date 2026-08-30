"""Rotation loss measured on the rotation manifold."""

from __future__ import annotations

from typing import Any

import torch

from poseydon.core.batch import MotionBatch
from poseydon.core.spec import Block
from poseydon.losses.base import LOSSES, LossTerm, masked_mean, raw_block


def rot6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """Gram-Schmidt two 3-vectors into a rotation matrix, columns layout."""
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = torch.nn.functional.normalize(a1, dim=-1)
    b2 = torch.nn.functional.normalize(a2 - (b1 * a2).sum(-1, keepdim=True) * b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)


def geodesic_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Angle of the relative rotation, in radians.

    Uses ``atan2`` of the rotation's sine and cosine rather than ``acos`` of the
    trace. ``acos`` is singular at both endpoints, so implementations clamp its
    argument -- which leaves a floor of ``acos(1 - eps)``, about 4.5e-4 radians
    for eps of 1e-7, and a spurious gradient at zero error. This form is exactly
    zero for identical rotations and has a finite derivative throughout.
    """
    relative = a.transpose(-1, -2) @ b
    trace = relative.diagonal(dim1=-2, dim2=-1).sum(-1)
    axis = torch.stack(
        [
            relative[..., 2, 1] - relative[..., 1, 2],
            relative[..., 0, 2] - relative[..., 2, 0],
            relative[..., 1, 0] - relative[..., 0, 1],
        ],
        dim=-1,
    )
    sine = 0.5 * torch.linalg.vector_norm(axis, dim=-1)
    cosine = (trace - 1.0) / 2.0
    return torch.atan2(sine, cosine)


@LOSSES.register("geodesic")
class GeodesicLoss(LossTerm):
    """Angular error between predicted and target joint rotations.

    Operates on RAW rotations. Measuring this on normalized values, as the
    reference does, re-orthonormalizes distorted 6D rows and reports the angle
    between two rotations that the animation never contained.
    """

    name = "geodesic"
    needs = (Block("rot6d", space="raw"),)

    def __call__(
        self,
        x0_hat: torch.Tensor,
        x0: torch.Tensor,
        batch: MotionBatch,
        aux: dict[str, Any],
    ) -> torch.Tensor:
        predicted = raw_block(batch, x0_hat, "rot6d").permute(0, 1, 3, 2)
        target = raw_block(batch, x0, "rot6d").permute(0, 1, 3, 2)

        angles = geodesic_distance(rot6d_to_matrix(predicted), rot6d_to_matrix(target))
        mask = (
            batch.masks.joints[:, :, None] & batch.masks.frames[:, None, :]
        ).to(angles.dtype)
        return masked_mean(angles, mask)

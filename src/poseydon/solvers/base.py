"""Inverse kinematics.

The solver is a seam rather than a fixed dependency. The reference calls a
fixed-iteration CPU Jacobian solver with no joint limits and no weighting, at
both ingest and export -- and the export call determines the visual quality of
every BVH the model produces.

Rotation-space parameterization means bone lengths are preserved exactly by
construction, so no term is needed to defend them.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar

import torch

from poseydon.core.registry import Registry


@dataclass(frozen=True)
class SolverSkeleton:
    """What a solver needs to know about the rig."""

    parents: torch.Tensor  # (J,) long
    offsets: torch.Tensor  # (J, 3)

    @property
    def n_joints(self) -> int:
        return int(self.parents.shape[0])


class IKTerm(ABC):
    """One weighted objective in the solve."""

    name: ClassVar[str]

    @abstractmethod
    def __call__(
        self,
        positions: torch.Tensor,  # (F, J, 3) current
        rotations: torch.Tensor,  # (F, J, 3, 3) current
        targets: torch.Tensor,  # (F, J, 3) desired
        skeleton: SolverSkeleton,
    ) -> torch.Tensor:
        """Return a scalar."""


class Solver(ABC):
    @abstractmethod
    def solve(
        self,
        targets: torch.Tensor,
        skeleton: SolverSkeleton,
        root_pos: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Fit local rotation matrices ``(F, J, 3, 3)`` to target positions."""


IK_TERMS: Registry[IKTerm] = Registry("ik term")
SOLVERS: Registry[Solver] = Registry("solver")

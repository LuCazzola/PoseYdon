"""Batched gradient-descent inverse kinematics."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from poseydon.core.torch_kinematics import forward_kinematics, rot6d_to_matrix
from poseydon.solvers.base import SOLVERS, IKTerm, Solver, SolverSkeleton


@SOLVERS.register("gradient_ik")
class GradientIK(Solver):
    """Solve every frame at once on whatever device the tensors live on.

    Three things this buys over the reference's fixed-iteration CPU Jacobian
    solver: the whole clip is one batched optimization rather than a loop; the
    objective is a weighted list of terms the caller composes, so joint limits
    and contact pinning are available rather than absent; and it runs on a GPU.

    Rotations are parameterized as 6D vectors, so every step of the optimizer
    lands on a valid rotation and bone lengths are exact by construction.
    """

    def __init__(
        self,
        terms: Sequence[tuple[float, IKTerm]],
        iterations: int = 60,
        optimizer: str = "adam",
        learning_rate: float = 0.1,
    ) -> None:
        if not terms:
            raise ValueError("a solve needs at least one objective term")
        if optimizer not in ("adam", "lbfgs"):
            raise ValueError(f"unknown optimizer `{optimizer}`; use adam or lbfgs")
        self.terms = list(terms)
        self.iterations = iterations
        self.optimizer = optimizer
        self.learning_rate = learning_rate

    def objective(
        self,
        parameters: torch.Tensor,
        targets: torch.Tensor,
        skeleton: SolverSkeleton,
        root_pos: torch.Tensor,
    ) -> torch.Tensor:
        rotations = rot6d_to_matrix(parameters)
        positions = forward_kinematics(rotations, root_pos, skeleton.offsets, skeleton.parents)
        total = parameters.new_zeros(())
        for weight, term in self.terms:
            total = total + weight * term(positions, rotations, targets, skeleton)
        return total

    def solve(
        self,
        targets: torch.Tensor,
        skeleton: SolverSkeleton,
        root_pos: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if targets.ndim != 3 or targets.shape[-1] != 3:
            raise ValueError(f"targets must be (F, J, 3), got {tuple(targets.shape)}")
        if targets.shape[1] != skeleton.n_joints:
            raise ValueError(
                f"targets describe {targets.shape[1]} joints, skeleton has "
                f"{skeleton.n_joints}"
            )

        n_frames = targets.shape[0]
        device, dtype = targets.device, targets.dtype
        if root_pos is None:
            root_pos = targets[:, 0]

        # Start from identity: the 6D vector whose Gram-Schmidt is the identity.
        identity = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], device=device, dtype=dtype)
        parameters = identity.repeat(n_frames, skeleton.n_joints, 1).requires_grad_(True)

        if self.optimizer == "adam":
            optimizer = torch.optim.Adam([parameters], lr=self.learning_rate)
        else:
            optimizer = torch.optim.LBFGS([parameters], max_iter=self.iterations, lr=self.learning_rate)

        def closure() -> torch.Tensor:
            optimizer.zero_grad()
            loss = self.objective(parameters, targets, skeleton, root_pos)
            loss.backward()
            return loss

        if self.optimizer == "lbfgs":
            optimizer.step(closure)
        else:
            for _ in range(self.iterations):
                optimizer.step(closure)

        return rot6d_to_matrix(parameters.detach())

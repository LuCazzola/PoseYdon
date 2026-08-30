"""Inverse kinematics: fitting rotations to desired positions."""

from poseydon.solvers import terms as _terms  # noqa: F401  (registers terms)
from poseydon.solvers.base import IK_TERMS, SOLVERS, IKTerm, Solver, SolverSkeleton
from poseydon.solvers.gradient_ik import GradientIK

__all__ = ["IK_TERMS", "SOLVERS", "GradientIK", "IKTerm", "Solver", "SolverSkeleton"]

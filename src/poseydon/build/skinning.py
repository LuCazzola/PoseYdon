"""Pure planning for topology reduction of skin-weight matrices."""

from dataclasses import dataclass
import numpy as np
from poseydon.features.reduce import COLLAPSE, DROP, RemovalOp

class SkinningPolicyError(ValueError): pass

@dataclass(frozen=True)
class SkinningInput:
    joint_names: tuple[str, ...]
    weights: np.ndarray
    animated: frozenset[str]
    removals: tuple[RemovalOp, ...]

@dataclass(frozen=True)
class JointDisposition:
    joint: str
    operation: str
    destination: str | None
    affected_vertices: int
    total_transferred_weight: float
    reason: str

@dataclass(frozen=True)
class SkinningPlan:
    output_joint_names: tuple[str, ...]
    output_weights: np.ndarray
    decisions: tuple[JointDisposition, ...]
    max_normalization_correction: float

def plan_skinning(source, policy):
    names, weights, decisions = list(source.joint_names), np.asarray(source.weights, dtype=float).copy(), []
    for removal in source.removals:
        column = names.index(removal.name)
        influence = weights[:, column]
        affected, total = int(np.count_nonzero(influence > 1e-12)), float(influence.sum())
        if affected and removal.name in source.animated and policy is None:
            raise SkinningPolicyError(f"weighted animated joint `{removal.name}` requires explicit policy")
        destination = removal.parent if removal.kind == COLLAPSE else None
        if destination is not None:
            weights[:, names.index(destination)] += influence
        elif affected:
            if policy not in {"exclude", "bake"}: raise SkinningPolicyError(f"weighted joint `{removal.name}` cannot be dropped")
            destination = removal.parent
            weights[:, names.index(destination)] += influence
        weights = np.delete(weights, column, axis=1); names.pop(column)
        decisions.append(JointDisposition(removal.name, "drop" if removal.kind == DROP else "collapse", destination, affected, total, "unweighted helper" if not affected else "explicit policy"))
    sums = weights.sum(axis=1, keepdims=True)
    if np.any(sums <= 0): raise SkinningPolicyError("vertex has zero total influence")
    normalized = weights/sums
    return SkinningPlan(tuple(names), normalized, tuple(decisions), float(np.abs(normalized-weights).max()))

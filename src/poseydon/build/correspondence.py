"""Validated name and bind-geometry mapping between model and presentation rigs."""

from dataclasses import dataclass
from typing import Mapping
import numpy as np

class CorrespondenceError(ValueError): pass

@dataclass(frozen=True)
class SkeletonGeometry:
    names: tuple[str, ...]
    parents: tuple[int, ...]
    bind_positions: np.ndarray

@dataclass(frozen=True)
class SkeletonCorrespondence:
    model_to_presentation: Mapping[str, str]
    max_bind_residual: float
    def presentation_for(self, model_joint: str) -> str:
        return self.model_to_presentation[model_joint]

def _residual(source, target):
    sm, tm = source.mean(0), target.mean(0)
    a, b = source-sm, target-tm
    u, singular, vt = np.linalg.svd(b.T @ a / len(a))
    correction = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0: correction[-1,-1] = -1
    rotation = u @ correction @ vt
    variance = float((a*a).sum()/len(a))
    scale = float((singular*np.diag(correction)).sum()/variance) if variance else 1.0
    fitted = scale*(source @ rotation.T) + tm - scale*(rotation @ sm)
    lengths = np.linalg.norm(np.diff(target, axis=0), axis=1)
    unit = float(lengths[lengths>1e-8].mean()) if np.any(lengths>1e-8) else 1.0
    return float(np.abs(fitted-target).max()/unit)

def build_correspondence(model, presentation, *, aliases=None, max_residual=0.05):
    aliases = aliases or {}
    if len(aliases.values()) != len(set(aliases.values())):
        raise CorrespondenceError("ambiguous aliases")
    mapping = {}
    for name in presentation.names:
        target = aliases.get(name, name)
        if target in model.names:
            if target in mapping: raise CorrespondenceError(f"ambiguous mapping for {target}")
            mapping[target] = name
    missing = [name for name in model.names if name not in mapping]
    if missing: raise CorrespondenceError(f"missing model joints: {', '.join(missing)}")
    indices = {name:i for i,name in enumerate(presentation.names)}
    target = np.stack([presentation.bind_positions[indices[mapping[name]]] for name in model.names])
    residual = _residual(np.asarray(model.bind_positions), target)
    if residual > max_residual:
        raise CorrespondenceError(f"bind residual {residual:.6g} exceeds threshold {max_residual:g}")
    return SkeletonCorrespondence(mapping, residual)

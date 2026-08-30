"""Per-skeleton feature statistics.

Statistics are computed FOR THE ACTIVE FEATURE SPEC, not for a fixed 13-dim
layout, and are invertible. That is what lets a structural loss ask for a block
in raw space: the reference computes its geodesic and foot-skate losses on
z-normalized values, where a per-channel divide by std is not a
rotation-preserving operation, so the quantity it optimizes is not the one its
paper describes.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from poseydon.core.spec import FeatureSpec

# Guards against dividing by the standard deviation of a constant channel, such
# as a contact flag that never fires or a joint that never moves.
STD_EPSILON = 1e-6


def spec_hash(spec: FeatureSpec) -> str:
    """Short stable digest of a layout, for cache keys."""
    payload = json.dumps(list(spec.blocks), sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


@dataclass(frozen=True)
class Normalizer:
    mean: np.ndarray  # (J, D)
    std: np.ndarray  # (J, D)
    spec: FeatureSpec

    def __post_init__(self) -> None:
        if self.mean.shape != self.std.shape:
            raise ValueError(f"mean {self.mean.shape} and std {self.std.shape} differ")
        if self.mean.shape[-1] != self.spec.dim:
            raise ValueError(
                f"statistics have width {self.mean.shape[-1]} but the spec is {self.spec.dim}"
            )

    @classmethod
    def fit(cls, arrays: Iterable[np.ndarray], spec: FeatureSpec) -> Normalizer:
        """Fit over clips of one skeleton, each ``(frames, joints, dim)``."""
        stacked = np.concatenate([np.asarray(a, dtype=np.float64) for a in arrays], axis=0)
        if stacked.ndim != 3:
            raise ValueError(f"expected (frames, joints, dim) arrays, got {stacked.ndim} dims")
        return cls(
            mean=stacked.mean(axis=0),
            std=stacked.std(axis=0) + STD_EPSILON,
            spec=spec,
        )

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return np.nan_to_num((x - self.mean) / self.std)

    def denormalize(self, x: np.ndarray) -> np.ndarray:
        return x * self.std + self.mean

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            mean=self.mean,
            std=self.std,
            blocks=np.array(json.dumps(list(self.spec.blocks))),
        )

    @classmethod
    def load(cls, path: str | Path) -> Normalizer:
        with np.load(Path(path), allow_pickle=False) as data:
            blocks = json.loads(str(data["blocks"]))
            return cls(
                mean=data["mean"],
                std=data["std"],
                spec=FeatureSpec(tuple((name, int(width)) for name, width in blocks)),
            )

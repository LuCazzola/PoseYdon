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
from collections.abc import Iterable, Sequence
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
class BlockPolicy:
    """How one feature block is normalized.

    Declared per block in the dataset config rather than fixed in code, because
    the right answer differs by block. Per-channel std makes a barely-moving toe
    unit-variance, so reconstruction loss weights it as heavily as the root;
    pooling preserves relative magnitude. More sharply: dividing a 6D rotation
    row by a UNIFORM scalar leaves the rotation unchanged after Gram-Schmidt,
    while dividing per-channel does not -- so pooling is what lets the rotation
    block survive normalization at all.
    """

    name: str
    center: bool = True
    #: ``channel`` per (joint, channel); ``joint_block`` one scalar per (joint,
    #: block); ``block`` one scalar per (root / non-root, block); ``none`` leaves
    #: std at 1, for flags.
    scale: str = "channel"


_SCALE_MODES = ("channel", "joint_block", "block", "none")


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
    def fit(
        cls,
        arrays: Iterable[np.ndarray],
        spec: FeatureSpec,
        policy: Sequence[BlockPolicy] | None = None,
    ) -> Normalizer:
        """Fit over clips of one skeleton, each ``(frames, joints, dim)``.

        ``policy`` declares per-block centring and pooling. ``None`` keeps the
        per-channel behaviour every existing caller expects.
        """
        stacked = np.concatenate([np.asarray(a, dtype=np.float64) for a in arrays], axis=0)
        if stacked.ndim != 3:
            raise ValueError(f"expected (frames, joints, dim) arrays, got {stacked.ndim} dims")

        mean = stacked.mean(axis=0)
        std = stacked.std(axis=0) + STD_EPSILON
        if policy is None:
            return cls(mean=mean, std=std, spec=spec)

        known = set(spec.names)
        for entry in policy:
            if entry.name not in known:
                raise ValueError(
                    f"policy names block `{entry.name}`, which the spec does not "
                    f"declare. Available: {', '.join(sorted(known))}"
                )
            if entry.scale not in _SCALE_MODES:
                raise ValueError(
                    f"block `{entry.name}` has unknown scale mode `{entry.scale}`. "
                    f"Available: {', '.join(_SCALE_MODES)}"
                )

        for entry in policy:
            block = spec.slice(entry.name)
            if not entry.center:
                mean[:, block] = 0.0

            if entry.scale == "channel":
                continue
            if entry.scale == "none":
                std[:, block] = 1.0
                continue

            # Pool the VARIANCE, not the std: pooling std would average
            # magnitudes rather than energies and is not the same statistic.
            variance = stacked[:, :, block].var(axis=0)  # (J, W)
            if entry.scale == "joint_block":
                pooled = np.sqrt(variance.mean(axis=-1, keepdims=True))  # (J, 1)
                std[:, block] = pooled + STD_EPSILON
            else:  # "block": one scalar for the root, one for everything else
                root = np.sqrt(variance[0].mean())
                rest = np.sqrt(variance[1:].mean()) if variance.shape[0] > 1 else root
                std[0, block] = root + STD_EPSILON
                std[1:, block] = rest + STD_EPSILON

        return cls(mean=mean, std=std, spec=spec)

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

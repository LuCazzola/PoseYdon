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

# Added to every standard deviation, matching the reference
# (`data_loaders/truebones/data/dataset.py:159`, `std += 1e-6 # for stability`).
STD_EPSILON = 1e-6

#: Below this, a channel is treated as CONSTANT and left unscaled (std 1.0)
#: rather than divided by something near zero.
#:
#: `STD_EPSILON` alone is not a guard, it is an amplifier. A channel that never
#: varies gets `std = 0 + 1e-6`, so dividing by it multiplies by a million. For
#: a CENTRED block that is harmless -- the constant becomes 0 before the divide
#: -- but `rot6d` ships `center: false` (a 6D rotation has no zero to centre
#: around), so a joint that never rotates carries a raw ~1.0 straight into
#: `1.0 / 1e-6 = 1e6`. Measured on the built corpus: 12 of 73 rigs, `rot6d`
#: only, 144 channels sitting exactly at the epsilon -- Alligator 30 channels,
#: Turtle 24, Ant 18, FireAnt/Roach/Scorpion-2 12 each. Under balanced sampling
#: that reached ~16% of batches and put the `simple` loss at ~1e10.
#:
#: 1e-4 is not a guess. Across all 73 rigs and every block, 168 channels fall
#: below 1e-4 and 168 below 1e-3 -- the SAME channels, because nothing at all
#: lies in that decade -- while the 1st percentile of real variation is 4.9e-3.
#: The threshold sits in an empty gap, so its exact value changes nothing.
#:
#: This is a deliberate divergence from the reference, which applies the bare
#: `+ 1e-6` and has no such guard. The arithmetic there is faithfully copied;
#: what differs is that this corpus contains zero-variance `rot6d` channels.
#: A channel with no variance carries no information, and scaling float noise
#: by a million does not create any -- it only drowns the channels that do.
STD_CONSTANT_THRESHOLD = 1e-4


def _unscale_constants(std: np.ndarray) -> np.ndarray:
    """Leave constant channels alone instead of dividing by ~zero.

    See :data:`STD_CONSTANT_THRESHOLD` for why this exists and why 1e-4.
    """
    return np.where(std < STD_CONSTANT_THRESHOLD, 1.0, std)


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
            return cls(mean=mean, std=_unscale_constants(std), spec=spec)

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

        # After pooling, not before: `joint_block` and `block` replace the
        # per-channel std outright, so flooring earlier would be overwritten.
        # `scale: none` blocks are already exactly 1.0 and unaffected.
        return cls(mean=mean, std=_unscale_constants(std), spec=spec)

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

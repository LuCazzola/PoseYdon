"""Skeleton-derived conditioning."""

from __future__ import annotations

from typing import Any

import torch

from poseydon.conditioners.base import CONDITIONERS, Conditioner
from poseydon.losses.base import NORM_STATS


def _pad_joints(tensor: torch.Tensor, max_joints: int, value: float = 0.0) -> torch.Tensor:
    """Pad the joint axis (dim 0) up to ``max_joints``."""
    if tensor.shape[0] == max_joints:
        return tensor
    pad = torch.full(
        (max_joints - tensor.shape[0], *tensor.shape[1:]), value, dtype=tensor.dtype
    )
    return torch.cat([tensor, pad], dim=0)


@CONDITIONERS.register("topology")
class Topology(Conditioner):
    """Parent indices and rest-pose bone offsets.

    This is what lets one model span skeletons with different joint counts:
    the structure travels with the batch instead of being baked into the
    architecture. Padded parents are -1, which no real joint uses.
    """

    name = "topology"

    def extract(self, item: Any) -> dict[str, torch.Tensor]:
        return {
            "parents": torch.from_numpy(item.anim.parents.astype("int64")),
            "offsets": torch.from_numpy(item.anim.offsets.astype("float32")),
        }

    def collate(self, payloads: list[dict], max_joints: int) -> dict[str, torch.Tensor]:
        return {
            "parents": torch.stack(
                [_pad_joints(p["parents"], max_joints, value=-1) for p in payloads]
            ),
            "offsets": torch.stack([_pad_joints(p["offsets"], max_joints) for p in payloads]),
        }


@CONDITIONERS.register(NORM_STATS)
class NormalizationStats(Conditioner):
    """Per-skeleton feature statistics, so a loss can undo normalization.

    Carried with the batch rather than looked up globally, because a batch may
    mix skeletons and each has its own statistics.
    """

    name = NORM_STATS

    def extract(self, item: Any) -> dict[str, torch.Tensor]:
        return {
            "mean": torch.from_numpy(item.normalizer.mean.astype("float32")),
            "std": torch.from_numpy(item.normalizer.std.astype("float32")),
        }

    def collate(self, payloads: list[dict], max_joints: int) -> dict[str, torch.Tensor]:
        return {
            "mean": torch.stack([_pad_joints(p["mean"], max_joints) for p in payloads]),
            "std": torch.stack([_pad_joints(p["std"], max_joints, value=1.0) for p in payloads]),
        }

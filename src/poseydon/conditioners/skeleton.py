"""Skeleton-derived conditioning."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from poseydon.conditioners.base import CONDITIONERS, Conditioner
from poseydon.core.topology import DEFAULT_MAX_PATH, edge_relations, hop_distances
from poseydon.losses.base import NORM_STATS
from poseydon.models.modiffae import JOINT_NAMES

#: Built once by `tools/build_joint_name_cache.py`; see `JointNames`.
DEFAULT_JOINT_NAME_CACHE = Path("data/truebones/joint_names_t5.npz")


def _pad_square(tensor: torch.Tensor, max_joints: int, value: int = 0) -> torch.Tensor:
    """Pad a (J, J) relation matrix to (max_joints, max_joints)."""
    joints = tensor.shape[0]
    if joints == max_joints:
        return tensor
    out = torch.full((max_joints, max_joints), value, dtype=tensor.dtype)
    out[:joints, :joints] = tensor
    return out


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

    def __init__(self, max_path: int = DEFAULT_MAX_PATH) -> None:
        self.max_path = max_path

    def extract(self, item: Any) -> dict[str, torch.Tensor]:
        parents = item.anim.parents.astype("int64")
        return {
            "parents": torch.from_numpy(parents),
            "offsets": torch.from_numpy(item.anim.offsets.astype("float32")),
            "relations": torch.from_numpy(edge_relations(parents)),
            "hops": torch.from_numpy(hop_distances(parents, self.max_path)),
        }

    def collate(self, payloads: list[dict], max_joints: int) -> dict[str, torch.Tensor]:
        return {
            "parents": torch.stack(
                [_pad_joints(p["parents"], max_joints, value=-1) for p in payloads]
            ),
            "offsets": torch.stack([_pad_joints(p["offsets"], max_joints) for p in payloads]),
            "relations": torch.stack([_pad_square(p["relations"], max_joints) for p in payloads]),
            "hops": torch.stack([_pad_square(p["hops"], max_joints) for p in payloads]),
        }


@CONDITIONERS.register("tpose")
class TPose(Conditioner):
    """A canonical rest pose, as one frame of features describing the rig.

    AnyTop consumes this as an extra leading frame, giving the model a
    description of the skeleton it is animating. The manifest may name a T-pose
    BVH; when it does not, the first frame of the clip stands in -- which is what
    the reference does too, falling back to the first file it finds for the
    character when no `tpos` file exists.
    """

    name = "tpose"

    def extract(self, item: Any) -> torch.Tensor:
        frame = item.rest_frame
        return torch.from_numpy(np.ascontiguousarray(frame, dtype="float32"))

    def collate(self, payloads: list[torch.Tensor], max_joints: int) -> torch.Tensor:
        return torch.stack([_pad_joints(p, max_joints) for p in payloads])


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


@CONDITIONERS.register(JOINT_NAMES)
class JointNames(Conditioner):
    """T5 embeddings of each joint's name, one row per joint.

    The reference trains with `skip_t5: False` and `cond_mask_prob: 0.0`, so
    this conditioning is present on every forward pass and never masked. It is
    what tells the model that a rig's `BN_Tail_L_01` is a tail and its
    `Bip01_Head` is a head -- structure alone cannot say which limb is which,
    and cross-topology transfer is exactly the task where that matters.

    Embeddings are read from a cache built once by
    `tools/build_joint_name_cache.py`, because computing them needs
    `transformers` and the training image deliberately does not carry it.
    """

    name = JOINT_NAMES

    #: A duplicated joint's name, from `augment.topology.duplicate_joint`.
    MID_SUFFIX = "__mid"

    def __init__(self, cache: str | Path = DEFAULT_JOINT_NAME_CACHE) -> None:
        path = Path(cache)
        if not path.exists():
            raise FileNotFoundError(
                f"joint-name embeddings are missing at {path}. Build them once with:\n"
                f"  docker compose run --rm --entrypoint python compat "
                f"tools/build_joint_name_cache.py\n"
                f"or drop `{JOINT_NAMES}` from `conditioners:` to train without them "
                f"(which is NOT what the reference does)."
            )
        with np.load(path, allow_pickle=True) as data:
            names = [str(n) for n in data["names"]]
            embeddings = data["embeddings"].astype("float32")
        self._table = dict(zip(names, embeddings))
        self._dim = int(embeddings.shape[1])

    def _row(self, names: Sequence[str], parents: np.ndarray, joint: int) -> np.ndarray:
        """One joint's embedding, resolving a duplicate against its parent.

        `DuplicateJoint` inserts `<source>__mid` between a joint and its parent
        and gives it the midpoint of their features. The reference does the same
        to the name embedding (`add_joint_augmentation` averages the joint's row
        with its parent's), so the inserted joint reads as what it geometrically
        is: halfway between the two.
        """
        name = names[joint]
        if not name.endswith(self.MID_SUFFIX):
            return self._lookup(name)
        source = self._lookup(name[: -len(self.MID_SUFFIX)])
        parent = int(parents[joint])
        if parent < 0:
            return source
        return (source + self._lookup(names[parent])) / 2.0

    def _lookup(self, name: str) -> np.ndarray:
        try:
            return self._table[name]
        except KeyError:
            raise KeyError(
                f"no cached embedding for joint name {name!r}. The cache at "
                f"{DEFAULT_JOINT_NAME_CACHE} predates this rig -- rebuild it with "
                f"tools/build_joint_name_cache.py."
            ) from None

    def extract(self, item: Any) -> torch.Tensor:
        names = list(item.anim.names)
        parents = np.asarray(item.anim.parents)
        rows = [self._row(names, parents, j) for j in range(len(names))]
        return torch.from_numpy(np.stack(rows).astype("float32"))

    def collate(self, payloads: list[torch.Tensor], max_joints: int) -> torch.Tensor:
        return torch.stack([_pad_joints(p, max_joints) for p in payloads])

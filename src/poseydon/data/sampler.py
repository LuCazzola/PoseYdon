"""Sampling weights that give every rig an equal voice.

Truebones is wildly unbalanced -- BrownBear ships 22 clips, a dozen rigs ship
one. Sampling uniformly over clips trains a bear model that has met some other
animals. The reference passes `--balanced` in both of its documented training
commands and implements it as a plain `WeightedRandomSampler`
(`data_loaders/truebones/data/dataset.py::TruebonesSampler`): equal share per
object type, split evenly across that type's clips. This mirrors it.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np


def balanced_weights(dataset: Any) -> np.ndarray:
    """`1 / (n_rigs * n_entries_of_rig)` per plan entry.

    Weighted per PLAN ENTRY -- a (clip, window) pair -- not per clip, because
    that is what `__len__` indexes and therefore what the sampler draws. A clip
    that yields more windows must not thereby win more of its rig's share.
    """
    rigs = [dataset.records[clip].skeleton for clip, _ in dataset._plan]
    per_rig = Counter(rigs)
    share = 1.0 / len(per_rig)
    return np.array([share / per_rig[rig] for rig in rigs], dtype=np.float64)

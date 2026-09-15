"""Fixtures for model-level tests.

A nine-joint rig, small enough to assert on by hand and shaped to exercise
every branch of the chain decomposition: a two-joint chain (neck, head), a
three-joint chain (the tail), three single-joint chains at a branch point, and
a root with two children so no chain runs through it.

    0 root
    |-- 1 spine ----- 2 neck -- 3 head
    |              |- 4 leg_l
    |              `- 5 leg_r
    `-- 6 tail1 ---- 7 tail2 -- 8 tail3

Chains: {1}, {2,3}, {4}, {5}, {6,7,8}, plus the root's own {0}.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from poseydon.core.batch import Cond, Masks
from poseydon.core.topology import edge_relations, hop_distances
from poseydon.models.base import CLEAN_MOTION
from poseydon.models.modiffae import MoDiffAE

PARENTS = (-1, 0, 1, 2, 1, 1, 0, 6, 7)
FEATURE_DIM = 6
N_FRAMES = 7


@pytest.fixture
def tiny_rig() -> torch.Tensor:
    return torch.tensor([PARENTS], dtype=torch.long)


@pytest.fixture
def tiny_valid() -> torch.Tensor:
    return torch.ones(1, len(PARENTS), dtype=torch.bool)


@pytest.fixture
def tiny_model():
    """A MoDiffAE small enough to run on CPU in a test.

    `dropout=0.0` deliberately: these tests compare two forward passes for
    equality, and dropout would make them differ for reasons that have nothing
    to do with what is being asserted.
    """

    def _build(**overrides):
        settings = {
            "feature_dim": FEATURE_DIM,
            "d_model": 16,
            "n_layers_semantic": 1,
            "n_layers_stochastic": 1,
            "n_heads": 2,
            "ff_size": 32,
            "dropout": 0.0,
            "n_virtual_joints": 2,
        }
        settings.update(overrides)
        return MoDiffAE(**settings)

    return _build


@pytest.fixture
def tiny_batch(tiny_rig, tiny_valid):
    """`(z_t, t, cond, masks)` for one sample of the nine-joint rig."""
    parents = np.asarray(PARENTS, dtype=np.int64)
    joints = len(PARENTS)
    clean = torch.randn(1, joints, FEATURE_DIM, N_FRAMES)
    cond = Cond(
        {
            "topology": {
                "parents": tiny_rig,
                "offsets": torch.zeros(1, joints, 3),
                "relations": torch.from_numpy(edge_relations(parents))[None],
                "hops": torch.from_numpy(hop_distances(parents))[None],
            },
            "tpose": torch.randn(1, joints, FEATURE_DIM),
            CLEAN_MOTION: clean,
        }
    )
    masks = Masks(
        frames=torch.ones(1, N_FRAMES, dtype=torch.bool),
        joints=tiny_valid.clone(),
    )
    return torch.randn(1, joints, FEATURE_DIM, N_FRAMES), torch.zeros(1, dtype=torch.long), cond, masks

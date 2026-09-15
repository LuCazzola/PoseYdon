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

import pytest
import torch

PARENTS = (-1, 0, 1, 2, 1, 1, 0, 6, 7)


@pytest.fixture
def tiny_rig() -> torch.Tensor:
    return torch.tensor([PARENTS], dtype=torch.long)


@pytest.fixture
def tiny_valid() -> torch.Tensor:
    return torch.ones(1, len(PARENTS), dtype=torch.bool)

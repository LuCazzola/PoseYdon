"""One Generator, forked into N workers, draws the same stream N times."""

from __future__ import annotations

import numpy as np


class _Tiny:
    """The bug in isolation: the object holds the Generator, fork copies it."""

    def __init__(self, seed=0):
        self._rng = np.random.default_rng(seed)

    def set_worker_seed(self, worker_id: int, base_seed: int = 0) -> None:
        self._rng = np.random.default_rng([base_seed, worker_id])

    def draw(self):
        return int(self._rng.integers(0, 1_000_000))


def test_forked_workers_would_draw_identically_without_reseeding():
    copies = [_Tiny(seed=0) for _ in range(4)]
    assert len({c.draw() for c in copies}) == 1, "this is the bug being fixed"


def test_reseeding_per_worker_decorrelates_them():
    copies = [_Tiny(seed=0) for _ in range(4)]
    for worker_id, copy in enumerate(copies):
        copy.set_worker_seed(worker_id)
    assert len({c.draw() for c in copies}) == 4


def test_the_dataset_exposes_the_hook_the_loader_calls():
    from poseydon.data.dataset import MotionDataset

    assert hasattr(MotionDataset, "set_worker_seed")

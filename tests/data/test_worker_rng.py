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


def test_the_real_dataset_reseeds_and_diverges_under_fork():
    """The tests above use a stand-in, so they prove the IDEA and not the
    WIRING. A regression that dropped `worker_init_fn=` from the DataLoader, or
    typo'd `info.dataset.set_worker_seed`, would leave every one of them green.

    This drives the real `MotionDataset`: it copies the object the way `fork`
    does, confirms the copies draw identically (the bug), reseeds them through
    the same call `MotionDataModule._worker_init` makes, and confirms they
    diverge. No corpus needed -- the RNG lives on the dataset, not in the data.
    """
    import copy

    from poseydon.data.dataset import MotionDataset

    dataset = MotionDataset.__new__(MotionDataset)
    dataset.seed = 0
    dataset._rng = np.random.default_rng(0)

    forked = [copy.deepcopy(dataset) for _ in range(4)]
    before = [int(d._rng.integers(0, 1_000_000)) for d in forked]
    assert len(set(before)) == 1, "fork copies the Generator; this is the bug"

    forked = [copy.deepcopy(dataset) for _ in range(4)]
    for worker_id, worker in enumerate(forked):
        worker.set_worker_seed(worker_id)
    after = [int(d._rng.integers(0, 1_000_000)) for d in forked]
    assert len(set(after)) == 4, f"workers still correlated: {after}"


def test_the_train_loader_actually_wires_the_worker_hook(monkeypatch):
    """`set_worker_seed` existing is not the same as the loader CALLING it.

    Two halves, because either can rot on its own: the DataLoader must carry a
    `worker_init_fn`, and that function must actually reach the dataset. The
    second half fakes `get_worker_info`, since outside a worker process the
    hook is a deliberate no-op and would otherwise assert nothing.
    """
    import torch

    from poseydon.training.lightning import MotionDataModule

    class _StubDataset:
        def __init__(self):
            self.records = []
            self._plan = []
            self.seed = 0
            self.seeded = None

        def __len__(self):
            return 4

        def __getitem__(self, index):
            raise AssertionError("this test never reads an item")

        def set_worker_seed(self, worker_id):
            self.seeded = worker_id

    dataset = _StubDataset()
    module = MotionDataModule(train=dataset, batch_size=1, num_workers=2, balanced=False)
    loader = module.train_dataloader()
    assert loader.worker_init_fn is not None, "workers would share one RNG stream"

    class _FakeInfo:
        pass

    info = _FakeInfo()
    info.dataset = dataset
    monkeypatch.setattr(torch.utils.data, "get_worker_info", lambda: info)
    loader.worker_init_fn(3)
    assert dataset.seeded == 3, "the hook never reached the dataset"


def test_cond_to_moves_nested_payloads():
    """`Cond.to` used to skip half of what it claimed to move.

    `topology` and `norm_stats` collate to DICTS of tensors, and a dict has no
    `.to`, so a flat comprehension left them behind while the caller believed
    the whole Cond had moved. It went unnoticed because MoDiffAE moves each
    cond tensor itself -- the bug only appears in a model that trusts the
    method, which is what the method exists for. Uses meta tensors so the test
    needs no second device.
    """
    import torch

    from poseydon.core.batch import Cond

    cond = Cond({
        "flat": torch.zeros(2),
        "topology": {"hops": torch.zeros(2, 2), "relations": torch.zeros(2, 2)},
        "listed": [torch.zeros(2)],
        "not_a_tensor": "left alone",
    })
    moved = cond.to("meta")

    assert moved["flat"].device.type == "meta"
    assert moved["topology"]["hops"].device.type == "meta", "nested dict was not moved"
    assert moved["topology"]["relations"].device.type == "meta"
    assert moved["listed"][0].device.type == "meta", "nested list was not moved"
    assert moved["not_a_tensor"] == "left alone"

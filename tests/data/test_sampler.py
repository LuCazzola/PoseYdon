"""Balanced sampling: every rig gets an equal share, not every clip."""

from __future__ import annotations

import numpy as np

from poseydon.data.sampler import balanced_weights


class _FakeRecord:
    def __init__(self, skeleton):
        self.skeleton = skeleton


class _FakeDataset:
    """Three rigs, wildly unequal clip counts -- BrownBear's real problem."""

    records = [_FakeRecord("BrownBear")] * 22 + [_FakeRecord("Crab")] * 2 + [_FakeRecord("Goat")]

    def __init__(self):
        # one window per clip keeps the arithmetic legible
        self._plan = [(i, 0) for i in range(len(self.records))]

    def __len__(self):
        return len(self._plan)


def test_each_rig_receives_an_equal_share():
    weights = balanced_weights(_FakeDataset())
    dataset = _FakeDataset()
    share = {}
    for weight, (clip, _) in zip(weights, dataset._plan, strict=True):
        share.setdefault(dataset.records[clip].skeleton, 0.0)
        share[dataset.records[clip].skeleton] += weight
    assert len(share) == 3
    for rig, total in share.items():
        # `isclose`, not `==`: BrownBear's third is a sum of 22 float64 terms,
        # which lands 2e-16 from 1/3. The brief wrote `==`; the arithmetic, not
        # the sampler, is what cannot satisfy it.
        assert np.isclose(total, 1 / 3), f"{rig} got {total}, not an equal third"


def test_clips_within_a_rig_are_equally_likely():
    weights = balanced_weights(_FakeDataset())
    bear = [w for w, (c, _) in zip(weights, _FakeDataset()._plan, strict=True)
            if _FakeDataset.records[c].skeleton == "BrownBear"]
    assert len(bear) == 22
    assert np.allclose(bear, bear[0])


def test_a_rig_with_one_clip_is_not_starved():
    """Goat has 1 clip to BrownBear's 22, so its single entry must carry 22x
    the weight of one bear entry. This is the whole point of the sampler.
    """
    dataset = _FakeDataset()
    weights = balanced_weights(dataset)
    goat = next(w for w, (c, _) in zip(weights, dataset._plan, strict=True)
                if dataset.records[c].skeleton == "Goat")
    bear = next(w for w, (c, _) in zip(weights, dataset._plan, strict=True)
                if dataset.records[c].skeleton == "BrownBear")
    assert np.isclose(goat / bear, 22.0)

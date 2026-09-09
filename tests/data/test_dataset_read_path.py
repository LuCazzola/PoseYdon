"""The read path against the real corpus A2 built.

Every test here skips when the corpus is absent (a fresh checkout has none of
it -- it is gitignored). That is a real weakness, so each test states what it
would catch, and `test_corpus_yield.py` carries the same caveat.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from poseydon.build.index import CorpusIndex
from poseydon.data.dataset import MotionDataset
from poseydon.data.normalize import Normalizer
from poseydon.data.window import RandomCrop

CORPUS = Path("data/truebones")
FEATURES = ["ric_pos", "rot6d", "local_vel", "foot_contact"]

pytestmark = pytest.mark.skipif(
    not (CORPUS / "index.jsonl").is_file(),
    reason="stage-2 corpus absent; run scripts/build_features.py",
)


@pytest.fixture(scope="module")
def dataset() -> MotionDataset:
    return MotionDataset(
        index=CorpusIndex.load(CORPUS / "index.jsonl"),
        root=CORPUS,
        manifest_dir=CORPUS / "rigs",
        features=FEATURES,
        window=RandomCrop(length=40),
        conditioners=["topology", "tpose", "norm_stats"],
        split="train",
        seed=0,
    )


def test_the_dataset_opens_against_the_entity_first_layout(dataset):
    """Before this task the constructor raised ManifestError on
    `rigs/Alligator.yaml` -- A2 moved manifests to `rigs/<Rig>/manifest.yaml`
    and nothing had followed.
    """
    assert len(dataset) > 0
    assert len(dataset.records) == 1145


def test_statistics_are_loaded_from_stats_npz_not_refitted(dataset):
    """Fitting re-derives per-rig moments from every clip at startup. Loading
    is not merely faster: a refit under a different `features:` silently
    produces different moments than the ones the checkpoint was trained with.
    """
    rig = dataset.records[0].skeleton
    loaded = Normalizer.load(CORPUS / "rigs" / rig / "stats.npz")
    used = dataset._normalizer(rig, dataset.spec)
    np.testing.assert_allclose(used.mean, loaded.mean)
    np.testing.assert_allclose(used.std, loaded.std)


def test_the_reduction_matches_the_one_skeleton_npz_records(dataset):
    """stats.npz's moments are per REDUCED joint. If the read path reduced
    differently from `build_stats`, normalization would silently pair a joint's
    values with another joint's statistics.
    """
    for rig in ("Goat", "Flamingo", "Crab", "Tukan"):
        with np.load(CORPUS / "rigs" / rig / "skeleton.npz", allow_pickle=True) as data:
            stored = tuple(int(i) for i in data["reduction_source_of"])
        assert dataset._reduction(rig).source_of == stored


def test_items_are_reduced_rigid_body_features(dataset):
    """The width and joint count a batch carries must be the reduced ones the
    statistics were fitted on -- 31 joints for Goat, not its 40 prepared ones.
    """
    index = next(
        i
        for i, (clip, _) in enumerate(dataset._plan)
        if dataset.records[clip].skeleton == "Goat"
    )
    item = dataset[index]
    assert item.features.shape[1] == 31
    assert item.features.shape[2] == item.spec.dim == 13


def test_startup_does_not_read_every_clip():
    """`__init__` used to extract features for all 1145 clips to count windows.
    The index already knows the frame count; the schema's frame cost is one
    constant. Guarded because the regression is invisible -- it only shows up
    as a slow start.

    Builds its OWN dataset rather than taking the module fixture: the fixture is
    shared, earlier tests have already pulled items through it, and asserting an
    empty cache on it would pass or fail on test ORDER rather than on behaviour.
    """
    fresh = MotionDataset(
        index=CorpusIndex.load(CORPUS / "index.jsonl"),
        root=CORPUS,
        manifest_dir=CORPUS / "rigs",
        features=FEATURES,
        window=RandomCrop(length=40),
        split="train",
        seed=0,
    )
    # One clip is loaded deliberately, to measure the schema's frame cost.
    assert len(fresh._anims) <= 1, "startup must not walk the corpus"
    assert len(fresh) > 0

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


def test_the_rest_frame_comes_from_the_declared_rest_pose(dataset):
    """Not from "the first clip that happened to be indexed", which is what the
    old fallback did, and not from stats.npz, which records no such frame.

    Crab is the case that proves it: its declared rest pose is `__Walk.bvh`,
    not a T-pose, and alphabetically its first clip is `attack1`. A fallback to
    the first clip would describe the rig with an attack pose.
    """
    from poseydon.build.corpus import rest_action

    manifest = dataset._manifest("Crab")
    assert rest_action(manifest) == "walk"

    # Compare against an INDEPENDENTLY built frame, not against
    # `dataset._extract_rest` -- `_rest_frame_of` just caches that call, so
    # comparing the two would assert nothing at all.
    from poseydon.core.animation import RigidBodyAnimation
    from poseydon.core.skeleton import resolve
    from poseydon.features import extract_features
    from poseydon.features.reduce import apply_reduction

    anim = RigidBodyAnimation.load(CORPUS / "clips" / "Crab" / "walk.npz")
    reduced = apply_reduction(anim, dataset._reduction("Crab"))
    raw, spec = extract_features(reduced, resolve(manifest, reduced.names), FEATURES)
    expected = dataset._normalizer("Crab", spec).normalize(raw[:1])[0]

    np.testing.assert_allclose(dataset._rest_frame_of("Crab"), expected)

    # And it must differ from what the old first-clip fallback produced --
    # otherwise the test would pass even if nothing changed.
    first_clip = RigidBodyAnimation.load(CORPUS / "clips" / "Crab" / "attack1.npz")
    reduced_first = apply_reduction(first_clip, dataset._reduction("Crab"))
    raw_first, _ = extract_features(
        reduced_first, resolve(manifest, reduced_first.names), FEATURES
    )
    fallback = dataset._normalizer("Crab", spec).normalize(raw_first[:1])[0]
    assert not np.allclose(expected, fallback), "the old fallback was not distinguishable"


def test_every_rig_can_produce_a_rest_frame(dataset):
    """A1 authored `rest_pose:` into all 73 manifests so this never silently
    falls back. If any rig cannot, the corpus is the problem, not the reader.
    """
    rigs = sorted({record.skeleton for record in dataset.records})
    assert len(rigs) == 73
    for rig in rigs:
        assert dataset._rest_frame_of(rig).shape[-1] == dataset.spec.dim

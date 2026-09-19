"""The corpus index, moved to build/ and made forward-compatible."""

from __future__ import annotations

import json

import pytest

from poseydon.build.index import ClipRecord, CorpusIndex, action_slug, clip_id


def _record(**over) -> ClipRecord:
    base = {
        "clip_id": "Goat__walk", "skeleton": "Goat", "action": "walk", "split": "train",
        "n_frames": 40, "fps": 30.0, "path": "clips/Goat/walk.npz", "tags": ("quadruped",),
    }
    return ClipRecord(**{**base, **over})


def test_an_index_written_by_a_newer_build_still_loads(tmp_path):
    """A row carrying a field this version does not know must not break the
    loader -- otherwise a corpus built by a colleague on a newer branch is
    unreadable, and the failure is a TypeError about an unexpected keyword."""
    path = tmp_path / "index.jsonl"
    payload = {
        "clip_id": "Goat__walk", "skeleton": "Goat", "action": "walk",
        "split": "train", "n_frames": 40, "fps": 30.0,
        "path": "clips/Goat/walk.npz", "tags": ["quadruped"],
        "a_field_from_the_future": {"nested": True},
    }
    path.write_text(json.dumps(payload) + "\n")

    index = CorpusIndex.load(path)
    assert len(index) == 1
    assert index.records[0].clip_id == "Goat__walk"


def test_round_trip_through_a_file(tmp_path):
    index = CorpusIndex()
    index.add(_record())
    index.add(_record(clip_id="Goat__run", action="run"))
    path = tmp_path / "index.jsonl"
    index.save(path)

    back = CorpusIndex.load(path)
    assert [r.clip_id for r in back.records] == ["Goat__walk", "Goat__run"]
    assert back.records[0].tags == ("quadruped",)


def test_helpers_moved_with_it():
    assert action_slug("__Attack3") == "attack3"
    assert clip_id("Goat", "walk") == "Goat__walk"


def _three() -> CorpusIndex:
    index = CorpusIndex()
    index.add(_record())
    index.add(_record(clip_id="Goat__run", action="run"))
    index.add(_record(clip_id="Crow__fly", skeleton="Crow", action="fly"))
    return index


def test_without_drops_a_whole_skeleton():
    kept = _three().without(skeletons=["Goat"])
    assert [r.clip_id for r in kept.records] == ["Crow__fly"]


def test_without_drops_one_clip_and_keeps_its_siblings():
    kept = _three().without(clips=["Goat__run"])
    assert [r.clip_id for r in kept.records] == ["Goat__walk", "Crow__fly"]


def test_without_rejects_an_entry_that_matches_nothing():
    """A typo'd rig name must not pass silently: the run would train for days
    on clips the config said to drop, and nothing would say so."""
    with pytest.raises(ValueError, match="Chicken"):
        _three().without(skeletons=["Goat", "Chicken"])
    with pytest.raises(ValueError, match="Goat__fly"):
        _three().without(clips=["Goat__fly"])

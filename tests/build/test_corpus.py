"""Clip discovery and label files (parity spec §2, §4).

Labels are AUTHORED, not owned by the build: a rebuild must never destroy a
hand-written `text:` or a changed `split:`.
"""

from __future__ import annotations

import pytest
import yaml

from poseydon.build.corpus import PerRigDirectory, read_labels, write_labels


def _corpus(tmp_path):
    for rig, actions in (("Goat", ("walk", "idle")), ("Crab", ("walk",))):
        d = tmp_path / "clips" / rig
        d.mkdir(parents=True)
        for action in actions:
            (d / f"{action}.bvh").write_text("HIERARCHY\n")
            (d / f"{action}.fbx").write_text("not a bvh")
    return tmp_path


def test_rigs_are_directories_under_clips(tmp_path):
    root = _corpus(tmp_path)
    assert PerRigDirectory().rigs(root) == ["Crab", "Goat"]


def test_clips_match_the_pattern_only(tmp_path):
    root = _corpus(tmp_path)
    found = PerRigDirectory().clips(root, "Goat")
    assert [p.stem for p in found] == ["idle", "walk"]
    assert all(p.suffix == ".bvh" for p in found), "the .fbx must not be picked up"


def test_a_rig_with_no_matching_clip_yields_an_empty_list(tmp_path):
    root = _corpus(tmp_path)
    (root / "clips" / "Empty").mkdir()
    assert PerRigDirectory().clips(root, "Empty") == []


def test_labels_are_written_when_absent(tmp_path):
    path = tmp_path / "walk.yaml"
    write_labels(path, {"split": "train", "action": "walk"})
    assert yaml.safe_load(path.read_text()) == {"split": "train", "action": "walk"}


def test_an_existing_label_file_is_left_alone(tmp_path):
    path = tmp_path / "walk.yaml"
    path.write_text("split: val\ntext: a goat walking uphill\n")
    write_labels(path, {"split": "train", "action": "walk"})
    assert read_labels(path) == {"split": "val", "text": "a goat walking uphill"}


def test_relabel_refreshes_derived_keys_and_preserves_the_rest(tmp_path):
    path = tmp_path / "walk.yaml"
    path.write_text("split: val\ntext: a goat walking uphill\naction: stale\n")
    write_labels(path, {"action": "walk"}, relabel=True)

    back = read_labels(path)
    assert back["action"] == "walk", "a derived key must be refreshed"
    assert back["text"] == "a goat walking uphill", "an authored key must survive"
    assert back["split"] == "val", "a key the build did not write must survive"


def test_reading_an_absent_label_file_gives_an_empty_mapping(tmp_path):
    assert read_labels(tmp_path / "nope.yaml") == {}


def test_a_malformed_label_file_is_refused_rather_than_ignored(tmp_path):
    path = tmp_path / "walk.yaml"
    path.write_text("- this is a list, not a mapping\n")
    with pytest.raises(ValueError, match="mapping"):
        read_labels(path)

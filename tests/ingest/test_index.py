import pytest

from poseydon.ingest.index import ClipRecord, CorpusIndex, action_slug, clip_id


def record(clip="Goat__head_butt", skeleton="Goat", action="head_butt", split="train"):
    return ClipRecord(
        clip_id=clip,
        skeleton=skeleton,
        action=action,
        split=split,
        n_frames=79,
        fps=24.0,
        path=f"aligned/{clip}.npz",
        tags=("quadruped",),
    )


@pytest.mark.parametrize(
    ("stem", "expected"),
    [
        ("Goat___HeadButt_395", "goat_headbutt_395"),
        ("Flamingo_Flamingo_OneLEgBEnt_353", "flamingo_flamingo_onelegbent_353"),
        ("Male Sitting Pose", "male_sitting_pose"),
        ("__leading__", "leading"),
        ("a---b", "a_b"),
    ],
)
def test_action_slug(stem, expected):
    assert action_slug(stem) == expected


def test_clip_id_uses_double_underscore_separator():
    assert clip_id("Goblin_m", "walking") == "Goblin_m__walking"


def test_clip_id_rejects_reserved_separator_in_skeleton():
    with pytest.raises(ValueError, match="__"):
        clip_id("Bad__Name", "walking")


def test_round_trip_jsonl(tmp_path):
    index = CorpusIndex()
    index.add(record())
    index.add(record(clip="Coyote__attack", skeleton="Coyote", action="attack"))
    path = tmp_path / "index.jsonl"
    index.save(path)

    back = CorpusIndex.load(path)
    assert len(back) == 2
    assert back.records[0] == index.records[0]
    assert back.records[1].skeleton == "Coyote"


def test_saved_file_is_one_json_object_per_line(tmp_path):
    index = CorpusIndex()
    index.add(record())
    index.add(record(clip="B__b", skeleton="B", action="b"))
    path = tmp_path / "index.jsonl"
    index.save(path)
    lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
    assert len(lines) == 2
    assert all(ln.startswith("{") and ln.endswith("}") for ln in lines)


def test_duplicate_clip_id_is_rejected():
    index = CorpusIndex()
    index.add(record())
    with pytest.raises(ValueError, match="duplicate"):
        index.add(record())


def test_query_filters_independently():
    index = CorpusIndex()
    index.add(record())
    index.add(record(clip="Coyote__attack", skeleton="Coyote", action="attack"))
    index.add(record(clip="Goat__walk", action="walk", split="test"))

    assert len(index.query(skeleton="Goat")) == 2
    assert len(index.query(split="test")) == 1
    assert len(index.query(action="attack")) == 1
    assert len(index.query(skeleton="Goat", split="train")) == 1
    assert len(index.query(tag="quadruped")) == 3
    assert len(index.query(tag="flying")) == 0


def test_retargeting_pairs_are_a_query_not_a_file():
    # Same action across different skeletons -- what the reference materialized
    # into txt files with a dedicated build script.
    index = CorpusIndex()
    index.add(record(clip="Goat__walk", action="walk"))
    index.add(record(clip="Coyote__walk", skeleton="Coyote", action="walk"))
    index.add(record(clip="Goat__attack", action="attack"))

    walkers = index.query(action="walk")
    assert {r.skeleton for r in walkers} == {"Goat", "Coyote"}


def test_skeletons_are_sorted_and_unique():
    index = CorpusIndex()
    index.add(record(clip="Z__a", skeleton="Zebra"))
    index.add(record(clip="A__a", skeleton="Ant"))
    index.add(record(clip="A__b", skeleton="Ant", action="b"))
    assert index.skeletons() == ["Ant", "Zebra"]

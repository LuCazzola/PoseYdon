import numpy as np
import pytest

from poseydon.core.anim import Anim
from poseydon.core.skeleton import ManifestError, SkeletonManifest, resolve
from poseydon.ingest.index import CorpusIndex
from poseydon.ingest.pipeline import ingest_clip, ingest_corpus
from poseydon.io.bvh import load_bvh
from tests.ingest.manifest_helper import MANIFEST_DIR, STEM_TO_SKELETON


def manifest_for(stem):
    return SkeletonManifest.load(MANIFEST_DIR / f"{STEM_TO_SKELETON[stem]}.yaml")


def test_ingest_clip_writes_aligned_anim(bvh_fixture, tmp_path):
    skeleton = STEM_TO_SKELETON[bvh_fixture.stem]
    record = ingest_clip(bvh_fixture, manifest_for(bvh_fixture.stem), tmp_path)

    written = tmp_path / record.path
    assert written.is_file()

    anim = Anim.load(written)
    assert anim.n_frames == record.n_frames
    assert anim.global_positions()[..., 1].min() == pytest.approx(0.0, abs=1e-9)
    assert record.skeleton == skeleton
    assert record.clip_id.startswith(f"{skeleton}__")


def test_ingest_never_chunks(bvh_fixture, tmp_path):
    # One BVH in, exactly one Anim out, at full length.
    record = ingest_clip(bvh_fixture, manifest_for(bvh_fixture.stem), tmp_path)
    assert record.n_frames == load_bvh(bvh_fixture).n_frames
    assert len(list(tmp_path.rglob("*.npz"))) == 1


def test_ingest_is_idempotent(bvh_fixture, tmp_path):
    manifest = manifest_for(bvh_fixture.stem)
    first = ingest_clip(bvh_fixture, manifest, tmp_path)
    second = ingest_clip(bvh_fixture, manifest, tmp_path)

    assert first.clip_id == second.clip_id
    assert first.path == second.path
    assert len(list(tmp_path.rglob("*.npz"))) == 1


def test_ingest_corpus_builds_index(truebones_dir, tmp_path):
    paths = sorted(truebones_dir.glob("*.bvh"))
    result = ingest_corpus(
        paths,
        MANIFEST_DIR,
        tmp_path,
        skeleton_of=lambda p: STEM_TO_SKELETON[p.stem],
    )

    assert len(result.index) == 7
    assert result.skipped == []
    assert result.index.skeletons() == sorted(set(STEM_TO_SKELETON.values()))

    index_path = tmp_path / "index.jsonl"
    result.index.save(index_path)
    assert len(CorpusIndex.load(index_path)) == 7


def test_ingest_corpus_reports_failures_without_aborting(truebones_dir, tmp_path):
    paths = sorted(truebones_dir.glob("*.bvh"))
    broken = tmp_path / "broken.bvh"
    broken.write_text("HIERARCHY\nROOT a\n{\nOFFSET 0 0 0\n}\n")

    result = ingest_corpus(
        [*paths, broken],
        MANIFEST_DIR,
        tmp_path / "out",
        skeleton_of=lambda p: STEM_TO_SKELETON.get(p.stem, "Goat"),
    )

    assert len(result.index) == 7
    assert len(result.skipped) == 1
    assert result.skipped[0][0] == broken


def test_manifest_facing_joints_exist_in_every_fixture(bvh_fixture):
    resolved = resolve(manifest_for(bvh_fixture.stem), load_bvh(bvh_fixture).names)
    assert len(resolved.facing_indices) == 2


def test_unknown_joint_name_fails_with_suggestion(truebones_dir, tmp_path):
    bad = tmp_path / "Bad.yaml"
    bad.write_text(
        "skeleton: Bad\n"
        "facing:\n"
        "  hips:      {right: Bip01_R_Thig, left: Bip01_L_Thigh}\n"
        "  shoulders: {right: Bip01_R_UpperArm, left: Bip01_L_UpperArm}\n"
        "scale: {mean_bone_length: 0.2}\n"
        "contact: {max_height: 0.3, max_speed: 0.045}\n"
    )
    manifest = SkeletonManifest.load(bad)
    with pytest.raises(ManifestError, match="Bip01_R_Thigh"):
        ingest_clip(truebones_dir / "Goat___HeadButt_395.bvh", manifest, tmp_path)


def test_rejects_fps_mismatch(truebones_dir, tmp_path):
    text = (MANIFEST_DIR / "Goat.yaml").read_text().replace("fps: null", "fps: 20")
    path = tmp_path / "Goat20.yaml"
    path.write_text(text.replace("skeleton: Goat", "skeleton: Goat20"))
    manifest = SkeletonManifest.load(path)

    with pytest.raises(ValueError, match="resampl"):
        ingest_clip(truebones_dir / "Goat___HeadButt_395.bvh", manifest, tmp_path)


def test_infer_skeleton_from_filename_prefix(truebones_dir):
    from poseydon.ingest.pipeline import infer_skeleton

    # No `__` anywhere in this stem -- the `__` convention describes ids we
    # generate, not source filenames we are given.
    path = truebones_dir / "Flamingo_Flamingo_OneLEgBEnt_353.bvh"
    assert infer_skeleton(path, MANIFEST_DIR) == "Flamingo"
    assert infer_skeleton(truebones_dir / "Goat___HeadButt_395.bvh", MANIFEST_DIR) == "Goat"


def test_infer_skeleton_prefers_containing_directory(tmp_path):
    from poseydon.ingest.pipeline import infer_skeleton

    nested = tmp_path / "Coyote"
    nested.mkdir()
    clip = nested / "some_unrelated_name.bvh"
    clip.touch()
    assert infer_skeleton(clip, MANIFEST_DIR) == "Coyote"


def test_infer_skeleton_prefers_the_longest_matching_name(tmp_path):
    from poseydon.ingest.pipeline import infer_skeleton

    for name in ("Goat", "GoatKid"):
        (tmp_path / f"{name}.yaml").write_text("skeleton: x\n")
    assert infer_skeleton(tmp_path / "GoatKid_walk.bvh", tmp_path) == "GoatKid"
    assert infer_skeleton(tmp_path / "Goat_walk.bvh", tmp_path) == "Goat"


def test_infer_skeleton_reports_known_names_when_it_cannot_tell(tmp_path):
    from poseydon.ingest.pipeline import infer_skeleton

    with pytest.raises(ValueError, match="Known:"):
        infer_skeleton(tmp_path / "mystery.bvh", MANIFEST_DIR)


def test_ingest_corpus_without_explicit_mapping(truebones_dir, tmp_path):
    result = ingest_corpus(sorted(truebones_dir.glob("*.bvh")), MANIFEST_DIR, tmp_path)
    assert len(result.index) == 7
    assert result.skipped == []


def test_infer_skeleton_honours_the_double_underscore_convention(tmp_path):
    from poseydon.ingest.pipeline import infer_skeleton

    # <Skeleton>__<whatever> is PoseYdon's dataset-sample convention and wins
    # over directory name and prefix matching.
    clip = tmp_path / "Coyote__anything_at_all.bvh"
    clip.touch()
    assert infer_skeleton(clip, MANIFEST_DIR) == "Coyote"


def test_double_underscore_convention_beats_a_longer_prefix_match(tmp_path):
    from poseydon.ingest.pipeline import infer_skeleton

    for name in ("Goat", "GoatKid"):
        (tmp_path / f"{name}.yaml").write_text("skeleton: x\n")
    clip = tmp_path / "Goat__kid_walk.bvh"
    clip.touch()
    assert infer_skeleton(clip, tmp_path) == "Goat"


def test_corpus_shares_alignment_params_across_a_skeletons_clips(truebones_dir, tmp_path):
    # Two clips of one skeleton must be aligned with the SAME constants, so their
    # relative height survives. Ingesting a clip twice under different names must
    # therefore give byte-identical geometry.
    from poseydon.core.anim import Anim

    src = truebones_dir / "Goat___HeadButt_395.bvh"
    a = tmp_path / "Goat__one.bvh"
    b = tmp_path / "Goat__two.bvh"
    a.write_text(src.read_text())
    b.write_text(src.read_text())

    result = ingest_corpus([a, b], MANIFEST_DIR, tmp_path / "out")
    assert len(result.index) == 2

    first, second = (Anim.load(tmp_path / "out" / r.path) for r in result.index.records)
    np.testing.assert_allclose(first.global_positions(), second.global_positions(), atol=0)


def test_ingest_does_not_reground_an_already_aligned_clip(truebones_dir, tmp_path):
    # The shipped assets are already-processed BVHs whose lowest joint is NOT at
    # y=0, because their ground height came from a T-pose. Ingesting them with
    # params derived from the same clip re-grounds them; that is expected. What
    # must not happen is a change of scale beyond renormalizing the file's own
    # rounding: BVH stores offsets to 6 decimals, so an already-scaled asset has
    # a mean bone length off by ~8e-8, and rescaling corrects exactly that.
    record = ingest_clip(
        truebones_dir / "Goat___HeadButt_395.bvh",
        manifest_for("Goat___HeadButt_395"),
        tmp_path,
    )
    from poseydon.core.anim import Anim

    anim = Anim.load(tmp_path / record.path)
    source = load_bvh(truebones_dir / "Goat___HeadButt_395.bvh")
    np.testing.assert_allclose(
        np.linalg.norm(anim.offsets[1:], axis=-1),
        np.linalg.norm(source.offsets[1:], axis=-1),
        rtol=1e-6,
    )

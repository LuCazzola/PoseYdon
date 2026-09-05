import shutil
from pathlib import Path

import numpy as np
import pytest
from scripts.create_truebones_dataset import clean_species_clips, resolve_rest_anim

from poseydon.core.anim import Anim
from poseydon.core.rotations import QUAT_IDENTITY
from poseydon.datasets.raw_bvh import load_raw_biped_bvh
from poseydon.ingest.pipeline import ingest_corpus
from poseydon.io.bvh import load_bvh

REAL_RAW_ROOT = Path(__file__).resolve().parents[2] / "data" / "truebones" / "Truebone_Z-OO"
MANIFEST_DIR = Path(__file__).resolve().parents[2] / "data" / "truebones" / "skeletons"

SAMPLE_CLIPS = {
    "Goat": "__HeadButt.bvh",
    "Crab": "__Attack3.bvh",
}


def _skip_if_raw_dump_missing():
    if not REAL_RAW_ROOT.is_dir():
        pytest.skip(f"raw Truebones dump not found at {REAL_RAW_ROOT}")


@pytest.fixture
def small_raw_root(tmp_path):
    _skip_if_raw_dump_missing()
    raw_root = tmp_path / "raw"
    for species, filename in SAMPLE_CLIPS.items():
        dest_dir = raw_root / species
        dest_dir.mkdir(parents=True)
        shutil.copy(REAL_RAW_ROOT / species / filename, dest_dir / filename)
    return raw_root


def test_clean_species_clips_produces_loadable_bvh(tmp_path, small_raw_root):
    scratch_root = tmp_path / "scratch"
    written = clean_species_clips("Goat", small_raw_root, scratch_root)

    assert len(written) == 1
    assert written[0] == scratch_root / "Goat" / "__HeadButt.bvh"
    anim = load_bvh(written[0])  # must round-trip through the STRICT loader
    assert isinstance(anim, Anim)
    assert anim.names[0] == "Bip01_Pelvis"


def test_resolve_rest_anim_prefers_a_t_pose_file():
    _skip_if_raw_dump_missing()
    # BrownBear has a raw T-pose file (BrownBear/__Tpose.bvh).
    rest_anim = resolve_rest_anim("BrownBear", REAL_RAW_ROOT)
    assert rest_anim.n_frames == 1
    assert rest_anim.names[0] == "Bip01_Pelvis"  # cleaned via load_raw_biped_bvh, so already merged


def test_resolve_rest_anim_falls_back_to_the_first_clips_first_frame():
    _skip_if_raw_dump_missing()
    # Goat has no raw T-pose file at all.
    assert not list((REAL_RAW_ROOT / "Goat").glob("*[Tt][Pp][Oo][Ss][Ee]*"))
    rest_anim = resolve_rest_anim("Goat", REAL_RAW_ROOT)
    assert rest_anim.n_frames == 1

    first_clip = sorted((REAL_RAW_ROOT / "Goat").glob("*.bvh"))[0]
    expected = load_raw_biped_bvh(first_clip)
    np.testing.assert_allclose(rest_anim.rotations[0], expected.rotations[0])


def test_clean_species_clips_preserves_positions_and_zeroes_the_rest_frame(tmp_path):
    _skip_if_raw_dump_missing()
    scratch_root = tmp_path / "scratch"
    written = clean_species_clips("Goat", REAL_RAW_ROOT, scratch_root)

    # Goat's rest fallback is its alphabetically-first raw clip -- that
    # clip's own cleaned frame 0 must now read as identity for every joint.
    first_clip_name = sorted((REAL_RAW_ROOT / "Goat").glob("*.bvh"))[0].name
    rest_clip = next(p for p in written if p.name == first_clip_name)
    rest_cleaned = load_bvh(rest_clip)
    np.testing.assert_allclose(
        rest_cleaned.rotations[0],
        np.broadcast_to(QUAT_IDENTITY, (rest_cleaned.n_joints, 4)),
        atol=1e-5,
    )

    # Every clip of a skeleton reuses the SAME (rest-pose) offsets -- they
    # are a skeleton-level constant, never re-rotated per clip -- matching
    # the reference's own compute_rots_from_tpos, which does the same.
    other_clip = next(p for p in written if p.name == "__HeadButt.bvh")
    after = load_bvh(other_clip)
    np.testing.assert_allclose(after.offsets, rest_cleaned.offsets, atol=1e-5)


def test_cleaned_clips_ingest_through_the_unmodified_pipeline(tmp_path, small_raw_root):
    scratch_root = tmp_path / "scratch"
    out_dir = tmp_path / "out"
    all_clean = []
    for species in SAMPLE_CLIPS:
        all_clean.extend(clean_species_clips(species, small_raw_root, scratch_root))

    result = ingest_corpus(all_clean, MANIFEST_DIR, out_dir, split="train")

    assert not result.skipped, result.skipped
    assert {record.skeleton for record in result.index.records} == {"Goat", "Crab"}
    for written_path in result.written:
        Anim.load(written_path)  # the aligned npz must itself be loadable

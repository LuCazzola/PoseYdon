import shutil
from pathlib import Path

import numpy as np
import pytest
from scripts.create_truebones_dataset import clean_species_clips, resolve_rest_anim

from poseydon.core.anim import Anim
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
    first_clip = sorted((REAL_RAW_ROOT / "Goat").glob("*.bvh"))[0]

    rest_anim = resolve_rest_anim("Goat", REAL_RAW_ROOT)
    assert rest_anim.n_frames == 1
    assert rest_anim.names == load_raw_biped_bvh(first_clip).names

    # resolve_rest_anim fits clean rotations/offsets via IK (establish_rest_pose)
    # rather than using the fallback file's raw declared rotations directly --
    # so its own global positions should closely match that file's TRUE
    # (rotation+translation) geometry, not equal its raw rotations exactly.
    from poseydon.datasets.raw_bvh import _damaged_global, _parse_and_merge

    names, parents, offsets, rotations, positions, fps = _parse_and_merge(first_clip)
    true_positions, _ = _damaged_global(rotations[:1], positions[:1], parents)
    scale = np.linalg.norm(offsets[1:], axis=1).mean()

    fitted_positions = rest_anim.global_positions()
    error = np.linalg.norm(fitted_positions[0] - true_positions[0], axis=-1) / scale
    assert np.median(error) < 0.05


def test_clean_species_clips_reuses_the_rest_poses_offsets_for_every_clip(tmp_path):
    _skip_if_raw_dump_missing()
    scratch_root = tmp_path / "scratch"
    written = clean_species_clips("Goat", REAL_RAW_ROOT, scratch_root)

    # Every clip of a skeleton reuses the SAME (rest-pose) offsets -- they
    # are a skeleton-level constant, never re-rotated per clip -- matching
    # the reference's own compute_rots_from_tpos, which does the same.
    rest_anim = resolve_rest_anim("Goat", REAL_RAW_ROOT)
    other_clip = next(p for p in written if p.name == "__HeadButt.bvh")
    after = load_bvh(other_clip)
    np.testing.assert_allclose(after.offsets, rest_anim.offsets, atol=1e-5)


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

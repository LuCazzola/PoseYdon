import shutil
from pathlib import Path

import pytest
from scripts.create_truebones_dataset import clean_species_clips

from poseydon.core.anim import Anim
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

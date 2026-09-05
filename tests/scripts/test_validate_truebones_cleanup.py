from pathlib import Path

import pytest

from scripts.validate_truebones_cleanup import compare_clip

RAW_ROOT = Path(__file__).resolve().parents[2] / "data" / "truebones" / "Truebone_Z-OO"
FIXTURE_ROOT = (
    Path(__file__).resolve().parents[2]
    / "external"
    / "neural_motion_blending"
    / "assets"
    / "truebones"
)


def _skip_if_data_missing():
    if not RAW_ROOT.is_dir():
        pytest.skip(f"raw Truebones dump not found at {RAW_ROOT}")
    if not FIXTURE_ROOT.is_dir():
        pytest.skip(f"fixture BVHs not found at {FIXTURE_ROOT}")


def test_compare_clip_returns_a_per_block_error_table():
    _skip_if_data_missing()
    report = compare_clip("Goat", "Goat/__HeadButt.bvh", "Goat___HeadButt_395")

    assert set(report) >= {"ric_pos", "rot6d"}
    for block_errors in report.values():
        assert block_errors["max_abs_error"] >= block_errors["mean_abs_error"] >= 0.0

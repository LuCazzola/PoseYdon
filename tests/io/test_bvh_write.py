import numpy as np
import pytest

from poseydon.core.rotations import quat_to_matrix
from poseydon.io.bvh import load_bvh, save_bvh


def test_round_trip_preserves_structure(bvh_fixture, tmp_path):
    original = load_bvh(bvh_fixture)
    out = tmp_path / "round_trip.bvh"
    save_bvh(original, out)
    back = load_bvh(out)

    assert back.names == original.names
    np.testing.assert_array_equal(back.parents, original.parents)
    np.testing.assert_allclose(back.offsets, original.offsets, atol=1e-6)
    assert back.n_frames == original.n_frames
    # Frame Time is written with 6 decimals, so fps survives only approximately:
    # 1/24 -> "0.041667" -> 23.99981...
    assert back.fps == pytest.approx(original.fps, rel=1e-4)


def test_round_trip_preserves_rotations_as_matrices(bvh_fixture, tmp_path):
    # Euler angles are not unique, so compare the rotations they represent.
    original = load_bvh(bvh_fixture)
    out = tmp_path / "round_trip.bvh"
    save_bvh(original, out)
    back = load_bvh(out)

    np.testing.assert_allclose(
        quat_to_matrix(back.rotations), quat_to_matrix(original.rotations), atol=1e-6
    )


def test_round_trip_preserves_root_trajectory(bvh_fixture, tmp_path):
    original = load_bvh(bvh_fixture)
    out = tmp_path / "round_trip.bvh"
    save_bvh(original, out)
    back = load_bvh(out)

    np.testing.assert_allclose(back.root_pos, original.root_pos, atol=1e-5)


def test_written_file_keeps_end_site_names(bvh_fixture, tmp_path):
    original = load_bvh(bvh_fixture)
    out = tmp_path / "round_trip.bvh"
    save_bvh(original, out)

    text = out.read_text()
    assert "End Site #name:" in text
    assert text.startswith("HIERARCHY")

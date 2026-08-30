"""The three ways to rebuild a skeleton from features."""

import numpy as np
import pytest

from poseydon.features import RECONSTRUCTORS, extract_features, reconstruct
from poseydon.io.bvh import load_bvh
from tests.ingest.manifest_helper import resolved_for

METHODS = ["fk", "positions", "positions_ik"]


def clip(bvh_path, frames=8):
    anim = load_bvh(bvh_path)
    array, spec = extract_features(anim, resolved_for(bvh_path, anim))
    return anim, array[:frames], spec


def test_all_three_are_registered():
    assert RECONSTRUCTORS.names() == sorted(METHODS)


@pytest.mark.parametrize("method", METHODS)
def test_every_method_returns_usable_positions(bvh_fixture, method):
    anim, array, spec = clip(bvh_fixture)
    kwargs = {"iterations": 5} if method == "positions_ik" else {}
    positions, _ = reconstruct(method, array, spec, anim, **kwargs)

    assert positions.shape == (array.shape[0], anim.n_joints, 3)
    assert np.isfinite(positions).all()


def test_positions_alone_cannot_produce_rotations(bvh_fixture):
    anim, array, spec = clip(bvh_fixture)
    _, produced = reconstruct("positions", array, spec, anim)
    assert produced is None, "a BVH stores rotations; this path has none"


def test_positions_reconstructor_says_why_it_has_no_anim(bvh_fixture):
    anim, array, spec = clip(bvh_fixture)
    with pytest.raises(NotImplementedError, match="BVH"):
        RECONSTRUCTORS.get("positions")().anim(array, spec, anim)


@pytest.mark.parametrize("method", ["fk", "positions_ik"])
def test_rigid_methods_preserve_bone_lengths_exactly(bvh_fixture, method):
    # Both parameterize by rotation, so bone lengths cannot drift.
    anim, array, spec = clip(bvh_fixture)
    kwargs = {"iterations": 5} if method == "positions_ik" else {}
    positions, _ = reconstruct(method, array, spec, anim, **kwargs)

    lengths = np.linalg.norm(positions[:, 1:] - positions[:, anim.parents[1:]], axis=-1)
    rest = np.linalg.norm(anim.offsets[1:], axis=-1)[None]
    np.testing.assert_allclose(lengths, np.broadcast_to(rest, lengths.shape), atol=1e-4)


def test_rotation_and_position_paths_agree_on_real_data(bvh_fixture):
    # A real clip has consistent rotations and positions, so these two describe
    # the same skeleton exactly. They diverge only on generated output.
    anim, array, spec = clip(bvh_fixture)
    from_fk, _ = reconstruct("fk", array, spec, anim)
    from_positions, _ = reconstruct("positions", array, spec, anim)
    np.testing.assert_allclose(from_positions, from_fk, atol=1e-9)


def test_ik_tracks_closely_but_cannot_be_exact(bvh_fixture):
    # Rotation about a bone's own axis does not move its children, so twist is
    # invisible to a solver fitting joint positions. IK therefore converges to a
    # plateau rather than to zero: on these clips the mean error settles near 1%
    # of a bone length while the worst joint stays around 15%.
    anim, array, spec = clip(bvh_fixture)
    reference, _ = reconstruct("fk", array, spec, anim)
    solved, _ = reconstruct("positions_ik", array, spec, anim, iterations=300)

    bone = float(np.linalg.norm(anim.offsets[1:], axis=-1).mean())
    error = np.abs(solved - reference)
    assert error.mean() < 0.05 * bone, f"mean {error.mean():.4f} vs bone {bone:.4f}"
    assert error.max() < 0.5 * bone


def test_unknown_method_is_rejected(bvh_fixture):
    anim, array, spec = clip(bvh_fixture)
    with pytest.raises(KeyError, match="positions"):
        reconstruct("postions", array, spec, anim)

"""Features back to an animation, and out to BVH."""

import numpy as np
import pytest

from poseydon.core.rotations import quat_to_matrix
from poseydon.core.spec import FeatureSpec
from poseydon.features import RecoveryError, extract_features, features_to_anim
from poseydon.io.bvh import load_bvh, save_bvh
from tests.ingest.manifest_helper import resolved_for


def round_trip(bvh_path, features=None):
    anim = load_bvh(bvh_path)
    resolved = resolved_for(bvh_path, anim)
    array, spec = extract_features(anim, resolved, features or ("ric_pos", "rot6d", "local_vel", "foot_contact"))
    return anim, array, spec, features_to_anim(array, spec, anim)


def test_rotations_recover_exactly(bvh_fixture):
    # Exact, not fitted: the rotations are in the representation, so there is
    # nothing to solve for.
    anim, _, _, back = round_trip(bvh_fixture)
    np.testing.assert_allclose(
        quat_to_matrix(back.rotations),
        quat_to_matrix(anim.rotations[: back.n_frames]),
        atol=1e-12,
    )


def test_root_height_recovers_exactly(bvh_fixture):
    anim, _, _, back = round_trip(bvh_fixture)
    original = anim.global_positions()[: back.n_frames, 0]
    np.testing.assert_allclose(back.root_pos[:, 1], original[:, 1], atol=1e-9)


def test_root_path_shape_recovers_exactly_up_to_placement(bvh_fixture):
    # Absolute horizontal position is discarded on purpose -- that is what
    # root-invariant means -- so the trajectory returns starting at the origin
    # and only its SHAPE can be checked.
    anim, _, _, back = round_trip(bvh_fixture)
    original = anim.global_positions()[: back.n_frames, 0]
    np.testing.assert_allclose(
        back.root_pos[:, [0, 2]] - back.root_pos[0, [0, 2]],
        original[:, [0, 2]] - original[0, [0, 2]],
        atol=1e-9,
    )


def test_recovered_trajectory_starts_at_the_origin(bvh_fixture):
    _, _, _, back = round_trip(bvh_fixture)
    np.testing.assert_allclose(back.root_pos[0, [0, 2]], [0.0, 0.0], atol=1e-12)


def test_pose_survives_the_round_trip(bvh_fixture):
    # Joint positions relative to the root -- the pose itself -- must be exact.
    anim, _, _, back = round_trip(bvh_fixture)
    recovered = back.global_positions()
    original = anim.global_positions()[: back.n_frames]
    np.testing.assert_allclose(
        recovered - recovered[:, :1], original - original[:, :1], atol=1e-9
    )


def test_skeleton_is_carried_over_unchanged(bvh_fixture):
    anim, _, _, back = round_trip(bvh_fixture)
    assert back.names == anim.names
    np.testing.assert_array_equal(back.parents, anim.parents)
    np.testing.assert_allclose(back.offsets, anim.offsets, atol=0)


def test_recovered_animation_writes_valid_bvh(bvh_fixture, tmp_path):
    _, _, _, back = round_trip(bvh_fixture)
    out = tmp_path / "recovered.bvh"
    save_bvh(back, out)

    reloaded = load_bvh(out)
    assert reloaded.names == back.names
    np.testing.assert_allclose(
        quat_to_matrix(reloaded.rotations), quat_to_matrix(back.rotations), atol=1e-6
    )


def test_recovery_needs_rotations(bvh_fixture):
    anim = load_bvh(bvh_fixture)
    array, spec = extract_features(anim, resolved_for(bvh_fixture, anim), ["ric_pos"])
    with pytest.raises(RecoveryError, match="rot6d"):
        features_to_anim(array, spec, anim)


def test_recovery_needs_velocity_for_the_trajectory(bvh_fixture):
    anim = load_bvh(bvh_fixture)
    array, spec = extract_features(anim, resolved_for(bvh_fixture, anim), ["rot6d"])
    with pytest.raises(RecoveryError, match="local_vel"):
        features_to_anim(array, spec, anim)


def test_mismatched_skeleton_is_rejected(bvh_fixture):
    anim, array, spec, _ = round_trip(bvh_fixture)
    with pytest.raises(RecoveryError, match="joints"):
        features_to_anim(array[:, :-1], spec, anim)


def test_mismatched_width_is_rejected(bvh_fixture):
    anim, array, _, _ = round_trip(bvh_fixture)
    with pytest.raises(RecoveryError, match="wide"):
        features_to_anim(array, FeatureSpec((("rot6d", 6),)), anim)

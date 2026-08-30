"""The golden gate: features from the shipped BVHs must reproduce the .npy files.

The seven assets are the reference's own processed output -- BVH and feature
array for the same clip -- so this compares PoseYdon's whole extraction path
against the published implementation.

Tolerance is bounded by the file format, not by float64. BVH stores six
decimals, so an already-scaled asset's mean bone length is off from target by
~8e-8; compounded through forward kinematics over up to 63 joints, agreement to
~1e-5 is the floor. Tightening below that would measure rounding, not
correctness.
"""

import numpy as np
import pytest

from poseydon.core.spec import FeatureSpec
from poseydon.features import extract_features
from poseydon.io.bvh import load_bvh
from tests.ingest.manifest_helper import resolved_for

GOLDEN_ATOL = 1e-5

# The reference's fixed 13-dim layout, which PoseYdon reproduces when asked for
# these four blocks in this order.
REFERENCE_LAYOUT = (("ric_pos", 3), ("rot6d", 6), ("local_vel", 3), ("foot_contact", 1))


def golden(bvh_path):
    return np.load(bvh_path.with_suffix(".npy"))


def extracted(bvh_path):
    anim = load_bvh(bvh_path)
    return extract_features(anim, resolved_for(bvh_path, anim))


def test_shape_matches_golden(bvh_fixture):
    array, spec = extracted(bvh_fixture)
    assert array.shape == golden(bvh_fixture).shape
    assert spec.dim == 13


def test_layout_matches_the_reference(bvh_fixture):
    _, spec = extracted(bvh_fixture)
    assert spec.blocks == REFERENCE_LAYOUT


def test_frame_count_is_one_less_than_the_animation(bvh_fixture):
    array, _ = extracted(bvh_fixture)
    assert array.shape[0] == load_bvh(bvh_fixture).n_frames - 1


@pytest.mark.parametrize("block", ["ric_pos", "rot6d", "local_vel", "foot_contact"])
def test_each_block_matches_golden(bvh_fixture, block):
    array, spec = extracted(bvh_fixture)
    where = spec.slice(block)
    np.testing.assert_allclose(
        array[..., where], golden(bvh_fixture)[..., where], atol=GOLDEN_ATOL
    )


def test_foot_contact_matches_golden_exactly(bvh_fixture):
    # Binary, so it is either right or it is not; no tolerance applies.
    array, spec = extracted(bvh_fixture)
    np.testing.assert_array_equal(
        array[..., spec.slice("foot_contact")],
        golden(bvh_fixture)[..., spec.slice("foot_contact")],
    )


def test_whole_tensor_matches_golden(bvh_fixture):
    array, _ = extracted(bvh_fixture)
    np.testing.assert_allclose(array, golden(bvh_fixture), atol=GOLDEN_ATOL)


def test_selecting_a_subset_matches_the_same_columns(bvh_fixture):
    # Composability is only real if a subset gives bit-identical values to the
    # corresponding slice of the full representation.
    anim = load_bvh(bvh_fixture)
    resolved = resolved_for(bvh_fixture, anim)
    full, full_spec = extract_features(anim, resolved)
    subset, subset_spec = extract_features(anim, resolved, ["rot6d", "local_vel"])

    assert subset_spec == FeatureSpec((("rot6d", 6), ("local_vel", 3)))
    np.testing.assert_array_equal(
        subset[..., subset_spec.slice("rot6d")], full[..., full_spec.slice("rot6d")]
    )
    np.testing.assert_array_equal(
        subset[..., subset_spec.slice("local_vel")], full[..., full_spec.slice("local_vel")]
    )


def test_whole_frame_features_keep_every_frame(bvh_fixture):
    anim = load_bvh(bvh_fixture)
    array, _ = extract_features(anim, resolved_for(bvh_fixture, anim), ["ric_pos", "rot6d"])
    assert array.shape[0] == anim.n_frames

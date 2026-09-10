"""The model's contact CLAIM, as distinct from contact re-derived from geometry."""

from __future__ import annotations

import numpy as np

from poseydon.core.spec import FeatureSpec
from poseydon.features import contact_flags_from

SPEC = FeatureSpec((("ric_pos", 3), ("rot6d", 6), ("local_vel", 3), ("foot_contact", 1)))


def test_the_channel_becomes_a_flag_at_the_midpoint():
    features = np.zeros((4, 2, SPEC.dim))
    features[..., SPEC.slice("foot_contact")] = np.array(
        [[0.9, 0.1], [0.5, 0.51], [0.0, 1.0], [0.49, 0.6]]
    )[..., None]
    flags = contact_flags_from(features, SPEC)
    np.testing.assert_array_equal(
        flags, [[True, False], [False, True], [False, True], [False, True]]
    )


def test_a_spec_without_the_block_returns_none():
    """A recipe may legitimately omit it, and a caller should not have to ask."""
    spec = FeatureSpec((("ric_pos", 3), ("rot6d", 6)))
    assert contact_flags_from(np.zeros((3, 2, spec.dim)), spec) is None


def test_the_threshold_is_adjustable():
    features = np.zeros((1, 3, SPEC.dim))
    features[..., SPEC.slice("foot_contact")] = np.array([[0.2, 0.6, 0.95]])[..., None]
    strict = contact_flags_from(features, SPEC, threshold=0.9)
    np.testing.assert_array_equal(strict, [[False, False, True]])

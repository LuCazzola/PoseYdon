"""`FeatureSpec.take`: the block layout is the spec's business, not the caller's.

The spec is constructed with the ordering, so nothing downstream should have to
remember it. Two things were being remembered by hand at 35 call sites, and both
are silent when wrong: which slice a name maps to, and which AXIS holds the
channels -- which is not the same axis everywhere in this repo.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from poseydon.core.spec import FeatureSpec

SPEC = FeatureSpec((("ric_pos", 3), ("rot6d", 6), ("local_vel", 3), ("foot_contact", 1)))


def test_it_reads_the_named_block_from_the_last_axis_by_default():
    features = np.arange(2 * 4 * SPEC.dim, dtype=float).reshape(2, 4, SPEC.dim)
    np.testing.assert_array_equal(SPEC.take(features, "rot6d"), features[..., 3:9])


def test_the_axis_is_explicit_because_it_differs_by_layout():
    """A MotionBatch is (B, J, D, T): channels at 2, frames last.

    A helper that assumed the last axis would read FRAMES as channels here --
    silently, and with plausibly-shaped output.
    """
    batch = torch.arange(2 * 4 * SPEC.dim * 5, dtype=torch.float32).reshape(
        2, 4, SPEC.dim, 5
    )
    torch.testing.assert_close(SPEC.take(batch, "rot6d", axis=2), batch[:, :, 3:9, :])
    assert SPEC.take(batch, "rot6d", axis=2).shape != SPEC.take(batch, "rot6d").shape


def test_it_works_on_statistics_with_channels_in_the_middle():
    stats = np.arange(4 * SPEC.dim, dtype=float).reshape(4, SPEC.dim)
    np.testing.assert_array_equal(SPEC.take(stats, "foot_contact", axis=1), stats[:, 12:13])


def test_drop_removes_the_axis_of_a_width_one_block():
    """The squeeze callers were writing as a bare `[..., 0]`."""
    features = np.zeros((2, 4, SPEC.dim))
    features[..., 12] = 1.0
    flags = SPEC.take(features, "foot_contact", drop=True)
    assert flags.shape == (2, 4)
    assert flags.all()


def test_drop_refuses_a_wider_block_rather_than_silently_taking_one_channel():
    """`[..., 0]` on rot6d returns a plausible array of the wrong thing."""
    features = np.zeros((2, 4, SPEC.dim))
    with pytest.raises(ValueError, match="6 wide, not 1"):
        SPEC.take(features, "rot6d", drop=True)


def test_an_absent_block_names_what_is_present():
    features = np.zeros((2, 4, SPEC.dim))
    with pytest.raises(KeyError, match="ric_pos"):
        SPEC.take(features, "not_a_block")


def test_take_is_a_view_so_it_costs_nothing():
    features = np.zeros((2, 4, SPEC.dim))
    SPEC.take(features, "foot_contact")[:] = 5.0
    assert (features[..., 12] == 5.0).all()

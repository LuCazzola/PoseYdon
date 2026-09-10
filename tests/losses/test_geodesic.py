"""The geodesic angle itself.

This loss trains at weight 1.0 and had no test of its own: `tests/losses` was
four foot-skate tests. Swapping `atan2`'s arguments -- which returns the
COMPLEMENTARY angle, so two identical rotations score pi/2 instead of 0 --
passed the entire suite.

The angle is the whole objective, so it is tested against rotations whose
answers are known by construction rather than against a recorded number.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from poseydon.losses.geodesic import geodesic_distance, rot6d_to_matrix


def _about_z(angle: float) -> torch.Tensor:
    c, s = math.cos(angle), math.sin(angle)
    return torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def test_identical_rotations_are_exactly_zero():
    """The property `atan2(sine, cosine)` exists for.

    `acos` of the trace is singular here: implementations clamp its argument and
    leave a floor of about 4.5e-4 radians, plus a spurious gradient at zero
    error. Swapping the arguments puts this at pi/2.
    """
    rotation = _about_z(0.7)[None]
    assert float(geodesic_distance(rotation, rotation)) == 0.0


@pytest.mark.parametrize("angle", [0.1, 0.5, math.pi / 4, math.pi / 2, 2.0, 3.0])
def test_it_recovers_the_angle_it_was_given(angle):
    a, b = _about_z(0.0)[None], _about_z(angle)[None]
    assert float(geodesic_distance(a, b)) == pytest.approx(angle, abs=1e-6)


def test_a_half_turn_is_pi():
    """The other endpoint `acos` is singular at."""
    a, b = _about_z(0.0)[None], _about_z(math.pi)[None]
    assert float(geodesic_distance(a, b)) == pytest.approx(math.pi, abs=1e-5)


def test_the_angle_is_symmetric():
    a, b = _about_z(0.3)[None], _about_z(1.9)[None]
    assert float(geodesic_distance(a, b)) == pytest.approx(
        float(geodesic_distance(b, a)), abs=1e-6
    )


def test_it_never_exceeds_pi():
    """A rotation and its inverse are at most half a turn apart, either way round."""
    rng = np.random.default_rng(0)
    for _ in range(20):
        angle = float(rng.uniform(-3 * math.pi, 3 * math.pi))
        value = float(geodesic_distance(_about_z(0.0)[None], _about_z(angle)[None]))
        assert 0.0 <= value <= math.pi + 1e-6


def test_it_is_the_shorter_way_round():
    """2*pi - 0.2 apart is 0.2 apart, not 6.08."""
    a, b = _about_z(0.0)[None], _about_z(2 * math.pi - 0.2)[None]
    assert float(geodesic_distance(a, b)) == pytest.approx(0.2, abs=1e-5)


def test_gram_schmidt_recovers_a_rotation_matrix():
    """`rot6d_to_matrix` feeds the angle, so a non-orthonormal result poisons it."""
    rng = np.random.default_rng(1)
    d6 = torch.tensor(rng.normal(size=(5, 6)), dtype=torch.float32)
    matrices = rot6d_to_matrix(d6)
    identity = matrices.transpose(-1, -2) @ matrices
    torch.testing.assert_close(
        identity, torch.eye(3).expand_as(identity), atol=1e-5, rtol=1e-5
    )
    torch.testing.assert_close(
        torch.linalg.det(matrices), torch.ones(5), atol=1e-5, rtol=1e-5
    )


def test_the_angle_between_recovered_6d_rotations_is_the_true_one():
    """End to end: 6D in, angle out, against a rotation built by construction."""
    angle = 0.9
    base = _about_z(0.0)
    turned = _about_z(angle)
    # Columns layout, matching `rot6d_to_matrix`.
    as_6d = lambda m: torch.cat([m[:, 0], m[:, 1]])[None]
    value = geodesic_distance(rot6d_to_matrix(as_6d(base)), rot6d_to_matrix(as_6d(turned)))
    assert float(value) == pytest.approx(angle, abs=1e-5)

"""The renderer's coordinate mapping."""

import numpy as np
import pytest

from poseydon.io.render import render_frame, render_skeleton, to_view


def test_mapping_preserves_handedness():
    # This is the whole point. Swapping two axes has determinant -1 and silently
    # mirrors the character, exchanging its left and right; a rotation does not.
    basis = np.eye(3)
    transformed = to_view(basis)
    assert np.linalg.det(transformed) == pytest.approx(1.0)


def test_up_axis_becomes_the_vertical_axis():
    # Motion is Y-up; matplotlib draws its third axis vertically.
    up = to_view(np.array([0.0, 1.0, 0.0]))
    np.testing.assert_allclose(up, [0.0, 0.0, 1.0], atol=1e-12)


def test_forward_axis_points_away_from_the_camera_side():
    forward = to_view(np.array([0.0, 0.0, 1.0]))
    np.testing.assert_allclose(forward, [0.0, -1.0, 0.0], atol=1e-12)


def test_mapping_preserves_distances():
    rng = np.random.default_rng(0)
    points = rng.normal(size=(32, 3))
    before = np.linalg.norm(points[:, None] - points[None, :], axis=-1)
    after = to_view(points)
    after = np.linalg.norm(after[:, None] - after[None, :], axis=-1)
    np.testing.assert_allclose(after, before, atol=1e-12)


def test_renders_a_video(tmp_path):
    parents = np.array([-1, 0, 1])
    positions = np.zeros((4, 3, 3))
    positions[:, 1, 1] = 1.0
    positions[:, 2, 1] = 2.0

    out = render_skeleton(tmp_path / "clip.mp4", parents, positions, fps=4, title="t")
    assert out.is_file() and out.stat().st_size > 0


def test_zoom_changes_the_framing(tmp_path):
    parents = np.array([-1, 0])
    positions = np.zeros((2, 2, 3))
    positions[:, 1, 1] = 1.0

    wide = render_frame(tmp_path / "wide.png", parents, positions, zoom=1.0)
    close = render_frame(tmp_path / "close.png", parents, positions, zoom=3.0)

    import imageio.v2 as imageio

    # Zooming in puts more non-background pixels on screen.
    def ink(path):
        image = imageio.imread(path)
        return int((image.reshape(-1, image.shape[-1]).min(axis=-1) < 200).sum())

    assert ink(close) > ink(wide)

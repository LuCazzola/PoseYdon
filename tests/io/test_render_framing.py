"""The camera must follow the character without cropping it.

Choosing a framing by eye on one frame is how a limb gets cropped somewhere
else in the clip: winding the old whole-trajectory framing in to a comparable
apparent size clipped 37-51% of joint-frames while looking fine on the frame
that happened to be inspected. These assert the property over every joint of
every frame instead.
"""

from __future__ import annotations

import numpy as np

from poseydon.io.render import _smooth_path, to_view


def _framing(positions: np.ndarray, zoom: float = 1.0):
    """The framing `render_skeleton` computes, without drawing anything."""
    view = to_view(np.asarray(positions, dtype=np.float64))
    track = _smooth_path(view.mean(axis=1))
    radius = float(np.abs(view - track[:, None]).max()) * 1.1
    return view, track, radius / max(zoom, 1e-6)


def _travelling_clip(frames: int = 120, joints: int = 12, travel: float = 12.0) -> np.ndarray:
    """A character of fixed size walking a long way -- the case that broke.

    The trajectory spans many times the character's own size, so a camera framed
    on the whole path renders the subject tiny.
    """
    rng = np.random.default_rng(0)
    body = rng.normal(scale=0.3, size=(joints, 3))
    path = np.linspace(0, travel, frames)
    clip = body[None] + np.stack(
        [path, np.zeros(frames), np.zeros(frames)], axis=-1
    )[:, None]
    # A limb that swings, so the extremes are not all at the same offset.
    clip[:, 0, 2] += 0.5 * np.sin(np.linspace(0, 8 * np.pi, frames))
    return clip


def test_no_joint_is_cropped_at_the_default_zoom():
    view, track, reach = _framing(_travelling_clip())
    outside = (np.abs(view - track[:, None]) > reach).any(axis=2)
    assert outside.sum() == 0, (
        f"{outside.sum()} joint-frames fall outside the frame at zoom=1.0"
    )


def test_the_subject_fills_the_frame_regardless_of_how_far_it_travels():
    """The regression: reach must scale with the CHARACTER, not the trajectory."""
    # Ten times the distance at the SAME speed -- a longer walk, not a
    # character teleporting across the floor. Holding speed fixed is what makes
    # this a test of the framing rule rather than of the smoothing lag.
    short = _travelling_clip(frames=120, travel=12.0)
    long = _travelling_clip(frames=1200, travel=120.0)

    _, _, reach_short = _framing(short)
    _, _, reach_long = _framing(long)

    # Not equality: a smoothed camera lags translation, and the reach must
    # cover that lag, so a faster clip legitimately frames a little wider. What
    # must NOT happen is reach tracking the trajectory -- ten times the travel
    # must not cost anything like ten times the reach.
    assert abs(reach_long - reach_short) / reach_short < 0.10, (
        f"reach grew from {reach_short:.3f} to {reach_long:.3f} for ten times "
        "the travel at the same speed -- the camera is framing the path, not "
        "the subject"
    )
    span = float(np.ptp(to_view(long).reshape(-1, 3), axis=0).max())
    assert reach_long < 0.1 * span, (
        f"reach {reach_long:.3f} is not small against the {span:.1f} trajectory"
    )


def test_the_camera_path_is_smooth():
    """A jittery centre reads as camera shake, which is worse than being far."""
    rng = np.random.default_rng(1)
    noisy = np.cumsum(rng.normal(scale=0.1, size=(200, 3)), axis=0)
    smoothed = _smooth_path(noisy)
    jitter = lambda p: np.abs(np.diff(p, n=2, axis=0)).mean()
    assert jitter(smoothed) < 0.25 * jitter(noisy)
    assert smoothed.shape == noisy.shape


def test_a_clip_too_short_to_smooth_is_returned_unchanged():
    path = np.zeros((2, 3))
    np.testing.assert_array_equal(_smooth_path(path), path)


def test_zoom_tightens_the_frame():
    _, _, wide = _framing(_travelling_clip())
    _, _, tight = _framing(_travelling_clip(), zoom=2.0)
    assert tight < wide

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


def _vertical(positions: np.ndarray, zoom: float = 1.0):
    """The vertical window `render_skeleton` computes."""
    from poseydon.io.render import MIN_VERTICAL, PAD

    view, track, reach = _framing(positions, zoom)
    floor, ceiling = float(view[..., 2].min()), float(view[..., 2].max())
    reach_z = max((ceiling - floor) / 2.0, reach * MIN_VERTICAL) * (1.0 + PAD)
    return view, track, reach, floor - reach_z * PAD, reach_z


def _low_wide_clip(frames: int = 80, joints: int = 20) -> np.ndarray:
    """A crab: wide, close to the ground, with the floor far below its centre.

    This is the shape that broke a vertical window derived from a per-frame
    rise and drop -- the floor sat well below the tracked centre, so the window
    stopped short of the character's own top.
    """
    rng = np.random.default_rng(2)
    body = rng.normal(scale=1.0, size=(joints, 3))
    body[:, 2] = np.abs(body[:, 2]) * 0.25          # flat, and above the floor
    path = np.linspace(0, 3.0, frames)
    clip = body[None] + np.stack(
        [path, path * 0.5, np.zeros(frames)], axis=-1
    )[:, None]
    clip[:, 0, 2] += 0.4                             # one joint reaching up
    return clip


def test_the_vertical_window_covers_the_character():
    """Regression: a low, wide rig had its top cropped at the default zoom."""
    view, _track, _reach, base, reach_z = _vertical(_low_wide_clip())
    above = (view[..., 2] > base + 2 * reach_z).sum()
    below = (view[..., 2] < base).sum()
    assert above == 0 and below == 0, (
        f"{above} joint-frames above the frame and {below} below it"
    )


def test_zoom_does_not_tighten_the_vertical_window():
    """There is no vertical slack to reclaim, so zooming it only crops heads."""
    *_, base_wide, reach_wide = _vertical(_low_wide_clip(), zoom=1.0)
    *_, base_tight, reach_tight = _vertical(_low_wide_clip(), zoom=1.3)
    assert reach_tight == reach_wide and base_tight == base_wide


def test_the_ground_is_never_above_the_bottom_of_the_frame():
    """The floor must stay visible: it is the only depth cue in the render."""
    view, _track, _reach, base, reach_z = _vertical(_low_wide_clip())
    floor = float(view[..., 2].min())
    assert base <= floor <= base + 2 * reach_z


def test_the_ground_covers_the_whole_path_not_just_one_view():
    """The camera follows the character, so the ground must exist ahead of it.

    Sized to the current view instead, the render shows the character walking
    to the edge of a floating slab.
    """
    from poseydon.io.render import _ground

    clip = _travelling_clip(frames=120, travel=12.0)
    view, _track, reach = _framing(clip)
    xs, ys, _colours, _floor = _ground(view, reach, float(view[..., 2].min()))

    flat = view.reshape(-1, 3)[:, :2]
    assert xs[0] < flat[:, 0].min() and xs[-1] > flat[:, 0].max()
    assert ys[0] < flat[:, 1].min() and ys[-1] > flat[:, 1].max()


def test_the_tiles_are_world_anchored_so_travel_is_visible():
    """The whole reason for a checkerboard.

    With a camera that follows the character, a ground drawn relative to the
    CAMERA moves with it, and the skeleton appears to run on the spot. The tile
    grid must be fixed in world space, so the character's offset within a tile
    changes as it travels.
    """
    from poseydon.io.render import _ground

    clip = _travelling_clip(frames=120, travel=12.0)
    view, track, reach = _framing(clip)
    xs, _ys, _colours, _floor = _ground(view, reach, float(view[..., 2].min()))
    tile = float(xs[1] - xs[0])

    phase = [(float(track[f, 0]) - xs[0]) / tile % 1.0 for f in (0, 40, 80, 119)]
    assert len(set(np.round(phase, 2))) > 1, (
        "the character keeps the same position within a tile across the whole "
        "clip -- the ground is moving with the camera"
    )


def test_only_the_visible_tiles_are_drawn():
    """A long walk must not put thousands of quads through matplotlib."""
    from poseydon.io.render import _ground, _visible_ground

    clip = _travelling_clip(frames=400, travel=200.0)
    view, track, reach = _framing(clip)
    ground = _ground(view, reach, float(view[..., 2].min()))
    whole = ground[2].size

    mesh_x, _mesh_y, _mesh_z, colours = _visible_ground(ground, track[0], reach)
    assert colours.size < whole / 10, (
        f"drew {colours.size} tiles of {whole} for one view"
    )
    # And it must actually cover the view, not merely be small.
    assert mesh_x.min() <= track[0, 0] - reach and mesh_x.max() >= track[0, 0] + reach


def test_the_tiles_alternate():
    from poseydon.io.render import _ground

    view, _track, reach = _framing(_travelling_clip())
    _xs, _ys, colours, _floor = _ground(view, reach, 0.0)
    assert not np.allclose(colours[0, 0], colours[0, 1])
    assert np.allclose(colours[0, 0], colours[1, 1])

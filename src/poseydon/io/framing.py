"""How a shot is framed: where the camera goes and where the floor lies.

Split out of :mod:`poseydon.io.render` so it can be used WITHOUT matplotlib.
The maths here is plain numpy, but `render` imports matplotlib and imageio at
module scope and raises without them -- and the `fbx` container, which is the
only image with Blender and so the only place a skinned character can be
rendered, has neither.

Copying the framing into that renderer instead was the obvious alternative and
is exactly what `camera`'s own docstring warns against: the tests used to
recompute it themselves, and a mutation that reframed the camera on the whole
trajectory left the entire suite green. One implementation, two renderers.

The vertical axis is the THIRD component throughout. That is matplotlib's
drawing convention (:func:`to_view` rotates motion coordinates into it) and
Blender's world convention, which is why one set of functions serves both;
motion data itself is Y-up and is never in this frame.
"""

from __future__ import annotations

import numpy as np

#: Breathing room around the vertical window, so a head does not touch the
#: top edge of the frame.
PAD = 0.06

#: Floor under the vertical reach, as a fraction of the horizontal one. A clip
#: whose character barely leaves the ground would otherwise be framed in a
#: sliver, which reads as a letterbox rather than a room.
MIN_VERTICAL = 0.45

#: Ground tile size, as a fraction of the horizontal reach. About four tiles
#: across the view: coarse enough to stay quiet behind the skeleton, fine enough
#: that a walking character visibly crosses them.
TILE = 0.5

#: How far the ground extends past the character's path, in tiles. The camera
#: follows the character, so the ground has to exist wherever the camera can
#: look or the render shows a cliff edge.
GROUND_MARGIN = 3

#: The two tile colours, RGBA. Deliberately close together and faint: the floor
#: is a motion cue, not something to read.
TILE_COLOURS = ((0.55, 0.55, 0.60, 0.20), (0.80, 0.80, 0.84, 0.11))

def to_view(points: np.ndarray) -> np.ndarray:
    """Motion coordinates to matplotlib's drawing frame.

    Motion data is Y-up and faces +Z; matplotlib draws its THIRD axis vertically.
    The mapping must be a ROTATION, not an axis swap: exchanging Y and Z has
    determinant -1, which silently mirrors the character and swaps its left and
    right. Rotating 90 degrees about X instead -- (x, y, z) -> (x, -z, y) --
    preserves handedness.
    """
    x, y, z = points[..., 0], points[..., 1], points[..., 2]
    return np.stack([x, -z, y], axis=-1)


def _smooth_path(path: np.ndarray, window: int = 9) -> np.ndarray:
    """Moving average of a camera path, so tracking does not jitter per frame.

    Edges are held rather than tapered: a taper would accelerate the camera into
    the first and last frames, which reads as a lurch.

    The window is short on purpose. Smoothing lags translation by roughly half
    the window times the speed, and the reach has to cover that lag, so a long
    window makes the camera back off exactly when the subject moves fast --
    measured at 15 frames, a tenfold increase in travel tripled the reach.
    Nine is enough to take the swing of the limbs out of the centroid.
    """
    if path.shape[0] < 3:
        return path
    window = min(window, path.shape[0] | 1)
    pad = window // 2
    padded = np.pad(path, ((pad, pad), (0, 0)), mode="edge")
    kernel = np.ones(window) / window
    return np.stack(
        [np.convolve(padded[:, axis], kernel, mode="valid")[: path.shape[0]]
         for axis in range(path.shape[1])],
        axis=1,
    )


def camera(view: np.ndarray, zoom: float = 1.0):
    """Where the camera sits and how much it sees: ``(track, reach, reach_z, floor)``.

    Split out of `render_skeleton` so it has exactly ONE implementation. The
    tests used to recompute this themselves, which meant they exercised a copy:
    a mutation that reframed the camera on the whole trajectory -- the very bug
    this logic exists to fix -- left the whole suite green.

    A camera that FOLLOWS the character at a fixed reach, rather than one framing
    wide enough for the whole trajectory. Framing the trajectory makes the
    subject small in exact proportion to how far it travels -- on the validation
    clips the character spans about 1 unit while the path spans 2-3, so it
    occupied roughly a third of the frame and read as "far away".

    The reach is FIXED for the clip and the centre is smoothed, so nothing
    rescales per frame. Smoothing comes FIRST and the reach is sized against the
    path actually used: sizing against the unsmoothed centre and framing on the
    smoothed one leaves the bound unguaranteed wherever smoothing lags a fast
    move.

    The vertical extent is its own quantity, taken from the clip's floor-to-
    ceiling span and NOT divided by ``zoom``: there is no vertical slack to
    reclaim, so dividing would crop heads to buy nothing.
    """
    track = _smooth_path(view.mean(axis=1))
    radius = float(np.abs(view - track[:, None]).max()) * 1.1
    reach = radius / max(zoom, 1e-6) + 1e-6
    floor = float(view[..., 2].min())
    ceiling = float(view[..., 2].max())
    reach_z = max((ceiling - floor) / 2.0, reach * MIN_VERTICAL) * (1.0 + PAD) + 1e-6
    return track, reach, reach_z, floor


def _ground(view: np.ndarray, reach: float, floor: float):
    """A checkerboard covering everywhere the character goes, plus a margin.

    Built from the PATH rather than from the camera, so the tiles stay put on
    the ground while the view moves over them. That is the whole point: with a
    camera that follows the character, a plain quad moving with it gives no cue
    that anything is travelling at all -- the skeleton appears to run on the
    spot.
    """
    flat = view.reshape(-1, 3)[:, :2]
    tile = max(reach * TILE, 1e-6)
    low = flat.min(axis=0) - tile * GROUND_MARGIN
    high = flat.max(axis=0) + tile * GROUND_MARGIN
    counts = np.maximum(np.ceil((high - low) / tile).astype(int), 1)
    xs = low[0] + np.arange(counts[0] + 1) * tile
    ys = low[1] + np.arange(counts[1] + 1) * tile
    parity = (np.arange(counts[0])[:, None] + np.arange(counts[1])[None, :]) % 2
    colours = np.array(TILE_COLOURS)[parity]
    return xs, ys, colours, floor

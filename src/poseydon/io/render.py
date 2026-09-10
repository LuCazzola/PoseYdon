"""Stick-figure MP4 rendering.

Deliberately independent of the reference's plot script, which calls
`FigureCanvasAgg.tostring_rgb` -- removed in matplotlib 3.8 -- and so cannot run
on a current install without pinning the whole stack backwards.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    import imageio.v2 as imageio
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as error:  # pragma: no cover - exercised only without the extra
    raise ImportError(
        "rendering needs the optional extra: install poseydon[render]"
    ) from error


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


def _bones(parents) -> list[tuple[int, int]]:
    return [(int(parent), joint) for joint, parent in enumerate(parents) if parent >= 0]



#: Breathing room around the vertical window, so a head does not touch the
#: top edge of the frame.
PAD = 0.06

#: Floor under the vertical reach, as a fraction of the horizontal one. A clip
#: whose character barely leaves the ground would otherwise be framed in a
#: sliver, which reads as a letterbox rather than a room.
MIN_VERTICAL = 0.45

#: Scales the axes cube within the figure. The data limits are unaffected, so
#: this crops empty margin rather than content. 1.55 was chosen by rendering
#: frames 10/60/150 of all three validation pairs and looking; at 1.7 the ground
#: plane is cut by the figure edge and reads as a platform rather than a floor.
BOX_ZOOM = 1.55

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


def _visible_ground(ground, centre, reach):
    """The tiles the camera can currently see, so a long path costs nothing.

    Drawing the whole floor every frame would put thousands of quads through
    matplotlib for the handful that land inside the axes limits.
    """
    xs, ys, colours, floor = ground
    i = np.searchsorted(xs, [centre[0] - reach, centre[0] + reach])
    j = np.searchsorted(ys, [centre[1] - reach, centre[1] + reach])
    i0, i1 = max(int(i[0]) - 1, 0), min(int(i[1]) + 1, len(xs) - 1)
    j0, j1 = max(int(j[0]) - 1, 0), min(int(j[1]) + 1, len(ys) - 1)
    if i1 <= i0 or j1 <= j0:
        return None
    mesh_x, mesh_y = np.meshgrid(xs[i0 : i1 + 1], ys[j0 : j1 + 1], indexing="ij")
    return mesh_x, mesh_y, np.full_like(mesh_x, floor), colours[i0:i1, j0:j1]


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


def render_skeleton(
    path: str | Path,
    parents,
    positions: np.ndarray,
    fps: int = 24,
    title: str = "",
    elev: float = 14.0,
    azim: float = -70.0,
    zoom: float = 1.15,
    dpi: int = 90,
    highlight: dict[str, list[int]] | None = None,
) -> Path:
    """Write an MP4 of a moving skeleton. ``positions`` is ``(F, J, 3)``.

    ``zoom`` scales the framing: values above 1 move the camera closer, so 2.0
    fills roughly twice the frame. ``highlight`` maps a colour to joint indices,
    for checking which side of the character is which.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    view = to_view(np.asarray(positions, dtype=np.float64))
    bones = _bones(parents)

    # A camera that FOLLOWS the character at a fixed reach, rather than one
    # framing wide enough for the whole trajectory. Framing the trajectory
    # makes the subject small in exact proportion to how far it travels -- on
    # these clips the character spans about 1 unit while the path spans 2-3, so
    # it occupied roughly a third of the frame and read as "far away".
    #
    # The reach is FIXED for the clip and the centre is smoothed, so nothing
    # rescales per frame: that rescaling is what the previous single framing was
    # avoiding, and it is avoided here too. Sized from the character's own
    # largest half-extent, which measured 0.00% clipped joint-frames across all
    # three validation pairs at `zoom=1.0`, against 37-51% for simply winding
    # the old framing in to a comparable size.
    # Smooth FIRST, then size the reach against the path actually used. Sizing
    # it against the unsmoothed centre and then framing on the smoothed one
    # leaves the bound unguaranteed: wherever smoothing lags a fast move, the
    # character can sit further from the camera centre than the reach allows.
    # Real clips measured 0.00% clipped either way, which is precisely why this
    # needed a test rather than an eyeball.
    track = _smooth_path(view.mean(axis=1))
    radius = float(np.abs(view - track[:, None]).max()) * 1.1
    reach = radius / max(zoom, 1e-6) + 1e-6
    floor = float(view[..., 2].min())

    # Vertical extent of its own. A cube spends half its height on sky the
    # character never enters -- these rigs stand about as tall as they are wide,
    # so an equal-sided box left a visibly empty band above every render. The
    # box aspect below is set to MATCH these limits, so units stay square in
    # every axis and nothing is stretched; only empty space is removed.
    #
    # Taken from the clip's own floor-to-ceiling span and NOT divided by
    # `zoom`. Vertically there is no slack to reclaim -- the window already
    # ends where the character does -- so dividing would crop heads to buy
    # nothing. `zoom` therefore tightens the horizontal framing only, which is
    # where the spare room actually is. Deriving this from a per-frame rise and
    # drop instead clipped 3.46% of Coyote->Crab at zoom 1.0, because the floor
    # can sit well below the tracked centre and the window stopped short of the
    # character's top.
    ground = _ground(view, reach, float(view[..., 2].min()))
    ceiling = float(view[..., 2].max())
    reach_z = max((ceiling - floor) / 2.0, reach * MIN_VERTICAL) * (1.0 + PAD) + 1e-6

    frames = []
    figure = plt.figure(figsize=(5, 5), dpi=dpi)
    axes = figure.add_subplot(111, projection="3d")

    for frame in range(view.shape[0]):
        axes.clear()
        joints = view[frame]

        centre = track[frame]
        tiles = _visible_ground(ground, centre, reach)
        if tiles is not None:
            mesh_a, mesh_b, mesh_c, facecolours = tiles
            axes.plot_surface(
                mesh_a, mesh_b, mesh_c, facecolors=facecolours, shade=False,
                linewidth=0, antialiased=False,
            )

        for parent, child in bones:
            axes.plot(
                *zip(joints[parent], joints[child], strict=True),
                color="#1f77b4", linewidth=1.8, solid_capstyle="round",
            )
        axes.scatter(*joints.T, s=6, color="#d62728", depthshade=False)

        for colour, indices in (highlight or {}).items():
            picked = joints[list(indices)]
            axes.scatter(*picked.T, s=55, color=colour, depthshade=False)

        axes.set_xlim(centre[0] - reach, centre[0] + reach)
        axes.set_ylim(centre[1] - reach, centre[1] + reach)
        # Anchored so the ground is visible without giving it half the frame:
        # a skeleton stands ON the floor, so a centred cube buries the lower
        # half below it.
        base = floor - reach_z * PAD
        axes.set_zlim(base, base + 2 * reach_z)
        # `zoom` on the box aspect scales the CUBE inside the figure, which is
        # where most of the empty margin was: matplotlib's 3D axes reserve room
        # for tick labels and a title that this render turns off anyway. The
        # data limits above are untouched, so nothing is clipped by this -- it
        # only stops drawing the same content so small.
        axes.set_box_aspect((reach, reach, reach_z), zoom=BOX_ZOOM)
        axes.view_init(elev=elev, azim=azim)
        axes.set_axis_off()
        axes.set_title(f"{title}\nframe {frame + 1}/{view.shape[0]}", fontsize=9)

        figure.canvas.draw()
        frames.append(np.asarray(figure.canvas.buffer_rgba())[..., :3].copy())

    plt.close(figure)
    imageio.mimsave(path, frames, fps=fps, macro_block_size=1)
    return path


def render_frame(
    path: str | Path, parents, positions: np.ndarray, frame: int = 0, **kwargs
) -> Path:
    """Write a single frame as a PNG, for checking framing or orientation."""
    single = np.asarray(positions)[frame : frame + 1]
    path = Path(path)
    video = path.with_suffix(".check.mp4")
    render_skeleton(video, parents, single, fps=1, **kwargs)
    imageio.imwrite(path, imageio.mimread(video)[0])
    video.unlink(missing_ok=True)
    return path

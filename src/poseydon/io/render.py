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



#: Scales the axes cube within the figure. The data limits are unaffected, so
#: this crops empty margin rather than content. 1.55 was chosen by rendering
#: frames 10/60/150 of all three validation pairs and looking; at 1.7 the ground
#: plane is cut by the figure edge and reads as a platform rather than a floor.
BOX_ZOOM = 1.55


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
    zoom: float = 1.0,
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

    frames = []
    figure = plt.figure(figsize=(5, 5), dpi=dpi)
    axes = figure.add_subplot(111, projection="3d")

    for frame in range(view.shape[0]):
        axes.clear()
        joints = view[frame]

        centre = track[frame]
        grid = np.linspace(-reach, reach, 2)
        mesh_a, mesh_b = np.meshgrid(grid + centre[0], grid + centre[1])
        axes.plot_surface(
            mesh_a, mesh_b, np.full_like(mesh_a, floor), alpha=0.12, color="#888888"
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
        # Anchored just below the floor rather than centred on the character:
        # a skeleton stands ON the ground, so centring would put half the frame
        # underneath it.
        base = min(floor, centre[2] - reach * 0.5)
        axes.set_zlim(base, base + 2 * reach)
        # `zoom` on the box aspect scales the CUBE inside the figure, which is
        # where most of the empty margin was: matplotlib's 3D axes reserve room
        # for tick labels and a title that this render turns off anyway. The
        # data limits above are untouched, so nothing is clipped by this -- it
        # only stops drawing the same content so small.
        axes.set_box_aspect((1, 1, 1), zoom=BOX_ZOOM)
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

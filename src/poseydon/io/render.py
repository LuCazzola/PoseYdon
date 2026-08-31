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

    # One framing for the whole clip, so the character does not appear to swim
    # as the view rescales per frame.
    flat = view.reshape(-1, 3)
    centre = flat.mean(axis=0)
    reach = float(np.abs(flat - centre).max()) * 1.1 / max(zoom, 1e-6) + 1e-6
    floor = float(view[..., 2].min())

    frames = []
    figure = plt.figure(figsize=(5, 5), dpi=dpi)
    axes = figure.add_subplot(111, projection="3d")

    for frame in range(view.shape[0]):
        axes.clear()
        joints = view[frame]

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
        axes.set_zlim(floor, floor + 2 * reach)
        axes.set_box_aspect((1, 1, 1))
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

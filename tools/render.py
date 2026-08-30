"""Stick-figure MP4 rendering.

Deliberately independent of the reference's plot script, which calls
`FigureCanvasAgg.tostring_rgb` -- removed in matplotlib 3.8 -- and so cannot run
on a current install without pinning the whole stack backwards.
"""

from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _bones(parents) -> list[tuple[int, int]]:
    return [(int(parent), joint) for joint, parent in enumerate(parents) if parent >= 0]


def render_skeleton(
    path: str | Path,
    parents,
    positions: np.ndarray,
    fps: int = 24,
    title: str = "",
    elev: float = 12.0,
    azim: float = 70.0,
    dpi: int = 90,
) -> Path:
    """Write an MP4 of a moving skeleton. ``positions`` is ``(F, J, 3)``.

    Motion data is Y-up and faces +Z after alignment, but matplotlib draws its
    THIRD axis vertically. Passing (x, y, z) straight through therefore stands
    the character on its face: the forward axis is drawn as up. Every plotting
    call below maps (x, y, z) -> (x, z, y) so motion-Y is vertical, and the
    camera sits on the +Z side so the character faces it.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    positions = np.asarray(positions, dtype=np.float64)
    bones = _bones(parents)

    # One set of limits for the whole clip, so the character does not appear to
    # swim as the view rescales per frame.
    centre = positions.reshape(-1, 3).mean(axis=0)
    reach = float(np.abs(positions.reshape(-1, 3) - centre).max()) * 1.15 + 1e-6
    floor = float(positions[..., 1].min())  # motion-Y is up

    frames = []
    figure = plt.figure(figsize=(5, 5), dpi=dpi)
    axes = figure.add_subplot(111, projection="3d")

    for frame in range(positions.shape[0]):
        axes.clear()
        joints = positions[frame]

        # A ground plane, so height and contact are readable.
        grid = np.linspace(-reach, reach, 2)
        mesh_x, mesh_depth = np.meshgrid(grid + centre[0], grid + centre[2])
        axes.plot_surface(
            mesh_x, mesh_depth, np.full_like(mesh_x, floor), alpha=0.12, color="#888888"
        )

        for parent, child in bones:
            axes.plot(
                [joints[parent, 0], joints[child, 0]],
                [joints[parent, 2], joints[child, 2]],
                [joints[parent, 1], joints[child, 1]],
                color="#1f77b4", linewidth=1.8, solid_capstyle="round",
            )
        axes.scatter(
            joints[:, 0], joints[:, 2], joints[:, 1], s=5, color="#d62728", depthshade=False
        )

        axes.set_xlim(centre[0] - reach, centre[0] + reach)
        axes.set_ylim(centre[2] - reach, centre[2] + reach)
        axes.set_zlim(floor, floor + 2 * reach)
        axes.set_box_aspect((1, 1, 1))
        axes.view_init(elev=elev, azim=azim)
        axes.set_axis_off()
        axes.set_title(f"{title}\nframe {frame + 1}/{positions.shape[0]}", fontsize=9)

        figure.canvas.draw()
        image = np.asarray(figure.canvas.buffer_rgba())[..., :3]
        frames.append(image.copy())

    plt.close(figure)
    imageio.mimsave(path, frames, fps=fps, macro_block_size=1)
    return path


def render_frame(
    path: str | Path, parents, positions: np.ndarray, frame: int = 0, **kwargs
) -> Path:
    """Write a single frame as a PNG, for checking orientation quickly."""
    single = np.asarray(positions)[frame : frame + 1]
    path = Path(path)
    frames_path = path.with_suffix(".mp4")
    render_skeleton(frames_path, parents, single, fps=1, **kwargs)
    figure_frames = imageio.mimread(frames_path)
    imageio.imwrite(path, figure_frames[0])
    frames_path.unlink(missing_ok=True)
    return path

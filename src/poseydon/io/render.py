"""Stick-figure MP4 rendering.

Deliberately independent of the reference's plot script, which calls
`FigureCanvasAgg.tostring_rgb` -- removed in matplotlib 3.8 -- and so cannot run
on a current install without pinning the whole stack backwards.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from poseydon.io.framing import (
    PAD,
    _ground,
    camera,
    to_view,
)

try:
    import imageio.v2 as imageio
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as error:  # pragma: no cover - exercised only without the extra
    raise ImportError(
        "rendering needs the optional extra: install poseydon[render]"
    ) from error


def _bones(parents) -> list[tuple[int, int]]:
    return [(int(parent), joint) for joint, parent in enumerate(parents) if parent >= 0]



#: Scales the axes cube within the figure. The data limits are unaffected, so
#: this crops empty margin rather than content. 1.55 was chosen by rendering
#: frames 10/60/150 of all three validation pairs and looking; at 1.7 the ground
#: plane is cut by the figure edge and reads as a platform rather than a floor.
BOX_ZOOM = 1.55

#: Joints the model marked as touching the ground. Green because the base joints
#: are drawn red -- a red highlight is invisible against them, which made the
#: first contact render useless.
CONTACT_COLOUR = "#2ca02c"



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


def _with_contacts(highlight, contacts, shape) -> dict[str, Any] | None:
    """Fold a contact mask into the highlight map, checking its shape first.

    A silently ignored mask of the wrong shape is worse than no mask: the render
    looks fine and shows nothing, which reads as "no contacts predicted".
    """
    if contacts is None:
        return highlight
    mask = np.asarray(contacts)
    if mask.shape != shape[:2]:
        raise ValueError(
            f"contacts must be one flag per (frame, joint) of positions "
            f"{shape[:2]}, got {mask.shape}"
        )
    return {CONTACT_COLOUR: mask.astype(bool), **(highlight or {})}


def _selected(selector, frame: int) -> np.ndarray:
    """Joint indices to highlight on `frame`, from a list or a per-frame mask."""
    array = np.asarray(selector)
    if array.ndim == 2:                       # (frames, joints) boolean mask
        return np.flatnonzero(array[frame])
    return array.astype(int)


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
    highlight: dict[str, Any] | None = None,
    contacts: np.ndarray | None = None,
) -> Path:
    """Write an MP4 of a moving skeleton. ``positions`` is ``(F, J, 3)``.

    ``zoom`` scales the framing: values above 1 move the camera closer, so 2.0
    fills roughly twice the frame.

    ``contacts`` is a ``(frames, joints)`` boolean mask of the model's own
    foot-contact channel; those joints are drawn green on the frames they are
    flagged. Every feature vector carries that channel, so a render that omits
    it is throwing away the one signal that says whether a foot is meant to be
    planted -- and a foot sliding while flagged down is the failure this is for.

    ``highlight`` maps a colour to either a fixed list of joint indices -- for
    checking which side of a character is which -- or a ``(frames, joints)``
    boolean mask, for any other property that changes over time. It composes
    with ``contacts``; an explicit ``CONTACT_COLOUR`` entry wins.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    view = to_view(np.asarray(positions, dtype=np.float64))
    bones = _bones(parents)
    highlight = _with_contacts(highlight, contacts, view.shape)

    track, reach, reach_z, floor = camera(view, zoom)
    ground = _ground(view, reach, floor)

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

        for colour, selector in (highlight or {}).items():
            picked = joints[_selected(selector, frame)]
            if len(picked):
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

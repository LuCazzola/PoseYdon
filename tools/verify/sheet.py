"""Fixed-camera sheets: the reference above, the file below coloured by its distance from the reference."""

from __future__ import annotations

from pathlib import Path

import numpy as np

# Error as a fraction of rig extent, fixed so sheets of different runs read alike: blue at the floor, red at 1e-1.
ERROR_LOW, ERROR_HIGH = 1e-6, 1e-1
VIEWS = (("front", 0, 1, 2), ("side", 2, 1, 0), ("top", 0, 2, 1), ("3/4", 0, 1, 2))


def _plt():
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt; return plt


def draw(ax, V, tris, P, parents, h, v, z, floor=True, colors=None):
    """Shaded mesh (painter's order along z) with the skeleton in red (if ``P``); the floor line blue.
    ``colors`` (one RGBA per triangle) replaces the grey shading."""
    from matplotlib.collections import PolyCollection
    plt = _plt(); t = V[tris]; nrm = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]); nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12
    shade = 0.25 + 0.75 * np.abs(nrm[:, z]); order = np.argsort(t[:, :, z].mean(1), kind="stable")
    face = plt.cm.Greys_r(0.15 + 0.7 * shade[order]) if colors is None else colors[order]
    ax.add_collection(PolyCollection(t[order][:, :, [h, v]], facecolors=face, edgecolors="none"))
    if P is not None:
        for j in range(1, len(parents)): ax.plot(P[[parents[j], j], h], P[[parents[j], j], v], color="tab:red", lw=0.6)
    if v == 1 and floor: ax.axhline(0, color="tab:blue", lw=0.8)
    if h == 0: ax.axvline(0, color="tab:blue", lw=0.4, ls=":")
    ax.autoscale_view(); ax.set_aspect("equal")


def sheet(case, stage: str, frame: int, got: np.ndarray, want: np.ndarray, error: np.ndarray, tris: np.ndarray, out: Path) -> Path:
    """Four views of ``want`` (grey, above) and ``got`` (coloured by ``error``, below) in one camera
    fixed from ``want``'s bounds: a mesh that sits in the wrong place looks it. Same input, same bytes."""
    from matplotlib.colors import LogNorm
    from scipy.spatial.transform import Rotation
    plt = _plt(); norm = LogNorm(ERROR_LOW, ERROR_HIGH); cmap = plt.cm.turbo
    got = np.where(np.isfinite(got), got, want)
    err = np.where(np.isfinite(error), error, ERROR_HIGH)
    colors = cmap(norm(np.clip(err[tris].mean(1), ERROR_LOW, ERROR_HIGH)))
    turn = Rotation.from_euler("xy", [20, -40], degrees=True).as_matrix()
    fig, axes = plt.subplots(2, 4, figsize=(20, 9), gridspec_kw={"width_ratios": [1, 2, 1, 1.6]})
    for col, (title, h, v, z) in enumerate(VIEWS):
        R = turn if title == "3/4" else np.eye(3)
        ref = want @ R.T; lo, hi = ref[:, [h, v]].min(0), ref[:, [h, v]].max(0); pad = 0.05 * (hi - lo).max()
        for row, (V, colour) in enumerate(((want, None), (got, colors))):
            ax = axes[row, col]; draw(ax, V @ R.T, tris, None, None, h, v, z, floor=False, colors=colour)
            ax.set_xlim(lo[0] - pad, hi[0] + pad); ax.set_ylim(lo[1] - pad, hi[1] + pad)
            ax.set_title(f"{'reference' if row == 0 else 'file'} {title}", fontsize=8)
    fig.suptitle(f"{case.rig} {case.action} {stage} frame {frame}", fontsize=10)
    fig.colorbar(plt.cm.ScalarMappable(norm, cmap), ax=axes, label=f"vertex error, fraction of rig extent (fixed log scale {ERROR_LOW:g} to {ERROR_HIGH:g}; beyond is clipped)")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=80, metadata={"Software": None}); plt.close(fig)
    return out

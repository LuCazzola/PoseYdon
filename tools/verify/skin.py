"""The reference skin: a vendor FBX's mesh deformed by plain linear blend skinning."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from poseydon.io.fbx import SkinData


def extent(points: np.ndarray) -> float:
    return float(np.ptp(points, axis=0).max())


def deform(skin: SkinData, world: Mapping[str, np.ndarray]) -> np.ndarray:
    """``(F, V, 3)`` positions of every control point of ``skin`` under ``world``
    (node name -> ``(F, 4, 4)``), as `FBX.skinned_vertices` does: weights
    normalised per vertex, a vertex no cluster weighs stays at ``skin.points``."""
    points = np.c_[skin.points, np.ones(len(skin.points))]
    n_frames = len(next(iter(world.values())))
    moved = np.zeros((n_frames, len(points), 4))
    total = np.zeros(len(points))
    for name, link_bind, mesh_bind, index, weight in zip(
        skin.links, skin.link_bind, skin.mesh_bind, skin.indices, skin.weights
    ):
        matrix = world[name] @ np.linalg.inv(link_bind) @ mesh_bind
        np.add.at(moved, (slice(None), index), weight[:, None] * (points[index] @ matrix.transpose(0, 2, 1)))
        np.add.at(total, index, weight)
    weighted = total > 0
    out = np.tile(skin.points, (n_frames, 1, 1))
    out[:, weighted] = moved[:, weighted, :3] / total[weighted, None]
    return out

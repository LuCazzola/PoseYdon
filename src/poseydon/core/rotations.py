"""Rotation representations.

Project-wide invariant: quaternions are scalar-last ``(x, y, z, w)``, matching
SciPy and roma. The 6D representation is the first two ROWS of the rotation
matrix, matching PyTorch3D.

SciPy already implements Euler/quaternion/matrix conversions correctly and
quickly; this module wraps those and adds only what SciPy lacks.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

QUAT_IDENTITY = np.array([0.0, 0.0, 0.0, 1.0])


def _as_rotation(q: np.ndarray) -> Rotation:
    return Rotation.from_quat(np.asarray(q, dtype=np.float64).reshape(-1, 4))


def euler_to_quat(angles_deg: np.ndarray, order: str) -> np.ndarray:
    """Intrinsic Euler angles (degrees) to quaternion.

    ``order`` is uppercase, e.g. ``"ZYX"``, and ``angles_deg[..., i]`` is the
    angle for ``order[i]``. Uppercase is SciPy's spelling for intrinsic
    rotations, which is what BVH means by its channel ordering.
    """
    angles = np.asarray(angles_deg, dtype=np.float64)
    flat = Rotation.from_euler(order.upper(), angles.reshape(-1, 3), degrees=True)
    return flat.as_quat().reshape(*angles.shape[:-1], 4)


def quat_to_euler(q: np.ndarray, order: str) -> np.ndarray:
    """Quaternion to intrinsic Euler angles in degrees, in ``order``."""
    q = np.asarray(q, dtype=np.float64)
    angles = _as_rotation(q).as_euler(order.upper(), degrees=True)
    return angles.reshape(*q.shape[:-1], 3)


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    return _as_rotation(q).as_matrix().reshape(*q.shape[:-1], 3, 3)


def matrix_to_quat(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=np.float64)
    flat = Rotation.from_matrix(m.reshape(-1, 3, 3))
    return flat.as_quat().reshape(*m.shape[:-2], 4)


def matrix_to_rot6d(m: np.ndarray, layout: str = "rows") -> np.ndarray:
    """Two basis vectors of the rotation matrix, flattened.

    ``layout="rows"`` takes the first two ROWS, matching PyTorch3D.
    ``layout="columns"`` takes the first two COLUMNS, matching Holden's
    ``Quaternions.rotation_matrix(cont6d=True)`` -- which is what the reference
    dataset's stored features actually use. The two differ by a transpose, so
    mixing them silently yields inverse rotations; verified empirically against
    the shipped `.npy` files, where the row layout is off by up to 2.0.
    """
    m = np.asarray(m, dtype=np.float64)
    if layout == "columns":
        m = np.swapaxes(m, -1, -2)
    elif layout != "rows":
        raise ValueError(f"layout must be 'rows' or 'columns', got {layout!r}")
    return m[..., :2, :].reshape(*m.shape[:-2], 6).copy()


def rot6d_to_matrix(d6: np.ndarray, layout: str = "rows") -> np.ndarray:
    """Gram-Schmidt the two 3-vectors back into an orthonormal matrix."""
    d6 = np.asarray(d6, dtype=np.float64)
    if layout not in ("rows", "columns"):
        raise ValueError(f"layout must be 'rows' or 'columns', got {layout!r}")
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)
    b2 = a2 - (b1 * a2).sum(axis=-1, keepdims=True) * b1
    b2 = b2 / np.linalg.norm(b2, axis=-1, keepdims=True)
    b3 = np.cross(b1, b2)
    out = np.stack([b1, b2, b3], axis=-2)
    return np.swapaxes(out, -1, -2) if layout == "columns" else out


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Composition: applying the result equals applying ``b`` then ``a``."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    ax, ay, az, aw = np.moveaxis(a, -1, 0)
    bx, by, bz, bw = np.moveaxis(b, -1, 0)
    return np.stack(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        axis=-1,
    )


def quat_apply(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector(s) ``v`` by quaternion(s) ``q``, broadcasting."""
    q = np.asarray(q, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    q, v = np.broadcast_arrays(q, np.concatenate([v, np.zeros_like(v[..., :1])], axis=-1))
    xyz, w = q[..., :3], q[..., 3:]
    t = 2.0 * np.cross(xyz, v[..., :3])
    return v[..., :3] + w * t + np.cross(xyz, t)


def quat_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Shortest-arc rotation taking unit vector ``a`` to unit vector ``b``."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b = b / np.linalg.norm(b, axis=-1, keepdims=True)

    axis = np.cross(a, b)
    w = 1.0 + (a * b).sum(axis=-1, keepdims=True)
    q = np.concatenate([axis, w], axis=-1)

    # Antiparallel: w collapses to 0 and the axis is degenerate. Any perpendicular
    # axis is a valid 180-degree rotation; pick one deterministically.
    antiparallel = w[..., 0] < 1e-12
    if np.any(antiparallel):
        fallback = np.cross(a, np.array([1.0, 0.0, 0.0]))
        degenerate = np.linalg.norm(fallback, axis=-1) < 1e-8
        alt = np.cross(a, np.array([0.0, 1.0, 0.0]))
        fallback = np.where(degenerate[..., None], alt, fallback)
        q = np.where(
            antiparallel[..., None],
            np.concatenate([fallback, np.zeros_like(w)], axis=-1),
            q,
        )

    return q / np.linalg.norm(q, axis=-1, keepdims=True)

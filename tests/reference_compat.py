"""Shared helpers for tests that compare PoseYdon against the real reference
implementation (Animation/BVH/Quaternions/InverseKinematics), available only
inside the `compat` Docker image:

    docker compose run --rm compat pytest tests/io/test_reference_parity.py \
        tests/preproc/test_reference_parity.py

Every test module that needs the reference calls require_reference() first,
so the main test image (which does not install these packages) skips
cleanly rather than failing to import.
"""

from __future__ import annotations

import numpy as np
import pytest


def require_reference() -> None:
    pytest.importorskip("BVH")
    pytest.importorskip("Animation")
    pytest.importorskip("Quaternions")
    pytest.importorskip("InverseKinematics")


def reference_quats_to_poseydon(qs: np.ndarray) -> np.ndarray:
    """(..., 4) scalar-first (w, x, y, z) -> scalar-last (x, y, z, w)."""
    return qs[..., [1, 2, 3, 0]]

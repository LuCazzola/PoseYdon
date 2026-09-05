"""Cross-checks poseydon.io.bvh against the real reference BVH reader/writer.

Compat-image only:
    docker compose run --rm compat pytest tests/io/test_reference_parity.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from poseydon.io.bvh import load_bvh, save_bvh
from tests.reference_compat import require_reference

require_reference()

import Animation  # noqa: E402  (only importable after require_reference)
import BVH  # noqa: E402

# BVH text stores 6 decimals; compounded through FK over up to 63 joints,
# agreement to 1e-4 is the practical floor -- see tests/features/test_golden_parity.py
# for the same reasoning applied to the existing golden-parity gate.
ATOL = 1e-4


def test_load_bvh_matches_the_real_reader(bvh_fixture):
    anim = load_bvh(bvh_fixture)
    ref_anim, ref_names, ref_frametime = BVH.load(str(bvh_fixture))

    assert list(ref_names) == list(anim.names)
    assert ref_frametime == pytest.approx(1.0 / anim.fps, abs=1e-9)

    ref_positions = Animation.positions_global(ref_anim)
    np.testing.assert_allclose(ref_positions, anim.global_positions(), atol=ATOL)


def test_save_bvh_round_trips_through_the_real_reader(bvh_fixture, tmp_path):
    anim = load_bvh(bvh_fixture)
    out_path = tmp_path / "roundtrip.bvh"
    save_bvh(anim, out_path)

    ref_anim, ref_names, _ = BVH.load(str(out_path))
    assert list(ref_names) == list(anim.names)

    ref_positions = Animation.positions_global(ref_anim)
    np.testing.assert_allclose(ref_positions, anim.global_positions(), atol=ATOL)

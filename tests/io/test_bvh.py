"""``BVH.read_names`` must agree with a full ``read().to_animation().names``."""

from __future__ import annotations

import pytest

from poseydon.io.bvh import BVH
from tests.conftest import CORPUS, SAMPLE_RIGS


def test_read_names_matches_full_read():
    clips = sorted((CORPUS / "source" / SAMPLE_RIGS[0]).glob("*.bvh"))
    if not clips:
        pytest.skip(f"no raw clips for {SAMPLE_RIGS[0]}")
    path = clips[0]

    names_only = BVH.read_names(path)
    full = BVH.read(path).to_animation().names

    assert names_only == full

"""Shared fixtures.

Real data is primary, but the suite must pass without it: a clean checkout has
no Truebones corpus (it is a paid collection) and no Blender, so every fixture
that needs either skips rather than fails.
"""

from __future__ import annotations

from pathlib import Path

import pytest

CORPUS = Path("data/truebones")

# One biped, one quadruped, one milliped, plus the rig whose reduction differs
# from the reference's (Scorpion keeps a zero-offset joint with siblings).
SAMPLE_RIGS = ("Flamingo", "BrownBear", "Crab", "Scorpion")


@pytest.fixture(scope="session")
def corpus_root() -> Path:
    if not (CORPUS / "source").is_dir() and not (CORPUS / "Truebone_Z-OO").is_dir():
        pytest.skip("Truebones corpus not present")
    return CORPUS


@pytest.fixture(scope="session")
def source_root(corpus_root: Path) -> Path:
    """The raw corpus, under whichever name it currently has."""
    for name in ("source", "Truebone_Z-OO"):
        candidate = corpus_root / name
        if candidate.is_dir():
            return candidate
    pytest.skip("no raw corpus directory")


@pytest.fixture
def raw_clips(source_root: Path):
    """`raw_clips(rig)` -> sorted list of that rig's raw .bvh paths."""

    def _get(rig: str) -> list[Path]:
        paths = sorted((source_root / rig).glob("*.bvh"))
        if not paths:
            pytest.skip(f"no raw clips for {rig}")
        return paths

    return _get

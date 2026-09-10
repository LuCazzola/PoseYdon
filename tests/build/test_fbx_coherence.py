"""The all-takes filter, and the pairing the FBX coherence sidecheck runs on.

Truebones ships, per rig, one compilation FBX holding every take of that rig
concatenated -- ``GoatAll.fbx``, ``scorpionALL.fbx``, ``Flamingo-ALL.fbx`` --
next to the per-clip files. It is not a distinct action and has no BVH
counterpart, so every consumer of the raw corpus has to exclude it.

Two consumers each grew their own approximation of that rule and each one was
wrong in its own direction:

* ``scripts/process_dataset_truebones_fbx.py`` used ``stem.lower().endswith("all")``,
  a SUFFIX test, which also dropped 13 real clips whose stem merely ends in
  those three letters -- twelve ``-Fall`` takes plus ``DEER-WalkCall``, which
  is the one that proves the mechanism is the suffix and not the word "fall".
* ``tests/build/test_roundtrip_fbx.py`` used ``"ALL" not in stem.upper()``, a
  SUBSTRING test, which additionally excluded every per-clip file of a rig
  whose export prefix contains "ALL" -- leaving Alligator, Anaconda, Crow,
  HermitCrab, Lion, SabreToothTiger and Tukan with no candidate at all, so the
  parametrised case for Tukan (a ``SAMPLE_RIGS`` rig) skipped green.

The convention the corpus actually follows is that a bundle's stem is the
rig's export PREFIX plus "ALL" and nothing else, so both now call one shared
predicate, :func:`poseydon.build.index.is_all_takes_bundle`, tested here.
"""

from __future__ import annotations

import pytest
from scripts.check_fbx_coherence import paired_clips

from poseydon.build.index import is_all_takes_bundle
from tests.conftest import CORPUS, SAMPLE_RIGS

# Every stem in the raw corpus that ends in the letters "all" but is a real,
# separate take. `(rig, stem)`; the rig is the source DIRECTORY it lives in.
REAL_CLIPS_ENDING_IN_ALL = [
    ("Buffalo", "Buffalo-Fall"),
    ("Camel", "Camel-Fall"),
    ("Deer", "DEER-WalkCall"),
    ("Gazelle", "Gazelle-Fall"),
    ("PolarBearB", "PolarBearB-Fall"),
    ("Raptor2", "Raptor-Fall"),
    ("Raptor2", "Raptor-FenceClimbFall"),
    ("Raptor2", "Raptor-RunFall"),
    ("Raptor2", "Raptor-RunJumpFall"),
    ("Roach", "roach-Fall"),
    ("Stego", "Stego-Fall"),
    ("Tricera", "Tricera-Fall"),
    ("Tyranno", "Tyranno-Fall"),
]

# The genuine compilations. Truebones exports several rigs under an alias, so
# the prefix is NOT always the rig's directory name: Dragon ships `WyvernALL`,
# Jaws ships `SharkALL`, SabreToothTiger ships `SABREALL`, HermitCrab ships
# `CrabAll`. A predicate that only ever accepted the rig's own name would turn
# those four bundles back into clips -- the opposite bug.
ALL_TAKES_BUNDLES = [
    ("Deer", "DEERALL"),
    ("Goat", "GoatAll"),
    ("Camel", "Camel_ALL"),
    ("Flamingo", "Flamingo-ALL"),
    ("Cat", "Cat-ALL"),
    ("Scorpion", "scorpionALL"),
    ("PolarBearB", "PolarBearBALL"),
    ("Raptor2", "RaptorALL"),
    ("Tukan", "TUKANALL"),
    ("Anaconda", "AnacondaALL"),
    ("SabreToothTiger", "SABREALL"),
    ("Dragon", "WyvernALL"),
    ("Jaws", "SharkALL"),
    ("HermitCrab", "CrabAll"),
]


@pytest.mark.parametrize("rig,stem", REAL_CLIPS_ENDING_IN_ALL)
def test_a_take_that_merely_ends_in_all_is_not_a_bundle(rig, stem):
    assert not is_all_takes_bundle(stem, rig=rig), (
        f"{rig}/{stem} is a real take, not an all-takes compilation"
    )


@pytest.mark.parametrize("rig,stem", ALL_TAKES_BUNDLES)
def test_the_export_prefix_plus_all_is_a_bundle(rig, stem):
    assert is_all_takes_bundle(stem, rig=rig), (
        f"{rig}/{stem} is Truebones' all-takes compilation for that rig"
    )


def test_a_take_that_merely_starts_with_the_bundle_name_is_not_a_bundle():
    """Anaconda ships `AnacondaALL-Twistrattle.fbx` beside `AnacondaALL.fbx`."""
    assert not is_all_takes_bundle("AnacondaALL-Twistrattle", rig="Anaconda")


def test_the_predicate_matches_the_raw_corpus():
    """Over the whole raw corpus: exactly one bundle per rig that ships one.

    A rig never has two compilations, so a predicate that swept up extra files
    would show here as a rig with two -- and one that missed the compilation
    would show as a stem ending in "ALL" that survived the filter with no
    hyphen or underscore separating a take name from the prefix.
    """
    source = CORPUS / "source"
    if not source.is_dir():
        pytest.skip("Truebones corpus not present")
    for rig_dir in sorted(p for p in source.iterdir() if p.is_dir()):
        bundles = [
            p.stem
            for p in sorted(rig_dir.glob("*.fbx"))
            if is_all_takes_bundle(p.stem, rig=rig_dir.name)
        ]
        assert len(bundles) <= 1, f"{rig_dir.name}: more than one bundle {bundles}"


@pytest.mark.parametrize("rig", SAMPLE_RIGS)
def test_every_sample_rig_has_a_per_clip_source_fbx(rig):
    """The precondition `test_roundtrip_fbx.py` silently skipped on for Tukan.

    Its old substring filter dropped every Tukan file (the export prefix is
    `TUKAN`, so `"ALL" in stem.upper()` is true of all of them), which made the
    round trip's Tukan case skip -- green, and asserting nothing.
    """
    source_dir = CORPUS / "source" / rig
    if not source_dir.is_dir():
        pytest.skip(f"{rig}: no source directory")
    per_clip = [
        p for p in sorted(source_dir.glob("*.fbx")) if not is_all_takes_bundle(p.stem, rig=rig)
    ]
    assert per_clip, f"{rig}: no per-clip source FBX survived the all-takes filter"


# --------------------------------------------------------------- the sidecheck

# The pairing half of `scripts/check_fbx_coherence.py` is pure path work and
# runs in the CPU image; the measurement half needs Blender and is exercised by
# running the script in the `fbx` container (see its docstring).
PAIRING_RIGS = ("Goat", "Crab", "Flamingo", "Camel", "Scorpion")


@pytest.mark.parametrize("rig", PAIRING_RIGS)
def test_the_sidecheck_pairs_a_bvh_and_an_fbx_per_action(rig):
    if not (CORPUS / "source" / rig).is_dir():
        pytest.skip(f"{rig}: no source directory")
    stage, pairs = paired_clips(rig, CORPUS)
    assert pairs, f"{rig}: the sidecheck found nothing to compare ({stage})"
    for action, bvh_path, fbx_path in pairs:
        assert bvh_path.suffix == ".bvh" and fbx_path.suffix == ".fbx"
        # Both arms of a comparison must come from the SAME stage: a prepared
        # BVH against a raw FBX would differ by the whole prepare chain.
        assert bvh_path.parent == fbx_path.parent, f"{rig}/{action}: mixed stages"
        assert not is_all_takes_bundle(fbx_path.stem, rig=rig)


@pytest.mark.parametrize("rig", ("BrownBear", "Tukan"))
@pytest.mark.xfail(
    strict=True,
    reason="Truebones exports these two rigs' FBX under a prefix that is not the "
    "rig name and not strippable by `strip_skeleton_prefix` -- `BEAR-Attack.fbx` "
    "against `__Attack.bvh`, and `TUKANALL-Fly.fbx` against `__Fly.bvh` -- so the "
    "two arms derive different action names and nothing pairs. Closing it needs an "
    "authored alias per rig (design spec, Limitations), not a filename rule; "
    "recorded here so the sidecheck's silent `no clip pairs` for them is not "
    "mistaken for agreement. Strict: this must start failing when an alias lands.",
)
def test_the_sidecheck_pairs_alias_prefixed_rigs(rig):
    if not (CORPUS / "source" / rig).is_dir():
        pytest.skip(f"{rig}: no source directory")
    _stage, pairs = paired_clips(rig, CORPUS)
    assert pairs

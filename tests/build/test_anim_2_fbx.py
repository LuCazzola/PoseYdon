from pathlib import Path
"""`scripts/anim_2_fbx.py`'s inversion half, with no Blender involved.

The export itself is covered in `tests/build/test_roundtrip_fbx.py`, which
needs Blender and is gated on it. What is covered HERE is everything the
export depends on and Blender has nothing to do with: reading a rig's
recorded transform back out of `skeleton.npz`, rebuilding the reduction that
`stats.npz` was fitted through, and putting a generated clip back onto the
joints the user supplied.

`tests/build/test_roundtrip.py` already proves the chain inverts. This proves
the SHIPPING path assembles the same parameters from the corpus artefacts --
a different claim, and the one an export actually rests on: that round trip
fits its chain in-process from the raw source, while `anim_2_fbx` has only
`rigs/<Rig>/skeleton.npz` and `rigs/<Rig>/prepare.npz` to work from.
"""

from __future__ import annotations

import numpy as np
import pytest
from scripts.anim_2_fbx import recorded_facing, rig_transform, unprepare

from poseydon.build.index import action_slug, strip_skeleton_prefix
from poseydon.core.animation import Animation
from poseydon.features.reduce import apply_reduction, build_reduction
from poseydon.io.bvh import BVH
from tests.conftest import CORPUS, SAMPLE_RIGS

# The prepared clip this compares against is not an in-memory animation: it
# reached `clips/<Rig>/<action>.npz` through `build_clips`, which reads the
# prepared **BVH text** stage 1 wrote -- and `io/bvh.py:341` formats every
# offset and every Euler angle as `.6f`. So the reference itself is quantized
# at 1e-6 before this test sees it, and no re-preparation can agree with it
# more closely than that quantization accumulated down the hierarchy.
# `tests/build/test_roundtrip.py` asserts the same property at 1e-9 because it
# never goes near a file. Measured worst case across the six rigs: 2.34e-6
# (BrownBear), i.e. 1.0e-6 of that rig's own extent; Flamingo 2.25e-6, Crab
# 2.23e-6, Camel 2.09e-6, Scorpion 1.63e-6, Tukan 1.31e-6. Set roughly an
# order of magnitude above that worst case -- tight enough that a genuine
# parameter mix-up (which moves joints by bone lengths, not microns) fails.
REPREPARE_ATOL = 2e-5


def _pair(rig: str):
    """``(prepared clip .npz, the source .bvh it came from)`` for one clip.

    Matched by ACTION, derived the way `process_dataset_truebones.py` derives
    it, rather than by position in a sorted listing: stage 1 drops clips
    rigged differently from their own rest pose (Elephant's `__Take_001`), so
    the two directories do not line up index for index.
    """
    clips = CORPUS / "clips" / rig
    source_dir = Path("data/_raw/truebones/source") / rig
    if not clips.is_dir() or not source_dir.is_dir():
        pytest.skip(f"{rig}: no prepared corpus")
    for path in sorted(source_dir.glob("*.bvh")):
        prepared = clips / f"{strip_skeleton_prefix(action_slug(path.stem), rig)}.npz"
        if prepared.is_file():
            return prepared, path
    pytest.skip(f"{rig}: no prepared clip with a source .bvh beside it")


def _reduced(prepared_path):
    """A prepared clip in the shape a model's output arrives in."""
    prepared = Animation.load(prepared_path).as_rigid_body(joint_translation="drop")
    reduction = build_reduction(prepared)
    reduced = apply_reduction(prepared, reduction)
    # A reduction that degenerated to the identity would satisfy every
    # assertion below without exercising the expansion at all.
    assert reduced.n_joints < prepared.n_joints
    return prepared, reduced


@pytest.mark.parametrize("rig", SAMPLE_RIGS)
def test_unprepare_returns_the_joints_the_user_supplied(rig):
    """Spec section 9's structure guarantee, reached through the corpus artefacts."""
    prepared_path, source_path = _pair(rig)
    _prepared, reduced = _reduced(prepared_path)

    restored = unprepare(reduced, rig, CORPUS)

    source = BVH.read(source_path).to_animation()
    assert restored.names == source.names, f"{rig}: joint names are not the source's"
    assert list(restored.parents) == list(source.parents), f"{rig}: hierarchy differs"
    np.testing.assert_allclose(
        restored.offsets,
        source.offsets,
        atol=1e-9,
        err_msg=f"{rig}: bone offsets are not the ones the user authored",
    )


@pytest.mark.parametrize("rig", SAMPLE_RIGS)
def test_recorded_facing_reproduces_the_prepared_clip(rig):
    """With the clip's own yaw put back, the loop closes in world space.

    `FaceAxis` is the one clip-scoped stage, so this is the assertion that
    catches a parameter set assembled from the wrong file or the wrong clip:
    every other stage is rig-scoped and would be equally wrong for all of
    them, while a misread facing is wrong for exactly this one.
    """
    prepared_path, _source_path = _pair(rig)
    prepared, reduced = _reduced(prepared_path)
    action = prepared_path.stem

    facing = recorded_facing(CORPUS, rig, action)
    restored = unprepare(reduced, rig, CORPUS, facing=facing)

    # Re-prepare with the recorded parameters, refitting nothing: `FaceAxis`
    # derives its rotation from frame-0 world positions, and `EnforceRigid`
    # replaced per-joint translation, so a refit would not reproduce them.
    chain, rig_params, _reduction = rig_transform(CORPUS, rig)
    params = {**rig_params, "face_axis": {"rotation": facing}}
    current = restored
    for stage in chain.stages:
        current = stage.apply(current, params[stage.name])

    np.testing.assert_allclose(
        current.global_positions(),
        prepared.global_positions(),
        rtol=0,
        atol=REPREPARE_ATOL,
        err_msg=f"{rig}: re-preparing the returned asset did not reproduce the clip",
    )


def test_unprepare_refuses_a_clip_from_another_rig():
    """A mismatch must name the problem, not surface as `tuple.index` failing.

    The expansion looks joints up BY NAME, so handing it another rig's clip
    fails deep inside `_reinsert` with `ValueError: tuple.index(x): x not in
    tuple` -- which says nothing about which rig was asked for.
    """
    content_path, _ = _pair("Flamingo")
    _prepared, reduced = _reduced(content_path)

    with pytest.raises(ValueError, match="Scorpion"):
        unprepare(reduced, "Scorpion", CORPUS)

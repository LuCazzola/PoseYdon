"""FBX IO-level round trips: what survives `FBX.read`/`.write`/`.from_animation`.

This covers the FBX IO layer ONLY (`poseydon.io.fbx`). Two round trips live
here, and they test different halves of it:

* `read -> write -> read` preserves the skeleton -- the original case, on the
  import/export settings `FBX.read` and `FBX.write` document.
* `Animation -> from_animation -> write -> read` preserves the posed skeleton
  -- the export half of `scripts/anim_2_fbx.py`. Its INVERSION half needs no
  Blender and is tested in `tests/build/test_anim_2_fbx.py` instead.

What is still absent is an FBX *reader* into an `Animation`: `FBX` has
`from_animation` but no `to_animation`, and
`scripts/process_dataset_truebones_fbx.py` calls
`FBX.read -> rotate -> scale_to_mean_bone_length -> write` directly on a Blender
scene without going through `PrepareChain` or recording fitted parameters -- so
an FBX-sourced corpus still has nothing to unprepare. The BVH-sourced one does,
which is the path `anim_2_fbx` takes.

Blender-gated: Dockerfile.compat and the test image deliberately ship no
Blender, so this runs against the `fbx` compose service, which installs
`python3-pytest` (Debian's distro package, importable from Blender's linked
system Python the same way PyYAML/numpy/scipy are) for exactly this purpose:

    docker compose run --rm fbx blender -b --python-expr \\
        "import sys, pytest; sys.exit(pytest.main(['tests/build/test_roundtrip_fbx.py','-v','-rs']))"

Skipping when Blender is absent is correct here -- the data is genuinely not
available -- but a test that can ONLY ever skip is worthless (this plan exists
because exactly that happened to a prior round-trip test for weeks). This one
is confirmed to actually execute and report failures inside the `fbx` image;
see task-8-report.md for the real pytest transcript and a deliberate-failure
check.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from scripts.anim_2_fbx import unprepare

from poseydon.build.index import is_all_takes_bundle
from poseydon.core.animation import Animation
from poseydon.core.rotations import QUAT_IDENTITY
from poseydon.features.reduce import apply_reduction, build_reduction
from tests.conftest import CORPUS, SAMPLE_RIGS

bpy = pytest.importorskip("bpy", reason="FBX round trip needs Blender")


from poseydon.io.fbx import FBX, _to_y_up

# Measured tolerance: joint offsets are bind-pose positions Blender derives
# from the armature's own bone-head matrices, not a value FBX stores and
# echoes back verbatim, so a read -> write -> read is not expected to be
# bit-exact. Measured directly (FBX.read -> FBX.write -> FBX.read) on the
# first per-clip source FBX of each SAMPLE_RIGS rig, in the rig's own
# (unscaled) units: max per-component offset error was 1.79e-7 (Flamingo),
# 2.38e-7 (BrownBear), 5.80e-6 (Crab, the worst), 3.58e-7 (Scorpion) and
# 7.15e-7 (Camel) -- all consistent with float32 round-off through Blender's
# FBX exporter/importer, not a structural drift. This bound is set roughly an
# order of magnitude above that measured worst case (Crab's 5.80e-6).
OFFSET_ATOL = 5e-5


def _first_per_clip_fbx(rig: str):
    source_dir = Path("data/_raw/truebones/source") / rig
    if not source_dir.is_dir():
        pytest.skip(f"{rig}: no source directory")
    # "ALL" files are Truebones' own all-takes compilations (e.g.
    # `Flamingo-ALL.fbx`, `CrabAll.fbx`, `RaptorALL.fbx`) -- not a distinct
    # clip. `process_dataset_truebones_fbx.py` excludes them the same way,
    # through this same predicate. It used to be `"ALL" not in stem.upper()`
    # here, a SUBSTRING test that excluded every file of a rig whose export
    # prefix contains "ALL" -- Tukan's is `TUKAN`, so this rig's case fell
    # through to the skip below and passed green while asserting nothing.
    fbxs = sorted(
        p for p in source_dir.glob("*.fbx") if not is_all_takes_bundle(p.stem, rig)
    )
    if not fbxs:
        pytest.skip(f"{rig}: no per-clip source FBX")
    return fbxs[0]


@pytest.mark.parametrize("rig", SAMPLE_RIGS)
def test_fbx_read_write_read_preserves_the_skeleton(rig, tmp_path):
    source_path = _first_per_clip_fbx(rig)

    source = FBX.read(source_path)
    source_names = source.joint_names
    source_parents = source.joint_parents
    source_offsets = source.joint_offsets

    out_path = tmp_path / "roundtrip.fbx"
    source.write(out_path)
    restored = FBX.read(out_path)

    assert restored.joint_names == source_names, (
        f"{rig}: joint names changed across an FBX read -> write -> read"
    )
    assert restored.joint_parents == source_parents, (
        f"{rig}: joint parents changed across an FBX read -> write -> read"
    )
    np.testing.assert_allclose(
        restored.joint_offsets,
        source_offsets,
        atol=OFFSET_ATOL,
        err_msg=f"{rig}: bind-pose joint offsets did not survive an FBX round trip",
    )


# `FBX.from_animation` places every joint's head from `anim.global_transforms()`
# and lets the exporter bake the curves, so the loss here is the exporter's
# float32 keyframes plus `bake_anim_simplify_factor=0.01`'s decimation --
# `FBX.write` documents that pairing and measures its own max positional error
# at 0.00004.
#
# Stated as a FRACTION of the rig's extent rather than as an absolute, because
# an animation reaches this function in whatever units it carries: prepared
# clips sit at a 0.209 mean bone length, while one that has been through
# `scripts/anim_2_fbx.py` is back in the source's own (Scorpion's offsets run
# to 8.4, and its extent to 398). A single absolute bound would be slack for
# one and impossible for the other.
#
# Measured over frames 1, n/2 and n of the first prepared clip of each rig:
# Flamingo 3.34e-6 of extent (the worst, and the longest clip at 201 frames),
# BrownBear 1.47e-6, Tukan 1.29e-6, Camel 7.23e-7, Crab 6.65e-7, Scorpion
# 5.51e-7. Set roughly an order of magnitude above that worst case.
POSE_RTOL_OF_EXTENT = 3e-5


def _prepared_clip(rig: str):
    """One prepared clip of `rig`, as an animation. Skips if the corpus is absent."""
    clips = sorted((CORPUS / "clips" / rig).glob("*.npz")) if (
        CORPUS / "clips" / rig
    ).is_dir() else []
    if not clips:
        pytest.skip(f"{rig}: no prepared clips")
    return Animation.load(clips[0])


@pytest.mark.parametrize("rig", SAMPLE_RIGS)
def test_from_animation_round_trips_the_posed_skeleton(rig, tmp_path):
    """`Animation -> FBX -> read` puts every joint where the animation had it.

    The claim `scripts/anim_2_fbx.py` rests on, and the half of the export no
    amount of numpy can check: that Blender's armature, keyed through
    `matrix_basis` against rest matrices it derived itself, reproduces the world
    transforms it was built from.

    End Sites are excluded on both sides rather than skipped past: they are not
    bones in the written file BY DESIGN (see `FBX.from_animation`), so the
    comparison is over the joints that are.
    """
    anim = _prepared_clip(rig)
    has_child = np.zeros(anim.n_joints, dtype=bool)
    real = anim.parents >= 0
    has_child[anim.parents[real]] = True
    bones = [j for j in range(anim.n_joints) if has_child[j]]

    out_path = tmp_path / "generated.fbx"
    FBX.from_animation(anim, name=rig).write(out_path)
    restored = FBX.read(out_path)

    assert restored.joint_names == [anim.names[j] for j in bones], (
        f"{rig}: the exported bones are not this animation's real joints"
    )

    expected = anim.global_positions()
    extent = float(np.ptp(expected.reshape(-1, 3), axis=0).max())
    # Frames 1 and n and one in the middle: the exporter's decimation is a
    # per-curve decision, so a sample spanning the clip catches it where a
    # single frame would not, without paying for every frame of every rig.
    for frame in (1, anim.n_frames // 2, anim.n_frames):
        restored.set_frame(frame)
        got = np.stack(
            [
                _to_y_up(restored.joint_position(anim.names[j]))
                for j in bones
            ]
        )
        np.testing.assert_allclose(
            got,
            expected[frame - 1][bones],
            atol=POSE_RTOL_OF_EXTENT * extent,
            rtol=0,
            err_msg=f"{rig}: joint positions drifted at frame {frame}",
        )


# `apply_animation` aligns the clip's T-pose onto the asset's bind pose and
# reports the fit residual in bone lengths. Measured on every per-clip source
# FBX of each rig -- the value is a property of the RIG, identical across its
# takes, so the asset chosen does not matter:
#
#   Flamingo 4.16e-5   BrownBear 2.76e-5   Scorpion 1.22e-5
#   Tukan 8.87e-1      Camel 1.95e0        Crab 2.38e0
#
# The split is four orders of magnitude wide and is a property of the CORPUS,
# not of this code: the three large ones are rigs whose BVH pipeline and FBX
# asset genuinely describe different skeletons, and each is xfailed below with
# what is known about it. `apply_animation` refuses them rather than draping a
# skin across the difference, which is why the bound sits between the groups.
BIND_ATOL_IN_BONE_LENGTHS = 1e-3

#: Rigs whose prepared BVH corpus and source FBX do not describe one skeleton,
#: so no skinned export is possible for them until the corpus is fixed. Two are
#: already xfailed for the same disagreement in
#: `tests/build/test_bvh_fbx_agreement.py`; Tukan is not, because that test
#: compares against `mesh.npz` and Tukan has none -- it SKIPS there, so this is
#: the first measurement of Tukan's agreement anywhere in the suite.
SKELETON_DISAGREEMENT = {
    "Crab": (
        "Crab's BVH T-pose declares a different skeleton from its animations "
        "(54 joints against 64), so its prepared corpus is built against a rest "
        "pose its own clips do not share. Bind fit 2.38 bone lengths. Already "
        "xfailed for the same disagreement in test_bvh_fbx_agreement.py, both "
        "halves."
    ),
    "Camel": (
        "Camel's prepared corpus and its FBX bind pose disagree by 1.95 bone "
        "lengths -- the same disagreement test_bvh_fbx_agreement.py xfails at "
        "1.56 against mesh.npz. Measured identical across all 18 of Camel's "
        "per-clip source FBX, so it is the corpus side, not the asset picked."
    ),
    "Tukan": (
        "Tukan's prepared corpus and its FBX bind pose disagree by 0.887 bone "
        "lengths. NOT previously known: test_bvh_fbx_agreement.py skips Tukan "
        "for want of a mesh.npz, so nothing in the suite had ever compared the "
        "two for this rig. Recorded here; diagnosing it is corpus work."
    ),
}


def _moving_clip(rig: str):
    """The prepared clip of `rig` with the widest rotation range.

    Not `sorted(...)[0]`: that is an alphabetical accident, and a rig whose
    first clip happens to be a T-pose would let
    `test_skinned_export_carries_the_mesh_and_moves_it` assert motion against a
    clip that has none.
    """
    clips = sorted((CORPUS / "clips" / rig).glob("*.npz")) if (
        CORPUS / "clips" / rig
    ).is_dir() else []
    if not clips:
        pytest.skip(f"{rig}: no prepared clips")
    def spread(path):
        with np.load(path) as data:
            return float(np.ptp(data["rotations"], axis=0).max())
    return max(clips, key=spread)


def _rest_relative(rig: str):
    """A prepared clip of `rig`, inverted back to the source's units and facing.

    Stops at `rest_relative` exactly as `scripts/anim_2_fbx.py --skin` does, so
    what is tested is the animation the shipping path actually hands to
    `apply_animation` -- the un-promoted joint set included, which is why this
    reaches Camel and Tukan at all where a raw prepared clip would not.
    """
    prepared = Animation.load(_moving_clip(rig)).as_rigid_body(joint_translation="drop")
    reduced = apply_reduction(prepared, build_reduction(prepared))
    return unprepare(reduced, rig, CORPUS, stop_before="rest_relative")


def _mean_bone_length(scene) -> float:
    lengths = np.linalg.norm(scene.joint_offsets[1:], axis=-1)
    real = lengths[lengths > 1e-8]
    return float(real.mean()) if real.size else 1.0


def _deformed(scene) -> np.ndarray:
    """Every mesh as the armature modifier leaves it, merged, in PoseYdon Y-up.

    The armature modifier only ever deforms the EVALUATED mesh, so `obj.data`
    is the bind pose no matter which frame is current -- which is exactly why
    `FBX.bind_pose_arrays` can read a bind pose without posing anything, and
    why this has to go through the depsgraph to see the posed one. Merged in
    `scene.meshes` order and converted to Y-up so the two are comparable
    row for row.
    """
    rows = []
    for obj in scene.meshes:
        evaluated = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
        rows.extend(
            _to_y_up(evaluated.matrix_world @ v.co) for v in evaluated.data.vertices
        )
    return np.stack(rows)


@pytest.mark.parametrize("rig", SAMPLE_RIGS)
def test_apply_animation_leaves_an_unrotated_clip_in_the_bind_pose(rig):
    """Identity rotation must mean the bind pose, or every skinned frame deforms.

    The one claim the whole `--skin` path rests on: `RestRelative` made zero
    mean "T-pose" on the corpus side, and an FBX bind pose means it on the asset
    side. If those two zeros disagree -- by a bone-frame convention, a stale
    rig, a missed inverse stage -- the character comes out mangled on EVERY
    frame, including ones that look plausible in a skeleton view. Driving with
    an all-identity clip is what isolates that from the motion.
    """
    if rig in SKELETON_DISAGREEMENT:
        pytest.xfail(SKELETON_DISAGREEMENT[rig])

    source_path = _first_per_clip_fbx(rig)
    anim = _rest_relative(rig)
    unrotated = dataclasses.replace(
        anim, rotations=np.tile(QUAT_IDENTITY, (anim.n_frames, anim.n_joints, 1))
    )

    scene = FBX.read(source_path)
    if not scene.meshes:
        pytest.skip(f"{rig}: source FBX carries no mesh")
    bind = scene.bind_pose_arrays()[0]
    tolerance = BIND_ATOL_IN_BONE_LENGTHS * _mean_bone_length(scene)

    scene.apply_animation(unrotated)
    scene.set_frame(1)

    np.testing.assert_allclose(
        _deformed(scene),
        bind,
        atol=tolerance,
        rtol=0,
        err_msg=f"{rig}: an unrotated clip moved the skin off the bind pose",
    )


@pytest.mark.parametrize("rig", SAMPLE_RIGS)
def test_skinned_export_carries_the_mesh_and_moves_it(rig, tmp_path):
    """`--skin` must write a character, not a skeleton -- and animate it.

    Both halves matter. A file with the mesh but no keys on the bones is a
    statue and would pass any check that only counts vertices; a file with keys
    and no mesh is what the bare-skeleton export already produces, and is the
    thing this option exists to stop being the only choice.
    """
    if rig in SKELETON_DISAGREEMENT:
        pytest.xfail(SKELETON_DISAGREEMENT[rig])

    source_path = _first_per_clip_fbx(rig)
    anim = _rest_relative(rig)

    scene = FBX.read(source_path)
    expected = [(len(o.data.vertices), len(o.data.polygons)) for o in scene.meshes]
    if not expected:
        pytest.skip(f"{rig}: source FBX carries no mesh to skin")
    scene.apply_animation(anim)
    out_path = tmp_path / "skinned.fbx"
    scene.write(out_path)

    restored = FBX.read(out_path)
    assert [
        (len(o.data.vertices), len(o.data.polygons)) for o in restored.meshes
    ] == expected, f"{rig}: the mesh did not survive the skinned export"
    assert any(m.type == "ARMATURE" for m in restored.meshes[0].modifiers), (
        f"{rig}: the exported mesh is not bound to the armature"
    )

    # Sampled across the clip, not just first-against-last: these are idle and
    # action loops, so a clip that moves a great deal can still END where it
    # began. Flamingo's `onelegbent` does exactly that -- 4.35e-3 bone lengths
    # between its first and last frame, which read as a statue.
    restored.set_frame(1)
    first = _deformed(restored)
    travel = 0.0
    for frame in np.linspace(1, anim.n_frames, 5, dtype=int):
        restored.set_frame(int(frame))
        travel = max(travel, float(np.abs(_deformed(restored) - first).max()))
    travel /= _mean_bone_length(restored)
    assert travel > 1e-2, (
        f"{rig}: the skin moved at most {travel:.2e} bone lengths from its first "
        "frame -- the export is a statue, not an animation"
    )

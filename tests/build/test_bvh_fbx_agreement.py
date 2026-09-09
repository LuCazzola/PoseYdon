"""The two prepared corpora must describe the same skeleton.

Split in two because the two claims tangled here have different statuses.
Joint-set SHAPE and parent-by-name HIERARCHY hold and are enforced. Rest
GEOMETRY at shared joints was parked as an unexplained disagreement (up to
8.1e-1 bone lengths on Flamingo) for one review cycle; it turned out to be
mostly Critical 1 of the final review, on the BVH side -- ScaleToMeanBoneLength
averaged over zero-length End Sites, inflating its factor by a rig-dependent
(J_all-1)/(J_real-1), while `FBX.write_mesh_npz` computed its factor over the
End-Site-free FBX joint set. Comparing an arbitrary BVH clip's OFFSET block
against the FBX bind pose was also wrong on its own terms (`FaceAxis` rotates
per clip, `mesh.npz`'s bind pose was faced from the rest pose alone) -- see
`_rest_path` below. Fixing both cut the disagreement roughly tenfold (Flamingo
8.1e-1 -> 6.8e-2, etc.) but did not close it; the residue is re-parked behind
the `xfail` on `test_bvh_and_fbx_agree_on_rest_geometry` below, with the
current measurements in its reason -- a Phase 2 problem (skin weights are
indexed by the FBX joint order, so the disagreement means weights applied to
BVH-driven motion would deform wrongly), not something resolved here.

Skips without the FBX artefacts, because the test image has no Blender.
"""

from __future__ import annotations

import numpy as np
import pytest
from scripts.process_dataset_truebones import rest_action

from poseydon.core.skeleton import HML_MEAN_BONE_LENGTH, SkeletonManifest
from poseydon.io.bvh import BVH
from tests.conftest import CORPUS, SAMPLE_RIGS


def _rest_path(bvhs):
    """The rig's prepared rest clip, named by its manifest.

    `FaceAxis` is clip-scoped and `rotate_rig` turns OFFSETS with the motion, so
    every prepared clip carries a differently-rotated offset block -- only the
    rest clip's offsets correspond to mesh.npz's bind pose. Matching on the
    filename was a stale copy of a rule that no longer holds: Crab's rest clip
    is `walk`.
    """
    rig = bvhs[0].parent.name
    manifest = SkeletonManifest.load(CORPUS / "rigs" / rig / "manifest.yaml")
    path = bvhs[0].parent / f"{rest_action(manifest)}.bvh"
    if not path.is_file():
        pytest.skip(f"{rig}: declared rest clip {path.name} is not in the prepared corpus")
    return path


def _pair(rig: str):
    clips = CORPUS / "clips" / rig
    bvhs = sorted(clips.glob("*.bvh"))
    if not bvhs:
        pytest.skip(f"{rig}: no prepared BVH")
    rest = _rest_path(bvhs)
    if rest is None:
        pytest.skip(f"{rig}: no rest-pose clip in the prepared corpus")
    mesh = CORPUS / "rigs" / rig / "mesh.npz"
    if not mesh.is_file():
        pytest.skip(f"{rig}: no mesh.npz -- run the fbx container")
    return rest, mesh


@pytest.mark.parametrize("rig", SAMPLE_RIGS)
def test_bvh_and_fbx_agree_on_the_skeleton_structure(rig):
    """Joint-set shape and hierarchy. This half holds and is enforced."""
    if rig == "Crab":
        pytest.xfail(
            "Crab's BVH T-pose reference declares a different skeleton from its "
            "animations (54 joints against 64), so its prepared corpus is the "
            "T-pose alone and carries a non-leaf joint the FBX bind pose lacks. "
            "A corpus data problem, recorded in the design spec's Limitations."
        )

    bvh_path, mesh_path = _pair(rig)
    anim = BVH.read(bvh_path).to_animation()
    with np.load(mesh_path, allow_pickle=True) as mesh:
        fbx_names = tuple(str(n) for n in mesh["joint_names"])
        fbx_parents = mesh["joint_parents"]

    fbx_set, bvh_set = set(fbx_names), set(anim.names)

    # BVH mandates an End Site per leaf bone and FBX has no counterpart, so the
    # two joint lists cannot be equal. Assert the SHAPE of the difference
    # instead: everything BVH carries that FBX does not must be a childless,
    # zero-offset End Site -- precisely the class joint reduction drops.
    has_child = np.zeros(anim.n_joints, dtype=bool)
    real = anim.parents >= 0
    has_child[anim.parents[real]] = True
    for name in (n for n in anim.names if n not in fbx_set):
        j = anim.names.index(name)
        assert not has_child[j], f"{rig}: `{name}` is BVH-only but has children"
        assert np.linalg.norm(anim.offsets[j]) < 1e-8, (
            f"{rig}: `{name}` is BVH-only but has a real offset"
        )

    # FBX must never carry a joint the BVH lacks.
    assert [n for n in fbx_names if n not in bvh_set] == []

    # Shared joints must agree on hierarchy, by NAME -- the index spaces
    # differ by exactly the End Sites, so an index comparison would be
    # meaningless.
    for name in (n for n in anim.names if n in fbx_set):
        b, f = anim.names.index(name), fbx_names.index(name)
        if anim.parents[b] >= 0 and fbx_parents[f] >= 0:
            assert anim.names[anim.parents[b]] == fbx_names[fbx_parents[f]], (
                f"{rig}: `{name}` has different parents in BVH and FBX"
            )


@pytest.mark.parametrize("rig", SAMPLE_RIGS)
def test_bvh_and_fbx_agree_on_rest_geometry(rig):
    """Rest offsets at shared joints, compared on the rig's T-pose clip.

    Used to be a blanket `xfail(strict=False)` across every sample rig, at
    "3-7% of a bone length: Flamingo 6.8e-2, BrownBear 5.5e-2, Scorpion 3.3e-2"
    -- residue left after fixing the dominant cause (`ScaleToMeanBoneLength`
    averaging over zero-length End Sites). Task 9's full corpus rebuild closed
    that residue for those three rigs to ~1-2e-5 (three orders of magnitude),
    so the assertion is enforced for them now rather than parked behind a
    marker that asserts nothing on either a pass OR a regression back to 6.8e-2.

    Crab alone stays known-failing, and by a lot: ~1.0 bone lengths at
    `BN_Leg_R_12`, not the 2.0e-5 the old blanket reason claimed (that number
    predates the rest-pose-selection fix, when Crab's BVH side compared
    against a different, wrong rest clip). The likely cause is that Crab's
    `mesh.npz` was built by the Blender/FBX pass against whatever rest
    reference it used before that fix, and the BVH side now resolves to
    `walk.bvh` instead -- comparing geometry from two different source clips.
    Regenerating `mesh.npz` needs the Blender/FBX container and belongs to a
    later phase; recorded here rather than fixed."""
    if rig == "Crab":
        pytest.xfail(
            "Crab's mesh.npz is stale against the rest-pose-selection fix: "
            "the BVH side now resolves its rest pose to walk.bvh, but "
            "mesh.npz was built against whatever rest reference the FBX pass "
            "used before that fix, so the two sides compare different source "
            "clips. Disagreement is ~1.0 bone lengths at BN_Leg_R_12 -- needs "
            "mesh.npz regenerated via the Blender/FBX container, not a code "
            "fix here."
        )

    bvh_path, mesh_path = _pair(rig)
    anim = BVH.read(bvh_path).to_animation()
    with np.load(mesh_path, allow_pickle=True) as mesh:
        fbx_names = tuple(str(n) for n in mesh["joint_names"])
        fbx_offsets = mesh["joint_offsets"]

    fbx_set = set(fbx_names)

    # The root is excluded: BVH's root OFFSET is a world placement, FBX's is
    # the bind position, and they are not the same quantity.
    scale = float(np.linalg.norm(anim.offsets[1:], axis=-1).mean())
    worst, worst_name = 0.0, ""
    for name in (n for n in anim.names if n in fbx_set):
        b, f = anim.names.index(name), fbx_names.index(name)
        if b == 0:
            continue
        error = float(np.abs(anim.offsets[b] - fbx_offsets[f]).max())
        if error > worst:
            worst, worst_name = error, name
    assert worst < 1e-3 * scale, (
        f"{rig}: worst rest-offset disagreement {worst / scale:.2e} bone lengths "
        f"at `{worst_name}`"
    )


@pytest.mark.parametrize("rig", SAMPLE_RIGS)
def test_mesh_npz_mean_bone_length_matches_target(rig):
    """``write_mesh_npz`` must scale to HML_MEAN_BONE_LENGTH like the BVH side.

    Regression for the bug where the mean was taken over EVERY entry of
    ``offsets[1:]``, including zero-length End-Site-equivalent joints, which
    inflates the factor by a rig-dependent (J_all-1)/(J_real-1) -- the same
    bug ``ScaleToMeanBoneLength.fit`` was fixed for on the BVH side. Only
    Crab has no zero-length bone in its FBX skeleton, so it is the only rig
    this passed for before the fix.
    """
    mesh_path = CORPUS / "rigs" / rig / "mesh.npz"
    if not mesh_path.is_file():
        pytest.skip(f"{rig}: no mesh.npz -- run the fbx container")

    with np.load(mesh_path, allow_pickle=True) as mesh:
        offsets = mesh["joint_offsets"]

    lengths = np.linalg.norm(offsets[1:], axis=-1)
    real = lengths[lengths > 1e-8]
    mean_length = float(real.mean())
    assert mean_length == pytest.approx(HML_MEAN_BONE_LENGTH, rel=1e-4), (
        f"{rig}: non-degenerate mean bone length {mean_length} != "
        f"HML_MEAN_BONE_LENGTH {HML_MEAN_BONE_LENGTH}"
    )


@pytest.mark.parametrize("rig", SAMPLE_RIGS)
def test_bvh_and_fbx_clips_share_basenames(rig):
    """The plan and the FBX script's own docstring assert this; check it.

    Both scripts derive `strip_skeleton_prefix(action_slug(stem), rig)`, so
    agreement is not automatic -- it depends on the two corpora's raw
    filenames actually sharing a prefix convention, which most but not all
    rigs do.
    """
    if rig == "BrownBear":
        pytest.xfail(
            "BrownBear's BVH files are named `__<Action>.bvh`, stripping the "
            "`brownbear_` prefix (its manifest name, lowercased), while its "
            "FBX files are named `BEAR-<Action>.fbx` -- `strip_skeleton_prefix` "
            "strips the RIG name, not `bear_`, so the two corpora derive "
            "completely different basenames (`attack` vs `bear_attack`) for "
            "the same clip. No filename-derivation rule can close this: it "
            "needs an authored alias, see the design spec's Limitations."
        )
    if rig == "Crab":
        pytest.xfail(
            "Unrelated to naming: Crab's BVH T-pose declares 54 joints against "
            "its animations' 64 (the RestRelative limitation recorded in the "
            "design spec), so only tpose.bvh survives prepare -- 1 BVH clip "
            "against 11 FBX clips, which cannot share a basename set no matter "
            "how either side derives its filename."
        )

    bvh_dir = CORPUS / "clips" / rig
    bvhs = sorted(bvh_dir.glob("*.bvh"))
    if not bvhs:
        pytest.skip(f"{rig}: no prepared BVH")
    fbxs = sorted(bvh_dir.glob("*.fbx"))
    if not fbxs:
        pytest.skip(f"{rig}: no prepared FBX -- run the fbx container")

    bvh_stems = {p.stem for p in bvhs}
    fbx_stems = {p.stem for p in fbxs}
    assert bvh_stems == fbx_stems, (
        f"{rig}: BVH-only {sorted(bvh_stems - fbx_stems)}, "
        f"FBX-only {sorted(fbx_stems - bvh_stems)}"
    )

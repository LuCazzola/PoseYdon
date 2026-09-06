"""The two prepared corpora must describe the same skeleton.

Split in two because the two claims tangled here have different statuses.
Joint-set SHAPE and parent-by-name HIERARCHY hold and are enforced. Rest
GEOMETRY at shared joints does not currently agree -- see the `xfail` on
``test_bvh_and_fbx_agree_on_rest_geometry`` below for the open question and
the measured numbers; this is a recorded Phase 2 problem (skin weights are
indexed by the FBX joint order, so the disagreement means weights applied to
BVH-driven motion would deform wrongly), not something resolved here.

Skips without the FBX artefacts, because the test image has no Blender.
"""

from __future__ import annotations

import numpy as np
import pytest

from poseydon.io.bvh import BVH

from tests.conftest import CORPUS, SAMPLE_RIGS


def _pair(rig: str):
    clips = CORPUS / "clips" / rig
    bvhs = sorted(clips.glob("*.bvh"))
    if not bvhs:
        pytest.skip(f"{rig}: no prepared BVH")
    mesh = CORPUS / "rigs" / rig / "mesh.npz"
    if not mesh.is_file():
        pytest.skip(f"{rig}: no mesh.npz -- run the fbx container")
    return bvhs[0], mesh


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


@pytest.mark.xfail(
    reason=(
        "BVH and FBX disagree on rest geometry at shared joints, unexplained. "
        "Worst disagreement in bone lengths: Flamingo 8.1e-1 at Bip01_R_Foot, "
        "BrownBear 5.5e-1 at Bip01_R_Forearm, Scorpion 4.7e-1 at Bip01_TailNub, "
        "against Crab's 2.0e-5. Errors grow distally along each chain. A "
        "coordinate-frame mismatch was ruled out by measuring parent-local and "
        "world-space forms and finding them identical to 1e-7. BrownBear is the "
        "rig behind the original 'joint positions agree to 0.0000 at frame 0' "
        "claim, so either that measured a different quantity -- world positions "
        "through FK rather than the rest OFFSET block -- or something regressed."
    ),
    strict=False,
)
@pytest.mark.parametrize("rig", SAMPLE_RIGS)
def test_bvh_and_fbx_agree_on_rest_geometry(rig):
    """Rest offsets at shared joints. Known-failing; see the xfail reason."""
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

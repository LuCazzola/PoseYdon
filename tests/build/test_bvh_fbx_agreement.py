"""The two prepared corpora must describe the same skeleton.

Verified once by hand in a commit message; a property this load-bearing needs a
test. Skips without the FBX artefacts, because the test image has no Blender.
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
def test_bvh_and_fbx_agree_on_the_skeleton(rig):
    bvh_path, mesh_path = _pair(rig)
    anim = BVH.read(bvh_path).to_animation()
    with np.load(mesh_path, allow_pickle=True) as mesh:
        fbx_names = tuple(str(n) for n in mesh["joint_names"])
        fbx_parents = mesh["joint_parents"]
        fbx_offsets = mesh["joint_offsets"]

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

    # Shared joints must agree on hierarchy and on rest geometry. The root is
    # excluded from the offset comparison: BVH's root OFFSET is a world
    # placement, FBX's is the bind position, and they are not the same quantity.
    scale = float(np.linalg.norm(anim.offsets[1:], axis=-1).mean())
    worst, worst_name = 0.0, ""
    for name in (n for n in anim.names if n in fbx_set):
        b, f = anim.names.index(name), fbx_names.index(name)
        if anim.parents[b] >= 0 and fbx_parents[f] >= 0:
            assert anim.names[anim.parents[b]] == fbx_names[fbx_parents[f]], (
                f"{rig}: `{name}` has different parents in BVH and FBX"
            )
        if b == 0:
            continue
        error = float(np.abs(anim.offsets[b] - fbx_offsets[f]).max())
        if error > worst:
            worst, worst_name = error, name
    assert worst < 1e-3 * scale, (
        f"{rig}: worst rest-offset disagreement {worst / scale:.2e} bone lengths "
        f"at `{worst_name}`"
    )

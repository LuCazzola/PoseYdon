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
        names = tuple(str(n) for n in mesh["joint_names"])
        parents = mesh["joint_parents"]
        offsets = mesh["joint_offsets"]

    assert anim.names == names
    assert list(anim.parents) == list(parents)

    scale = float(np.linalg.norm(anim.offsets[1:], axis=-1).mean())
    error = np.abs(anim.offsets - offsets).max()
    assert error < 1e-3 * scale, f"{rig}: rest offsets differ by {error / scale:.2e}"

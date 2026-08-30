import numpy as np
import pytest

from poseydon.core.anim import Anim
from poseydon.io.bvh import BvhParseError, load_bvh


def test_loads_expected_shape(bvh_fixture, fixture_facts):
    facts = fixture_facts[bvh_fixture.stem]
    anim = load_bvh(bvh_fixture)

    assert isinstance(anim, Anim)
    assert anim.n_joints == facts["joints"], "End Sites must be counted as joints"
    assert anim.n_frames == facts["frames"]


def test_fps_is_24(bvh_fixture):
    assert load_bvh(bvh_fixture).fps == pytest.approx(24.0, abs=0.01)


def test_rotations_are_unit_quaternions(bvh_fixture):
    anim = load_bvh(bvh_fixture)
    norms = np.linalg.norm(anim.rotations, axis=-1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-10)


def test_topology_is_valid(bvh_fixture):
    anim = load_bvh(bvh_fixture)
    assert anim.parents[0] == -1
    assert np.all(anim.parents[1:] < np.arange(1, anim.n_joints))


def test_end_sites_have_identity_rotation(bvh_fixture):
    # End Sites are leaves with no channels, so their local rotation never changes.
    anim = load_bvh(bvh_fixture)
    is_leaf = np.ones(anim.n_joints, dtype=bool)
    is_leaf[anim.parents[1:]] = False
    leaf_rots = anim.rotations[:, is_leaf]
    expected = np.broadcast_to(np.array([0.0, 0.0, 0.0, 1.0]), leaf_rots.shape)
    np.testing.assert_allclose(leaf_rots, expected, atol=1e-12)


def test_flamingo_root_offset_and_name(truebones_dir):
    anim = load_bvh(truebones_dir / "Flamingo_Flamingo_OneLEgBEnt_353.bvh")
    assert anim.names[0] == "Bip01_Pelvis"
    np.testing.assert_allclose(anim.offsets[0], [0.0, 1.328589, 0.0], atol=1e-9)
    assert "BN_Tail_R_01" in anim.names, "named End Site must be kept"


def test_rejects_truncated_motion_block(tmp_path):
    path = tmp_path / "bad.bvh"
    path.write_text(
        "HIERARCHY\n"
        "ROOT a\n{\nOFFSET 0 0 0\n"
        "CHANNELS 6 Xposition Yposition Zposition Zrotation Yrotation Xrotation\n"
        "End Site #name: a_end\n{\nOFFSET 0 1 0\n}\n}\n"
        "MOTION\nFrames: 2\nFrame Time: 0.041667\n"
        "0 0 0 0 0 0\n"
    )
    with pytest.raises(BvhParseError, match="2 frames"):
        load_bvh(path)

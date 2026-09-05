from pathlib import Path

import numpy as np
import pytest

from poseydon.core.anim import Anim
from poseydon.core.rotations import quat_mul
from poseydon.datasets.raw_bvh import (
    freeze_non_root_translation,
    load_raw_biped_bvh,
    merge_redundant_root,
)

RAW_ROOT = Path(__file__).resolve().parents[2] / "data" / "truebones" / "Truebone_Z-OO"

# (relative path, expected root name, expected joint count). Joint counts and
# merge outcomes were verified by direct inspection of these exact files --
# see the design spec's §3 addendum.
PILOT_RAW_CLIPS = [
    ("BrownBear/__RiseSwat.bvh", "Bip01_Pelvis", 48),
    ("Coyote/__Attack3.bvh", "Bip01_Pelvis", 49),
    ("Crab/__Attack3.bvh", "Hips", 64),
    ("Flamingo/Flamingo_OneLEgBEnt.bvh", "Bip01_Pelvis", 52),
    ("Goat/__HeadButt.bvh", "Bip01_Pelvis", 39),
    ("Scorpion/__SlowForward.bvh", "Hips", 78),
    ("Skunk/__Spray.bvh", "Bip01_Pelvis", 46),
]


def _skip_if_raw_dump_missing():
    if not RAW_ROOT.is_dir():
        pytest.skip(f"raw Truebones dump not found at {RAW_ROOT}")


def test_loads_every_pilot_species_raw_clip():
    _skip_if_raw_dump_missing()
    for relative, expected_root, expected_joints in PILOT_RAW_CLIPS:
        anim = load_raw_biped_bvh(RAW_ROOT / relative)
        assert isinstance(anim, Anim), relative
        assert anim.names[0] == expected_root, relative
        assert anim.n_joints == expected_joints, relative
        assert anim.parents[0] == -1, relative


def test_does_not_merge_when_root_has_two_children():
    _skip_if_raw_dump_missing()
    # Scorpion's root (Hips) has two children (Bip01_Neck1, Bip01_Spine), so
    # nothing merges: root name and joint count are unchanged from the raw
    # hierarchy (78 joints total, including End Sites).
    anim = load_raw_biped_bvh(RAW_ROOT / "Scorpion/__SlowForward.bvh")
    assert anim.names[0] == "Hips"
    assert anim.n_joints == 78


def test_freeze_non_root_translation_keeps_only_the_root_column():
    positions = np.array(
        [
            [[1.0, 2.0, 3.0], [10.0, 20.0, 30.0], [100.0, 200.0, 300.0]],
            [[4.0, 5.0, 6.0], [40.0, 50.0, 60.0], [400.0, 500.0, 600.0]],
        ]
    )  # (F=2, J=3, 3)

    root_pos = freeze_non_root_translation(positions)

    np.testing.assert_array_equal(root_pos, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    assert root_pos.shape == (2, 3)


def _chain(offset1, rot0, rot1, pos0, pos1):
    """A 3-joint chain: root -> joint1 (candidate for merging) -> joint2."""
    names = ("Hips", "Bip01_Pelvis", "Bip01_Spine")
    parents = np.array([-1, 0, 1], dtype=np.int32)
    offsets = np.array([[0.0, 0.0, 0.0], offset1, [1.0, 0.0, 0.0]])
    rotations = np.array([[rot0, rot1, [0.0, 0.0, 0.0, 1.0]]])  # (1 frame, 3 joints, 4)
    positions = np.array([[pos0, pos1, [0.0, 0.0, 0.0]]])       # (1 frame, 3 joints, 3)
    return names, parents, offsets, rotations, positions


def test_merges_zero_offset_single_child_of_root():
    root_rot = [0.0, 0.0, 0.0, 1.0]
    child_rot = [0.0, 0.7071067811865476, 0.0, 0.7071067811865476]  # 90 deg about Y
    names, parents, offsets, rotations, positions = _chain(
        offset1=[0.0, 0.0, 0.0],
        rot0=root_rot,
        rot1=child_rot,
        pos0=[5.0, 0.0, 0.0],
        pos1=[0.1, 0.0, 0.0],
    )

    new_names, new_parents, new_offsets, new_rotations, new_positions = merge_redundant_root(
        names, parents, offsets, rotations, positions
    )

    assert new_names == ("Bip01_Pelvis", "Bip01_Spine")
    np.testing.assert_array_equal(new_parents, [-1, 0])
    # New root's offset is the OLD root's offset (the true world-space rest position).
    np.testing.assert_allclose(new_offsets[0], [0.0, 0.0, 0.0])
    np.testing.assert_allclose(new_offsets[1], [1.0, 0.0, 0.0])
    # New root's rotation composes root-then-child, same order forward_kinematics uses.
    np.testing.assert_allclose(new_rotations[0, 0], quat_mul(np.array(root_rot), np.array(child_rot)))
    np.testing.assert_allclose(new_rotations[0, 1], [0.0, 0.0, 0.0, 1.0])
    # New root's translation is the two old translations summed.
    np.testing.assert_allclose(new_positions[0, 0], [5.1, 0.0, 0.0])


def test_raises_when_root_has_two_children():
    names = ("Hips", "A", "B")
    parents = np.array([-1, 0, 0], dtype=np.int32)
    offsets = np.zeros((3, 3))
    rotations = np.broadcast_to([0.0, 0.0, 0.0, 1.0], (1, 3, 4)).copy()
    positions = np.zeros((1, 3, 3))

    with pytest.raises(ValueError, match="exactly one child"):
        merge_redundant_root(names, parents, offsets, rotations, positions)


def test_raises_when_single_child_offset_is_not_near_zero():
    names, parents, offsets, rotations, positions = _chain(
        offset1=[3.0, 0.0, 0.0],  # not ~zero -- this is a real bone, not a redundant wrapper
        rot0=[0.0, 0.0, 0.0, 1.0],
        rot1=[0.0, 0.0, 0.0, 1.0],
        pos0=[0.0, 0.0, 0.0],
        pos1=[0.0, 0.0, 0.0],
    )

    with pytest.raises(ValueError, match="not.*zero"):
        merge_redundant_root(names, parents, offsets, rotations, positions)

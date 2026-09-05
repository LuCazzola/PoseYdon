import numpy as np
import pytest

from poseydon.core.rotations import quat_mul
from poseydon.datasets.raw_bvh import merge_redundant_root


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

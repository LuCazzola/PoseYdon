import numpy as np
import pytest

from poseydon.core.kinematics import check_topological_order, forward_kinematics
from poseydon.core.rotations import QUAT_IDENTITY, euler_to_quat


def test_two_joint_chain_child_follows_root_rotation():
    # Child sits 1 unit up the Y axis from the root. Rotating the root 90 degrees
    # about Z must swing the child onto the negative X axis.
    parents = np.array([-1, 0], dtype=np.int32)
    offsets = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    root_pos = np.zeros((1, 3))
    rotations = np.stack(
        [euler_to_quat(np.array([90.0, 0.0, 0.0]), "ZYX"), QUAT_IDENTITY]
    )[None]

    positions, _ = forward_kinematics(rotations, root_pos, offsets, parents)

    np.testing.assert_allclose(positions[0, 0], [0.0, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(positions[0, 1], [-1.0, 0.0, 0.0], atol=1e-12)


def test_identity_rotations_reproduce_offset_sums():
    parents = np.array([-1, 0, 1], dtype=np.int32)
    offsets = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    root_pos = np.array([[5.0, 0.0, 0.0]])
    rotations = np.broadcast_to(QUAT_IDENTITY, (1, 3, 4)).copy()

    positions, _ = forward_kinematics(rotations, root_pos, offsets, parents)

    np.testing.assert_allclose(positions[0, 2], [6.0, 2.0, 0.0], atol=1e-12)


def test_root_translation_moves_whole_skeleton():
    parents = np.array([-1, 0], dtype=np.int32)
    offsets = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    rotations = np.broadcast_to(QUAT_IDENTITY, (2, 2, 4)).copy()
    root_pos = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 7.0]])

    positions, _ = forward_kinematics(rotations, root_pos, offsets, parents)

    np.testing.assert_allclose(positions[1, 1], [3.0, 1.0, 7.0], atol=1e-12)


def test_rejects_non_topological_parents():
    with pytest.raises(ValueError, match="topological"):
        check_topological_order(np.array([-1, 2, 0], dtype=np.int32))


def test_rejects_missing_root():
    with pytest.raises(ValueError, match="root"):
        check_topological_order(np.array([1, 0], dtype=np.int32))

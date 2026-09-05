from pathlib import Path

import numpy as np
import pytest

from poseydon.core.anim import Anim
from poseydon.core.kinematics import forward_kinematics
from poseydon.core.rotations import QUAT_IDENTITY, euler_to_quat, quat_mul
from poseydon.datasets.raw_bvh import (
    freeze_non_root_translation,
    load_raw_biped_bvh,
    merge_redundant_root,
    remove_bind_pose,
)


def test_remove_bind_pose_makes_the_rest_frame_read_as_identity_and_preserves_positions():
    # Root and child each carry a large, arbitrary "bind" rotation (mimicking
    # the real raw Biped rig quirk), plus an extra rotation on the animated
    # frame that represents genuine motion.
    names = ("Root", "Child")
    parents = np.array([-1, 0], dtype=np.int32)
    offsets = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])

    root_bind = euler_to_quat(np.array([[90.0, 0.0, 0.0]]), "ZYX")[0]
    child_bind = euler_to_quat(np.array([[0.0, 0.0, 90.0]]), "ZYX")[0]
    extra_root = euler_to_quat(np.array([[0.0, 30.0, 0.0]]), "ZYX")[0]
    extra_child = euler_to_quat(np.array([[20.0, 0.0, 0.0]]), "ZYX")[0]

    rest_anim = Anim(
        rotations=np.array([[root_bind, child_bind]]),
        root_pos=np.zeros((1, 3)),
        offsets=offsets,
        parents=parents,
        names=names,
        fps=30.0,
    )
    anim = Anim(
        rotations=np.array(
            [
                [root_bind, child_bind],  # frame 0: exactly at rest
                [quat_mul(extra_root, root_bind), quat_mul(extra_child, child_bind)],  # frame 1: animated
            ]
        ),
        root_pos=np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]]),
        offsets=offsets,
        parents=parents,
        names=names,
        fps=30.0,
    )

    new_anim = remove_bind_pose(anim, rest_anim)

    # Zero rotation at the rest frame, for every joint.
    np.testing.assert_allclose(
        new_anim.rotations[0], np.broadcast_to(QUAT_IDENTITY, (2, 4)), atol=1e-9
    )

    # Global joint positions are preserved exactly at every frame -- only the
    # rotation/offset CONVENTION changed, not the physically observed motion.
    old_positions, _ = forward_kinematics(anim.rotations, anim.root_pos, anim.offsets, anim.parents)
    new_positions, _ = forward_kinematics(
        new_anim.rotations, new_anim.root_pos, new_anim.offsets, new_anim.parents
    )
    np.testing.assert_allclose(new_positions, old_positions, atol=1e-9)


def test_remove_bind_pose_preserves_positions_through_a_three_joint_chain():
    # A deeper chain (root -> mid -> tip) with distinct bind rotations at
    # every level, to guard against a formula that only happens to work when
    # there is just one non-root joint.
    names = ("Root", "Mid", "Tip")
    parents = np.array([-1, 0, 1], dtype=np.int32)
    offsets = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    bind = euler_to_quat(np.array([[40.0, -60.0, 15.0], [70.0, 10.0, -25.0], [-30.0, 50.0, 5.0]]), "ZYX")
    extra = euler_to_quat(np.array([[12.0, 8.0, -4.0], [5.0, -15.0, 20.0], [-10.0, 3.0, 9.0]]), "ZYX")

    rest_anim = Anim(
        rotations=bind[None, :, :],
        root_pos=np.zeros((1, 3)),
        offsets=offsets,
        parents=parents,
        names=names,
        fps=30.0,
    )
    animated_rotations = quat_mul(extra, bind)
    anim = Anim(
        rotations=np.stack([bind, animated_rotations]),
        root_pos=np.array([[0.0, 0.0, 0.0], [0.5, -1.0, 2.0]]),
        offsets=offsets,
        parents=parents,
        names=names,
        fps=30.0,
    )

    new_anim = remove_bind_pose(anim, rest_anim)

    np.testing.assert_allclose(
        new_anim.rotations[0], np.broadcast_to(QUAT_IDENTITY, (3, 4)), atol=1e-9
    )
    old_positions, _ = forward_kinematics(anim.rotations, anim.root_pos, anim.offsets, anim.parents)
    new_positions, _ = forward_kinematics(
        new_anim.rotations, new_anim.root_pos, new_anim.offsets, new_anim.parents
    )
    np.testing.assert_allclose(new_positions, old_positions, atol=1e-9)


def test_remove_bind_pose_tolerates_an_end_site_missing_from_the_rest_reference():
    # Some raw T-pose files omit End Sites that action clips still declare
    # (confirmed on real data: Crab's T-pose has 54 joints, its action clips
    # have 64 -- exactly the 10 End Sites). An End Site never has rotation
    # channels (always identity) and never has children, so there is nothing
    # to remove for it and nothing downstream it could affect.
    names = ("Root", "Child", "Child_End")
    parents = np.array([-1, 0, 1], dtype=np.int32)
    offsets = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    root_bind = euler_to_quat(np.array([[40.0, -20.0, 10.0]]), "ZYX")[0]
    child_bind = euler_to_quat(np.array([[15.0, 5.0, -30.0]]), "ZYX")[0]
    extra_root = euler_to_quat(np.array([[5.0, 0.0, 0.0]]), "ZYX")[0]
    extra_child = euler_to_quat(np.array([[0.0, 10.0, 0.0]]), "ZYX")[0]

    rest_anim = Anim(  # no End Site entry at all
        rotations=np.array([[root_bind, child_bind]]),
        root_pos=np.zeros((1, 3)),
        offsets=offsets[:2],
        parents=parents[:2],
        names=names[:2],
        fps=30.0,
    )
    anim = Anim(
        rotations=np.array(
            [
                [root_bind, child_bind, QUAT_IDENTITY],
                [
                    quat_mul(extra_root, root_bind),
                    quat_mul(extra_child, child_bind),
                    QUAT_IDENTITY,
                ],
            ]
        ),
        root_pos=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        offsets=offsets,
        parents=parents,
        names=names,
        fps=30.0,
    )

    new_anim = remove_bind_pose(anim, rest_anim)

    # Joints WITH a rest reference read as identity at the rest frame. The
    # End Site has none, so its own rotation value is whatever the general
    # formula produces (not necessarily identity) -- but since it has no
    # children, that value affects nothing; only its POSITION (checked
    # below) has to be right.
    np.testing.assert_allclose(new_anim.rotations[0, :2], np.broadcast_to(QUAT_IDENTITY, (2, 4)), atol=1e-9)

    old_positions, _ = forward_kinematics(anim.rotations, anim.root_pos, anim.offsets, anim.parents)
    new_positions, _ = forward_kinematics(
        new_anim.rotations, new_anim.root_pos, new_anim.offsets, new_anim.parents
    )
    np.testing.assert_allclose(new_positions, old_positions, atol=1e-9)


def test_remove_bind_pose_falls_back_to_the_clips_own_frame_zero_for_a_missing_non_leaf_joint():
    # "Child" is missing from the rest reference and HAS a child of its own
    # ("Grandchild") -- unlike a childless End Site, this really could be a
    # meaningfully-animated joint. Rather than fail the whole clip, it falls
    # back to THIS clip's own frame-0 rotation as an approximate bind (real
    # data: joints missing from a T-pose have turned out to barely move at
    # all, so a clip's own first frame is a reasonable stand-in).
    names = ("Root", "Child", "Grandchild")
    parents = np.array([-1, 0, 1], dtype=np.int32)
    offsets = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    root_bind = euler_to_quat(np.array([[20.0, 0.0, 0.0]]), "ZYX")[0]
    child_bind = euler_to_quat(np.array([[0.0, 35.0, 0.0]]), "ZYX")[0]  # anim's own frame-0 value
    extra_root = euler_to_quat(np.array([[5.0, 0.0, 0.0]]), "ZYX")[0]
    extra_child = euler_to_quat(np.array([[0.0, 10.0, 0.0]]), "ZYX")[0]

    rest_anim = Anim(  # only names "Child" and "Grandchild" onward
        rotations=np.array([[root_bind]]),
        root_pos=np.zeros((1, 3)),
        offsets=offsets[:1],
        parents=parents[:1],
        names=names[:1],
        fps=30.0,
    )
    anim = Anim(
        rotations=np.array(
            [
                [root_bind, child_bind, QUAT_IDENTITY],
                [
                    quat_mul(extra_root, root_bind),
                    quat_mul(extra_child, child_bind),
                    QUAT_IDENTITY,
                ],
            ]
        ),
        root_pos=np.array([[0.0, 0.0, 0.0], [0.2, 0.3, -0.1]]),
        offsets=offsets,
        parents=parents,
        names=names,
        fps=30.0,
    )

    new_anim = remove_bind_pose(anim, rest_anim)

    # Root reads as identity at frame 0 (it HAS a rest reference); Child's
    # own frame-0 was USED as its bind, so it also reads as identity there.
    np.testing.assert_allclose(new_anim.rotations[0, :2], np.broadcast_to(QUAT_IDENTITY, (2, 4)), atol=1e-9)

    old_positions, _ = forward_kinematics(anim.rotations, anim.root_pos, anim.offsets, anim.parents)
    new_positions, _ = forward_kinematics(
        new_anim.rotations, new_anim.root_pos, new_anim.offsets, new_anim.parents
    )
    np.testing.assert_allclose(new_positions, old_positions, atol=1e-9)

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

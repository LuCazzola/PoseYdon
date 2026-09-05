import numpy as np

from poseydon.core.anim import Anim
from poseydon.core.rotations import QUAT_IDENTITY, euler_to_quat, quat_mul
from poseydon.preproc.rest_pose import remove_bind_pose


def test_remove_bind_pose_makes_the_rest_frame_read_as_identity():
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

    # Zero rotation at the rest frame, for every joint -- this is the whole
    # point: "if you took the BVH when all rotation channels are zero, you
    # get the T-pose."
    np.testing.assert_allclose(
        new_anim.rotations[0], np.broadcast_to(QUAT_IDENTITY, (2, 4)), atol=1e-9
    )


def test_remove_bind_pose_rest_frame_identity_holds_through_a_three_joint_chain():
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
    # children, nothing downstream depends on it.
    np.testing.assert_allclose(new_anim.rotations[0, :2], np.broadcast_to(QUAT_IDENTITY, (2, 4)), atol=1e-9)


def test_remove_bind_pose_falls_back_to_the_clips_own_frame_zero_for_a_missing_non_leaf_joint():
    # "Child" is missing from the rest reference and HAS a child of its own
    # ("Grandchild") -- unlike a childless End Site, this really could be a
    # meaningfully-animated joint. Rather than fail the whole clip, it falls
    # back to THIS clip's own frame-0 rotation as an approximate bind.
    names = ("Root", "Child", "Grandchild")
    parents = np.array([-1, 0, 1], dtype=np.int32)
    offsets = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    root_bind = euler_to_quat(np.array([[20.0, 0.0, 0.0]]), "ZYX")[0]
    child_bind = euler_to_quat(np.array([[0.0, 35.0, 0.0]]), "ZYX")[0]  # anim's own frame-0 value
    extra_root = euler_to_quat(np.array([[5.0, 0.0, 0.0]]), "ZYX")[0]
    extra_child = euler_to_quat(np.array([[0.0, 10.0, 0.0]]), "ZYX")[0]

    rest_anim = Anim(  # only names "Root" -- "Child" and "Grandchild" are missing
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

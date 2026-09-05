import numpy as np
import pytest

from poseydon.augment.topology import (
    DropEndEffector,
    DuplicateJoint,
    child_count,
    drop_joints,
    duplicate_joint,
    leaves,
)
from poseydon.core.kinematics import check_topological_order
from poseydon.core.skeleton import ResolvedSkeleton
from poseydon.io.bvh import load_bvh
from tests.ingest.manifest_helper import resolved_for


def _excluded(resolved) -> set[int]:
    facing = {i for pair in resolved.facing_indices for i in pair}
    return facing | set(resolved.foot_indices)


def _droppable_leaf(anim, resolved) -> int:
    excluded = _excluded(resolved)
    candidates = [j for j in leaves(anim.parents) if j not in excluded and j != 0]
    if not candidates:
        pytest.skip("this fixture has no droppable end-effector")
    return candidates[0]


def test_leaves_are_joints_with_no_children(bvh_fixture):
    anim = load_bvh(bvh_fixture)
    leaf_set = set(leaves(anim.parents))
    has_child = {int(p) for p in anim.parents if p >= 0}
    assert leaf_set == set(range(anim.n_joints)) - has_child


def test_drop_joints_shrinks_every_structural_array_by_one(bvh_fixture):
    anim = load_bvh(bvh_fixture)
    resolved = resolved_for(bvh_fixture, anim)
    j = _droppable_leaf(anim, resolved)

    new_anim, new_resolved, edit = drop_joints(anim, resolved, {j})

    assert new_anim.n_joints == anim.n_joints - 1
    check_topological_order(new_anim.parents)
    assert len(edit.source_of) == new_anim.n_joints
    assert j not in edit.source_of
    assert len(new_resolved.foot_indices) == len(resolved.foot_indices)


def test_drop_joints_leaves_surviving_positions_bit_identical(bvh_fixture):
    anim = load_bvh(bvh_fixture)
    resolved = resolved_for(bvh_fixture, anim)
    j = _droppable_leaf(anim, resolved)

    original_positions = anim.global_positions()
    new_anim, _, edit = drop_joints(anim, resolved, {j})
    new_positions = new_anim.global_positions()

    for new_index, old_index in enumerate(edit.source_of):
        np.testing.assert_array_equal(
            new_positions[:, new_index], original_positions[:, old_index]
        )


def test_drop_end_effector_never_selects_a_facing_or_foot_joint(bvh_fixture):
    anim = load_bvh(bvh_fixture)
    resolved = resolved_for(bvh_fixture, anim)
    excluded = _excluded(resolved)
    augmentation = DropEndEffector(p=1.0)

    for seed in range(50):
        _, _, edit = augmentation.apply_structural(anim, resolved, np.random.default_rng(seed))
        dropped = set(range(anim.n_joints)) - set(edit.source_of)
        assert dropped.isdisjoint(excluded)


def test_drop_end_effector_is_a_no_op_when_every_leaf_is_excluded():
    # A 2-joint chain: root (0) -> foot (1). The only leaf is also the only
    # declared foot, so there is nothing legal to drop.
    from poseydon.core.anim import Anim
    from poseydon.core.rotations import QUAT_IDENTITY

    n_joints, n_frames = 2, 3
    anim = Anim(
        rotations=np.tile(QUAT_IDENTITY, (n_frames, n_joints, 1)),
        root_pos=np.zeros((n_frames, 3)),
        offsets=np.array([[0.0, 0.0, 0.0], [0.0, -1.0, 0.0]]),
        parents=np.array([-1, 0], dtype=np.int32),
        names=("root", "foot"),
        fps=30.0,
    )
    resolved = ResolvedSkeleton(manifest=None, facing_indices=(), foot_indices=(1,))
    augmentation = DropEndEffector(p=1.0)

    new_anim, new_resolved, edit = augmentation.apply_structural(
        anim, resolved, np.random.default_rng(0)
    )

    assert new_anim is anim
    assert new_resolved is resolved
    assert edit.source_of == (0, 1)


def _duplicatable_joint(anim, resolved) -> int:
    excluded = _excluded(resolved)
    counts = child_count(anim.parents)
    candidates = [
        j
        for j in range(1, anim.n_joints)
        if counts[j] == 1 and anim.parents[j] != 0 and j not in excluded
    ]
    if not candidates:
        pytest.skip("this fixture has no duplicatable joint")
    return candidates[0]


def test_duplicate_joint_adds_exactly_one_joint(bvh_fixture):
    anim = load_bvh(bvh_fixture)
    resolved = resolved_for(bvh_fixture, anim)
    j = _duplicatable_joint(anim, resolved)

    new_anim, _, edit = duplicate_joint(anim, resolved, j)

    assert new_anim.n_joints == anim.n_joints + 1
    check_topological_order(new_anim.parents)
    assert len(edit.source_of) == new_anim.n_joints
    assert new_anim.parents[j] == anim.parents[j]  # midpoint keeps j's old parent
    assert new_anim.parents[j + 1] == j            # j itself now hangs off the midpoint


def test_duplicate_joint_preserves_downstream_positions(bvh_fixture):
    anim = load_bvh(bvh_fixture)
    resolved = resolved_for(bvh_fixture, anim)
    j = _duplicatable_joint(anim, resolved)

    original_positions = anim.global_positions()
    new_anim, _, edit = duplicate_joint(anim, resolved, j)
    new_positions = new_anim.global_positions()

    for new_index, old_index in enumerate(edit.source_of):
        if new_index == j:
            continue  # the freshly inserted midpoint has no "before" to compare
        np.testing.assert_allclose(
            new_positions[:, new_index], original_positions[:, old_index], atol=1e-9
        )


def test_duplicate_joint_new_joint_sits_at_the_bone_midpoint(bvh_fixture):
    anim = load_bvh(bvh_fixture)
    resolved = resolved_for(bvh_fixture, anim)
    j = _duplicatable_joint(anim, resolved)
    parent = int(anim.parents[j])

    original_positions = anim.global_positions()
    new_anim, _, _ = duplicate_joint(anim, resolved, j)
    new_positions = new_anim.global_positions()

    midpoint = (original_positions[:, parent] + original_positions[:, j]) / 2
    np.testing.assert_allclose(new_positions[:, j], midpoint, atol=1e-9)


def test_duplicate_joint_never_selects_a_facing_or_foot_joint(bvh_fixture):
    from collections import Counter

    anim = load_bvh(bvh_fixture)
    resolved = resolved_for(bvh_fixture, anim)
    excluded = _excluded(resolved)
    augmentation = DuplicateJoint(p=1.0)

    for seed in range(50):
        new_anim, _, edit = augmentation.apply_structural(
            anim, resolved, np.random.default_rng(seed)
        )
        if new_anim.n_joints == anim.n_joints:
            continue  # no candidate this draw
        # The duplicated joint is the one value that appears twice in
        # source_of (the midpoint copies it, and the original joint itself
        # still maps back to it).
        counts = Counter(edit.source_of)
        duplicated = {value for value, count in counts.items() if count > 1}
        assert duplicated.isdisjoint(excluded)


def test_duplicate_joint_is_a_no_op_when_nothing_qualifies():
    # A 2-joint chain: root (0) -> child (1). Child's parent IS the root, so
    # it fails the "parent is not the root" rule -- no legal candidate.
    from poseydon.core.anim import Anim
    from poseydon.core.rotations import QUAT_IDENTITY

    n_joints, n_frames = 2, 3
    anim = Anim(
        rotations=np.tile(QUAT_IDENTITY, (n_frames, n_joints, 1)),
        root_pos=np.zeros((n_frames, 3)),
        offsets=np.array([[0.0, 0.0, 0.0], [0.0, -1.0, 0.0]]),
        parents=np.array([-1, 0], dtype=np.int32),
        names=("root", "child"),
        fps=30.0,
    )
    resolved = ResolvedSkeleton(manifest=None, facing_indices=(), foot_indices=())
    augmentation = DuplicateJoint(p=1.0)

    new_anim, new_resolved, edit = augmentation.apply_structural(
        anim, resolved, np.random.default_rng(0)
    )

    assert new_anim is anim
    assert new_resolved is resolved
    assert edit.source_of == (0, 1)

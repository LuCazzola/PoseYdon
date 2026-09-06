"""Joint reduction: what comes out, and that nothing moves."""

from __future__ import annotations

import numpy as np
import pytest

from poseydon.core.animation import RigidBodyAnimation
from poseydon.core.rotations import QUAT_IDENTITY, euler_to_quat
from poseydon.features.reduce import (
    apply_reduction,
    build_reduction,
    invert_reduction,
)


def _rig(offsets, parents, names, n_frames=4, spin=()):
    """A rigid animation; joints in `spin` get a non-identity rotation."""
    offsets = np.asarray(offsets, dtype=np.float64)
    rotations = np.tile(QUAT_IDENTITY, (n_frames, len(names), 1))
    for joint in spin:
        rotations[:, joint] = euler_to_quat(np.array([0.0, 0.0, 25.0]), "ZYX")
    return RigidBodyAnimation.from_root_motion(
        rotations=rotations,
        root_pos=np.zeros((n_frames, 3)),
        offsets=offsets,
        parents=np.asarray(parents, dtype=np.int32),
        names=tuple(names),
        fps=30.0,
    )


def _leafy():
    """root -> mid -> tip, plus a zero-offset End Site under tip."""
    return _rig(
        offsets=[[0, 0, 0], [0, 1, 0], [0, 1, 0], [0, 0, 0]],
        parents=[-1, 0, 1, 2],
        names=("root", "mid", "tip", "end"),
        spin=(1, 2),
    )


def _only_child():
    """root -> dummy(zero offset, sole child) -> a, b."""
    return _rig(
        offsets=[[0, 0, 0], [0, 0, 0], [0, 1, 0], [1, 0, 0]],
        parents=[-1, 0, 1, 1],
        names=("root", "dummy", "a", "b"),
        spin=(1, 2),
    )


def _with_sibling():
    """root -> {dummy(zero offset), other}; dummy is NOT an only child."""
    return _rig(
        offsets=[[0, 0, 0], [0, 0, 0], [0, 1, 0], [1, 0, 0]],
        parents=[-1, 0, 1, 0],
        names=("root", "dummy", "a", "other"),
        spin=(1, 3),
    )


def _leaf_with_sibling():
    """P -> {Z zero-offset internal with a zero-offset leaf child, other}.
    Z begins with a sibling, so a rule reading the ORIGINAL hierarchy keeps it.
    Reducing at removal time drops Z's leaf, which makes Z a leaf, and drops Z."""
    return _rig(
        offsets=[[0, 0, 0], [0, 0, 0], [0, 0, 0], [1, 0, 0]],
        parents=[-1, 0, 1, 0],
        names=("root", "z", "z_end", "other"),
        spin=(1,),
    )


def _real_internal_with_siblings():
    """P -> {Z zero-offset with real-offset child, other}.
    Z is internal and has siblings, so it is kept; it will never become a leaf."""
    return _rig(
        offsets=[[0, 0, 0], [0, 0, 0], [0, 2, 0], [1, 0, 0]],
        parents=[-1, 0, 1, 0],
        names=("root", "z", "a", "other"),
        spin=(1, 2),
    )


def test_a_zero_offset_leaf_is_dropped():
    reduction = build_reduction(_leafy())
    reduced = apply_reduction(_leafy(), reduction)
    assert reduced.names == ("root", "mid", "tip")


def test_a_zero_offset_only_child_is_collapsed_into_its_parent():
    reduction = build_reduction(_only_child())
    reduced = apply_reduction(_only_child(), reduction)
    assert reduced.names == ("root", "a", "b")
    assert list(reduced.parents) == [-1, 0, 0]


def test_a_zero_offset_joint_with_siblings_is_kept():
    """Folding it into the parent would turn `other` too, so it stays."""
    reduction = build_reduction(_with_sibling())
    reduced = apply_reduction(_with_sibling(), reduction)
    assert "dummy" in reduced.names


@pytest.mark.parametrize(
    "factory",
    [_leafy, _only_child, _with_sibling, _leaf_with_sibling, _real_internal_with_siblings],
    ids=["leaf", "only-child", "with-sibling", "leaf-with-sibling",
         "real-internal-with-siblings"],
)
def test_reduction_does_not_move_any_surviving_joint(factory):
    source = factory()
    reduction = build_reduction(source)
    reduced = apply_reduction(source, reduction)

    source_positions = source.global_positions()
    reduced_positions = reduced.global_positions()
    for new, old in enumerate(reduction.source_of):
        np.testing.assert_allclose(
            reduced_positions[:, new], source_positions[:, old], atol=1e-9,
            err_msg=f"joint {source.names[old]} moved",
        )


@pytest.mark.parametrize(
    "factory",
    [_leafy, _only_child, _with_sibling, _leaf_with_sibling, _real_internal_with_siblings],
    ids=["leaf", "only-child", "with-sibling", "leaf-with-sibling",
         "real-internal-with-siblings"],
)
def test_expansion_restores_structure_and_world_positions(factory):
    source = factory()
    reduction = build_reduction(source)
    restored = invert_reduction(apply_reduction(source, reduction), reduction)

    assert restored.names == source.names
    assert list(restored.parents) == list(source.parents)
    np.testing.assert_allclose(restored.offsets, source.offsets, atol=1e-9)
    np.testing.assert_allclose(
        restored.global_positions(), source.global_positions(), atol=1e-9
    )


def test_reduction_repeats_until_no_zero_offset_leaf_remains():
    """Dropping a zero-offset leaf can leave its parent a zero-offset leaf."""
    anim = _rig(
        offsets=[[0, 0, 0], [0, 1, 0], [0, 0, 0], [0, 0, 0]],
        parents=[-1, 0, 1, 2],
        names=("root", "mid", "a", "b"),
        spin=(1,),
    )
    reduced = apply_reduction(anim, build_reduction(anim))
    assert reduced.names == ("root", "mid")


def test_reduction_never_removes_the_root():
    anim = _rig(
        offsets=[[0, 0, 0], [0, 1, 0]], parents=[-1, 0], names=("root", "tip")
    )
    assert apply_reduction(anim, build_reduction(anim)).names == ("root", "tip")


def test_a_joint_that_becomes_a_leaf_is_dropped_even_though_it_had_siblings():
    """Z initially has a sibling and is internal, but becomes a leaf after its
    zero-offset child is dropped, then is itself dropped."""
    source = _leaf_with_sibling()
    reduced = apply_reduction(source, build_reduction(source))
    assert reduced.names == ("root", "other")


def test_a_zero_offset_internal_joint_with_real_children_and_siblings_is_kept():
    """The genuine case 3: its child has a real offset, so it never becomes a
    leaf, and folding it would turn its sibling."""
    source = _real_internal_with_siblings()
    reduced = apply_reduction(source, build_reduction(source))
    assert "z" in reduced.names

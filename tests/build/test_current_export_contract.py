"""Executable characterization of the motion export boundary.

These tests pin behavior observed in the implementation before the
character-aware exporter changes it.  Expectations are hand-derived from the
fixture, not computed by the code under test.
"""

import numpy as np

from poseydon.core.animation import Animation
from poseydon.core.rotations import QUAT_IDENTITY
from poseydon.features.reduce import (
    apply_reduction,
    build_reduction,
    invert_reduction,
)
from poseydon.io.bvh import BVH


def _animation() -> Animation:
    rotations = np.tile(QUAT_IDENTITY, (2, 4, 1))
    translations = np.tile(
        np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        (2, 1, 1),
    )
    translations[1, 0, 0] = 0.25
    return Animation(
        rotations=rotations,
        translations=translations,
        offsets=np.array(
            [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
        ),
        parents=np.array([-1, 0, 1, 2], dtype=np.int32),
        names=("Root", "Joint", "Zero", "End"),
        fps=30.0,
    )


def test_bvh_writer_preserves_names_parents_offsets_and_childless_joint():
    source = _animation()

    reread = BVH.from_animation(source).to_animation()

    assert reread.names == ("Root", "Joint", "Zero", "End")
    np.testing.assert_array_equal(reread.parents, [-1, 0, 1, 2])
    np.testing.assert_allclose(
        reread.offsets,
        [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
    )


def test_reduction_round_trip_preserves_world_positions():
    source = _animation().as_rigid_body(joint_translation="drop")

    reduction = build_reduction(source)
    reduced = apply_reduction(source, reduction)
    restored = invert_reduction(reduced, reduction)

    assert reduced.names == ("Root", "Joint", "End")
    np.testing.assert_allclose(
        restored.global_positions(), source.global_positions(), atol=1e-8
    )

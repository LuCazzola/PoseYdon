"""Blender-side characterization of the current FBX API."""

import numpy as np
import pytest

from poseydon.core.animation import Animation
from poseydon.core.rotations import QUAT_IDENTITY
pytest.importorskip("bpy", reason="FBX characterization needs Blender")

from poseydon.io.fbx import FBX


def _animation(names=("Root", "Leaf", "End")) -> Animation:
    return Animation(
        rotations=np.tile(QUAT_IDENTITY, (2, len(names), 1)),
        translations=np.tile(
            np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]])[: len(names)],
            (2, 1, 1),
        ),
        offsets=np.array(
            [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]]
        )[: len(names)],
        parents=np.array([-1, 0, 1], dtype=np.int32)[: len(names)],
        names=tuple(names),
        fps=30.0,
    )


def test_bare_fbx_has_no_mesh_and_omits_childless_bvh_joint():
    scene = FBX.from_animation(_animation())

    assert scene.meshes == []
    assert scene.joint_names == ["Root", "Leaf"]


def test_apply_animation_rejects_armature_bone_absent_from_animation():
    scene = FBX.from_animation(_animation())
    animation_without_leaf = _animation(("Root",))

    with pytest.raises(ValueError, match="animation does not drive"):
        scene.apply_animation(animation_without_leaf)

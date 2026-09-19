import numpy as np
import pytest

from poseydon.core.animation import Animation
from poseydon.core.rotations import QUAT_IDENTITY, euler_to_quat

pytest.importorskip("bpy", reason="FBX extraction needs Blender")

from poseydon.io.fbx import FBX


def _keyed_animation():
    rotations = np.tile(QUAT_IDENTITY, (3, 3, 1))
    rotations[1:, 1] = euler_to_quat(
        np.array([[0.0, 0.0, 30.0], [0.0, 0.0, -20.0]]), "XYZ"
    )
    translations = np.tile(
        np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]),
        (3, 1, 1),
    )
    translations[:, 0, 0] = [0.0, 0.5, 1.0]
    return Animation(
        rotations, translations, translations[0].copy(),
        np.array([-1, 0, 1], dtype=np.int32), ("Root", "Joint", "End"), 30.0,
    )


def test_fbx_to_animation_preserves_hierarchy_rest_and_evaluated_positions():
    scene = FBX.from_animation(_keyed_animation())

    extracted = scene.to_animation()

    assert extracted.names == ("Root", "Joint")
    np.testing.assert_array_equal(extracted.parents, [-1, 0])
    np.testing.assert_allclose(extracted.offsets, scene.joint_offsets, atol=1e-8)
    assert extracted.n_frames == 3
    for sample, frame in enumerate((1, 2, 3)):
        scene.set_frame(frame)
        expected = np.stack([
            np.array((-scene.joint_position(name).x,
                      scene.joint_position(name).z,
                      scene.joint_position(name).y))
            for name in scene.joint_names
        ])
        np.testing.assert_allclose(
            extracted.global_positions()[sample], expected, atol=1e-6
        )

from pathlib import Path

import pytest

pytest.importorskip("bpy", reason="character assets need Blender")

from poseydon.build.character_assets import CharacterAssetContext, build_character_assets


def test_bvh_only_returns_typed_absence(tmp_path):
    context = CharacterAssetContext(
        rig="NoMesh", raw_fbx=None, model_names=("Root",),
        model_parents=(-1,), removals=(),
    )

    assets = build_character_assets(context, tmp_path)

    assert assets.model is None
    assert assets.presentation is None
    assert assets.reason == "source_surface_missing"


def test_missing_declared_fbx_is_not_treated_as_absence(tmp_path):
    context = CharacterAssetContext(
        rig="Broken", raw_fbx=Path("missing.fbx"), model_names=("Root",),
        model_parents=(-1,), removals=(),
    )

    with pytest.raises(FileNotFoundError, match="missing.fbx"):
        build_character_assets(context, tmp_path)

import numpy as np
import bpy
from poseydon.core.animation import Animation
from poseydon.core.rotations import QUAT_IDENTITY
from poseydon.io.fbx import FBX


def _raw_skinned_fbx(path):
    anim = Animation(
        rotations=np.tile(QUAT_IDENTITY, (1, 4, 1)),
        translations=np.array([[[0, 0, 0], [0, 1, 0], [0, 1, 0], [0, 0, 0]]], dtype=float),
        offsets=np.array([[0, 0, 0], [0, 1, 0], [0, 1, 0], [0, 0, 0]], dtype=float),
        parents=np.array([-1, 0, 1, 2], dtype=np.int32),
        names=("Root", "Spine", "ClothSecondary", "End"), fps=30.0,
    )
    scene = FBX.from_bare_animation(anim, name="Character")
    mesh = bpy.data.meshes.new("BodyMesh")
    mesh.from_pydata([(0, 0, 0), (1, 0, 0), (0, 1, 0)], [], [(0, 1, 2)])
    mesh.update()
    obj = bpy.data.objects.new("Body", mesh)
    bpy.context.scene.collection.objects.link(obj)
    obj.parent = scene.armature
    modifier = obj.modifiers.new("Armature", "ARMATURE")
    modifier.object = scene.armature
    for name, weights in {"Root": (1.0, 0.0, 0.0), "Spine": (0.0, 1.0, 0.0), "ClothSecondary": (0.0, 0.0, 1.0)}.items():
        group = obj.vertex_groups.new(name=name)
        for index, weight in enumerate(weights):
            if weight:
                group.add([index], weight, "REPLACE")
    scene.meshes = [obj]
    scene.write(path)
    return anim


def test_model_and_presentation_assets_preserve_mesh_and_auxiliary(tmp_path):
    raw = tmp_path / "raw.fbx"
    _raw_skinned_fbx(raw)
    model_rest = Animation(
        rotations=np.tile(QUAT_IDENTITY, (1, 2, 1)),
        translations=np.array([[[0, 0, 0], [0, 1, 0]]], dtype=float),
        offsets=np.array([[0, 0, 0], [0, 1, 0]], dtype=float),
        parents=np.array([-1, 0], dtype=np.int32),
        names=("Root", "Spine"), fps=30.0,
    )
    context = CharacterAssetContext(
        rig="Character", raw_fbx=raw, model_names=("Root", "Spine"),
        model_parents=(-1, 0), removals=(), model_rest=model_rest,
    )
    assets = build_character_assets(context, tmp_path / "out")
    model = FBX.read(assets.model)
    assert tuple(model.joint_names) == ("Root", "Spine")
    assert tuple(model.joint_parents) == (-1, 0)
    assert model.meshes
    # The removed weighted auxiliary is transferred to its surviving parent;
    # reducing topology must not leave a vertex with lost influence.
    assert model.vertex_group_weight("ClothSecondary") == 0
    assert model.vertex_group_weight("Spine") > 0
    presentation = FBX.read(assets.presentation)
    assert "ClothSecondary" in presentation.joint_names
    assert presentation.vertex_group_weight("ClothSecondary") > 0

# Task 6 focused fixture follows

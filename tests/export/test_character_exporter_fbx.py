from pathlib import Path

import pytest

pytest.importorskip("bpy")

from poseydon.core.animation import Animation
from poseydon.export import CharacterExporter
from poseydon.features.reduce import apply_reduction, build_reduction


CORPUS = Path("data/truebones")
RAW = Path("data/_raw/truebones")


def _fixture():
    exporter = CharacterExporter.load(CORPUS, RAW, "Fox")
    animation = Animation.load(CORPUS / "clips/Fox/attack1.npz").as_rigid_body(
        joint_translation="drop"
    )
    animation = apply_reduction(animation, build_reduction(animation))
    return exporter, animation


@pytest.mark.parametrize("method", ["prepared_fbx", "presentation_fbx", "original_fbx"])
def test_character_exporter_fbx_modes_keep_mesh(method):
    exporter, animation = _fixture()
    scene = getattr(exporter, method)(animation)
    assert scene.meshes, method
    assert tuple(scene.joint_names)


def test_prepared_and_presentation_use_distinct_canonical_assets():
    exporter, animation = _fixture()
    assert exporter.model_fbx != exporter.presentation_asset
    prepared = exporter.prepared_fbx(animation)
    prepared_joints = len(prepared.joint_names)
    presentation = exporter.presentation_fbx(animation)
    assert prepared_joints <= len(presentation.joint_names)

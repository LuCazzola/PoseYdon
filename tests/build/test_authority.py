from pathlib import Path

import pytest
import yaml

from poseydon.build.authority import ArtifactAvailability
from poseydon.core.skeleton import ManifestError, SkeletonManifest


def _load(tmp_path, *, authority=None, artifacts=None):
    data = {
        "skeleton": "Fixture",
        "rest_pose": "__Idle.bvh",
        "facing": {"hips": {"right": "Right", "left": "Left"}},
        "contact": {"max_height": 0.1, "max_speed": 0.2},
        "authority": authority
        or {"skeleton": "bvh", "motion": "bvh", "surface": "none", "auxiliary_joints": "exclude"},
        "artifacts": artifacts if artifacts is not None else {"bvh": "__Idle.bvh", "fbx": None},
    }
    path = tmp_path / "manifest.yaml"
    path.write_text(yaml.safe_dump(data))
    return SkeletonManifest.load(path)


def test_bvh_only_manifest_has_no_surface_outputs(tmp_path):
    manifest = _load(tmp_path, artifacts={"bvh": "__Idle.bvh", "fbx": None})

    availability = ArtifactAvailability.from_manifest(manifest)

    assert availability.prepared_bvh
    assert availability.original_bvh
    assert not availability.prepared_fbx
    assert not availability.presentation_fbx
    assert not availability.original_fbx
    assert availability.reasons["prepared_fbx"] == "source_surface_missing"


def test_fbx_only_manifest_accepts_null_bvh(tmp_path):
    manifest = _load(
        tmp_path,
        authority={
            "skeleton": "fbx",
            "motion": "fbx",
            "surface": "fbx",
            "auxiliary_joints": "bake",
        },
        artifacts={"bvh": None, "fbx": "__Idle.fbx"},
    )

    assert manifest.artifacts.bvh is None
    assert manifest.artifacts.fbx == Path("__Idle.fbx")
    assert ArtifactAvailability.from_manifest(manifest).original_fbx


def test_declared_authority_requires_its_modality(tmp_path):
    with pytest.raises(ManifestError, match="motion authority.*bvh"):
        _load(
            tmp_path,
            authority={
                "skeleton": "bvh",
                "motion": "bvh",
                "surface": "none",
                "auxiliary_joints": "exclude",
            },
            artifacts={"bvh": None, "fbx": None},
        )


def test_unknown_authority_value_is_rejected(tmp_path):
    with pytest.raises(ManifestError, match="authority.skeleton"):
        _load(
            tmp_path,
            authority={
                "skeleton": "gltf",
                "motion": "bvh",
                "surface": "none",
                "auxiliary_joints": "exclude",
            },
        )


def test_unavailable_error_carries_machine_readable_context(tmp_path):
    availability = ArtifactAvailability.from_manifest(_load(tmp_path))

    error = availability.require("dataset", "Fixture", "prepared", "fbx")

    assert error.reason == "source_surface_missing"
    assert error.rig == "Fixture"

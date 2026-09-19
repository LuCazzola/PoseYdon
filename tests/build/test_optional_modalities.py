from pathlib import Path

import pytest

from poseydon.build.authority import ArtifactAvailability
from poseydon.build.character_assets import CharacterAssetContext, build_character_assets
from poseydon.core.skeleton import SkeletonManifest

def test_bvh_only_build_has_typed_surface_absence(tmp_path):
    context=CharacterAssetContext("BVHOnly",None,("Root",),(-1,))
    assets=build_character_assets(context,tmp_path)
    assert assets.model is None and assets.presentation is None
    assert assets.reason == "source_surface_missing"

def test_fbx_only_manifest_exposes_bvh_serializer_without_vendor_bvh(tmp_path):
    path=tmp_path/"manifest.yaml"
    path.write_text("""skeleton: Fixture
rest_pose: null
facing: {hips: {right: Right, left: Left}}
contact: {max_height: 0.1, max_speed: 0.2}
authority: {skeleton: fbx, motion: fbx, surface: fbx, auxiliary_joints: bake}
artifacts: {bvh: null, fbx: motion.fbx}
""")
    manifest=SkeletonManifest.load(path)
    availability=ArtifactAvailability.from_manifest(manifest)
    assert availability.prepared_bvh and availability.original_bvh
    assert availability.original_fbx
    assert manifest.artifacts.bvh is None

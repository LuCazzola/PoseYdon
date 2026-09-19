import json
from pathlib import Path
from scripts.build_skeleton_visual_debug import build_bundle

def test_bundle_records_typed_absence_and_hash(tmp_path):
    bvh=tmp_path/"x.bvh"; bvh.write_text("fixture")
    build_bundle([{"rig":"BvhOnly","action":"x","original_bvh":bvh}], tmp_path/"out")
    data=json.loads((tmp_path/"out"/"BvhOnly"/"metadata.json").read_text())
    assert data["original_bvh"]["sha256"]
    assert data["prepared_fbx"] is None
    assert data["unavailable"]["prepared_fbx"] == "source_surface_missing"

import pytest

from poseydon.core.skeleton import (
    HML_MEAN_BONE_LENGTH,
    ManifestError,
    SkeletonManifest,
)

MINIMAL = """
skeleton: Testy
facing:
  hips:      {right: R_Thigh, left: L_Thigh}
  shoulders: {right: R_Arm,   left: L_Arm}
scale: {mean_bone_length: 0.2}
contact: {max_height: 0.3, max_speed: 0.045}
"""


def write(tmp_path, text, name="Testy.yaml"):
    path = tmp_path / name
    path.write_text(text)
    return path


def test_loads_minimal_manifest(tmp_path):
    manifest = SkeletonManifest.load(write(tmp_path, MINIMAL))

    assert manifest.name == "Testy"
    assert [p.right for p in manifest.facing] == ["R_Thigh", "R_Arm"]
    assert [p.left for p in manifest.facing] == ["L_Thigh", "L_Arm"]
    assert manifest.extra_yaw_deg == 0.0
    assert manifest.mean_bone_length == 0.2
    assert manifest.contact.max_speed == 0.045
    assert manifest.tags == ()
    assert manifest.fps is None


def test_hml_constant_matches_reference():
    # mean of the 21 SMPL bone lengths in the reference's param_utils
    assert HML_MEAN_BONE_LENGTH == pytest.approx(0.20921428571428569, abs=1e-15)


def test_facing_order_is_hips_then_shoulders(tmp_path):
    # The forward axis sums the across-body vectors, so order does not change the
    # result -- but it must be deterministic for reproducibility.
    manifest = SkeletonManifest.load(write(tmp_path, MINIMAL))
    assert manifest.facing[0].right == "R_Thigh"
    assert manifest.facing[1].right == "R_Arm"


def test_base_inheritance_merges_and_overrides(tmp_path):
    (tmp_path / "_base.yaml").write_text(
        "facing:\n"
        "  hips:      {right: R_Thigh, left: L_Thigh}\n"
        "  shoulders: {right: R_Arm,   left: L_Arm}\n"
        "scale: {mean_bone_length: 0.2}\n"
        "contact: {max_height: 0.3, max_speed: 0.045}\n"
        "strip_joint_prefix: 'mixamorig:'\n"
        "tags: [humanoid]\n"
    )
    child = write(
        tmp_path,
        "base: _base.yaml\nskeleton: Goblin\ntags: [humanoid, goblin]\n",
        name="Goblin.yaml",
    )
    manifest = SkeletonManifest.load(child)

    assert manifest.name == "Goblin"
    assert manifest.strip_joint_prefix == "mixamorig:"
    assert manifest.facing[0].right == "R_Thigh"
    assert manifest.tags == ("humanoid", "goblin")


def test_tpose_path_resolves_relative_to_manifest(tmp_path):
    nested = tmp_path / "skeletons"
    nested.mkdir()
    path = write(nested, MINIMAL + "tpose: ../tposes/Testy.bvh\n")
    manifest = SkeletonManifest.load(path)
    assert manifest.tpose == (tmp_path / "tposes" / "Testy.bvh").resolve()


def test_rejects_double_underscore_in_skeleton_name(tmp_path):
    text = MINIMAL.replace("skeleton: Testy", "skeleton: Bad__Name")
    with pytest.raises(ManifestError, match="__"):
        SkeletonManifest.load(write(tmp_path, text))


def test_rejects_missing_required_section(tmp_path):
    text = "skeleton: Testy\ncontact: {max_height: 0.3, max_speed: 0.045}\n"
    with pytest.raises(ManifestError, match="facing"):
        SkeletonManifest.load(write(tmp_path, text))


def test_rejects_unknown_key_with_suggestion(tmp_path):
    text = MINIMAL + "foot_joint: [a]\n"
    with pytest.raises(ManifestError, match="foot_joints"):
        SkeletonManifest.load(write(tmp_path, text))


def test_rejects_facing_pair_missing_a_side(tmp_path):
    text = MINIMAL.replace("{right: R_Arm,   left: L_Arm}", "{right: R_Arm}")
    with pytest.raises(ManifestError, match="left"):
        SkeletonManifest.load(write(tmp_path, text))

from pathlib import Path

import pytest

from poseydon.core.skeleton import (
    ContactParams,
    FacingPair,
    ManifestError,
    SkeletonManifest,
    resolve,
    resolve_joint,
    strip_prefix,
)

NAMES = ["Hips", "R_Thigh", "L_Thigh", "R_Arm", "L_Arm", "R_Foot", "L_Foot"]


def make_manifest(foot=("R_Foot", "L_Foot")) -> SkeletonManifest:
    return SkeletonManifest(
        name="Testy",
        facing=(FacingPair("R_Thigh", "L_Thigh"), FacingPair("R_Arm", "L_Arm")),
        contact=ContactParams(0.3, 0.045),
        source=Path("Testy.yaml"),
        foot_joints=foot,
    )


def test_resolves_exact_name():
    assert resolve_joint("R_Thigh", NAMES) == 1


def test_unknown_name_suggests_nearest_match():
    with pytest.raises(ManifestError) as excinfo:
        resolve_joint("R_Thig", NAMES)
    message = str(excinfo.value)
    assert "R_Thig" in message
    assert "R_Thigh" in message


def test_unknown_name_lists_available_when_nothing_is_close():
    with pytest.raises(ManifestError) as excinfo:
        resolve_joint("zzzzzz", NAMES)
    assert "Hips" in str(excinfo.value)


def test_resolve_produces_index_pairs_in_order():
    resolved = resolve(make_manifest(), NAMES)
    assert resolved.facing_indices == ((1, 2), (3, 4))
    assert resolved.foot_indices == (5, 6)


def test_resolve_reports_context_on_failure():
    manifest = make_manifest(foot=("NoSuchFoot",))
    with pytest.raises(ManifestError, match="foot_joints"):
        resolve(manifest, NAMES)


def test_strip_prefix_removes_only_the_declared_prefix():
    names = ["mixamorig:Hips", "mixamorig:Spine", "Extra"]
    assert strip_prefix(names, "mixamorig:") == ("Hips", "Spine", "Extra")


def test_strip_prefix_is_identity_when_none():
    assert strip_prefix(NAMES, None) == tuple(NAMES)


def test_strip_prefix_rejects_collisions():
    with pytest.raises(ManifestError, match="collide"):
        strip_prefix(["a:Hips", "Hips"], "a:")

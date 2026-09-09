"""A rig's rest pose is declared in its manifest and validated against the corpus.

The manifest names the raw clip whose first frame is this rig's rest pose. That
choice is authored -- no naming rule can pick it reliably, since matching
"idle" as a substring picks Lion's __DeathIdle.bvh, Jaguar's __LieIdle.bvh and
Trex's __idle_attack.bvh. But an authored value must never override the
corpus's own evidence: the declared file must carry the rig's MODAL joint set,
because a rest pose disagreeing with the clips makes `RestRelative` reject
every one of them. `rest_source` enforces that, and raises rather than falling
back -- a rig whose rest pose cannot be resolved is a data error that must
stop the build, not one that quietly gets an arbitrary rest geometry.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.conftest import CORPUS

_HEADER_TAIL = "MOTION\nFrames: 1\nFrame Time: 0.0083333\n0.0\n"


def _bvh_text(joint_names: list[str]) -> str:
    lines = ["HIERARCHY", f"ROOT {joint_names[0]}", "{", "OFFSET 0.0 0.0 0.0",
              "CHANNELS 3 Xposition Yposition Zposition"]
    for name in joint_names[1:]:
        lines += [
            f"JOINT {name}",
            "{",
            "OFFSET 1.0 0.0 0.0",
            "CHANNELS 3 Zrotation Xrotation Yrotation",
            "End Site",
            "{",
            "OFFSET 1.0 0.0 0.0",
            "}",
            "}",
        ]
    lines.append("}")
    return "\n".join(lines) + "\n" + _HEADER_TAIL


def _write(tmp_path: Path, name: str, joint_names: list[str]) -> Path:
    path = tmp_path / name
    path.write_text(_bvh_text(joint_names))
    return path


def test_manifest_round_trips_rest_pose(tmp_path):
    """A rig's rest pose is data about the rig, so it lives in the manifest."""
    from poseydon.core.skeleton import SkeletonManifest

    path = tmp_path / "manifest.yaml"
    path.write_text(
        "skeleton: Testy\n"
        "rest_pose: __IdleLoop.bvh\n"
        "facing:\n"
        "  hips: {right: R, left: L}\n"
        "contact: {max_height: 0.3, max_speed: 0.04}\n"
    )
    assert SkeletonManifest.load(path).rest_pose == "__IdleLoop.bvh"


def test_the_dead_tpose_key_is_gone(tmp_path):
    """`tpose` was parsed, read by two consumers, and declared by no manifest,
    so both consumers silently took their fallback forever. A key that reads as
    working is worse than one that is obviously dead."""
    from poseydon.core.skeleton import ManifestError, SkeletonManifest

    path = tmp_path / "manifest.yaml"
    path.write_text(
        "skeleton: Testy\n"
        "tpose: ../tposes/Testy.bvh\n"
        "facing:\n"
        "  hips: {right: R, left: L}\n"
        "contact: {max_height: 0.3, max_speed: 0.04}\n"
    )
    with pytest.raises(ManifestError, match="tpose"):
        SkeletonManifest.load(path)


def test_rest_source_refuses_a_rig_that_declares_nothing(tmp_path):
    """No silent fallback. A rig with no declared rest pose is a data error
    that must stop the build, not a rig that quietly gets an arbitrary one."""
    from poseydon.build.corpus import rest_source
    from poseydon.core.skeleton import SkeletonManifest

    path = tmp_path / "manifest.yaml"
    path.write_text(
        "skeleton: Testy\n"
        "facing:\n"
        "  hips: {right: R, left: L}\n"
        "contact: {max_height: 0.3, max_speed: 0.04}\n"
    )
    manifest = SkeletonManifest.load(path)
    clip = _write(tmp_path, "__Walk.bvh", ["Hips", "Spine", "Head"])
    with pytest.raises(ValueError, match="declares no `rest_pose`"):
        rest_source(manifest, [clip])


def test_rest_source_refuses_a_declaration_outside_the_modal_set(tmp_path):
    """Trex's natural neutral pick, __STILL.bvh, IS that rig's outlier file.
    An authored value must never override the corpus's own evidence."""
    from poseydon.build.corpus import rest_source
    from poseydon.core.skeleton import SkeletonManifest

    path = tmp_path / "manifest.yaml"
    path.write_text(
        "skeleton: Testy\n"
        "rest_pose: __Odd.bvh\n"
        "facing:\n"
        "  hips: {right: R, left: L}\n"
        "contact: {max_height: 0.3, max_speed: 0.04}\n"
    )
    manifest = SkeletonManifest.load(path)
    majority = ["Hips", "Spine", "Head"]
    clips = [
        _write(tmp_path, "__Walk.bvh", majority),
        _write(tmp_path, "__Run.bvh", majority),
        _write(tmp_path, "__Odd.bvh", ["Hips", "Spine"]),
    ]
    with pytest.raises(ValueError, match="modal"):
        rest_source(manifest, clips)


def test_rest_source_returns_the_declared_file(tmp_path):
    from poseydon.build.corpus import rest_source
    from poseydon.core.skeleton import SkeletonManifest

    path = tmp_path / "manifest.yaml"
    path.write_text(
        "skeleton: Testy\n"
        "rest_pose: __Run.bvh\n"
        "facing:\n"
        "  hips: {right: R, left: L}\n"
        "contact: {max_height: 0.3, max_speed: 0.04}\n"
    )
    manifest = SkeletonManifest.load(path)
    majority = ["Hips", "Spine", "Head"]
    clips = [
        _write(tmp_path, "__Walk.bvh", majority),
        _write(tmp_path, "__Run.bvh", majority),
    ]
    assert rest_source(manifest, clips).name == "__Run.bvh"


def test_every_rig_declares_a_rest_pose_that_exists_and_is_modal():
    """The declaration is a judgement about pose content and cannot be tested.
    What can be: it exists, and it agrees with the rig's own clips."""
    from poseydon.build.corpus import rest_source
    from poseydon.core.skeleton import SkeletonManifest

    source = CORPUS / "source"
    if not source.is_dir():
        pytest.skip("Truebones corpus not present")

    manifests = sorted((CORPUS / "rigs").glob("*/manifest.yaml"))
    assert manifests, "no rig manifests found"

    checked = 0
    for path in manifests:
        manifest = SkeletonManifest.load(path)
        rig_dir = source / manifest.name
        if not rig_dir.is_dir():
            continue
        clips = sorted(rig_dir.glob("*.bvh"))
        if not clips:
            continue
        # Raises on: no declaration, missing file, or non-modal skeleton.
        rest_source(manifest, clips)
        checked += 1

    # `continue` above can silently drop a rig from the loop without failing
    # anything else -- exactly the shape of bug this plan exists to catch
    # (a check that can pass while checking nothing). All 73 rigs currently
    # have a source directory and clips, so this must be exact, not a floor:
    # an exact count is what notices a rig quietly falling out of coverage.
    assert checked == len(manifests), (
        f"only checked {checked}/{len(manifests)} rig manifests; a rig with "
        "no source directory or no clips was silently skipped"
    )

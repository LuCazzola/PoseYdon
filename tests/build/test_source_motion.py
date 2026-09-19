from pathlib import Path
from types import SimpleNamespace

from poseydon.build.source import SourceMotion
from poseydon.core.authority import AuthorityPolicy, SourceArtifacts


def _manifest(kind):
    return SimpleNamespace(
        name="Rig",
        authority=AuthorityPolicy(skeleton=kind, motion=kind),
        artifacts=SourceArtifacts(**{kind: Path(f"clip.{kind}")}),
    )


def test_source_motion_dispatches_only_to_bvh_authority(tmp_path):
    calls = []
    result = SourceMotion.load(
        _manifest("bvh"), tmp_path,
        bvh_loader=lambda path: calls.append(("bvh", path)) or "bvh-motion",
        fbx_loader=lambda path: calls.append(("fbx", path)) or "fbx-motion",
    )
    assert result == "bvh-motion"
    assert calls == [("bvh", tmp_path / "clip.bvh")]


def test_source_motion_dispatches_only_to_fbx_authority(tmp_path):
    calls = []
    result = SourceMotion.load(
        _manifest("fbx"), tmp_path,
        bvh_loader=lambda path: calls.append(("bvh", path)) or "bvh-motion",
        fbx_loader=lambda path: calls.append(("fbx", path)) or "fbx-motion",
    )
    assert result == "fbx-motion"
    assert calls == [("fbx", tmp_path / "clip.fbx")]

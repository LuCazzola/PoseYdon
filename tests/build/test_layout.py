"""The entity-first layout: a rig is a directory."""

from __future__ import annotations

import pytest

from poseydon.core.skeleton import SkeletonManifest
from poseydon.ingest.pipeline import available_rigs
from tests.conftest import CORPUS


def test_available_rigs_lists_directories_not_yaml_stems(tmp_path):
    rigs = tmp_path / "rigs"
    for name in ("Goat", "Crab"):
        (rigs / name).mkdir(parents=True)
        (rigs / name / "manifest.yaml").write_text(f"skeleton: {name}\n")
    # A shared fragment is a file, not a directory, so it is not a rig -- which
    # is what retires the `_`-prefix convention the old flat layout needed.
    (rigs / "_base.yaml").write_text("fps: 30\n")
    # A directory with no manifest is not a rig either: rigs/<Rig>/ also holds
    # derived artefacts, so one can exist before anyone authors a manifest.
    (rigs / "Scratch").mkdir()

    assert available_rigs(rigs) == ["Crab", "Goat"]


def test_manifests_live_beside_their_rig():
    if not (CORPUS / "rigs").is_dir():
        pytest.skip("corpus not migrated")
    manifest = SkeletonManifest.load(CORPUS / "rigs" / "Goat" / "manifest.yaml")
    assert manifest.name == "Goat"
    assert "quadruped" in manifest.tags


def test_strip_joint_prefix_is_gone():
    """It was declared, documented and read by nothing; joint-name humanizing
    computes the prefix instead of requiring it to be declared."""
    assert not hasattr(SkeletonManifest, "strip_joint_prefix")

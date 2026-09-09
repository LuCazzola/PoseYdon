"""FBX IO-level round trip: `FBX.read -> write -> read` preserves the skeleton.

This covers the FBX IO layer ONLY (`poseydon.io.fbx.FBX.read`/`.write`). It is
NOT the prepare/unprepare round trip that the other `tests/build/` round-trip
tests exercise for BVH. The FBX arm of that round trip has no inverse to test:
`FBX` exposes `read`/`write` plus geometry helpers, never a `read_animation`/
`write_animation`/`to_animation` pair, and `scripts/process_dataset_truebones_fbx.py`
calls `FBX.read -> rotate -> scale_to_mean_bone_length -> write` directly on a
Blender scene without going through `PrepareChain` or recording fitted
parameters -- so there is nothing to unprepare. That gap is recorded in the
design spec's Sec 9; closing it is build-stage work for a later plan, not this
test.

Blender-gated: Dockerfile.compat and the test image deliberately ship no
Blender, so this runs against the `fbx` compose service, which installs
`python3-pytest` (Debian's distro package, importable from Blender's linked
system Python the same way PyYAML/numpy/scipy are) for exactly this purpose:

    docker compose run --rm fbx blender -b --python-expr \\
        "import sys, pytest; sys.exit(pytest.main(['tests/build/test_roundtrip_fbx.py','-v','-rs']))"

Skipping when Blender is absent is correct here -- the data is genuinely not
available -- but a test that can ONLY ever skip is worthless (this plan exists
because exactly that happened to a prior round-trip test for weeks). This one
is confirmed to actually execute and report failures inside the `fbx` image;
see task-8-report.md for the real pytest transcript and a deliberate-failure
check.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.conftest import CORPUS, SAMPLE_RIGS

bpy = pytest.importorskip("bpy", reason="FBX round trip needs Blender")

from poseydon.io.fbx import FBX

# Measured tolerance: joint offsets are bind-pose positions Blender derives
# from the armature's own bone-head matrices, not a value FBX stores and
# echoes back verbatim, so a read -> write -> read is not expected to be
# bit-exact. Measured directly (FBX.read -> FBX.write -> FBX.read) on the
# first per-clip source FBX of each SAMPLE_RIGS rig, in the rig's own
# (unscaled) units: max per-component offset error was 1.79e-7 (Flamingo),
# 2.38e-7 (BrownBear), 5.80e-6 (Crab, the worst), 3.58e-7 (Scorpion) and
# 7.15e-7 (Camel) -- all consistent with float32 round-off through Blender's
# FBX exporter/importer, not a structural drift. This bound is set roughly an
# order of magnitude above that measured worst case (Crab's 5.80e-6).
OFFSET_ATOL = 5e-5


def _first_per_clip_fbx(rig: str):
    source_dir = CORPUS / "source" / rig
    if not source_dir.is_dir():
        pytest.skip(f"{rig}: no source directory")
    # "ALL" files are Truebones' own all-takes compilations (e.g.
    # `Flamingo-ALL.fbx`, `CrabAll.fbx`, `RaptorALL.fbx`) -- not a distinct
    # clip. `process_dataset_truebones_fbx.py` excludes them the same way.
    fbxs = sorted(p for p in source_dir.glob("*.fbx") if "ALL" not in p.stem.upper())
    if not fbxs:
        pytest.skip(f"{rig}: no per-clip source FBX")
    return fbxs[0]


@pytest.mark.parametrize("rig", SAMPLE_RIGS)
def test_fbx_read_write_read_preserves_the_skeleton(rig, tmp_path):
    source_path = _first_per_clip_fbx(rig)

    source = FBX.read(source_path)
    source_names = source.joint_names
    source_parents = source.joint_parents
    source_offsets = source.joint_offsets

    out_path = tmp_path / "roundtrip.fbx"
    source.write(out_path)
    restored = FBX.read(out_path)

    assert restored.joint_names == source_names, (
        f"{rig}: joint names changed across an FBX read -> write -> read"
    )
    assert restored.joint_parents == source_parents, (
        f"{rig}: joint parents changed across an FBX read -> write -> read"
    )
    np.testing.assert_allclose(
        restored.joint_offsets,
        source_offsets,
        atol=OFFSET_ATOL,
        err_msg=f"{rig}: bind-pose joint offsets did not survive an FBX round trip",
    )

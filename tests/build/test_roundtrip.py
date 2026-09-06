"""source -> prepare -> reduce -> expand -> unprepare -> source.

If either inverse is wrong this fails, which is the point: an application that
cannot return a user's rig in the representation they supplied is a benchmark,
not a product.
"""

from __future__ import annotations

import numpy as np
import pytest

from poseydon.build.prepare import (
    CentreXZ,
    EnforceRigid,
    FaceAxis,
    PrepareChain,
    PutOnGround,
    RestRelative,
    ScaleToMeanBoneLength,
)
from poseydon.core.skeleton import SkeletonManifest, resolve
from poseydon.features.reduce import apply_reduction, build_reduction, invert_reduction
from poseydon.io.bvh import BVH

from tests.conftest import CORPUS, SAMPLE_RIGS

CHAIN = PrepareChain(
    (
        RestRelative(),
        FaceAxis(axis="+Z"),
        EnforceRigid(joint_translation="drop"),
        CentreXZ(),
        ScaleToMeanBoneLength(),
        PutOnGround(),
    )
)


def _manifest(rig: str) -> SkeletonManifest:
    for candidate in (
        CORPUS / "rigs" / rig / "manifest.yaml",
        CORPUS / "skeletons" / f"{rig}.yaml",
    ):
        if candidate.is_file():
            return SkeletonManifest.load(candidate)
    pytest.skip(f"no manifest for {rig}")


def _rest_path(clips):
    for path in clips:
        if "tpos" in path.name.lower():
            return path
    for path in clips:
        if path.name.lower().lstrip("_").startswith("idle"):
            return path
    pytest.skip("no rest-pose file for this rig")


@pytest.mark.parametrize("rig", SAMPLE_RIGS)
def test_round_trip_returns_the_source_rig(rig, raw_clips):
    clips = raw_clips(rig)
    manifest = _manifest(rig)
    rest = BVH.read(_rest_path(clips)).to_animation()
    resolved = resolve(manifest, rest.names)
    rig_params = CHAIN.fit_rig(rest, resolved)

    source_bvh = BVH.read(next(p for p in clips if p != _rest_path(clips)))
    source = source_bvh.to_animation()
    if tuple(source.names) != tuple(rest.names):
        pytest.skip(f"{rig}: this clip is rigged differently from its own rest pose")

    prepared, params = CHAIN.apply(source, resolve(manifest, source.names), rig_params)

    reduction = build_reduction(prepared)
    reduced = apply_reduction(prepared, reduction)
    expanded = invert_reduction(reduced, reduction)
    restored = CHAIN.invert(expanded, params)

    assert restored.names == source.names
    assert list(restored.parents) == list(source.parents)

    # EnforceRigid discards animated per-joint translation, so compare against
    # the source made rigid -- the structure round-trips, the values do not.
    reference = source.as_rigid_body(joint_translation="drop")
    scale = float(np.linalg.norm(reference.offsets[1:], axis=-1).mean())
    error = np.abs(restored.global_positions() - reference.global_positions()).max()
    assert error < 1e-4 * scale, f"{rig}: worst joint off by {error / scale:.2e} bone lengths"


@pytest.mark.parametrize("rig", SAMPLE_RIGS)
def test_prepared_clips_face_plus_z_and_stand_on_the_ground(rig, raw_clips):
    clips = raw_clips(rig)
    manifest = _manifest(rig)
    rest = BVH.read(_rest_path(clips)).to_animation()
    rig_params = CHAIN.fit_rig(rest, resolve(manifest, rest.names))

    prepared, _params = CHAIN.apply(rest, resolve(manifest, rest.names), rig_params)

    lengths = np.linalg.norm(prepared.offsets[1:], axis=-1)
    assert lengths.mean() == pytest.approx(0.20921428571428569, rel=1e-9)
    assert prepared.global_positions()[..., 1].min() == pytest.approx(0.0, abs=1e-9)

"""source -> prepare -> reduce -> expand -> unprepare -> source.

If either inverse is wrong this fails, which is the point: an application that
cannot return a user's rig in the representation they supplied is a benchmark,
not a product.

The last arrow is deliberately not asserted against the source's world
positions. The pipeline builds its canonical skeleton from the T-pose's
MEASURED geometry, because raw Biped exports "carry the real skeleton in
their per-joint POSITION channels while the OFFSET block describes
something else" (see poseydon.core.animation.rest_geometry). Comparing
against `source.as_rigid_body("drop")` pins joints to that distrusted
header and demands a skeleton the pipeline never claimed to preserve. What
returns is the source's exact STRUCTURE with per-joint translation replaced
by the rest pose's -- the documented EnforceRigid limitation.
"""

from __future__ import annotations

import numpy as np
import pytest
from scripts.process_dataset_truebones import _chain_for, rest_source

from poseydon.core.rotations import QUAT_IDENTITY
from poseydon.core.skeleton import SkeletonManifest, resolve
from poseydon.features.reduce import apply_reduction, build_reduction, invert_reduction
from poseydon.ingest.align import axis_vector, facing_quats
from poseydon.io.bvh import BVH
from tests.conftest import CORPUS, SAMPLE_RIGS


def _manifest(rig: str) -> SkeletonManifest:
    for candidate in (
        CORPUS / "rigs" / rig / "manifest.yaml",
        CORPUS / "skeletons" / f"{rig}.yaml",
    ):
        if candidate.is_file():
            return SkeletonManifest.load(candidate)
    pytest.skip(f"no manifest for {rig}")


def _rest_path(clips):
    """The rig's rest-pose file, from its manifest.

    This used to carry its own copy of a filename-matching rule -- the rule
    b2b0151 fixed in the script and not here. Crab therefore resolved to its
    54-joint __TPOSE.bvh, every 64-joint clip disagreed, and the case skipped.
    """
    return rest_source(_manifest(_rig_of(clips)), list(clips))


def _rig_of(clips) -> str:
    return clips[0].parent.name


def _apply_with(chain, anim, params):
    """Apply the chain reusing recorded parameters, refitting nothing.

    `PrepareChain.apply` refits clip-scoped stages, and `FaceAxis` derives its
    rotation from frame-0 world positions -- which differ between the source
    and the restored clip, because EnforceRigid replaced per-joint translation.
    Reusing the recorded parameters is what makes the comparison exact.
    """
    current = anim
    for stage in chain.stages:
        current = stage.apply(current, params[stage.name])
    return current


@pytest.mark.parametrize("rig", SAMPLE_RIGS)
def test_round_trip_returns_the_source_rig(rig, raw_clips):
    clips = raw_clips(rig)
    manifest = _manifest(rig)
    rest_bvh = BVH.read(_rest_path(clips))
    rest = rest_bvh.to_animation()
    resolved = resolve(manifest, rest.names)
    # Use the production chain -- built with THIS rig's real BVH channel
    # layout, exactly as scripts.process_dataset_truebones.process_species does
    # -- so the round trip exercises PromoteRoot for the 14 ground-locator
    # rigs (Camel among them) instead of a private copy that omits it.
    chain = _chain_for(rest_bvh.channels, rig)
    rig_params = chain.fit_rig(rest, resolved)

    source_bvh = BVH.read(next(p for p in clips if p != _rest_path(clips)))
    source = source_bvh.to_animation()
    if tuple(source.names) != tuple(rest.names):
        pytest.skip(f"{rig}: this clip is rigged differently from its own rest pose")

    prepared, params = chain.apply(source, resolve(manifest, source.names), rig_params)

    reduction = build_reduction(prepared)
    reduced = apply_reduction(prepared, reduction)
    # A reduction that degenerated to the identity would satisfy the assertion
    # below unchanged.
    assert reduced.n_joints < prepared.n_joints
    expanded = invert_reduction(reduced, reduction)
    restored = chain.invert(expanded, params)

    # Structure: the rig comes back exactly as the user supplied it.
    assert restored.names == source.names
    assert list(restored.parents) == list(source.parents)
    np.testing.assert_allclose(restored.offsets, source.offsets, atol=1e-9)

    # EnforceRigid replaced per-joint translation with the rest pose's, so the
    # returned clip's non-root translations are constant over time. This is the
    # documented loss, asserted rather than assumed.
    non_root = restored.translations[:, 1:]
    np.testing.assert_allclose(
        non_root, np.broadcast_to(non_root[:1], non_root.shape), atol=1e-9
    )

    # Reduction and expansion are world-exact, not representation-exact: a
    # collapsed joint's rotation is folded into its parent and cannot be split
    # back out, because the two are coincident and the split is unobservable.
    # So every geometric claim below is made in world space.
    #
    # First, isolate the reduction: expanding what we reduced must put every
    # joint back where it was.
    np.testing.assert_allclose(
        expanded.global_positions(), prepared.global_positions(), rtol=0, atol=1e-9
    )

    # Then the whole loop: re-preparing the returned asset reproduces the
    # prepared clip. This is the property the application needs -- a generated
    # clip, unprepared onto the user's rig and prepared again, is the clip we
    # started from.
    reprepared = _apply_with(chain, restored, params)
    np.testing.assert_allclose(
        reprepared.global_positions(), prepared.global_positions(), rtol=0, atol=1e-9
    )


@pytest.mark.parametrize("rig", SAMPLE_RIGS)
def test_prepared_clips_face_plus_z_and_stand_on_the_ground(rig, raw_clips):
    clips = raw_clips(rig)
    manifest = _manifest(rig)
    rest_bvh = BVH.read(_rest_path(clips))
    rest = rest_bvh.to_animation()
    resolved = resolve(manifest, rest.names)
    chain = _chain_for(rest_bvh.channels, rig)
    rig_params = chain.fit_rig(rest, resolved)

    prepared, _params = chain.apply(rest, resolved, rig_params)

    # Facing cannot be checked by the round-trip test: FaceAxis.invert mirrors
    # whatever fit recorded, so a wrong facing rotation cancels between apply
    # and invert and leaves that test green. Assert it here, absolutely, by
    # asking FaceAxis's own derivation what correction the PREPARED clip still
    # needs -- if it already faces +Z, that correction is the identity.
    residual = facing_quats(
        prepared.global_positions()[:1],
        resolved.facing_indices,
        resolved.manifest.extra_yaw_deg,
        axis_vector("+Z"),
    )[0]
    np.testing.assert_allclose(np.abs(residual), QUAT_IDENTITY, atol=1e-9)

    # ScaleToMeanBoneLength fits its factor over non-degenerate bones only
    # (build/prepare.py); averaging over the full offset block, End Sites
    # included, would restate that exclusion's inverse rather than test it --
    # every rig would trivially fail or pass depending on how many End Sites it
    # happens to declare. So measure the same set the stage measures.
    lengths = np.linalg.norm(prepared.offsets[1:], axis=-1)
    real_lengths = lengths[lengths > 1e-8]
    assert real_lengths.mean() == pytest.approx(0.20921428571428569, rel=1e-9)
    assert prepared.global_positions()[..., 1].min() == pytest.approx(0.0, abs=1e-9)

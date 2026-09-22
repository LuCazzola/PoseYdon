"""What `CharacterExporter` must preserve when it returns a clip to source space.

Three separable promises, one test each:

* the restored pose does not depend on WHICH yaw is put back -- a generated
  clip has no historical facing, so the exporter substitutes a default, and
  that substitution has to be a rigid rotation of the whole rig and nothing
  else. It was not: the stages that run after `FaceAxis` in the inverse order
  (`PromoteRoot`, `RestRelative`) stamp constants captured in the SOURCE frame,
  so leaving the rig in canonical +Z scrambled every joint's relative framing.
* the original representation returns the source file's own CHANNELS block,
  which `EnforceRigid.invert` documents as the writer's job.
* the rotation order in that block is the one the joint declares, not a
  writer-wide constant.
"""

from __future__ import annotations

import numpy as np
import pytest

from poseydon.build.corpus import rest_action
from poseydon.build.index import action_slug, strip_skeleton_prefix
from poseydon.build.prepare import RigTransform
from poseydon.core.skeleton import SkeletonManifest
from poseydon.core.kinematics import forward_kinematics
from poseydon.export import CharacterExporter
from poseydon.features.reduce import apply_reduction
from poseydon.io.bvh import BVH
from tests.conftest import CORPUS, SAMPLE_RIGS

RAW = CORPUS.parent / "_raw" / "truebones"


def _shape_residual(a, b) -> float:
    """Largest per-joint disagreement after the best per-frame rigid alignment.

    Reported as a fraction of rig extent, so a pure global rotation or
    translation scores zero and only a genuine change of SHAPE registers.
    """
    frames = min(a.rotations.shape[0], b.rotations.shape[0])
    pa, _ = forward_kinematics(a.rotations[:frames], a.translations[:frames], a.parents)
    pb, _ = forward_kinematics(b.rotations[:frames], b.translations[:frames], b.parents)
    ca = pa - pa.mean(axis=1, keepdims=True)
    cb = pb - pb.mean(axis=1, keepdims=True)
    worst = 0.0
    for frame in range(frames):
        u, _s, vt = np.linalg.svd(ca[frame].T @ cb[frame])
        flip = np.diag([1.0, 1.0, float(np.sign(np.linalg.det(u @ vt)))])
        rotation = u @ flip @ vt
        worst = max(worst, float(np.linalg.norm(cb[frame] @ rotation.T - ca[frame], axis=-1).max()))
    return worst / float(np.linalg.norm(ca, axis=-1).max())


def _manifest(rig: str) -> SkeletonManifest:
    path = CORPUS / "rigs" / rig / "manifest.yaml"
    if not path.is_file():
        pytest.skip(f"no manifest for {rig}")
    return SkeletonManifest.load(path)


def _case(rig: str):
    exporter = CharacterExporter.load(CORPUS, RAW, rig)
    clips = sorted((CORPUS / "clips" / rig).glob("*.bvh"))
    if not clips:
        pytest.skip(f"no prepared clips for {rig}")
    clip = next((p for p in clips if p.stem != "tpose"), clips[0])
    prepared = BVH.read(clip).to_animation().as_rigid_body(joint_translation="drop")
    return exporter, clip.stem, apply_reduction(prepared, exporter.reduction)


@pytest.mark.parametrize("rig", SAMPLE_RIGS)
def test_default_facing_is_the_rigs_own_rest_facing(rig):
    """A generated clip has no recorded yaw. The default must still be a FRAME.

    `PromoteRoot.invert` and `RestRelative.invert` stamp constants captured in
    the source frame -- the reinserted locator chain's offsets, the dead
    subtree's, every joint's bind rotation. Substituting identity for the
    missing facing leaves the body in canonical +Z while those constants stay
    where they were, which is not a rig that is merely turned: on the 14
    ground-locator rigs it detached the root by up to 1.7 rig extents.

    The rig's own rest pose is the one yaw that makes the two agree, and it is
    fitted during `fit_rig` already. So restoring with no `facing_from` has to
    give exactly what borrowing the REST clip's recorded yaw gives -- not
    approximately, since both are the same fit against the same rest pose.
    """
    manifest = _manifest(rig)
    rest_clip = rest_action(manifest)
    exporter = CharacterExporter.load(CORPUS, RAW, rig)
    if rest_clip not in exporter.clip_params:
        pytest.skip(f"{rig}: rest clip was not prepared")
    prepared = CORPUS / "clips" / rig / f"{rest_clip}.bvh"
    model = apply_reduction(
        BVH.read(prepared).to_animation().as_rigid_body(joint_translation="drop"),
        exporter.reduction,
    )
    np.testing.assert_allclose(
        exporter.params["face_axis"]["rotation"],
        exporter.clip_params[rest_clip]["face_axis"]["rotation"],
        atol=1e-12,
    )
    default = exporter._restore(model)
    borrowed = exporter._restore(model, facing_from=rest_clip)
    np.testing.assert_allclose(default.translations, borrowed.translations, atol=1e-12)
    np.testing.assert_allclose(default.offsets, borrowed.offsets, atol=1e-12)
    assert _shape_residual(borrowed, default) < 1e-9


@pytest.mark.parametrize("rig", SAMPLE_RIGS)
def test_restoring_a_real_clip_returns_the_source_body_pose(rig):
    """The product promise, on the one case where an exact answer exists.

    A clip that CAME from the corpus carries a recorded yaw, so restoring it
    with that yaw must put every joint back where the source file had it. The
    tolerance is not slack: it absorbs the one documented loss, `EnforceRigid`
    pairing the source's OFFSET header with rotations fitted against the rest
    pose's MEASURED geometry (see `poseydon.core.animation.rest_geometry`),
    and nothing else.
    """
    clips = sorted((RAW / "source" / rig).glob("*.bvh"))
    if not clips:
        pytest.skip("raw corpus not present")
    exporter, clip, model = _case(rig)
    by_action = {
        strip_skeleton_prefix(action_slug(path.stem), rig): path for path in clips
    }
    if clip not in by_action:
        pytest.skip(f"{rig}: no raw file for {clip}")
    restored = exporter._restore(model, facing_from=clip)
    source = BVH.read(by_action[clip]).to_animation().as_rigid_body(
        joint_translation="drop"
    )
    assert restored.names == source.names
    assert _shape_residual(source, restored) < 0.1


@pytest.mark.parametrize("rig", SAMPLE_RIGS)
def test_original_bvh_restores_the_source_channel_declaration(rig):
    """The source's own CHANNELS block, verbatim -- `EnforceRigid.invert`'s contract."""
    if not (CORPUS / "rigs" / rig / "preproc.npz").is_file():
        pytest.skip("no prepared rig")
    exporter, clip, model = _case(rig)
    declared = RigTransform.load(CORPUS / "rigs" / rig / "preproc.npz").rig_params[
        "enforce_rigid"
    ]["source_channels"]
    written = exporter.original_bvh(model, facing_from=clip)
    assert written.channels == tuple(tuple(spec) for spec in declared)


@pytest.mark.parametrize("rig", SAMPLE_RIGS)
def test_original_bvh_survives_a_write_read_cycle(tmp_path, rig):
    """Declared channels are worthless if the writer does not honour them."""
    if not (CORPUS / "rigs" / rig / "preproc.npz").is_file():
        pytest.skip("no prepared rig")
    exporter, clip, model = _case(rig)
    written = exporter.original_bvh(model, facing_from=clip)
    path = tmp_path / f"{rig}.bvh"
    written.write(path)
    reread = BVH.read(path)
    assert reread.channels == written.channels
    assert reread.names == written.names
    np.testing.assert_allclose(reread.offsets, written.offsets, atol=1e-5)
    assert _shape_residual(written.to_animation(), reread.to_animation()) < 1e-5


def test_restore_names_a_stage_whose_parameters_were_never_recorded():
    """A chain that has outrun its corpus must say so, not invert half of it."""
    exporter = CharacterExporter.load(CORPUS, RAW, SAMPLE_RIGS[0])
    clips = sorted((CORPUS / "clips" / SAMPLE_RIGS[0]).glob("*.bvh"))
    if not clips:
        pytest.skip("no prepared clips")
    model = apply_reduction(
        BVH.read(clips[0]).to_animation().as_rigid_body(joint_translation="drop"),
        exporter.reduction,
    )
    exporter.params = {k: v for k, v in exporter.params.items() if k != "scale"}
    with pytest.raises(KeyError, match="scale"):
        exporter._restore(model)

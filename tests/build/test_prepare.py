"""Stage contract and chain composition."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from poseydon.build.prepare import (
    CentreXZ,
    EnforceRigid,
    FaceAxis,
    PrepareChain,
    PrepareStage,
    PutOnGround,
    RestRelative,
    ScaleToMeanBoneLength,
)
from poseydon.core.animation import Animation
from poseydon.core.rotations import QUAT_IDENTITY, euler_to_quat
from poseydon.core.skeleton import HML_MEAN_BONE_LENGTH


def _anim(n_frames: int = 3, n_joints: int = 2) -> Animation:
    rotations = np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (n_frames, n_joints, 1))
    offsets = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    translations = np.broadcast_to(offsets, (n_frames, n_joints, 3)).copy()
    # A non-zero root trajectory, so `Shift` is not a no-op. With a zero root
    # the order tests below cannot tell reverse-order inversion from
    # forward-order inversion: adding zero commutes with everything.
    translations[:, 0] = np.array([5.0, 2.0, -3.0])
    return Animation(
        rotations=rotations,
        translations=translations,
        offsets=offsets,
        parents=np.array([-1, 0], dtype=np.int32),
        names=("root", "child"),
        fps=30.0,
    )


class Shift(PrepareStage):
    """Adds a fitted constant to the root trajectory. Rig-scoped."""

    name = "shift"
    scope = "rig"

    def fit(self, anim, resolved):
        return {"amount": anim.translations[0, 0].copy()}

    def apply(self, anim, params):
        translations = anim.translations.copy()
        translations[:, 0] = translations[:, 0] - params["amount"]
        return Animation(
            rotations=anim.rotations, translations=translations, offsets=anim.offsets,
            parents=anim.parents, names=anim.names, fps=anim.fps,
        )

    def invert(self, anim, params):
        translations = anim.translations.copy()
        translations[:, 0] = translations[:, 0] + params["amount"]
        return Animation(
            rotations=anim.rotations, translations=translations, offsets=anim.offsets,
            parents=anim.parents, names=anim.names, fps=anim.fps,
        )


class Double(PrepareStage):
    """Scales every offset. Clip-scoped, so it refits per animation."""

    name = "double"
    scope = "clip"

    def fit(self, anim, resolved):
        return {"factor": np.float64(2.0)}

    def apply(self, anim, params):
        return Animation(
            rotations=anim.rotations, translations=anim.translations * params["factor"],
            offsets=anim.offsets * params["factor"], parents=anim.parents,
            names=anim.names, fps=anim.fps,
        )

    def invert(self, anim, params):
        return Animation(
            rotations=anim.rotations, translations=anim.translations / params["factor"],
            offsets=anim.offsets / params["factor"], parents=anim.parents,
            names=anim.names, fps=anim.fps,
        )


def test_fit_rig_keeps_only_rig_scoped_parameters():
    chain = PrepareChain((Shift(), Double()))
    rig = chain.fit_rig(_anim(), resolved=None)
    assert set(rig) == {"shift"}


def test_apply_then_invert_is_the_identity():
    chain = PrepareChain((Shift(), Double()))
    source = _anim()
    rig = chain.fit_rig(source, resolved=None)

    prepared, params = chain.apply(source, resolved=None, rig_params=rig)
    restored = chain.invert(prepared, params)

    np.testing.assert_allclose(restored.translations, source.translations, atol=1e-12)
    np.testing.assert_allclose(restored.offsets, source.offsets, atol=1e-12)


def test_invert_runs_stages_in_reverse_order():
    """Shift-then-Double does not commute. Unwinding in the wrong order divides
    the restored root trajectory by two, so the correct order recovers the
    source and the swapped order must not."""
    chain = PrepareChain((Shift(), Double()))
    source = _anim()
    rig = chain.fit_rig(source, resolved=None)
    prepared, params = chain.apply(source, resolved=None, rig_params=rig)

    restored = chain.invert(prepared, params)
    np.testing.assert_allclose(
        restored.translations[:, 0], source.translations[:, 0], atol=1e-12
    )

    # The same stages unwound forward instead of reversed must NOT recover it.
    wrong = prepared
    for stage in chain.stages:
        wrong = stage.invert(wrong, params[stage.name])
    assert not np.allclose(
        wrong.translations[:, 0], source.translations[:, 0], atol=1e-9
    )


def test_a_clip_scoped_stage_refits_on_each_animation():
    chain = PrepareChain((Double(),))
    rig = chain.fit_rig(_anim(), resolved=None)
    assert rig == {}
    _prepared, params = chain.apply(_anim(), resolved=None, rig_params=rig)
    assert "double" in params


def test_apply_rejects_a_missing_rig_parameter():
    chain = PrepareChain((Shift(),))
    with pytest.raises(KeyError, match="shift"):
        chain.apply(_anim(), resolved=None, rig_params={})


def _bent(n_frames: int = 4) -> Animation:
    """A three-joint chain whose rest pose is NOT the identity pose."""
    rest = euler_to_quat(np.array([0.0, 0.0, 20.0]), "ZYX")
    rotations = np.tile(QUAT_IDENTITY, (n_frames, 3, 1))
    rotations[:, 1] = rest
    rotations[:, 2] = euler_to_quat(np.array([0.0, 0.0, 35.0]), "ZYX")
    offsets = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 2.0, 0.0]])
    translations = np.broadcast_to(offsets, (n_frames, 3, 3)).copy()
    translations[:, 0] = np.arange(n_frames)[:, None] * np.array([1.0, 0.0, 0.0])
    return Animation(
        rotations=rotations, translations=translations, offsets=offsets,
        parents=np.array([-1, 0, 1], dtype=np.int32),
        names=("root", "mid", "tip"), fps=30.0,
    )


def test_rest_relative_makes_the_rest_pose_read_as_identity():
    stage = RestRelative()
    rest = _bent(n_frames=1)
    params = stage.fit(rest, resolved=None)
    prepared = stage.apply(rest, params)

    dots = np.abs(np.sum(prepared.rotations * QUAT_IDENTITY, axis=-1))
    np.testing.assert_allclose(dots, 1.0, atol=1e-9)


def test_rest_relative_preserves_world_positions():
    stage = RestRelative()
    source = _bent()
    params = stage.fit(_bent(n_frames=1), resolved=None)
    prepared = stage.apply(source, params)

    np.testing.assert_allclose(
        prepared.global_positions(), source.global_positions(), atol=1e-9
    )


def test_rest_relative_inverts_exactly():
    stage = RestRelative()
    source = _bent()
    params = stage.fit(_bent(n_frames=1), resolved=None)
    restored = stage.invert(stage.apply(source, params), params)

    np.testing.assert_allclose(restored.rotations, source.rotations, atol=1e-9)
    np.testing.assert_allclose(restored.translations, source.translations, atol=1e-9)
    np.testing.assert_allclose(restored.offsets, source.offsets, atol=1e-9)


def test_rest_relative_refuses_a_rig_it_was_not_fitted_to():
    stage = RestRelative()
    params = stage.fit(_bent(n_frames=1), resolved=None)
    other = _anim()
    with pytest.raises(ValueError, match="fitted"):
        stage.apply(other, params)


@dataclass(frozen=True)
class _FakeManifest:
    extra_yaw_deg: float = 0.0


@dataclass(frozen=True)
class _FakeResolved:
    facing_indices: tuple[tuple[int, int], ...]
    manifest: _FakeManifest = _FakeManifest()


def _wide() -> Animation:
    """Root with a left and a right joint, facing +X rather than +Z."""
    offsets = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, -1.0]])
    translations = np.broadcast_to(offsets, (2, 3, 3)).copy()
    return Animation(
        rotations=np.tile(QUAT_IDENTITY, (2, 3, 1)),
        translations=translations, offsets=offsets,
        parents=np.array([-1, 0, 0], dtype=np.int32),
        names=("root", "right", "left"), fps=30.0,
    )


def test_face_axis_turns_frame_zero_onto_plus_z():
    stage = FaceAxis(axis="+Z")
    resolved = _FakeResolved(facing_indices=((1, 2),))
    source = _wide()

    params = stage.fit(source, resolved)
    prepared = stage.apply(source, params)

    positions = prepared.global_positions()
    across = positions[0, 1] - positions[0, 2]
    forward = np.cross(np.array([0.0, 1.0, 0.0]), across / np.linalg.norm(across))
    np.testing.assert_allclose(forward, [0.0, 0.0, 1.0], atol=1e-9)


def test_face_axis_inverts_exactly():
    stage = FaceAxis(axis="+Z")
    resolved = _FakeResolved(facing_indices=((1, 2),))
    source = _wide()

    params = stage.fit(source, resolved)
    restored = stage.invert(stage.apply(source, params), params)

    np.testing.assert_allclose(restored.rotations, source.rotations, atol=1e-9)
    np.testing.assert_allclose(restored.offsets, source.offsets, atol=1e-9)
    np.testing.assert_allclose(
        restored.global_positions(), source.global_positions(), atol=1e-9
    )


def _rigid(n_frames: int = 3) -> Animation:
    offsets = np.array([[0.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 6.0, 0.0]])
    translations = np.broadcast_to(offsets, (n_frames, 3, 3)).copy()
    translations[:, 0] = np.array([7.0, 3.0, -2.0])
    return Animation(
        rotations=np.tile(QUAT_IDENTITY, (n_frames, 3, 1)),
        translations=translations, offsets=offsets,
        parents=np.array([-1, 0, 1], dtype=np.int32),
        names=("root", "mid", "tip"), fps=30.0,
    )


def test_centre_xz_moves_frame_zero_to_the_origin_in_xz_only():
    stage = CentreXZ()
    source = _rigid()
    params = stage.fit(source, resolved=None)
    prepared = stage.apply(source, params)

    np.testing.assert_allclose(prepared.translations[0, 0], [0.0, 3.0, 0.0], atol=1e-12)


def test_scale_makes_the_mean_bone_length_the_target():
    stage = ScaleToMeanBoneLength()
    source = _rigid()
    prepared = stage.apply(source, stage.fit(source, resolved=None))

    lengths = np.linalg.norm(prepared.offsets[1:], axis=-1)
    assert lengths.mean() == pytest.approx(HML_MEAN_BONE_LENGTH)


def test_ground_puts_the_lowest_joint_at_zero():
    stage = PutOnGround()
    source = _rigid()
    prepared = stage.apply(source, stage.fit(source, resolved=None))

    assert prepared.global_positions()[..., 1].min() == pytest.approx(0.0)


@pytest.mark.parametrize(
    "stage", [CentreXZ(), ScaleToMeanBoneLength(), PutOnGround()],
    ids=["centre", "scale", "ground"],
)
def test_geometry_stages_invert_exactly(stage):
    source = _rigid()
    params = stage.fit(source, resolved=None)
    restored = stage.invert(stage.apply(source, params), params)

    np.testing.assert_allclose(restored.translations, source.translations, atol=1e-12)
    np.testing.assert_allclose(restored.offsets, source.offsets, atol=1e-12)


def test_enforce_rigid_records_the_source_channel_layout():
    """The values are lost; the channel DECLARATION is not, so a source rig
    that gave six channels per joint gets six channels back."""
    from poseydon.io.bvh import BVH

    source = _rigid()
    moving = source.translations.copy()
    moving[:, 1] += np.array([0.0, 0.1, 0.0])
    source = Animation(
        rotations=source.rotations, translations=moving, offsets=source.offsets,
        parents=source.parents, names=source.names, fps=source.fps,
    )

    stage = EnforceRigid()
    params = stage.fit(source, resolved=None)
    rigid = stage.apply(source, params)
    restored = stage.invert(rigid, params)

    assert rigid.is_rigid()
    written = BVH.from_animation(restored, channels=params["source_channels"])
    assert "Xposition" in written.channels[1]

    # The leaf is written as an End Site, which declares an OFFSET and no
    # CHANNELS -- so the fitted layout must record the empty tuple for it,
    # matching what BVH.read produces. This is the whole point of the stage
    # recording a layout at all: an approximation of the source's declaration
    # is not the source's declaration.
    assert params["source_channels"][-1] == ()
    assert written.channels[-1] == ()

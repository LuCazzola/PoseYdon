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
    PromoteRoot,
    PutOnGround,
    RestRelative,
    RigTransform,
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


def test_scale_ignores_zero_length_end_sites():
    """A rig differing only in how many End Sites it declares must come out the
    same size. Averaging over them makes the factor depend on joint count."""
    bare = _rigid()
    stage = ScaleToMeanBoneLength()
    scaled_bare = stage.apply(bare, stage.fit(bare, resolved=None))

    # The same skeleton with two extra zero-length End Sites hung off its leaf.
    offsets = np.concatenate([bare.offsets, np.zeros((2, 3))])
    n_frames = bare.n_frames
    padded = Animation(
        rotations=np.concatenate(
            [bare.rotations, np.tile(QUAT_IDENTITY, (n_frames, 2, 1))], axis=1
        ),
        translations=np.concatenate(
            [bare.translations, np.zeros((n_frames, 2, 3))], axis=1
        ),
        offsets=offsets,
        parents=np.array([*list(bare.parents), 2, 2], dtype=np.int32),
        names=(*bare.names, "tip_end", "tip_end2"),
        fps=bare.fps,
    )
    scaled_padded = stage.apply(padded, stage.fit(padded, resolved=None))

    real = slice(1, bare.n_joints)
    np.testing.assert_allclose(
        np.linalg.norm(scaled_padded.offsets[real], axis=-1),
        np.linalg.norm(scaled_bare.offsets[real], axis=-1),
        rtol=1e-12,
    )


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


def test_enforce_rigid_records_an_explicit_source_channels_verbatim():
    """A caller with the source file to hand should not get the inferred
    layout -- a static rest pose under-declares (root only), and the file's
    own `.channels` is authoritative. `fit` must record it unchanged."""
    source = _rigid()
    explicit = (
        ("Xposition", "Yposition", "Zposition", "Zrotation", "Xrotation", "Yrotation"),
        ("Xposition", "Yposition", "Zposition", "Zrotation", "Xrotation", "Yrotation"),
        (),
    )

    stage = EnforceRigid(source_channels=explicit)
    params = stage.fit(source, resolved=None)

    assert tuple(params["source_channels"]) == explicit


def test_rig_transform_round_trips_through_a_file(tmp_path):
    chain = PrepareChain((Shift(), Double()))
    source = _anim()
    rig = chain.fit_rig(source, resolved=None)
    _prepared, params = chain.apply(source, resolved=None, rig_params=rig)

    transform = RigTransform(rig_params=rig, clip_params={"walk": params})
    path = tmp_path / "prepare.npz"
    transform.save(path)
    loaded = RigTransform.load(path)

    np.testing.assert_allclose(
        loaded.rig_params["shift"]["amount"], rig["shift"]["amount"]
    )
    np.testing.assert_allclose(
        loaded.clip_params["walk"]["double"]["factor"], 2.0
    )


def test_rig_transform_reports_an_unknown_clip_clearly(tmp_path):
    transform = RigTransform(rig_params={}, clip_params={})
    transform.save(tmp_path / "prepare.npz")
    with pytest.raises(KeyError, match="jump"):
        RigTransform.load(tmp_path / "prepare.npz").params_for("jump")


def test_rig_transform_rejects_a_clip_name_that_would_corrupt_the_keys(tmp_path):
    """Clip names become npz key segments, so a slash would misparse on load."""
    transform = RigTransform(
        rig_params={}, clip_params={"walk/take1": {"double": {"factor": np.float64(2.0)}}}
    )
    with pytest.raises(ValueError, match="walk/take1"):
        transform.save(tmp_path / "prepare.npz")


def _locator_rig(n_frames: int = 4) -> Animation:
    """Hips -> Cog -> Pelvis -> {LeftLeg, RightLeg}: a two-step locator chain.

    Hips sits at the origin and Pelvis a real distance above it, which is the
    shape of Camel, Horse and Trex. Cog and Hips both carry a non-identity
    rotation, so a promotion that merely dropped them would move the body.
    """
    names = ("Hips", "Cog", "Pelvis", "LeftLeg", "RightLeg")
    parents = np.array([-1, 0, 1, 2, 2], dtype=np.int32)
    offsets = np.array(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 5.0, 0.0],
         [-1.0, -1.0, 0.0], [1.0, -1.0, 0.0]]
    )
    rng = np.random.default_rng(0)
    rotations = rng.normal(size=(n_frames, len(names), 4))
    rotations /= np.linalg.norm(rotations, axis=-1, keepdims=True)

    translations = np.broadcast_to(offsets, (n_frames, len(names), 3)).copy()
    translations[:, 0] = rng.normal(size=(n_frames, 3))
    return Animation(
        rotations=rotations, translations=translations, offsets=offsets,
        parents=parents, names=names, fps=30.0,
    )


def test_promote_root_moves_no_surviving_joint():
    """The promoted joint and its descendants keep their world positions.

    This is the assertion that matters. Checking the joint COUNT would restate
    the table; checking world positions measures the transform.
    """
    anim = _locator_rig()
    stage = PromoteRoot(target="Pelvis")
    params = stage.fit(anim, None)
    promoted = stage.apply(anim, params)

    before = anim.global_positions()
    after = promoted.global_positions()
    for name in ("Pelvis", "LeftLeg", "RightLeg"):
        np.testing.assert_allclose(
            after[:, promoted.names.index(name)],
            before[:, anim.names.index(name)],
            atol=1e-9,
            err_msg=f"{name} moved",
        )


def test_promote_root_makes_the_target_the_root():
    anim = _locator_rig()
    stage = PromoteRoot(target="Pelvis")
    promoted = stage.apply(anim, stage.fit(anim, None))

    assert promoted.names == ("Pelvis", "LeftLeg", "RightLeg")
    assert list(promoted.parents) == [-1, 0, 0]


def test_promote_root_restores_the_source_structure():
    """invert returns the hierarchy the user supplied -- spec §9's contract."""
    anim = _locator_rig()
    stage = PromoteRoot(target="Pelvis")
    params = stage.fit(anim, None)
    restored = stage.invert(stage.apply(anim, params), params)

    assert restored.names == anim.names
    assert list(restored.parents) == list(anim.parents)
    np.testing.assert_allclose(restored.offsets, anim.offsets, atol=1e-12)


def test_promote_root_round_trip_keeps_the_body_in_place():
    """Structure returns exactly; the body returns exactly; the locators do not.

    The chain's rotations all turn the same subtree, so only their product is
    observable and invert puts that product on the promoted joint with identity
    above it. Pelvis and its descendants therefore land exactly where they
    were. Hips and Cog do not, and asserting they would is asserting something
    the design explicitly does not promise (spec §9).
    """
    anim = _locator_rig()
    stage = PromoteRoot(target="Pelvis")
    params = stage.fit(anim, None)
    restored = stage.invert(stage.apply(anim, params), params)

    before = anim.global_positions()
    after = restored.global_positions()
    for name in ("Pelvis", "LeftLeg", "RightLeg"):
        index = anim.names.index(name)
        np.testing.assert_allclose(after[:, index], before[:, index], atol=1e-9)


def test_promote_root_with_no_target_is_the_identity():
    """59 of 73 rigs are not in the table and must be untouched."""
    anim = _locator_rig()
    stage = PromoteRoot(target=None)
    params = stage.fit(anim, None)

    for produced in (stage.apply(anim, params), stage.invert(anim, params)):
        assert produced.names == anim.names
        np.testing.assert_allclose(
            produced.global_positions(), anim.global_positions(), atol=1e-12
        )


def test_promote_root_refuses_a_target_that_is_not_on_the_root_chain():
    """LeftLeg has a sibling, so the joints above it are not a single-child
    chain and slicing them away would delete RightLeg's subtree."""
    anim = _locator_rig()
    stage = PromoteRoot(target="LeftLeg")
    with pytest.raises(ValueError, match="single-child chain"):
        stage.fit(anim, None)


def test_promote_root_refuses_an_unknown_target():
    anim = _locator_rig()
    stage = PromoteRoot(target="NoSuchJoint")
    with pytest.raises(ValueError, match="NoSuchJoint"):
        stage.fit(anim, None)

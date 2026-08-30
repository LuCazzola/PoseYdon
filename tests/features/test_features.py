import numpy as np
import pytest

from poseydon.core.rotations import QUAT_IDENTITY, quat_to_matrix, rot6d_to_matrix
from poseydon.features import FEATURES, extract_features
from poseydon.features.base import FeatureContext
from poseydon.io.bvh import load_bvh
from tests.ingest.manifest_helper import resolved_for


def context(bvh_path):
    anim = load_bvh(bvh_path)
    return FeatureContext.build(anim, resolved_for(bvh_path, anim))


def test_all_four_reference_features_are_registered():
    assert FEATURES.names() == ["foot_contact", "local_vel", "ric_pos", "rot6d"]


def test_unknown_feature_is_rejected_with_a_suggestion(bvh_fixture):
    anim = load_bvh(bvh_fixture)
    with pytest.raises(KeyError, match="rot6d"):
        extract_features(anim, resolved_for(bvh_fixture, anim), ["rot6"])


def test_empty_feature_list_is_rejected(bvh_fixture):
    anim = load_bvh(bvh_fixture)
    with pytest.raises(ValueError, match="at least one"):
        extract_features(anim, resolved_for(bvh_fixture, anim), [])


def test_feature_order_follows_the_request(bvh_fixture):
    anim = load_bvh(bvh_fixture)
    _, spec = extract_features(anim, resolved_for(bvh_fixture, anim), ["rot6d", "ric_pos"])
    assert spec.names == ("rot6d", "ric_pos")
    assert spec.slice("rot6d") == slice(0, 6)


def test_ric_pos_puts_the_root_at_the_xz_origin(bvh_fixture):
    # Root-invariant means exactly that: the root sits on the Y axis every frame.
    ctx = context(bvh_fixture)
    ric = FEATURES.get("ric_pos")()(ctx)
    np.testing.assert_allclose(ric[:, 0, 0], 0.0, atol=1e-9)
    np.testing.assert_allclose(ric[:, 0, 2], 0.0, atol=1e-9)


def test_ric_pos_preserves_root_height(bvh_fixture):
    ctx = context(bvh_fixture)
    ric = FEATURES.get("ric_pos")()(ctx)
    np.testing.assert_allclose(ric[:, 0, 1], ctx.positions[:, 0, 1], atol=1e-9)


def test_ric_pos_is_rigid_within_a_frame(bvh_fixture):
    # XZ translation plus a rotation preserves all pairwise distances.
    ctx = context(bvh_fixture)
    ric = FEATURES.get("ric_pos")()(ctx)
    a, b = ctx.positions[0], ric[0]
    da = np.linalg.norm(a[:, None] - a[None, :], axis=-1)
    db = np.linalg.norm(b[:, None] - b[None, :], axis=-1)
    np.testing.assert_allclose(db, da, atol=1e-7)


def test_rot6d_decodes_back_to_valid_rotations(bvh_fixture):
    ctx = context(bvh_fixture)
    block = FEATURES.get("rot6d")()(ctx)
    matrices = rot6d_to_matrix(block, layout="columns")
    gram = np.einsum("...ij,...kj->...ik", matrices, matrices)
    eye = np.broadcast_to(np.eye(3), gram.shape)
    np.testing.assert_allclose(gram, eye, atol=1e-9)


def test_rot6d_reslots_each_joint_to_its_parents_rotation(bvh_fixture):
    ctx = context(bvh_fixture)
    block = FEATURES.get("rot6d")()(ctx)
    parents = ctx.anim.parents
    expected = quat_to_matrix(ctx.anim.rotations[:, parents[1:]])
    got = rot6d_to_matrix(block[:, 1:], layout="columns")
    np.testing.assert_allclose(got, expected, atol=1e-9)


def test_local_vel_is_one_frame_shorter(bvh_fixture):
    ctx = context(bvh_fixture)
    assert FEATURES.get("local_vel")()(ctx).shape[0] == ctx.anim.n_frames - 1


def test_local_vel_is_zero_for_a_static_clip(bvh_fixture):
    ctx = context(bvh_fixture)
    static = ctx.anim.slice(0, 1)
    frozen = type(ctx.anim)(
        rotations=np.repeat(static.rotations, 5, axis=0),
        root_pos=np.repeat(static.root_pos, 5, axis=0),
        offsets=static.offsets,
        parents=static.parents,
        names=static.names,
        fps=static.fps,
    )
    resolved = resolved_for(bvh_fixture, frozen)
    block = FEATURES.get("local_vel")()(FeatureContext.build(frozen, resolved))
    np.testing.assert_allclose(block, 0.0, atol=1e-12)


def test_foot_contact_is_binary_and_only_on_declared_feet(bvh_fixture):
    ctx = context(bvh_fixture)
    block = FEATURES.get("foot_contact")()(ctx)[..., 0]
    assert set(np.unique(block)) <= {0.0, 1.0}

    feet = set(ctx.resolved.foot_indices)
    non_feet = [j for j in range(ctx.n_joints) if j not in feet]
    np.testing.assert_array_equal(block[:, non_feet], 0.0)


def test_foot_contact_fires_when_a_foot_is_planted(bvh_fixture):
    # A frozen clip has zero foot speed, so every declared foot below the height
    # threshold must register contact.
    ctx = context(bvh_fixture)
    if not ctx.resolved.foot_indices:
        pytest.skip("this rig declares no foot joints")

    anim = ctx.anim
    frozen = type(anim)(
        rotations=np.repeat(anim.rotations[:1], 4, axis=0),
        root_pos=np.repeat(anim.root_pos[:1], 4, axis=0),
        offsets=anim.offsets,
        parents=anim.parents,
        names=anim.names,
        fps=anim.fps,
    )
    frozen_ctx = FeatureContext.build(frozen, resolved_for(bvh_fixture, frozen))
    block = FEATURES.get("foot_contact")()(frozen_ctx)[..., 0]

    feet = list(frozen_ctx.resolved.foot_indices)
    low = np.abs(frozen_ctx.positions[1:, feet, 1]) <= ctx.resolved.manifest.contact.max_height
    np.testing.assert_array_equal(block[:, feet].astype(bool), low)


def test_identity_rotations_give_identity_6d(bvh_fixture):
    ctx = context(bvh_fixture)
    anim = ctx.anim
    rest = type(anim)(
        rotations=np.broadcast_to(QUAT_IDENTITY, anim.rotations.shape).copy(),
        root_pos=anim.root_pos,
        offsets=anim.offsets,
        parents=anim.parents,
        names=anim.names,
        fps=anim.fps,
    )
    rest_ctx = FeatureContext.build(rest, resolved_for(bvh_fixture, rest))
    block = FEATURES.get("rot6d")()(rest_ctx)
    np.testing.assert_allclose(
        rot6d_to_matrix(block[:, 1:], layout="columns"),
        np.broadcast_to(np.eye(3), (*block.shape[:2], 3, 3))[:, 1:],
        atol=1e-12,
    )

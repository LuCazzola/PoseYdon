from pathlib import Path

import numpy as np
import pytest

from poseydon.core.anim import Anim
from poseydon.core.rotations import QUAT_IDENTITY, quat_apply
from poseydon.core.skeleton import ContactParams, FacingPair, SkeletonManifest, resolve
from poseydon.ingest.align import (
    align,
    compute_alignment_params,
    facing_quat,
    put_on_ground,
    scale_to_mean_bone_length,
)
from poseydon.io.bvh import load_bvh
from tests.ingest.manifest_helper import resolved_for

TOY_NAMES = ("root", "r_hip", "l_hip", "r_sdr", "l_sdr", "foot")


def toy_anim(n_frames=3) -> Anim:
    # Root, two hips (right +X, left -X), two shoulders, one foot below origin.
    offsets = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [1.0, 2.0, 0.0],
            [-1.0, 2.0, 0.0],
            [0.0, -3.0, 0.0],
        ]
    )
    return Anim(
        rotations=np.broadcast_to(QUAT_IDENTITY, (n_frames, 6, 4)).copy(),
        root_pos=np.tile(np.array([5.0, 4.0, 7.0]), (n_frames, 1)),
        offsets=offsets,
        parents=np.array([-1, 0, 0, 0, 0, 0], dtype=np.int32),
        names=TOY_NAMES,
        fps=24.0,
    )


def toy_resolved():
    manifest = SkeletonManifest(
        name="Toy",
        facing=(FacingPair("r_hip", "l_hip"), FacingPair("r_sdr", "l_sdr")),
        contact=ContactParams(0.3, 0.045),
        source=Path("Toy.yaml"),
    )
    return resolve(manifest, list(TOY_NAMES))


def test_facing_quat_turns_x_facing_character_to_face_z():
    # across = right - left = +X, so forward = cross(+Y, +X) = -Z.
    # The rotation must take -Z onto +Z.
    q = facing_quat(toy_anim().global_positions(), toy_resolved().facing_indices)
    np.testing.assert_allclose(
        quat_apply(q, np.array([0.0, 0.0, -1.0])), [0.0, 0.0, 1.0], atol=1e-10
    )


def test_extra_yaw_is_applied_on_top():
    positions = toy_anim().global_positions()
    plain = facing_quat(positions, toy_resolved().facing_indices)
    yawed = facing_quat(positions, toy_resolved().facing_indices, extra_yaw_deg=90.0)
    assert not np.allclose(plain, yawed)


def test_scale_sets_mean_bone_length():
    scaled, factor = scale_to_mean_bone_length(toy_anim(), 0.2)
    lengths = np.linalg.norm(scaled.offsets[1:], axis=-1)
    assert lengths.mean() == pytest.approx(0.2, rel=1e-12)
    assert factor > 0


def test_put_on_ground_puts_lowest_joint_at_zero():
    grounded, height = put_on_ground(toy_anim())
    assert grounded.global_positions()[..., 1].min() == pytest.approx(0.0, abs=1e-12)
    assert height == pytest.approx(1.0, abs=1e-12)  # foot at 4 - 3 = 1


def test_align_is_idempotent_when_params_come_from_the_clip_itself():
    anim, resolved = toy_anim(), toy_resolved()
    once = align(anim, resolved, compute_alignment_params(anim, resolved))
    twice = align(once, resolved, compute_alignment_params(once, resolved))
    np.testing.assert_allclose(twice.global_positions(), once.global_positions(), atol=1e-9)


def test_shared_params_preserve_relative_height_between_clips():
    # The whole point of skeleton-level params: a clip that never touches the
    # ground must NOT be slammed onto it.
    resolved = toy_resolved()
    grounded = toy_anim()
    params = compute_alignment_params(grounded, resolved)

    airborne = Anim(
        rotations=grounded.rotations,
        root_pos=grounded.root_pos + np.array([0.0, 10.0, 0.0]),
        offsets=grounded.offsets,
        parents=grounded.parents,
        names=grounded.names,
        fps=grounded.fps,
    )

    a = align(grounded, resolved, params).global_positions()[..., 1].min()
    b = align(airborne, resolved, params).global_positions()[..., 1].min()

    assert a == pytest.approx(0.0, abs=1e-12)
    assert b == pytest.approx(10.0 * params.scale_factor, abs=1e-9)


def test_align_grounds_and_centres_real_fixture(bvh_fixture):
    anim = load_bvh(bvh_fixture)
    resolved = resolved_for(bvh_fixture, anim)
    params = compute_alignment_params(anim, resolved)
    aligned = align(anim, resolved, params)

    positions = aligned.global_positions()
    assert positions[..., 1].min() == pytest.approx(0.0, abs=1e-9)
    np.testing.assert_allclose(aligned.root_pos[0, [0, 2]], [0.0, 0.0], atol=1e-9)
    lengths = np.linalg.norm(aligned.offsets[1:], axis=-1)
    assert lengths.mean() == pytest.approx(0.20921428571428569, rel=1e-9)
    assert aligned.n_frames == anim.n_frames
    assert aligned.names == anim.names
    assert params.scale_factor > 0


def test_align_preserves_bone_length_ratios(bvh_fixture):
    anim = load_bvh(bvh_fixture)
    resolved = resolved_for(bvh_fixture, anim)
    params = compute_alignment_params(anim, resolved)
    aligned = align(anim, resolved, params)
    before = np.linalg.norm(anim.offsets[1:], axis=-1)
    after = np.linalg.norm(aligned.offsets[1:], axis=-1)
    np.testing.assert_allclose(after, before * params.scale_factor, atol=1e-9)


def test_rotation_is_rigid(bvh_fixture):
    anim = load_bvh(bvh_fixture)
    resolved = resolved_for(bvh_fixture, anim)
    params = compute_alignment_params(anim, resolved)
    aligned = align(anim, resolved, params)
    # Pairwise joint distances are preserved up to the uniform scale factor.
    a = anim.global_positions()[0]
    b = aligned.global_positions()[0]
    da = np.linalg.norm(a[:, None] - a[None, :], axis=-1)
    db = np.linalg.norm(b[:, None] - b[None, :], axis=-1)
    np.testing.assert_allclose(db, da * params.scale_factor, atol=1e-7)


def test_align_makes_character_face_positive_z(bvh_fixture):
    anim = load_bvh(bvh_fixture)
    resolved = resolved_for(bvh_fixture, anim)
    aligned = align(anim, resolved, compute_alignment_params(anim, resolved))
    # Recomputing the facing rotation on the aligned clip must be a no-op.
    residual = facing_quat(aligned.global_positions(), resolved.facing_indices)
    np.testing.assert_allclose(
        quat_apply(residual, np.array([0.0, 0.0, 1.0])), [0.0, 0.0, 1.0], atol=1e-9
    )

import pytest
import torch

from poseydon.core.batch import Cond, Masks, MotionBatch, WindowInfo
from poseydon.core.spec import FeatureSpec
from poseydon.losses import LOSSES, NORM_STATS

SPEC = FeatureSpec((("ric_pos", 3), ("rot6d", 6), ("local_vel", 3), ("foot_contact", 1)))


def make_batch(batch=2, joints=3, frames=6, with_stats=True, real_joints=None):
    joint_mask = torch.ones(batch, joints, dtype=torch.bool)
    if real_joints is not None:
        joint_mask[:, real_joints:] = False
    x = torch.zeros(batch, joints, SPEC.dim, frames, dtype=torch.float64)
    # A valid identity rotation in columns layout.
    x[:, :, 3:9, :] = torch.tensor([1.0, 0, 0, 0, 1.0, 0], dtype=torch.float64)[
        None, None, :, None
    ]
    payloads = {}
    if with_stats:
        payloads[NORM_STATS] = {
            "mean": torch.zeros(batch, joints, SPEC.dim, dtype=torch.float64),
            "std": torch.ones(batch, joints, SPEC.dim, dtype=torch.float64),
        }
    return MotionBatch(
        x=x,
        spec=SPEC,
        masks=Masks(frames=torch.ones(batch, frames, dtype=torch.bool), joints=joint_mask),
        window=WindowInfo(
            torch.zeros(batch, dtype=torch.long), torch.full((batch,), frames, dtype=torch.long)
        ),
        cond=Cond(payloads),
    )


def test_all_terms_are_registered():
    assert LOSSES.names() == ["footskate", "geodesic", "kl", "simple"]


def test_simple_is_zero_for_a_perfect_prediction():
    batch = make_batch()
    loss = LOSSES.get("simple")()(batch.x, batch.x, batch, {})
    assert loss.item() == pytest.approx(0.0)


def test_simple_ignores_padded_joints():
    # Padding must not dilute the loss. An error confined to a padded joint is
    # invisible; the same error on a real joint is not.
    batch = make_batch(joints=4, real_joints=2)
    wrong_padding = batch.x.clone()
    wrong_padding[:, 2:] += 5.0
    assert LOSSES.get("simple")()(wrong_padding, batch.x, batch, {}).item() == pytest.approx(0.0)

    wrong_real = batch.x.clone()
    wrong_real[:, :2] += 5.0
    assert LOSSES.get("simple")()(wrong_real, batch.x, batch, {}).item() > 1.0


def test_geodesic_is_zero_for_identical_rotations():
    batch = make_batch()
    loss = LOSSES.get("geodesic")()(batch.x, batch.x, batch, {})
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_geodesic_measures_a_known_angle():
    # A 90 degree rotation about Z must read as pi/2 radians.
    batch = make_batch()
    rotated = batch.x.clone()
    rotated[:, :, 3:9, :] = torch.tensor([0.0, 1.0, 0, -1.0, 0.0, 0], dtype=torch.float64)[
        None, None, :, None
    ]
    loss = LOSSES.get("geodesic")()(rotated, batch.x, batch, {})
    assert loss.item() == pytest.approx(torch.pi / 2, abs=1e-4)


def test_geodesic_declares_it_needs_rotations():
    term = LOSSES.get("geodesic")()
    term.validate(SPEC)
    with pytest.raises(ValueError, match="rot6d"):
        term.validate(FeatureSpec((("ric_pos", 3),)))


def test_raw_space_requires_normalization_statistics():
    batch = make_batch(with_stats=False)
    with pytest.raises(ValueError, match=NORM_STATS):
        LOSSES.get("geodesic")()(batch.x, batch.x, batch, {})


def test_geodesic_uses_raw_values_not_normalized_ones():
    # With a non-unit std, a normalized 6D pair is not a rotation. Scaling the
    # statistics must change nothing, because the loss undoes them first.
    batch = make_batch()
    scaled_stats = {
        "mean": torch.zeros_like(batch.cond[NORM_STATS]["mean"]),
        "std": torch.full_like(batch.cond[NORM_STATS]["std"], 4.0),
    }
    scaled = MotionBatch(
        x=batch.x / 4.0,
        spec=batch.spec,
        masks=batch.masks,
        window=batch.window,
        cond=Cond({NORM_STATS: scaled_stats}),
    )
    plain = LOSSES.get("geodesic")()(batch.x, batch.x, batch, {})
    unscaled = LOSSES.get("geodesic")()(scaled.x, scaled.x, scaled, {})
    assert unscaled.item() == pytest.approx(plain.item(), abs=1e-9)


def test_footskate_is_zero_when_nothing_is_in_contact():
    batch = make_batch()
    loss = LOSSES.get("footskate")()(batch.x + 1.0, batch.x, batch, {})
    assert loss.item() == pytest.approx(0.0)


def test_footskate_penalizes_movement_under_contact():
    batch = make_batch()
    target = batch.x.clone()
    target[:, :, 12, :] = 1.0  # everything planted
    moving = batch.x.clone()
    moving[:, :, 0:3, :] = torch.arange(6, dtype=torch.float64)[None, None, None, :]

    grounded = MotionBatch(
        x=target, spec=SPEC, masks=batch.masks, window=batch.window, cond=batch.cond
    )
    loss = LOSSES.get("footskate")()(moving, target, grounded, {})
    assert loss.item() > 0.0


def test_footskate_declares_both_blocks():
    term = LOSSES.get("footskate")()
    assert {block.name for block in term.needs} == {"ric_pos", "foot_contact"}


def test_kl_is_zero_for_a_standard_normal_posterior():
    batch = make_batch()
    aux = {"mu": torch.zeros(2, 4), "logvar": torch.zeros(2, 4)}
    assert LOSSES.get("kl")()(batch.x, batch.x, batch, aux).item() == pytest.approx(0.0)


def test_kl_grows_as_the_posterior_drifts():
    batch = make_batch()
    aux = {"mu": torch.full((2, 4), 3.0), "logvar": torch.zeros(2, 4)}
    assert LOSSES.get("kl")()(batch.x, batch.x, batch, aux).item() > 1.0


def test_kl_says_so_when_the_model_has_no_bottleneck():
    batch = make_batch()
    with pytest.raises(ValueError, match="logvar"):
        LOSSES.get("kl")()(batch.x, batch.x, batch, {"mu": torch.zeros(2, 4)})

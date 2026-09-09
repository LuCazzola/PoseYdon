"""The contact flag must be read raw, whatever the normalization policy says.

Today `foot_contact` ships as `center: false, scale: none`, so the flag is
untouched and the old threshold worked. This pins the behaviour under a policy
that DOES normalize it -- where a mostly-planted foot's z-scored 1 falls below
0.5, every planted frame is discarded, and the loss quietly reads zero.
"""

from __future__ import annotations

import torch

from poseydon.core.batch import Cond, Masks, MotionBatch, WindowInfo
from poseydon.core.spec import Block, FeatureSpec
from poseydon.losses.base import NORM_STATS
from poseydon.losses.footskate import FootSkateLoss

SPEC = FeatureSpec((("ric_pos", 3), ("foot_contact", 1)))

# A foot planted most of the time: mean 0.9, std 0.3. A raw flag of 1 z-scores
# to (1 - 0.9) / 0.3 = 0.33, which the old `> 0.5` threshold reads as "swing".
CONTACT_MEAN, CONTACT_STD = 0.9, 0.3


def _batch(contact_mean: float, contact_std: float) -> MotionBatch:
    """One joint, two frames, planted throughout, under a given contact policy.

    Positions are left at mean 0 / std 1, so the position half of the loss is
    the same in both spaces and only the contact policy varies.
    """
    mean = torch.tensor([[[0.0, 0.0, 0.0, contact_mean]]])
    std = torch.tensor([[[1.0, 1.0, 1.0, contact_std]]])

    x0 = torch.zeros(1, 1, SPEC.dim, 2)
    # The raw flag is 1 -- genuinely planted -- stored normalized, as `x` is.
    x0[:, :, SPEC.slice("foot_contact"), :] = (1.0 - contact_mean) / contact_std

    return MotionBatch(
        x=x0,
        spec=SPEC,
        masks=Masks(
            frames=torch.ones(1, 2, dtype=torch.bool),
            joints=torch.ones(1, 1, dtype=torch.bool),
        ),
        window=WindowInfo(start=torch.zeros(1, dtype=torch.long),
                          source_length=torch.full((1,), 2, dtype=torch.long)),
        cond=Cond({NORM_STATS: {"mean": mean, "std": std}}),
    )


def _moving_foot(batch: MotionBatch) -> torch.Tensor:
    """A prediction whose one joint slides a unit along x between the frames."""
    x0_hat = batch.x.clone()
    x0_hat[:, :, SPEC.slice("ric_pos"), 1] = torch.tensor([1.0, 0.0, 0.0])
    return x0_hat


def test_a_planted_foot_is_penalized_under_a_normalizing_policy():
    batch = _batch(CONTACT_MEAN, CONTACT_STD)
    loss = FootSkateLoss()(_moving_foot(batch), batch.x, batch, {})
    assert loss > 0, "the planted frame was discarded: the flag was read normalized"


def test_the_normalized_read_is_what_would_discard_it():
    """Guards against a vacuous pass: the old read really does fall below 0.5."""
    batch = _batch(CONTACT_MEAN, CONTACT_STD)
    normalized = batch.x[:, :, SPEC.slice("foot_contact"), :]
    assert normalized.max() < 0.5
    planted = (normalized[..., :-1] > 0.5).to(batch.x.dtype)
    assert planted.sum() == 0


def test_the_shipped_identity_policy_still_penalizes():
    """`center: false, scale: none` is mean 0 / std 1: the flag passes through."""
    batch = _batch(0.0, 1.0)
    loss = FootSkateLoss()(_moving_foot(batch), batch.x, batch, {})
    assert loss > 0


def test_the_term_declares_the_contact_flag_raw():
    assert Block("foot_contact", space="raw") in FootSkateLoss.needs

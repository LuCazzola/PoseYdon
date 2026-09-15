"""What the model is not told, and what stands in for it."""

from __future__ import annotations

import pytest
import torch

from poseydon.core.batch import Cond
from poseydon.models.base import COND_DROP
from poseydon.models.modules.masking import (
    ConditionMasker,
    MotionMasker,
    chain_ids,
    one_chain_per_sample,
)

CHAINS = ({1}, {2, 3}, {4}, {5}, {6, 7, 8})


def test_chain_ids_group_a_run_and_cut_at_a_branch(tiny_rig, tiny_valid):
    ids = chain_ids(tiny_rig, tiny_valid)[0].tolist()
    assert ids[2] == ids[3], "neck and head are one chain"
    assert ids[6] == ids[7] == ids[8], "the tail is one chain"
    assert ids[1] != ids[2], "the spine is cut where the legs branch"
    assert ids[4] != ids[5], "two leaves of one parent are two chains"
    # Five chains plus the root's own.
    assert len(set(ids)) == 6


def test_a_chain_never_runs_through_the_root(tiny_rig, tiny_valid):
    ids = chain_ids(tiny_rig, tiny_valid)[0].tolist()
    assert ids[0] != ids[1] and ids[0] != ids[6]


def test_one_chain_per_sample_picks_exactly_one_whole_chain(tiny_rig, tiny_valid):
    for seed in range(25):
        torch.manual_seed(seed)
        mask = one_chain_per_sample(tiny_rig, tiny_valid)
        picked = {joint for joint, hit in enumerate(mask[0].tolist()) if hit}
        assert picked in CHAINS, f"seed {seed} picked {picked}"
        assert not mask[0, 0], "the root slot carries the trajectory, never mask it"


def test_padded_joints_are_not_part_of_any_chain(tiny_rig):
    """The tail is padding here, so it must never be drawn."""
    valid = torch.ones(1, 9, dtype=torch.bool)
    valid[0, 6:] = False
    for seed in range(15):
        torch.manual_seed(seed)
        mask = one_chain_per_sample(tiny_rig, valid)
        assert not mask[0, 6:].any()


def test_an_unconfigured_masker_is_a_true_no_op(tiny_rig, tiny_valid):
    masker = MotionMasker(d_model=4)
    assert not masker.enabled
    assert masker.mask(tiny_valid, tiny_rig, n_frames=5) is None
    motion = torch.randn(5, 1, 9, 4)
    assert masker.apply(motion, None) is motion


def test_eval_mode_never_masks(tiny_rig, tiny_valid):
    masker = MotionMasker(d_model=4, frames=1.0, joints=1.0, chains=1.0).eval()
    assert masker.mask(tiny_valid, tiny_rig, n_frames=5) is None


def test_the_null_token_lands_only_at_masked_slots():
    masker = MotionMasker(d_model=4)
    motion = torch.randn(3, 1, 9, 4)
    mask = torch.zeros(3, 1, 9, dtype=torch.bool)
    mask[1, 0, 2] = True

    out = masker.apply(motion, mask)
    assert torch.equal(out[1, 0, 2], masker.null_token)
    untouched = ~mask[..., None].expand_as(motion)
    assert torch.equal(out[untouched], motion[untouched])


def test_frame_masking_hides_a_frame_across_every_joint(tiny_rig, tiny_valid):
    torch.manual_seed(0)
    masker = MotionMasker(d_model=4, frames=1.0)
    mask = masker.mask(tiny_valid, tiny_rig, n_frames=40)
    per_frame = mask[:, 0].sum(dim=1)
    assert set(per_frame.tolist()) <= {0, 9}, "a masked frame hides all of its joints"
    assert mask.any(), "rate 1.0 over 40 frames must hide something"


def test_padding_is_never_masked(tiny_rig):
    torch.manual_seed(0)
    valid = torch.ones(1, 9, dtype=torch.bool)
    valid[0, 5:] = False
    masker = MotionMasker(d_model=4, frames=1.0, joints=1.0, chains=1.0)
    mask = masker.mask(valid, tiny_rig, n_frames=20)
    assert not mask[:, :, 5:].any()


@pytest.mark.parametrize("rate", [-0.1, 1.5])
def test_a_rate_outside_zero_to_one_is_rejected(rate):
    with pytest.raises(ValueError, match="must be in"):
        MotionMasker(d_model=4, frames=rate)


def test_cond_drop_replaces_the_whole_latent():
    torch.manual_seed(0)
    masker = ConditionMasker(d_model=4, cond_drop=1.0)
    z_sem = torch.randn(3, 2, 1, 4)
    out = masker.drop(z_sem, Cond({}))
    assert torch.equal(out[0, 0, 0], masker.null_latent)
    assert torch.equal(out[2, 1, 0], masker.null_latent)


def test_cond_drop_is_off_in_eval_unless_asked(tiny_valid):
    masker = ConditionMasker(d_model=4, cond_drop=1.0).eval()
    z_sem = torch.randn(3, 2, 1, 4)
    assert torch.equal(masker.drop(z_sem, Cond({})), z_sem)
    forced = masker.drop(z_sem, Cond({COND_DROP: True}))
    assert torch.equal(forced[0, 0, 0], masker.null_latent)


def test_cond_drop_honours_a_per_sample_selection():
    masker = ConditionMasker(d_model=4).eval()
    z_sem = torch.randn(3, 2, 1, 4)
    selection = torch.tensor([True, False])
    out = masker.drop(z_sem, Cond({COND_DROP: selection}))
    assert torch.equal(out[0, 0, 0], masker.null_latent)
    assert torch.equal(out[:, 1], z_sem[:, 1])

"""Masked conditioning and guidance, through a real MoDiffAE."""

from __future__ import annotations

import torch

from poseydon.models.base import CLEAN_MOTION


def _latent(model, cond, masks):
    z_sem, _ = model.encode(cond[CLEAN_MOTION], cond, masks)
    return z_sem


def test_masking_off_leaves_the_latent_untouched(tiny_model, tiny_batch):
    _, _, cond, masks = tiny_batch
    model = tiny_model().train()
    torch.manual_seed(0)
    first = _latent(model, cond, masks)
    torch.manual_seed(0)
    second = _latent(model, cond, masks)
    assert torch.equal(first, second)


def test_masking_changes_the_latent_while_training(tiny_model, tiny_batch):
    _, _, cond, masks = tiny_batch
    model = tiny_model(masking={"frames": 1.0, "joints": 1.0, "chains": 1.0}).train()
    baseline = tiny_model()
    baseline.load_state_dict(model.state_dict(), strict=False)

    torch.manual_seed(0)
    masked = _latent(model, cond, masks)
    unmasked = _latent(baseline.eval(), cond, masks)
    assert not torch.allclose(masked, unmasked)


def test_masking_is_inert_in_eval(tiny_model, tiny_batch):
    """Sampling and retargeting must see whole conditioning."""
    _, _, cond, masks = tiny_batch
    model = tiny_model(masking={"frames": 1.0, "joints": 1.0, "chains": 1.0}).eval()
    baseline = tiny_model().eval()
    baseline.load_state_dict(model.state_dict(), strict=False)
    assert torch.equal(_latent(model, cond, masks), _latent(baseline, cond, masks))


def test_the_rest_frame_is_never_masked(tiny_model, tiny_batch):
    """Token 0 describes the rig. Masking it would corrupt what both CFG
    branches have to agree on, so the masked and unmasked latents must still
    agree there."""
    _, _, cond, masks = tiny_batch
    model = tiny_model(masking={"frames": 1.0}).train()
    baseline = tiny_model().eval()
    baseline.load_state_dict(model.state_dict(), strict=False)

    torch.manual_seed(0)
    # The rest token attends only itself (`temporal_pair_mask`), so its latent
    # row cannot be reached by any masked motion frame.
    assert torch.allclose(
        _latent(model, cond, masks)[0], _latent(baseline, cond, masks)[0], atol=1e-6
    )


def test_a_masked_slot_still_carries_its_position(tiny_model, tiny_batch):
    """The null token replaces the MOTION embedding; names and frame positions
    are added on top of it. If the substitution ever moves after that addition
    every masked slot becomes the same vector, and the encoder can no longer
    tell WHERE the hole is -- which is the whole reason to learn a token
    instead of dropping the slot."""
    _, _, cond, _ = tiny_batch
    model = tiny_model()
    masker = model.motion_masking
    clean = cond[CLEAN_MOTION]

    every_slot = torch.ones(clean.shape[3], 1, clean.shape[1], dtype=torch.bool)
    tokens = model.semantic_encoder.input_process(
        clean, cond["tpose"], None, torch.zeros(1, dtype=torch.long), masker, every_slot
    )
    # Token 0 is the rest frame; 1 onwards are the fully-masked motion frames.
    assert not torch.allclose(tokens[1, 0, 0], tokens[2, 0, 0]), "positions differ"
    assert not torch.allclose(tokens[1, 0, 0], masker.null_token), "position was added"

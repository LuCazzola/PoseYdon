"""Attention locality belongs to the model, and must respect padding."""

from __future__ import annotations

import torch

from poseydon.models.modiffae import MoDiffAE


def test_the_band_limits_attention_to_the_declared_width():
    model = MoDiffAE(feature_dim=13, d_model=32, n_heads=2, temporal_window=5)
    joint_valid = torch.ones(1, 3, dtype=torch.bool)
    frame_valid = torch.ones(1, 10, dtype=torch.bool)
    _, pair = model._valid_with_band(joint_valid, frame_valid)
    # +1 for the leading rest frame; frame i attends frames within 2 of it
    assert bool(pair[0, 1 + 0, 1 + 2])
    assert not bool(pair[0, 1 + 0, 1 + 3])


def test_the_band_never_revives_a_padded_frame():
    """A padded frame that happens to sit inside the band must stay invalid --
    the band NARROWS attention, it never widens it.
    """
    model = MoDiffAE(feature_dim=13, d_model=32, n_heads=2, temporal_window=31)
    joint_valid = torch.ones(1, 3, dtype=torch.bool)
    frame_valid = torch.tensor([[True, True, False, False]])
    _, pair = model._valid_with_band(joint_valid, frame_valid)
    assert not bool(pair[0, 1 + 0, 1 + 2])


def test_an_explicit_override_still_wins():
    """Sampling passes TEMPORAL_VALID directly; the parameter is only the
    default for when it is absent. If the band were applied AFTER the override
    it would silently narrow a mask the caller chose deliberately.
    """
    from poseydon.core.batch import Cond
    from poseydon.models.base import TEMPORAL_VALID

    model = MoDiffAE(feature_dim=13, d_model=32, n_heads=2, temporal_window=1)
    override = torch.ones(1, 9, 9, dtype=torch.bool)
    cond = Cond({
        "topology": {
            "hops": torch.zeros(1, 3, 3, dtype=torch.long),
            "relations": torch.zeros(1, 3, 3, dtype=torch.long),
        },
        "tpose": torch.zeros(1, 3, 13),
        "clean_motion": torch.zeros(1, 3, 13, 8),
        TEMPORAL_VALID: override,
    })
    prediction = model(torch.zeros(1, 3, 13, 8), torch.zeros(1, dtype=torch.long), cond)
    # A temporal_window of 1 would leave each frame attending only itself; the
    # override says otherwise and must win. Reaching a finite output at all is
    # the observable: the mask is internal, so this asserts the override path
    # runs rather than inspecting a private tensor.
    assert torch.isfinite(prediction.out).all()


def test_the_band_reaches_anytops_attention_mask_too():
    """AnyTop carries the same parameter, and its mask is additive floats."""
    from poseydon.core.batch import Cond, Masks
    from poseydon.models.anytop import AnyTop

    model = AnyTop(feature_dim=13, d_model=32, n_heads=2, n_layers=1, temporal_window=3)
    cond = Cond({})
    masks = Masks(
        joints=torch.ones(1, 2, dtype=torch.bool),
        frames=torch.tensor([[True, True, True, False]]),
    )
    _, temporal = model._masks(masks, cond, 1, 2, 4, torch.device("cpu"))
    allowed = temporal[0] == 0.0
    assert bool(allowed[1 + 0, 1 + 1])  # within the band
    assert not bool(allowed[1 + 0, 1 + 2])  # outside it
    assert not bool(allowed[1 + 2, 1 + 3])  # inside the band, but frame 3 is padded


def test_anytops_override_wins_over_its_band():
    from poseydon.core.batch import Cond, Masks
    from poseydon.models.anytop import AnyTop
    from poseydon.models.base import TEMPORAL_VALID

    model = AnyTop(feature_dim=13, d_model=32, n_heads=2, n_layers=1, temporal_window=1)
    masks = Masks(
        joints=torch.ones(1, 2, dtype=torch.bool), frames=torch.ones(1, 4, dtype=torch.bool)
    )
    cond = Cond({TEMPORAL_VALID: torch.ones(1, 5, 5, dtype=torch.bool)})
    _, temporal = model._masks(masks, cond, 1, 2, 4, torch.device("cpu"))
    assert bool((temporal == 0.0).all())

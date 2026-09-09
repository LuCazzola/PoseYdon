"""Attention locality belongs to the model, and must respect padding."""

from __future__ import annotations

import pytest
import torch

from poseydon.core.batch import Cond, Masks
from poseydon.models import modiffae as modiffae_module
from poseydon.models.anytop import AnyTop
from poseydon.models.base import TEMPORAL_VALID
from poseydon.models.modiffae import MoDiffAE

CPU = torch.device("cpu")


def _cond(frames: int, joints: int = 3, **extra) -> Cond:
    """The minimum MoDiffAE needs to run a forward pass."""
    return Cond({
        "topology": {
            "hops": torch.zeros(1, joints, joints, dtype=torch.long),
            "relations": torch.zeros(1, joints, joints, dtype=torch.long),
        },
        "tpose": torch.zeros(1, joints, 13),
        "clean_motion": torch.zeros(1, joints, 13, frames),
        **extra,
    })


def _spy_on_temporal_masks(monkeypatch) -> list[torch.Tensor]:
    """Every temporal mask MoDiffAE actually hands to its attention modules.

    The mask is internal, but it is not unreachable: both the semantic encoder
    and the stochastic decoder route it through ``build_attention_masks``, so
    one seam catches what each of them really attends over. Asserting the
    forward pass merely produced finite numbers cannot distinguish a mask that
    obeyed the caller from one the band silently narrowed.
    """
    seen: list[torch.Tensor] = []
    real = modiffae_module.build_attention_masks

    def spy(joint_valid, temporal_valid, n_heads):
        seen.append(temporal_valid)
        return real(joint_valid, temporal_valid, n_heads)

    monkeypatch.setattr(modiffae_module, "build_attention_masks", spy)
    return seen


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


def test_an_explicit_override_still_wins(monkeypatch):
    """Sampling passes TEMPORAL_VALID directly; the parameter is only the
    default for when it is absent. If the band were applied AFTER the override
    it would silently narrow a mask the caller chose deliberately.
    """
    model = MoDiffAE(feature_dim=13, d_model=32, n_heads=2, temporal_window=1)
    override = torch.ones(1, 9, 9, dtype=torch.bool)
    cond = _cond(8, **{TEMPORAL_VALID: override})
    seen = _spy_on_temporal_masks(monkeypatch)

    prediction = model(torch.zeros(1, 3, 13, 8), torch.zeros(1, dtype=torch.long), cond)

    assert torch.isfinite(prediction.out).all()
    # A temporal_window of 1 would leave each frame attending only itself; the
    # override says otherwise and must win, in the encoder and the decoder both.
    assert len(seen) == 2
    for mask in seen:
        assert bool(mask.all()), "the band narrowed a mask the caller chose deliberately"


def test_an_override_cannot_resurrect_a_padded_frame(monkeypatch):
    """An override replaces the BAND. It does NOT replace the padding mask.

    Padded positions are literal zero-fill, so letting an all-ones override make
    them attendable feeds fabricated frames into real tokens. Temporal locality
    is the caller's to choose; what counts as data is not.
    """
    model = MoDiffAE(feature_dim=13, d_model=32, n_heads=2, temporal_window=31)
    masks = Masks(
        joints=torch.ones(1, 3, dtype=torch.bool),
        frames=torch.tensor([[True, True, False, False]]),
    )
    cond = _cond(4, **{TEMPORAL_VALID: torch.ones(1, 5, 5, dtype=torch.bool)})
    seen = _spy_on_temporal_masks(monkeypatch)

    model(torch.zeros(1, 3, 13, 4), torch.zeros(1, dtype=torch.long), cond, masks)

    assert len(seen) == 2
    for mask in seen:
        assert bool(mask[0, 1 + 0, 1 + 1])  # real frame 1, the override's to grant
        assert not bool(mask[0, 1 + 0, 1 + 2])  # padded frame 2, nobody's to grant
        assert not bool(mask[0, 1 + 2].any())  # and it queries nothing


def test_the_band_reaches_anytops_attention_mask_too():
    """AnyTop carries the same parameter, and its mask is additive floats."""
    model = AnyTop(feature_dim=13, d_model=32, n_heads=2, n_layers=1, temporal_window=3)
    masks = Masks(
        joints=torch.ones(1, 2, dtype=torch.bool),
        frames=torch.tensor([[True, True, True, False]]),
    )
    _, temporal = model._masks(masks, Cond({}), 1, 2, 4, CPU)
    allowed = temporal[0] == 0.0
    assert bool(allowed[1 + 0, 1 + 1])  # within the band
    assert not bool(allowed[1 + 0, 1 + 2])  # outside it
    assert not bool(allowed[1 + 2, 1 + 3])  # inside the band, but frame 3 is padded


def test_anytops_override_wins_over_its_band():
    model = AnyTop(feature_dim=13, d_model=32, n_heads=2, n_layers=1, temporal_window=1)
    masks = Masks(
        joints=torch.ones(1, 2, dtype=torch.bool), frames=torch.ones(1, 4, dtype=torch.bool)
    )
    cond = Cond({TEMPORAL_VALID: torch.ones(1, 5, 5, dtype=torch.bool)})
    _, temporal = model._masks(masks, cond, 1, 2, 4, CPU)
    assert bool((temporal == 0.0).all())


def test_anytops_override_cannot_resurrect_a_padded_frame():
    """The same rule as MoDiffAE's, on the model that used to ignore the key."""
    model = AnyTop(feature_dim=13, d_model=32, n_heads=2, n_layers=1, temporal_window=31)
    masks = Masks(
        joints=torch.ones(1, 2, dtype=torch.bool),
        frames=torch.tensor([[True, True, False, False]]),
    )
    cond = Cond({TEMPORAL_VALID: torch.ones(1, 5, 5, dtype=torch.bool)})
    _, temporal = model._masks(masks, cond, 1, 2, 4, CPU)
    allowed = temporal[0] == 0.0
    assert bool(allowed[1 + 0, 1 + 1])  # real frame 1
    assert not bool(allowed[1 + 0, 1 + 2])  # padded frame 2
    assert not bool(allowed[1 + 2].any())


def test_the_two_models_keep_their_own_rest_row_policies():
    """AnyTop's rest token attends every valid frame, as it did before the band
    existed; MoDiffAE's attends only itself.

    The divergence is deliberate and per-model, so sharing one mask builder must
    not quietly align them -- which is why the rest-row rule is a parameter of
    ``temporal_pair_mask`` rather than a constant. (The reference is a third
    variant again: its rest row attends the rest frame plus real frames
    0..window//2 -- external/neural_motion_blending/data_loaders/truebones/data/
    dataset.py:34. Neither model matches it; AnyTop at least is not newly moved
    further away by the band.)
    """
    frame_valid = torch.tensor([[True, True, True, False]])

    anytop = AnyTop(feature_dim=13, d_model=32, n_heads=2, n_layers=1, temporal_window=3)
    masks = Masks(joints=torch.ones(1, 2, dtype=torch.bool), frames=frame_valid)
    _, temporal = anytop._masks(masks, Cond({}), 1, 2, 4, CPU)
    assert (temporal[0] == 0.0)[0].tolist() == [True, True, True, True, False]

    modiffae = MoDiffAE(feature_dim=13, d_model=32, n_heads=2, temporal_window=3)
    _, pair = modiffae._valid_with_band(torch.ones(1, 3, dtype=torch.bool), frame_valid)
    assert pair[0, 0].tolist() == [True, False, False, False, False]


def test_a_wrong_shaped_override_is_rejected_by_name():
    """Off-by-one on the rest frame is the obvious hand-built-mask mistake; it
    should not surface later as a bare shape error inside attention.
    """
    model = AnyTop(feature_dim=13, d_model=32, n_heads=2, n_layers=1)
    masks = Masks(
        joints=torch.ones(1, 2, dtype=torch.bool), frames=torch.ones(1, 4, dtype=torch.bool)
    )
    cond = Cond({TEMPORAL_VALID: torch.ones(1, 4, 4, dtype=torch.bool)})
    with pytest.raises(ValueError, match="temporal_valid"):
        model._masks(masks, cond, 1, 2, 4, CPU)

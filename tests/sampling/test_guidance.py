"""Classifier-free guidance, as a wrapper around a denoiser."""

from __future__ import annotations

import pytest
import torch

from poseydon.core.batch import Cond
from poseydon.models.base import COND_DROP, Denoiser, Prediction
from poseydon.ops.generate import Generate
from poseydon.sampling.guidance import GuidedDenoiser


class Branchy(Denoiser):
    """Returns 1.0 conditioned and 3.0 unconditioned, and counts its calls."""

    requires = ("topology",)
    optional = ("joint_names",)

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def forward(self, z_t, t, cond, masks=None):
        self.calls += 1
        value = 3.0 if cond.get(COND_DROP) else 1.0
        return Prediction(out=torch.full_like(z_t, value), aux={"branch": value})


class Encoding(Branchy):
    def encode(self, clean, cond, masks=None, temporal_valid=None):
        return torch.zeros(1), {}


def test_the_wrapper_extrapolates_away_from_the_unconditional_branch():
    guided = GuidedDenoiser(Branchy(), scale=2.0)
    out = guided(torch.zeros(1, 2, 3, 4), torch.zeros(1), Cond({})).out
    # u + s * (c - u) = 3 + 2 * (1 - 3) = -1
    assert torch.allclose(out, torch.full_like(out, -1.0))


def test_scale_zero_costs_one_forward_and_returns_the_conditional():
    inner = Branchy()
    guided = GuidedDenoiser(inner, scale=0.0)
    out = guided(torch.zeros(1, 2, 3, 4), torch.zeros(1), Cond({})).out
    assert torch.allclose(out, torch.ones_like(out))
    assert inner.calls == 1, "the shipped default must not pay for a second pass"


def test_the_aux_comes_from_the_conditional_branch():
    guided = GuidedDenoiser(Branchy(), scale=2.0)
    aux = guided(torch.zeros(1, 2, 3, 4), torch.zeros(1), Cond({})).aux
    assert aux["branch"] == 1.0


def test_a_negative_scale_is_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        GuidedDenoiser(Branchy(), scale=-1.0)


def test_the_wrapper_asks_for_the_inner_model_s_conditioners():
    guided = GuidedDenoiser(Branchy(), scale=1.0)
    assert guided.requires == ("topology",)
    with pytest.raises(ValueError, match="topology"):
        guided.check_conditioners(set())


def test_encode_delegates_when_the_inner_model_has_one():
    guided = GuidedDenoiser(Encoding(), scale=1.0)
    latent, _aux = guided.encode(torch.zeros(1), Cond({}))
    assert latent.shape == (1,)


def test_a_model_without_encode_still_looks_like_one_through_the_wrapper():
    """`Retarget` branches on `getattr(model, "encode", None)`; the wrapper
    must not invent an `encode` that a bare denoiser does not have."""
    guided = GuidedDenoiser(Branchy(), scale=1.0)
    assert getattr(guided, "encode", None) is None


def test_generate_wraps_only_when_guidance_is_asked_for():
    inner = Branchy()
    assert Generate(guidance_scale=0.0).wrap_model(inner) is inner
    wrapped = Generate(guidance_scale=2.5).wrap_model(inner)
    assert isinstance(wrapped, GuidedDenoiser)
    assert wrapped.scale == 2.5


def test_generate_declares_no_controls():
    assert list(Generate(guidance_scale=2.5).build_controls(Cond({}))) == []


def test_the_wrapper_starts_in_the_inner_model_s_training_mode():
    """Wrapping an `eval()` model must not report `training=True`: the
    `was_training = model.training` / restore-in-`finally` idiom in
    `training/validation.py` reads the wrapper's own flag."""
    inner = Branchy()
    inner.eval()
    assert GuidedDenoiser(inner, scale=1.0).training is False


def test_every_operation_inherits_the_guidance_hook():
    """`wrap_model` lives on `Operation`, so an op that declares a scale is
    guided without reimplementing the combination -- which is the whole reason
    guidance is not inside any model's `forward`."""
    from poseydon.ops.base import Operation

    class Bare(Operation):
        def build_controls(self, cond):
            return []

    inner = Branchy()
    assert Operation.guidance_scale == 0.0
    assert Bare().wrap_model(inner) is inner, "identity unless a scale is set"

    scaled = Bare()
    scaled.guidance_scale = 3.0
    wrapped = scaled.wrap_model(inner)
    assert isinstance(wrapped, GuidedDenoiser)
    assert wrapped.scale == 3.0


def test_retarget_is_guided_when_asked():
    """The invariant the design argues hardest for -- `COND_DROP` beating a
    pinned `Z_SEM` -- is only reachable through a guided retarget, so this is
    what stops it living in tests alone."""
    from poseydon.core.batch import Cond as _Cond
    from poseydon.ops.retarget import Retarget

    content, cond = torch.zeros(1, 2, 3, 4), _Cond({})
    assert Retarget(content, cond).wrap_model(Branchy()) is not None
    assert not isinstance(Retarget(content, cond).wrap_model(Branchy()), GuidedDenoiser)

    guided = Retarget(content, cond, guidance_scale=2.0).wrap_model(Branchy())
    assert isinstance(guided, GuidedDenoiser)
    assert guided.scale == 2.0

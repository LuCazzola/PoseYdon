"""Samplers, controls and operations."""

import numpy as np
import pytest
import torch

from poseydon.core.batch import Cond
from poseydon.models.base import Denoiser, Prediction
from poseydon.ops import OPERATIONS, Blend, Generate, Inbetween
from poseydon.process import FlowMatching, GaussianDiffusion
from poseydon.sampling import CONTROLS, DDIM, DDPM, SAMPLERS, Euler
from poseydon.sampling.controls.latent_mix import build_schedule

SHAPE = (2, 3, 13, 8)


class ConstantDenoiser(Denoiser):
    """Always predicts the same clean sample, so sampling has a known fixed point."""

    def __init__(self, value=0.0):
        super().__init__()
        self.value = value
        self.calls = 0

    def forward(self, z_t, t, cond, masks=None):
        self.calls += 1
        return Prediction(out=torch.full_like(z_t, self.value))


class LinearDenoiser(Denoiser):
    """Predicts a scaled copy of its input, so the output depends on the state."""

    def __init__(self, scale=0.5):
        super().__init__()
        self.scale = scale

    def forward(self, z_t, t, cond, masks=None):
        return Prediction(out=self.scale * z_t)


class RecordingControl(CONTROLS.get("cfg").__mro__[1]):  # subclass of Control
    def __init__(self):
        self.before = 0
        self.after = 0

    def before_step(self, z_t, t, cond):
        self.before += 1
        return z_t, cond

    def after_step(self, z_t, t, cond):
        self.after += 1
        return z_t


def test_registries_are_populated():
    assert SAMPLERS.names() == ["ddim", "ddpm", "euler"]
    assert CONTROLS.names() == ["cfg", "inbetween", "latent_mix"]
    assert OPERATIONS.names() == ["blend", "generate", "inbetween"]


def test_ddpm_converges_to_the_predicted_sample():
    # If the model always says "the answer is 0.7", sampling must end near 0.7.
    torch.manual_seed(0)
    out = DDPM().sample(
        ConstantDenoiser(0.7), GaussianDiffusion(), SHAPE, Cond({})
    )
    assert out.shape == SHAPE
    assert out.mean().item() == pytest.approx(0.7, abs=0.05)


def test_ddim_converges_and_is_deterministic():
    process, model = GaussianDiffusion(), ConstantDenoiser(-0.3)
    start = torch.randn(SHAPE, generator=torch.Generator().manual_seed(0))

    first = DDIM().sample(model, process, SHAPE, Cond({}), start=start)
    second = DDIM().sample(model, process, SHAPE, Cond({}), start=start)

    torch.testing.assert_close(first, second)
    assert first.mean().item() == pytest.approx(-0.3, abs=0.05)


def test_ddim_striding_reduces_model_calls():
    process = GaussianDiffusion(num_steps=100)
    full, strided = ConstantDenoiser(), ConstantDenoiser()

    DDIM().sample(full, process, SHAPE, Cond({}))
    DDIM(steps=10).sample(strided, process, SHAPE, Cond({}))

    assert full.calls == 100
    assert strided.calls == 10


def test_ddim_rejects_more_steps_than_the_process_has():
    with pytest.raises(ValueError, match="cannot take"):
        DDIM(steps=500).sample(ConstantDenoiser(), GaussianDiffusion(), SHAPE, Cond({}))


def test_ddim_last_step_lands_on_the_models_prediction():
    # DDIM's final step has alpha_prev = 1, so it returns x0_hat exactly. A
    # constant denoiser therefore ignores its starting noise entirely -- worth
    # pinning, because it is why inversion cannot be tested with one.
    process, model = GaussianDiffusion(num_steps=20), ConstantDenoiser(0.42)
    out = DDIM().sample(model, process, SHAPE, Cond({}))
    torch.testing.assert_close(out, torch.full(SHAPE, 0.42))


def test_ddim_inversion_is_deterministic_and_finite():
    process, model = GaussianDiffusion(num_steps=50), LinearDenoiser(0.5)
    z0 = torch.randn(SHAPE, generator=torch.Generator().manual_seed(3))

    first = DDIM(steps=50).invert(model, process, z0, Cond({}))
    second = DDIM(steps=50).invert(model, process, z0, Cond({}))

    torch.testing.assert_close(first, second)
    assert torch.isfinite(first).all()
    assert first.shape == z0.shape


def test_ddim_inversion_carries_information_about_its_input():
    # Inversion evaluates the model one rung behind the level it enters, so the
    # round trip is approximate for an arbitrary model and exact for none. What
    # must hold is that the recovered noise is genuinely about THIS sample:
    # sampling from it lands much closer to the original than sampling from
    # unrelated noise does. Index misalignment destroys that gap.
    process, model = GaussianDiffusion(num_steps=100), LinearDenoiser(0.5)
    z0 = torch.randn(SHAPE, generator=torch.Generator().manual_seed(3))

    inverted = DDIM(steps=100).invert(model, process, z0, Cond({}))
    from_inversion = DDIM(steps=100).sample(model, process, SHAPE, Cond({}), start=inverted)
    from_noise = DDIM(steps=100).sample(
        model, process, SHAPE, Cond({}), start=torch.randn_like(z0)
    )

    informed = (from_inversion - z0).abs().mean()
    blind = (from_noise - z0).abs().mean()
    assert informed < 0.5 * blind, f"inversion {informed:.3f} vs noise {blind:.3f}"


def test_samplers_reject_a_mismatched_process():
    with pytest.raises(TypeError, match="not a gaussian"):
        DDPM().sample(ConstantDenoiser(), FlowMatching(), SHAPE, Cond({}))
    with pytest.raises(TypeError, match="not a flow"):
        Euler().sample(ConstantDenoiser(), GaussianDiffusion(), SHAPE, Cond({}))


def test_euler_integrates_flow_to_the_predicted_sample():
    torch.manual_seed(0)
    # A constant velocity field of zero leaves the sample where it started.
    out = Euler(steps=20).sample(ConstantDenoiser(0.0), FlowMatching(), SHAPE, Cond({}))
    assert out.shape == SHAPE
    assert torch.isfinite(out).all()


def test_controls_are_invoked_on_every_step():
    control = RecordingControl()
    process = GaussianDiffusion(num_steps=7)
    DDPM().sample(ConstantDenoiser(), process, SHAPE, Cond({}), controls=[control])
    assert control.before == 7
    assert control.after == 7


def test_inbetween_holds_the_known_frames():
    torch.manual_seed(0)
    process = GaussianDiffusion(num_steps=25)
    reference = torch.full(SHAPE, 5.0)

    operation = Inbetween.from_endpoints(reference, process, head=2, tail=2)
    out = operation.run(ConstantDenoiser(0.0), process, DDPM(), SHAPE, Cond({}))

    # Known frames track the reference; the gap does not.
    assert out[..., :2].mean().item() == pytest.approx(5.0, abs=0.5)
    assert out[..., -2:].mean().item() == pytest.approx(5.0, abs=0.5)
    assert abs(out[..., 3:5].mean().item()) < 2.0


def test_inbetween_rejects_a_gap_that_does_not_exist():
    with pytest.raises(ValueError, match="nothing to generate"):
        Inbetween.from_endpoints(torch.zeros(SHAPE), GaussianDiffusion(), head=5, tail=5)


def test_generate_adds_guidance_only_when_asked():
    assert Generate().build_controls(Cond({})) == []
    assert len(Generate(guidance_scale=2.5).build_controls(Cond({}))) == 1


def test_guidance_publishes_its_scale_to_the_model():
    control = Generate(guidance_scale=3.0).build_controls(Cond({}))[0]
    _, cond = control.before_step(torch.zeros(1), torch.zeros(1), Cond({}))
    assert cond["uncond"] == 3.0


@pytest.mark.parametrize("name", ["static", "linear", "ease"])
def test_blend_schedules_stay_in_range(name):
    alpha = build_schedule(name, 32)
    assert alpha.shape == (32,)
    assert alpha.min() >= 0.0 and alpha.max() <= 1.0


def test_linear_schedule_spans_the_full_range():
    alpha = build_schedule("linear", 10)
    assert alpha[0] == pytest.approx(0.0)
    assert alpha[-1] == pytest.approx(1.0)


def test_ease_schedule_is_monotone():
    alpha = build_schedule("ease", 50)
    assert np.all(np.diff(alpha) >= -1e-12)


def test_unknown_schedule_is_rejected():
    with pytest.raises(ValueError, match="unknown blend schedule"):
        build_schedule("bouncy", 10)


def test_blend_mixes_endpoints_into_conditioning():
    reference = torch.zeros(8, 4)
    target = torch.ones(8, 4)
    control = Blend(reference, target, schedule="linear").build_controls(Cond({}))[0]

    _, cond = control.before_step(torch.zeros(1), torch.zeros(1), Cond({}))
    mixed = cond["z_sem"]
    torch.testing.assert_close(mixed[0], reference[0])
    torch.testing.assert_close(mixed[-1], target[-1])


def test_blend_rejects_mismatched_endpoints():
    with pytest.raises(ValueError, match="same shape"):
        Blend(torch.zeros(8, 4), torch.zeros(8, 5))

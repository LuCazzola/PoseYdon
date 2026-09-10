"""The diffusion math itself.

A mutation pass over `process/gaussian.py` and `sampling/diffusion.py` broke
eleven real behaviours -- swapped signal and noise in `corrupt`, reversed the
DDPM loop, dropped DDIM's eps term, and eight more -- and the full suite passed
every time. The math was correct (both samplers reproduce the reference to
within 3e-06), but nothing in CI would have noticed it breaking.

These assert closed-form identities rather than recorded numbers, so they say
what the math MEANS instead of what it happened to output.
"""

from __future__ import annotations

import pytest
import torch

from poseydon.core.batch import Cond
from poseydon.models.base import Prediction
from poseydon.process.gaussian import GaussianDiffusion, cosine_betas, linear_betas
from poseydon.sampling.base import Control
from poseydon.sampling.diffusion import DDIM, DDPM

SHAPE = (2, 3, 13, 5)


def _process(steps: int = 20, parameterization: str = "x0") -> GaussianDiffusion:
    return GaussianDiffusion(
        num_steps=steps, schedule="cosine", parameterization=parameterization
    )


class _Constant:
    """A denoiser that always predicts the same clean signal."""

    def __init__(self, z0: torch.Tensor) -> None:
        self.z0 = z0
        self.seen: list[int] = []

    def __call__(self, z_t, t, cond, masks=None) -> Prediction:
        self.seen.append(int(t[0]))
        return Prediction(out=self.z0.clone(), aux={})


class _Contraction:
    """A denoiser whose estimate DEPENDS on the state it is given.

    A constant model hides anything that happens mid-trajectory: DDIM's last
    step lands on `alpha_prev = 1`, so the output is the prediction exactly,
    no matter what the earlier steps did. Randomness, controls and schedule
    order are all invisible through it -- so anything about the path, rather
    than the destination, needs a model that actually reads `z_t`.
    """

    def __init__(self, gain: float = 0.9) -> None:
        self.gain = gain

    def __call__(self, z_t, t, cond, masks=None) -> Prediction:
        return Prediction(out=self.gain * z_t, aux={})


# --- the forward process -------------------------------------------------


def test_corrupt_interpolates_signal_towards_noise():
    """At t=0 the sample is essentially clean; at the end it is essentially noise.

    Swapping the two coefficients -- the mutation that passed -- inverts this.
    """
    process = _process()
    z0 = torch.ones(SHAPE)
    noise = -torch.ones(SHAPE)

    early = process.corrupt(z0, torch.zeros(SHAPE[0], dtype=torch.long), noise)
    late = process.corrupt(
        z0, torch.full((SHAPE[0],), process.num_steps - 1, dtype=torch.long), noise
    )
    assert float(early.mean()) > 0.9, "t=0 should be almost entirely signal"
    assert float(late.mean()) < float(early.mean()), "later steps must carry more noise"


def test_corrupt_preserves_variance():
    """Variance-preserving: signal^2 + noise^2 = 1 at every step."""
    process = _process()
    total = process.sqrt_alphas_cumprod**2 + process.sqrt_one_minus_alphas_cumprod**2
    torch.testing.assert_close(total, torch.ones_like(total), atol=1e-6, rtol=0)


def test_to_eps_inverts_corrupt():
    """Recovering the noise from a corrupted sample is exact, by construction."""
    process = _process()
    torch.manual_seed(0)
    z0, noise = torch.randn(SHAPE), torch.randn(SHAPE)
    t = torch.full((SHAPE[0],), 7, dtype=torch.long)

    z_t = process.corrupt(z0, t, noise)
    torch.testing.assert_close(process.to_eps(z0, z_t, t), noise, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("parameterization", ["x0", "eps", "v"])
def test_to_z0_inverts_the_training_target(parameterization):
    """Whatever the model is trained to predict must map back to the clean signal."""
    process = _process(parameterization=parameterization)
    torch.manual_seed(1)
    z0, noise = torch.randn(SHAPE), torch.randn(SHAPE)
    t = torch.full((SHAPE[0],), 5, dtype=torch.long)

    z_t = process.corrupt(z0, t, noise)
    target = process.target(z0, noise, t)
    torch.testing.assert_close(process.to_z0(target, z_t, t), z0, atol=1e-3, rtol=1e-3)


def test_sample_t_can_draw_the_first_and_last_step():
    """Never drawing t=0 leaves the least-noisy step untrained, silently."""
    process = _process(steps=8)
    drawn = {int(v) for _ in range(400) for v in process.sample_t(16)}
    assert drawn == set(range(8))


# --- the posterior -------------------------------------------------------


def test_the_posterior_is_exact_when_z_t_came_from_z0():
    """q(z_{t-1} | z_t, z_0) must reproduce the true previous state in the mean.

    Checked at t=0, where the posterior collapses onto z0 itself: coefficients
    swapped or a wrong variance both break this.
    """
    process = _process()
    torch.manual_seed(2)
    z0, noise = torch.randn(SHAPE), torch.randn(SHAPE)
    t = torch.zeros(SHAPE[0], dtype=torch.long)

    mean, variance = process.posterior(z0, process.corrupt(z0, t, noise), t)
    torch.testing.assert_close(mean, z0, atol=1e-4, rtol=1e-4)
    assert float(variance.max()) == 0.0, "the posterior at t=0 carries no variance"


def test_the_posterior_variance_is_the_small_one():
    """`sigma_small`/FIXED_SMALL: beta-tilde, not beta.

    The reference trains with `sigma_small: True`, so this is a parity property,
    not a preference. beta-tilde is strictly smaller than beta after t=0.
    """
    process = _process()
    torch.testing.assert_close(
        process.posterior_variance,
        process.betas * (1.0 - process.alphas_cumprod_prev) / (1.0 - process.alphas_cumprod),
    )
    assert (process.posterior_variance[1:] < process.betas[1:]).all()


# --- schedules -----------------------------------------------------------


@pytest.mark.parametrize("schedule", [linear_betas, cosine_betas])
def test_betas_stay_in_range_and_alpha_bar_decreases(schedule):
    betas = schedule(100)
    assert (betas > 0).all() and (betas < 1).all()
    alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)
    assert (alphas_cumprod.diff() < 0).all(), "alpha-bar must decrease monotonically"
    assert float(alphas_cumprod[0]) > 0.9, "the first step must barely corrupt"


def test_the_cosine_schedule_is_the_published_one():
    """Nichol & Dhariwal's alpha-bar, which is what the reference uses."""
    import math

    betas = cosine_betas(50)
    alpha_bar = lambda s: math.cos((s + 0.008) / 1.008 * math.pi / 2) ** 2
    expected = [min(1 - alpha_bar((i + 1) / 50) / alpha_bar(i / 50), 0.999) for i in range(50)]
    torch.testing.assert_close(betas, torch.tensor(expected, dtype=torch.float64))


# --- samplers ------------------------------------------------------------


def test_ddpm_walks_the_schedule_backwards():
    """Forwards, the model is asked to denoise a sample that is getting noisier."""
    process = _process(steps=6)
    model = _Constant(torch.zeros(SHAPE))
    DDPM().sample(model, process, SHAPE, Cond({}), generator=torch.Generator().manual_seed(0))
    assert model.seen == [5, 4, 3, 2, 1, 0]


def test_ddpm_is_deterministic_on_the_final_step():
    """The last step takes the posterior MEAN. Adding noise there makes the
    output random even with a fixed generator and a deterministic model."""
    process = _process(steps=4)
    z0 = torch.full(SHAPE, 0.5)
    runs = [
        DDPM().sample(_Constant(z0), process, SHAPE, Cond({}),
                      generator=torch.Generator().manual_seed(0))
        for _ in range(2)
    ]
    torch.testing.assert_close(runs[0], runs[1])
    torch.testing.assert_close(runs[0], z0, atol=1e-4, rtol=1e-4)


def test_ddim_at_eta_zero_is_deterministic_and_ignores_the_generator():
    process = _process(steps=6)
    z0 = torch.full(SHAPE, -0.25)
    a = DDIM().sample(_Constant(z0), process, SHAPE, Cond({}),
                      generator=torch.Generator().manual_seed(0))
    b = DDIM().sample(_Constant(z0), process, SHAPE, Cond({}),
                      generator=torch.Generator().manual_seed(999))
    torch.testing.assert_close(a, b)


def test_ddim_lands_on_the_predicted_signal():
    """The terminal alpha-bar is 1, so the last step is exactly z0.

    Setting it to 0 instead -- a mutation that passed -- returns pure eps.
    """
    process = _process(steps=6)
    z0 = torch.full(SHAPE, 0.3)
    out = DDIM().sample(_Constant(z0), process, SHAPE, Cond({}),
                        start=torch.zeros(SHAPE))
    torch.testing.assert_close(out, z0, atol=1e-4, rtol=1e-4)


def test_ddim_eta_is_actually_used():
    """It was accepted and ignored for the life of the class.

    Needs a state-dependent model: through a constant one the final step lands
    on the prediction exactly, so injected noise leaves no trace in the output
    and a stored-but-unread eta is indistinguishable from a used one.
    """
    process = _process(steps=6)
    start = torch.full(SHAPE, 0.4)

    def run(eta: float, seed: int) -> torch.Tensor:
        return DDIM(eta=eta).sample(
            _Contraction(), process, SHAPE, Cond({}), start=start,
            generator=torch.Generator().manual_seed(seed),
        )

    assert not torch.allclose(run(1.0, 0), run(1.0, 1)), "eta > 0 must add noise"
    torch.testing.assert_close(run(1.0, 0), run(1.0, 0))
    # And eta=0 stays deterministic across generators, as before.
    torch.testing.assert_close(run(0.0, 0), run(0.0, 7))
    # The two regimes must not coincide.
    assert not torch.allclose(run(1.0, 0), run(0.0, 0))


def test_a_negative_eta_is_refused():
    with pytest.raises(ValueError, match="non-negative"):
        DDIM(eta=-0.5)


def test_the_ddim_schedule_starts_at_zero_and_fits_the_process():
    process = _process(steps=100)
    for steps in (10, 25, 100):
        schedule = DDIM(steps=steps)._schedule(process)
        assert schedule[0] == 0, "the schedule must reach the cleanest step"
        assert max(schedule) < process.num_steps
        assert schedule == sorted(schedule)


def test_more_ddim_steps_than_the_process_has_is_refused():
    with pytest.raises(ValueError, match="cannot take"):
        DDIM(steps=200)._schedule(_process(steps=100))


def test_ddim_inversion_round_trips():
    """invert() then sample() returns the clip it started from.

    Exact here because the model is constant, so the estimate never depends on
    the state it is given -- which isolates the ladder arithmetic from model
    error, and is the part the mutations broke.
    """
    process = _process(steps=8)
    z0 = torch.full(SHAPE, 0.6)
    model = _Constant(z0)

    noise = DDIM().invert(model, process, z0, Cond({}))
    back = DDIM().sample(model, process, SHAPE, Cond({}), start=noise)
    torch.testing.assert_close(back, z0, atol=1e-3, rtol=1e-3)


class _Recorder(Control):
    """Captures the state after every step, through the real control hook.

    `sample` only returns the destination, and through a constant model the
    destination is exactly the prediction -- so anything that happens along the
    way is invisible from the outside. This is how the trajectory itself gets
    asserted on.
    """

    def __init__(self) -> None:
        self.states: dict[int, torch.Tensor] = {}

    def after_step(self, z_t, t, cond):
        self.states[int(t[0])] = z_t.clone()
        return z_t


def test_the_ancestral_noise_is_scaled_by_the_standard_deviation():
    """`variance.sqrt()`, not `variance`. Asserted as a LOWER BOUND.

    The step injects noise of standard deviation sqrt(posterior_variance), and
    that noise is independent of the posterior mean, so the observed spread
    across seeds cannot fall below it -- it is larger, because the state also
    carries noise accumulated from every later step. Any tighter claim would
    mean recomputing the update here, which is testing a copy.

    Scaling by the variance instead is a 4x error at this step (0.238 against
    0.488) and shrinks the accumulated spread at every step above it, so the
    bound catches it. A determinism check cannot: a fixed generator reproduces
    a wrongly-scaled sample just as faithfully.
    """
    process = _process(steps=10)
    z0 = torch.zeros(SHAPE)
    step = 5

    seen = []
    for seed in range(48):
        recorder = _Recorder()
        DDPM().sample(
            _Constant(z0), process, SHAPE, Cond({}), controls=[recorder],
            generator=torch.Generator().manual_seed(seed),
        )
        seen.append(recorder.states[step])

    spread = float(torch.stack(seen).std(dim=0).mean())
    injected = float(process.posterior_variance[step].sqrt())
    assert spread >= injected * 0.95, (
        f"spread across seeds is {spread:.5f}, below the {injected:.5f} the step "
        f"injects on its own -- the noise is being scaled by the variance "
        f"({float(process.posterior_variance[step]):.5f}) rather than its root"
    )


def test_every_step_of_the_trajectory_is_visited_once():
    """The loop must not skip or repeat a step -- a schedule bug the
    destination cannot reveal."""
    process = _process(steps=7)
    recorder = _Recorder()
    DDPM().sample(
        _Constant(torch.zeros(SHAPE)), process, SHAPE, Cond({}), controls=[recorder],
        generator=torch.Generator().manual_seed(0),
    )
    assert sorted(recorder.states) == list(range(7))


def test_ddim_carries_the_noise_term_along_the_trajectory():
    """Dropping the eps term is invisible at the destination.

    The terminal alpha-bar is 1, so the final step is `1 * z0 + 0 * eps` and
    the output is the prediction either way. The term only does work in the
    middle of the ladder, where it carries the noise the state has not shed
    yet.

    Made visible with a model that predicts ZERO: correct DDIM keeps the
    trajectory populated by the eps term alone, while dropping it collapses
    every intermediate state to exactly zero.
    """
    process = _process(steps=10)
    recorder = _Recorder()
    DDIM().sample(
        _Constant(torch.zeros(SHAPE)), process, SHAPE, Cond({}),
        controls=[recorder], start=torch.randn(SHAPE, generator=torch.Generator().manual_seed(0)),
    )

    middle = recorder.states[process.num_steps // 2]
    assert float(middle.abs().max()) > 1e-3, (
        "the trajectory collapsed to zero -- the eps term is not being carried"
    )
    # And it decays towards the prediction rather than staying put.
    early = float(recorder.states[process.num_steps - 1].std())
    late = float(recorder.states[0].std())
    assert late < early, "the state must approach the prediction as t falls"

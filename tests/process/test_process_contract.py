"""Contract tests every registered process must satisfy.

A new process inherits these by registering; that is the mechanical half of
"new components work with existing ones".
"""

import pytest
import torch

from poseydon.process import PROCESSES, FlowMatching, GaussianDiffusion

# Every concrete configuration under test. Adding a process to the registry
# without adding it here is caught by test_every_registered_process_is_covered.
CONFIGURATIONS = [
    pytest.param(GaussianDiffusion(parameterization="x0"), id="gaussian-x0"),
    pytest.param(GaussianDiffusion(parameterization="eps"), id="gaussian-eps"),
    pytest.param(GaussianDiffusion(parameterization="v"), id="gaussian-v"),
    pytest.param(GaussianDiffusion(schedule="linear"), id="gaussian-linear"),
    pytest.param(FlowMatching(), id="flow"),
]


def sample(process, batch=8, shape=(3, 13, 5), seed=0):
    generator = torch.Generator().manual_seed(seed)
    z0 = torch.randn(batch, *shape, generator=generator, dtype=torch.float64)
    noise = torch.randn(batch, *shape, generator=generator, dtype=torch.float64)
    t = process.sample_t(batch)
    return z0, noise, t


def test_every_registered_process_is_covered():
    covered = {"gaussian", "flow"}
    assert set(PROCESSES.names()) == covered


@pytest.mark.parametrize("process", CONFIGURATIONS)
def test_to_z0_inverts_target(process):
    # The central contract: whatever the network predicts, a structural loss
    # gets back a clean sample.
    z0, noise, t = sample(process)
    z_t = process.corrupt(z0, t, noise)
    recovered = process.to_z0(process.target(z0, noise, t), z_t, t)
    torch.testing.assert_close(recovered, z0, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("process", CONFIGURATIONS)
def test_sample_t_has_one_entry_per_item(process):
    t = process.sample_t(11)
    assert t.shape == (11,)


@pytest.mark.parametrize("process", CONFIGURATIONS)
def test_corrupt_preserves_shape(process):
    z0, noise, t = sample(process)
    assert process.corrupt(z0, t, noise).shape == z0.shape


@pytest.mark.parametrize("process", CONFIGURATIONS)
def test_corrupt_is_deterministic_given_noise(process):
    z0, noise, t = sample(process)
    torch.testing.assert_close(
        process.corrupt(z0, t, noise), process.corrupt(z0, t, noise), rtol=0, atol=0
    )


@pytest.mark.parametrize("process", CONFIGURATIONS)
def test_num_steps_is_positive(process):
    assert process.num_steps > 0


def test_gaussian_barely_corrupts_at_the_first_step():
    process = GaussianDiffusion()
    z0 = torch.ones(1, 4, dtype=torch.float64)
    noise = torch.zeros_like(z0)
    out = process.corrupt(z0, torch.zeros(1, dtype=torch.long), noise)
    assert out.mean() > 0.99


def test_gaussian_is_almost_pure_noise_at_the_last_step():
    process = GaussianDiffusion()
    z0 = torch.ones(1, 4, dtype=torch.float64)
    noise = torch.zeros_like(z0)
    last = torch.full((1,), process.num_steps - 1, dtype=torch.long)
    assert process.corrupt(z0, last, noise).abs().max() < 1e-3


def test_eps_parameterization_is_ill_conditioned_at_the_last_step():
    # Recorded rather than hidden: recovering x0 from epsilon divides by
    # sqrt(alpha_bar), which the cosine schedule drives to ~5e-4. The 0.999 beta
    # cap keeps it nonzero, so the inverse exists, but it amplifies error ~2000x.
    process = GaussianDiffusion(parameterization="eps")
    amplification = 1.0 / process.sqrt_alphas_cumprod[-1]
    assert 1e3 < amplification < 1e4


def test_flow_interpolates_between_data_and_noise():
    process = FlowMatching()
    z0 = torch.zeros(1, 4, dtype=torch.float64)
    noise = torch.ones(1, 4, dtype=torch.float64)

    torch.testing.assert_close(
        process.corrupt(z0, torch.zeros(1, dtype=torch.float64), noise), z0
    )
    torch.testing.assert_close(
        process.corrupt(z0, torch.ones(1, dtype=torch.float64), noise), noise
    )
    half = process.corrupt(z0, torch.full((1,), 0.5, dtype=torch.float64), noise)
    torch.testing.assert_close(half, torch.full_like(half, 0.5))


def test_flow_target_is_the_constant_velocity():
    process = FlowMatching()
    z0 = torch.zeros(2, 3, dtype=torch.float64)
    noise = torch.ones(2, 3, dtype=torch.float64)
    torch.testing.assert_close(
        process.target(z0, noise, process.sample_t(2)), noise - z0
    )


def test_gaussian_rejects_bad_configuration():
    with pytest.raises(ValueError, match="schedule"):
        GaussianDiffusion(schedule="quadratic")
    with pytest.raises(ValueError, match="parameterization"):
        GaussianDiffusion(parameterization="mu")
    with pytest.raises(ValueError, match="num_steps"):
        GaussianDiffusion(num_steps=0)


def test_schedules_are_monotone_and_bounded():
    for schedule in ("cosine", "linear"):
        process = GaussianDiffusion(schedule=schedule)
        alphas = process.alphas_cumprod
        assert torch.all(alphas[1:] <= alphas[:-1]), schedule
        assert alphas.min() > 0.0 and alphas.max() <= 1.0, schedule

"""Gaussian diffusion."""

from __future__ import annotations

import math
from typing import Literal

import torch

from poseydon.process.base import PROCESSES, Process, expand_to

Parameterization = Literal["x0", "eps", "v"]


def linear_betas(num_steps: int) -> torch.Tensor:
    """The DDPM schedule, rescaled so step count does not change the endpoints."""
    scale = 1000.0 / num_steps
    return torch.linspace(scale * 1e-4, scale * 0.02, num_steps, dtype=torch.float64)


def cosine_betas(num_steps: int, max_beta: float = 0.999) -> torch.Tensor:
    """Nichol and Dhariwal's cosine schedule, via its alpha-bar."""

    def alpha_bar(step: float) -> float:
        return math.cos((step + 0.008) / 1.008 * math.pi / 2) ** 2

    betas = [
        min(1 - alpha_bar((i + 1) / num_steps) / alpha_bar(i / num_steps), max_beta)
        for i in range(num_steps)
    ]
    return torch.tensor(betas, dtype=torch.float64)


_SCHEDULES = {"linear": linear_betas, "cosine": cosine_betas}


@PROCESSES.register("gaussian")
class GaussianDiffusion(Process):
    """Variance-preserving diffusion.

    Defaults mirror the reference: 100 steps, a cosine schedule, and predicting
    x0 rather than epsilon.
    """

    def __init__(
        self,
        num_steps: int = 100,
        schedule: str = "cosine",
        parameterization: Parameterization = "x0",
    ) -> None:
        if num_steps < 1:
            raise ValueError(f"num_steps must be positive, got {num_steps}")
        if schedule not in _SCHEDULES:
            raise ValueError(
                f"unknown schedule `{schedule}`. Available: {', '.join(sorted(_SCHEDULES))}"
            )
        if parameterization not in ("x0", "eps", "v"):
            raise ValueError(f"unknown parameterization `{parameterization}`")

        self._num_steps = num_steps
        self.schedule = schedule
        self.parameterization = parameterization

        # The schedule is built and accumulated in float64. A cumulative product
        # over a hundred steps in float32 drifts enough to move sampling by ~2e-5
        # relative -- small, but larger than the model's own error, so it would
        # dominate any parity measurement. Coefficients are cast to the working
        # dtype only when used.
        betas = _SCHEDULES[schedule](num_steps).to(torch.float64)
        alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)
        self.betas = betas
        self.alphas_cumprod = alphas_cumprod
        self.sqrt_alphas_cumprod = alphas_cumprod.sqrt()
        self.sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod).sqrt()

        # Posterior q(z_{t-1} | z_t, z_0), needed by ancestral sampling.
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])
        self.alphas_cumprod_prev = alphas_cumprod_prev
        self.posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.posterior_mean_coef_z0 = (
            betas * alphas_cumprod_prev.sqrt() / (1.0 - alphas_cumprod)
        )
        self.posterior_mean_coef_zt = (
            (1.0 - alphas_cumprod_prev) * (1.0 - betas).sqrt() / (1.0 - alphas_cumprod)
        )

    @property
    def num_steps(self) -> int:
        return self._num_steps

    def sample_t(self, batch_size: int, device: torch.device | str = "cpu") -> torch.Tensor:
        return torch.randint(0, self._num_steps, (batch_size,), device=device)

    def _coefficients(
        self, t: torch.Tensor, like: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        index = t.to(self.sqrt_alphas_cumprod.device).long()
        signal = self.sqrt_alphas_cumprod[index].to(like.device, like.dtype)
        noise = self.sqrt_one_minus_alphas_cumprod[index].to(like.device, like.dtype)
        return expand_to(signal, like), expand_to(noise, like)

    def corrupt(self, z0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        signal, noise_scale = self._coefficients(t, z0)
        return signal * z0 + noise_scale * noise

    def target(self, z0: torch.Tensor, noise: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if self.parameterization == "x0":
            return z0
        if self.parameterization == "eps":
            return noise
        signal, noise_scale = self._coefficients(t, z0)
        return signal * noise - noise_scale * z0

    def to_eps(self, z0: torch.Tensor, z_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """The noise implied by a clean estimate, used by DDIM."""
        signal, noise_scale = self._coefficients(t, z_t)
        return (z_t - signal * z0) / noise_scale

    def posterior(
        self, z0: torch.Tensor, z_t: torch.Tensor, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Mean and variance of ``q(z_{t-1} | z_t, z_0)``."""
        index = t.to(self.betas.device).long()
        coef_z0 = expand_to(self.posterior_mean_coef_z0[index].to(z_t.device, z_t.dtype), z_t)
        coef_zt = expand_to(self.posterior_mean_coef_zt[index].to(z_t.device, z_t.dtype), z_t)
        variance = expand_to(self.posterior_variance[index].to(z_t.device, z_t.dtype), z_t)
        return coef_z0 * z0 + coef_zt * z_t, variance

    def to_z0(self, pred: torch.Tensor, z_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if self.parameterization == "x0":
            return pred
        signal, noise_scale = self._coefficients(t, z_t)
        if self.parameterization == "eps":
            return (z_t - noise_scale * pred) / signal
        return signal * z_t - noise_scale * pred

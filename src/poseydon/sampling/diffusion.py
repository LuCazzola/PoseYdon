"""Samplers for gaussian diffusion."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from poseydon.core.batch import Cond, Masks
from poseydon.models.base import Denoiser
from poseydon.process.base import Process
from poseydon.process.gaussian import GaussianDiffusion
from poseydon.sampling.base import SAMPLERS, Control, Sampler, apply_after, apply_before


def _require_gaussian(process: Process) -> GaussianDiffusion:
    if not isinstance(process, GaussianDiffusion):
        raise TypeError(
            f"{type(process).__name__} is not a gaussian diffusion; use a sampler "
            "matched to it, such as Euler for flow matching"
        )
    return process


@SAMPLERS.register("ddpm")
class DDPM(Sampler):
    """Ancestral sampling: every step draws from the true posterior."""

    def sample(
        self,
        model: Denoiser,
        process: Process,
        shape: tuple[int, ...],
        cond: Cond,
        controls: Sequence[Control] = (),
        device: torch.device | str = "cpu",
        generator: torch.Generator | None = None,
        masks: Masks | None = None,
    ) -> torch.Tensor:
        gaussian = _require_gaussian(process)
        z_t = torch.randn(shape, device=device, generator=generator)

        for step in reversed(range(gaussian.num_steps)):
            t = torch.full((shape[0],), step, device=device, dtype=torch.long)
            z_t, cond = apply_before(controls, z_t, t, cond)

            z0 = gaussian.to_z0(model(z_t, t, cond, masks).out, z_t, t)
            mean, variance = gaussian.posterior(z0, z_t, t)
            if step > 0:
                noise = torch.randn(shape, device=device, generator=generator)
                z_t = mean + variance.sqrt() * noise
            else:
                z_t = mean

            z_t = apply_after(controls, z_t, t, cond)
        return z_t


@SAMPLERS.register("ddim")
class DDIM(Sampler):
    """Deterministic implicit sampling, optionally strided, and invertible.

    Inversion -- recovering the noise that produces a given sample -- is a
    first-class mode rather than a separate script, because editing operations
    need it to start from real motion instead of from noise.
    """

    def __init__(self, steps: int | None = None, eta: float = 0.0) -> None:
        self.steps = steps
        self.eta = eta

    def _schedule(self, process: GaussianDiffusion) -> list[int]:
        steps = self.steps or process.num_steps
        if steps > process.num_steps:
            raise ValueError(
                f"cannot take {steps} DDIM steps from a {process.num_steps}-step process"
            )
        stride = process.num_steps / steps
        return sorted({int(i * stride) for i in range(steps)})

    def _alpha_bar(self, process: GaussianDiffusion, step: int, device) -> torch.Tensor:
        if step < 0:
            return torch.ones((), device=device)
        return process.alphas_cumprod[step].to(device)

    def sample(
        self,
        model: Denoiser,
        process: Process,
        shape: tuple[int, ...],
        cond: Cond,
        controls: Sequence[Control] = (),
        device: torch.device | str = "cpu",
        generator: torch.Generator | None = None,
        masks: Masks | None = None,
        start: torch.Tensor | None = None,
    ) -> torch.Tensor:
        gaussian = _require_gaussian(process)
        schedule = self._schedule(gaussian)
        z_t = (
            torch.randn(shape, device=device, generator=generator)
            if start is None
            else start.to(device)
        )

        for position in reversed(range(len(schedule))):
            step = schedule[position]
            previous = schedule[position - 1] if position > 0 else -1

            t = torch.full((shape[0],), step, device=device, dtype=torch.long)
            z_t, cond = apply_before(controls, z_t, t, cond)

            z0 = gaussian.to_z0(model(z_t, t, cond, masks).out, z_t, t)
            eps = gaussian.to_eps(z0, z_t, t)
            alpha_prev = self._alpha_bar(gaussian, previous, device)
            z_t = alpha_prev.sqrt() * z0 + (1.0 - alpha_prev).sqrt() * eps

            z_t = apply_after(controls, z_t, t, cond)
        return z_t

    def invert(
        self,
        model: Denoiser,
        process: Process,
        z0: torch.Tensor,
        cond: Cond,
        device: torch.device | str = "cpu",
        masks: Masks | None = None,
    ) -> torch.Tensor:
        """Run the trajectory forwards, recovering the noise behind ``z0``.

        This walks the same ladder :meth:`sample` descends, in reverse: the state
        starts clean and is pushed to each scheduled noise level in turn. Like
        every DDIM inversion it evaluates the model at the level it is leaving
        rather than the one it is entering, so it is exact only in the limit of
        many steps -- fewer steps trade fidelity for speed, in both directions.
        """
        gaussian = _require_gaussian(process)
        schedule = self._schedule(gaussian)
        z_t = z0.to(device)

        for position, step in enumerate(schedule):
            leaving = schedule[position - 1] if position > 0 else 0
            t = torch.full((z0.shape[0],), leaving, device=device, dtype=torch.long)

            estimate = gaussian.to_z0(model(z_t, t, cond, masks).out, z_t, t)
            eps = gaussian.to_eps(estimate, z_t, t)
            alpha_next = self._alpha_bar(gaussian, step, device)
            z_t = alpha_next.sqrt() * estimate + (1.0 - alpha_next).sqrt() * eps
        return z_t

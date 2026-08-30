"""ODE solvers for flow matching."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from poseydon.core.batch import Cond, Masks
from poseydon.models.base import Denoiser
from poseydon.process.base import Process
from poseydon.process.flow import FlowMatching
from poseydon.sampling.base import SAMPLERS, Control, Sampler, apply_after, apply_before


def _require_flow(process: Process) -> FlowMatching:
    if not isinstance(process, FlowMatching):
        raise TypeError(
            f"{type(process).__name__} is not a flow process; use DDPM or DDIM for "
            "gaussian diffusion"
        )
    return process


@SAMPLERS.register("euler")
class Euler(Sampler):
    """First-order integration from noise at t=1 back to data at t=0."""

    def __init__(self, steps: int | None = None) -> None:
        self.steps = steps

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
        flow = _require_flow(process)
        steps = self.steps or flow.num_steps
        z_t = torch.randn(shape, device=device, generator=generator)
        dt = 1.0 / steps

        for step in reversed(range(steps)):
            time = (step + 1) * dt
            t = torch.full((shape[0],), time, device=device)
            z_t, cond = apply_before(controls, z_t, t, cond)
            z_t = z_t - dt * model(z_t, t, cond, masks).out
            z_t = apply_after(controls, z_t, t, cond)
        return z_t

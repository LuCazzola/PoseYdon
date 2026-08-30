"""Positional and timestep embeddings."""

from __future__ import annotations

import torch


def sinusoidal_embedding(
    positions: torch.Tensor, dim: int, max_period: float = 10000.0
) -> torch.Tensor:
    """Classic sinusoidal embedding of integer positions.

    ``positions`` of shape ``(...)`` becomes ``(..., dim)``. Used for both the
    diffusion timestep and the frame index.
    """
    if dim % 2 != 0:
        raise ValueError(f"embedding dim must be even, got {dim}")
    half = dim // 2
    positions = positions.to(torch.float32)
    index = torch.arange(half, device=positions.device, dtype=torch.float32)
    phase = positions[..., None] / (max_period ** (index / (half - 1)))
    return torch.cat([torch.cos(phase), torch.sin(phase)], dim=-1)

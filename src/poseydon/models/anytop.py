"""AnyTop: a topology-aware spatio-temporal denoiser.

Ported from the reference implementation. The architecture is faithful; what
changed is everything around it -- conditioning is declared rather than pulled
from a fourteen-key dict, masks arrive named instead of reshaped inline, and the
model no longer owns any blending logic.

The shape of the idea: alternate attention across joints (biased by skeleton
topology) with attention across time, and prepend a rest-pose frame so the model
is told which skeleton it is animating.
"""

from __future__ import annotations

import torch
from torch import nn

from poseydon.core.batch import Cond, Masks
from poseydon.core.topology import DEFAULT_MAX_PATH, EdgeType
from poseydon.models.base import (
    MODELS,
    TEMPORAL_VALID,
    Denoiser,
    Prediction,
    temporal_pair_mask,
)
from poseydon.models.modules.embeddings import sinusoidal_embedding
from poseydon.models.modules.graph_attention import MASK_FILL, GraphAttention


class AnyTopLayer(nn.Module):
    """Spatial graph attention, then temporal attention, then a feed-forward block.

    Post-norm throughout, matching the reference. The diffusion timestep is added
    to every token at the start of the layer rather than injected once at the
    input, so depth cannot wash it out.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        ff_size: int,
        dropout: float,
        max_path: int,
        value_bias: bool,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.spatial = GraphAttention(
            d_model,
            n_heads,
            dropout=dropout,
            n_hops=max_path + 1,
            n_edges=len(EdgeType),
            value_bias=value_bias,
        )
        self.temporal = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.timestep = nn.Linear(d_model, d_model)

        self.norm_spatial = nn.LayerNorm(d_model)
        self.norm_temporal = nn.LayerNorm(d_model)
        self.norm_ff = nn.LayerNorm(d_model)

        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_size, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,  # (B, T, J, C)
        timestep: torch.Tensor,  # (B, C)
        hops: torch.Tensor,  # (B, J, J)
        edges: torch.Tensor,  # (B, J, J)
        spatial_mask: torch.Tensor | None,  # (B, 1, J, J)
        temporal_mask: torch.Tensor | None,  # (B*J, T+1, T+1) or None
    ) -> torch.Tensor:
        batch, frames, joints, channels = x.shape
        x = x + self.timestep(timestep)[:, None, None, :]

        flat = x.reshape(batch * frames, joints, channels)
        hops_flat = hops.repeat_interleave(frames, dim=0)
        edges_flat = edges.repeat_interleave(frames, dim=0)
        mask_flat = (
            spatial_mask.repeat_interleave(frames, dim=0) if spatial_mask is not None else None
        )
        attended = self.spatial(flat, hops_flat, edges_flat, mask_flat)
        x = self.norm_spatial(x + self.dropout(attended.view(batch, frames, joints, channels)))

        # Temporal attention runs per joint: every joint attends over its own
        # trajectory, so joints stay distinguishable through time.
        per_joint = x.permute(0, 2, 1, 3).reshape(batch * joints, frames, channels)
        # A 3-D attn_mask is indexed by (batch * heads), so the per-sequence mask
        # has to be repeated once per head.
        expanded = (
            temporal_mask.repeat_interleave(self.n_heads, dim=0)
            if temporal_mask is not None
            else None
        )
        temporal, _ = self.temporal(
            per_joint, per_joint, per_joint, attn_mask=expanded, need_weights=False
        )
        temporal = temporal.view(batch, joints, frames, channels).permute(0, 2, 1, 3)
        x = self.norm_temporal(x + self.dropout(temporal))

        return self.norm_ff(x + self.dropout(self.ff(x)))


@MODELS.register("anytop")
class AnyTop(Denoiser):
    """Denoiser for skeletons of arbitrary topology."""

    requires = ("topology", "tpose")
    optional = ("joint_names",)

    def __init__(
        self,
        feature_dim: int,
        d_model: int = 256,
        n_layers: int = 8,
        n_heads: int = 4,
        ff_size: int = 1024,
        dropout: float = 0.1,
        max_path: int = DEFAULT_MAX_PATH,
        value_bias: bool = False,
        name_embedding_dim: int | None = None,
        temporal_window: int = 31,
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.d_model = d_model
        self.temporal_window = temporal_window

        # The root carries a different quantity from the other joints -- its
        # block-0 slot holds the facing rotation, not a bone rotation -- so it
        # gets its own projection rather than sharing one.
        self.embed_root = nn.Linear(feature_dim, d_model)
        self.embed_joint = nn.Linear(feature_dim, d_model)
        self.embed_rest_root = nn.Linear(feature_dim, d_model)
        self.embed_rest_joint = nn.Linear(feature_dim, d_model)

        self.name_projection = (
            nn.Linear(name_embedding_dim, d_model) if name_embedding_dim else None
        )
        self.name_dropout = nn.Dropout(0.1)

        self.layers = nn.ModuleList(
            AnyTopLayer(d_model, n_heads, ff_size, dropout, max_path, value_bias)
            for _ in range(n_layers)
        )

        self.project_root = nn.Linear(d_model, feature_dim)
        self.project_joint = nn.Linear(d_model, feature_dim)

    def _embed(self, x: torch.Tensor, root: nn.Linear, joint: nn.Linear) -> torch.Tensor:
        """``(B, T, J, D)`` to ``(B, T, J, C)``, root projected separately."""
        return torch.cat([root(x[:, :, :1]), joint(x[:, :, 1:])], dim=2)

    def forward(
        self,
        z_t: torch.Tensor,  # (B, J, D, T)
        t: torch.Tensor,  # (B,)
        cond: Cond,
        masks: Masks | None = None,
    ) -> Prediction:
        batch, joints, _, frames = z_t.shape
        device = z_t.device

        topology = cond["topology"]
        hops = topology["hops"].to(device)
        edges = topology["relations"].to(device)

        # (B, J, D, T) -> (B, T, J, D)
        x = z_t.permute(0, 3, 1, 2)
        tokens = self._embed(x, self.embed_root, self.embed_joint)

        # The rest pose becomes an extra leading frame describing the skeleton.
        rest = cond["tpose"].to(device)[:, None]  # (B, 1, J, D)
        rest_tokens = self._embed(rest, self.embed_rest_root, self.embed_rest_joint)
        tokens = torch.cat([rest_tokens, tokens], dim=1)

        if self.name_projection is not None and "joint_names" in cond:
            names = self.name_projection(self.name_dropout(cond["joint_names"].to(device)))
            tokens = tokens + names[:, None]

        tokens = tokens + self._positions(batch, frames, cond, device)

        timestep = sinusoidal_embedding(t.to(device), self.d_model)
        spatial_mask, temporal_mask = self._masks(masks, cond, batch, joints, frames, device)

        for layer in self.layers:
            tokens = layer(tokens, timestep, hops, edges, spatial_mask, temporal_mask)

        # Drop the rest frame before projecting back.
        tokens = tokens[:, 1:]
        out = torch.cat(
            [self.project_root(tokens[:, :, :1]), self.project_joint(tokens[:, :, 1:])], dim=2
        )
        return Prediction(out=out.permute(0, 2, 3, 1))

    def _positions(
        self, batch: int, frames: int, cond: Cond, device: torch.device
    ) -> torch.Tensor:
        """Frame-index embedding, offset by where the window starts.

        The offset makes a window drawn from mid-animation encode its true
        position, which is meaningful here only because ingest never chunks.
        """
        index = torch.arange(frames + 1, device=device)[None].repeat(batch, 1)
        # `crop_start`, matching MoDiffAE. This used to read `window_start`, a
        # key nothing ever wrote -- so this branch has never once fired.
        start = cond.get("crop_start")
        if start is not None:
            index[:, 1:] = index[:, 1:] + start.to(device)[:, None]
        return sinusoidal_embedding(index, self.d_model)[:, :, None, :]

    def _valid_with_band(
        self, joint_valid: torch.Tensor, frame_valid: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Joint validity and pairwise frame validity, band-limited.

        Index 0 of the pair mask is the REST frame, which conditions everything
        and attends only itself; the band applies to the real frames after it.
        The band is ANDed with the padding mask, never ORed: it NARROWS
        attention and must never make a padded frame visible.
        """
        return joint_valid, temporal_pair_mask(frame_valid, self.temporal_window)

    def _masks(
        self,
        masks: Masks | None,
        cond: Cond,
        batch: int,
        joints: int,
        frames: int,
        device: torch.device,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        spatial = masks.spatial().to(device) if masks is not None else None  # (B, 1, J, J)

        # An explicit mask from the caller wins outright -- the band is only the
        # default for when none was supplied, and applying it afterwards would
        # silently narrow a mask sampling chose deliberately.
        override = cond.get(TEMPORAL_VALID)
        if override is not None:
            pair = override.to(device)
        elif masks is None and not self.temporal_window:
            return spatial, None
        else:
            frame_valid = (
                masks.frames.to(device)
                if masks is not None
                else torch.ones(batch, frames, dtype=torch.bool, device=device)
            )
            _, pair = self._valid_with_band(
                torch.ones(batch, joints, dtype=torch.bool, device=device), frame_valid
            )

        temporal = pair.repeat_interleave(joints, dim=0)  # (B * J, T+1, T+1)
        # MultiheadAttention wants an additive float mask when it is not bool.
        return spatial, torch.zeros_like(temporal, dtype=torch.float32).masked_fill(
            ~temporal, MASK_FILL
        )

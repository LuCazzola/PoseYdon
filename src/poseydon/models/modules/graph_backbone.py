"""The spatio-temporal graph backbone shared by MoDiffAE's two halves.

Module and parameter names deliberately mirror the reference implementation, so
its published checkpoints load with no key remapping at all. Everything else --
what the modules are called from outside, how conditioning reaches them, what is
documented -- is PoseYdon's.
"""

from __future__ import annotations

import torch
from torch import nn

MASK_FILL = -1e9


class GraphRelEmbedder(nn.Module):
    """Learned biases for how two joints relate.

    Two vocabularies. Hop distance saturates at ``max_path_len``; edge type is
    self / parent / child / sibling / none / end-effector. With virtual joints
    present each gains two more entries, for real-to-virtual and
    virtual-to-virtual, so pooling tokens are distinguishable from bones.
    """

    def __init__(self, d_model: int, max_path_len: int = 5, virtual: bool = False) -> None:
        super().__init__()
        n_topo = max_path_len + 3 if virtual else max_path_len + 1
        n_edge = 8 if virtual else 6
        self.topo_q = nn.Embedding(n_topo, d_model)
        self.topo_k = nn.Embedding(n_topo, d_model)
        self.edge_q = nn.Embedding(n_edge, d_model)
        self.edge_k = nn.Embedding(n_edge, d_model)


class GraphMultiHeadAttention(nn.Module):
    """Attention over joints, with relative biases from the skeleton graph."""

    def __init__(
        self, d_model: int, dropout: float, nheads: int, max_path_len: int = 5,
        virtual: bool = False,
    ) -> None:
        super().__init__()
        self.nheads = nheads
        self.att_size = d_model // nheads
        self.scale = self.att_size**-0.5

        self.linear_q = nn.Linear(d_model, d_model)
        self.linear_k = nn.Linear(d_model, d_model)
        self.linear_v = nn.Linear(d_model, d_model)
        self.output_layer = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.graph_embed = GraphRelEmbedder(d_model, max_path_len, virtual)

    def _heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, _ = x.shape
        return x.view(batch, length, self.nheads, self.att_size).transpose(1, 2)

    def _table(self, embedding: nn.Embedding) -> torch.Tensor:
        count = embedding.weight.shape[0]
        return embedding.weight.view(1, count, self.nheads, self.att_size).transpose(1, 2)

    def _bias(self, projected, embedding, index) -> torch.Tensor:
        scores = torch.matmul(projected, self._table(embedding).transpose(2, 3))
        return torch.gather(scores, 3, index.unsqueeze(1).expand(-1, self.nheads, -1, -1))

    def forward(self, x, distance, edge_attr, mask=None) -> torch.Tensor:
        batch, joints, _ = x.shape
        q, k, v = self._heads(self.linear_q(x)), self._heads(self.linear_k(x)), self._heads(self.linear_v(x))

        logits = torch.matmul(q, k.transpose(2, 3))
        logits = logits + self._bias(q, self.graph_embed.topo_q, distance)
        logits = logits + self._bias(k, self.graph_embed.topo_k, distance)
        logits = logits + self._bias(q, self.graph_embed.edge_q, edge_attr)
        logits = logits + self._bias(k, self.graph_embed.edge_k, edge_attr)
        logits = logits * self.scale

        if mask is not None:
            logits = logits + mask

        weights = self.dropout(torch.softmax(logits, dim=3))
        attended = torch.matmul(weights, v).transpose(1, 2).contiguous()
        return self.output_layer(attended.view(batch, joints, -1))


class SpatialAttention(nn.Module):
    """Joints exchange information within each frame."""

    def __init__(self, d_model, nhead, dropout, max_path_len=5, virtual=False) -> None:
        super().__init__()
        self.model = GraphMultiHeadAttention(d_model, dropout, nhead, max_path_len, virtual)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, topo_rel, edge_rel, mask) -> torch.Tensor:
        frames, batch, joints, channels = x.shape
        flat = x.view(frames * batch, joints, channels)
        topo = topo_rel.unsqueeze(0).repeat(frames, 1, 1, 1).view(-1, joints, joints)
        edge = edge_rel.unsqueeze(0).repeat(frames, 1, 1, 1).view(-1, joints, joints)
        out = self.model(flat, topo, edge, mask)
        return self.dropout(out.reshape(frames, batch, joints, channels))


class TemporalAttention(nn.Module):
    """Each joint attends over its own trajectory."""

    def __init__(self, d_model, num_heads, dropout) -> None:
        super().__init__()
        self.model = nn.MultiheadAttention(d_model, num_heads, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask) -> torch.Tensor:
        frames, batch, joints, channels = x.shape
        flat = x.view(frames, batch * joints, channels)
        out, _ = self.model(flat, flat, flat, attn_mask=attn_mask, need_weights=False)
        return self.dropout(out.view(frames, batch, joints, channels))


class ConditionAttention(nn.Module):
    """Gated semantic conditioning.

    Rather than cross-attending over a single latent -- where attention weights
    necessarily collapse to 1.0 and carry no information -- the latent is mapped
    to a value and gated by a projection of ``[joint, latent]``. The gate is what
    varies per joint.
    """

    def __init__(self, d_model: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.query_projector = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.gate_gen = nn.Sigmoid()
        self.value_map = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        joints = x.shape[2]
        expanded = memory.expand(-1, -1, joints, -1)
        gate = self.gate_gen(self.query_projector(torch.cat([x, expanded], dim=-1)))
        return self.dropout(self.out_proj(self.value_map(expanded) * gate))


class GraphMotionDecoderLayer(nn.Module):
    """Spatial, then temporal, then optional conditioning, then feed-forward.

    Post-norm throughout. The diffusion timestep is added at the top of every
    layer rather than once at the input, so depth cannot wash it out; the
    encoder omits it, since it sees clean motion.
    """

    def __init__(
        self,
        d_model: int = 128,
        num_heads: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_path_len: int = 5,
        timestep: bool = True,
        cross: bool = True,
        virtual: bool = False,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.spatial_attn = SpatialAttention(d_model, num_heads, dropout, max_path_len, virtual)
        self.spatial_norm = nn.LayerNorm(d_model)
        self.temporal_attn = TemporalAttention(d_model, num_heads, dropout)
        self.temporal_norm = nn.LayerNorm(d_model)

        self.cross_attn = ConditionAttention(d_model, dropout) if cross else None
        self.cross_norm = nn.LayerNorm(d_model) if cross else None

        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
        self.ffn_norm = nn.LayerNorm(d_model)
        self.timestep_embed = nn.Linear(d_model, d_model) if timestep else None

    def forward(self, x, topo_rel, edge_rel, timesteps_emb, memory, spatial_mask, temporal_mask):
        batch = x.shape[1]
        if self.timestep_embed is not None:
            x = x + self.timestep_embed(timesteps_emb).view(1, batch, 1, self.d_model)
        x = self.spatial_norm(x + self.spatial_attn(x, topo_rel, edge_rel, spatial_mask))
        x = self.temporal_norm(x + self.temporal_attn(x, temporal_mask))
        if self.cross_attn is not None:
            x = self.cross_norm(x + self.cross_attn(x, memory))
        return self.ffn_norm(x + self.ffn(x))


class GraphMotionDecoder(nn.Module):
    """A stack of :class:`GraphMotionDecoderLayer`."""

    def __init__(self, make_layer, num_layers: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(make_layer() for _ in range(num_layers))

    def forward(self, x, topo_rel, edge_rel, timesteps_embs, memory, spatial_mask, temporal_mask):
        for layer in self.layers:
            x = layer(x, topo_rel, edge_rel, timesteps_embs, memory, spatial_mask, temporal_mask)
        return x


def build_attention_masks(
    joint_valid: torch.Tensor, temporal_valid: torch.Tensor, n_heads: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Additive masks for the spatial and temporal attention passes.

    ``joint_valid`` is ``(B, J)`` and ``temporal_valid`` is ``(B, T, T)``, where
    T already includes the leading rest-pose frame.
    """
    batch, joints = joint_valid.shape
    frames = temporal_valid.shape[1]

    spatial_pair = joint_valid[:, :, None] & joint_valid[:, None, :]
    spatial = torch.zeros(batch, joints, joints, dtype=torch.float32, device=joint_valid.device)
    spatial = spatial.masked_fill(~spatial_pair, MASK_FILL)
    spatial = (
        spatial[:, None, None]
        .repeat(1, frames, n_heads, 1, 1)
        .reshape(-1, n_heads, joints, joints)
    )

    temporal = torch.zeros(
        batch, frames, frames, dtype=torch.float32, device=joint_valid.device
    ).masked_fill(~temporal_valid, MASK_FILL)
    temporal = (
        temporal[:, None, None].repeat(1, joints, n_heads, 1, 1).reshape(-1, frames, frames)
    )
    return spatial, temporal

"""Attention biased by skeleton topology.

Standard attention treats joints as an unordered set. This adds a learned bias
for how each pair of joints relates -- parent, child, sibling, or how many hops
apart -- which is what lets one model span skeletons whose joint counts and
shapes differ.
"""

from __future__ import annotations

import torch
from torch import nn

MASK_FILL = -1e9


class GraphAttention(nn.Module):
    """Multi-head attention with relative biases drawn from the skeleton graph.

    Biases enter the logits through both query and key projections, following
    Shaw-style relative position encoding generalized from a sequence to a tree.
    Value-side biases are optional and off by default, matching the reference's
    ``value_emb`` flag.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        n_hops: int = 6,
        n_edges: int = 7,
        value_bias: bool = False,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model {d_model} is not divisible by n_heads {n_heads}")

        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim**-0.5
        self.value_bias = value_bias

        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        self.hop_query = nn.Embedding(n_hops, d_model)
        self.hop_key = nn.Embedding(n_hops, d_model)
        self.edge_query = nn.Embedding(n_edges, d_model)
        self.edge_key = nn.Embedding(n_edges, d_model)
        if value_bias:
            self.hop_value = nn.Embedding(n_hops, d_model)
            self.edge_value = nn.Embedding(n_edges, d_model)

    def _heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, _ = x.shape
        return x.view(batch, length, self.n_heads, self.head_dim).transpose(1, 2)

    def _table(self, embedding: nn.Embedding) -> torch.Tensor:
        count = embedding.weight.shape[0]
        return embedding.weight.view(1, count, self.n_heads, self.head_dim).transpose(1, 2)

    def _relative(
        self, projected: torch.Tensor, table: torch.Tensor, index: torch.Tensor
    ) -> torch.Tensor:
        """Pick, for every (i, j) pair, the bias for their relation type."""
        scores = torch.matmul(projected, table.transpose(2, 3))
        return torch.gather(scores, 3, index.unsqueeze(1).expand(-1, self.n_heads, -1, -1))

    def forward(
        self,
        x: torch.Tensor,
        hops: torch.Tensor,
        edges: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """``x`` is ``(B, J, C)``; ``hops`` and ``edges`` are ``(B, J, J)`` indices."""
        batch, joints, _ = x.shape
        q, k, v = self._heads(self.query(x)), self._heads(self.key(x)), self._heads(self.value(x))

        logits = torch.matmul(q, k.transpose(2, 3))
        logits = logits + self._relative(q, self._table(self.hop_query), hops)
        logits = logits + self._relative(q, self._table(self.edge_query), edges)
        logits = logits + self._relative(k, self._table(self.hop_key), hops)
        logits = logits + self._relative(k, self._table(self.edge_key), edges)
        logits = logits * self.scale

        if mask is not None:
            logits = logits.masked_fill(~mask, MASK_FILL)

        weights = self.dropout(torch.softmax(logits, dim=-1))
        attended = torch.matmul(weights, v)

        if self.value_bias:
            attended = attended + self._value_bias(weights, hops, self.hop_value)
            attended = attended + self._value_bias(weights, edges, self.edge_value)

        attended = attended.transpose(1, 2).contiguous().view(batch, joints, -1)
        return self.out(attended)

    def _value_bias(
        self, weights: torch.Tensor, index: torch.Tensor, embedding: nn.Embedding
    ) -> torch.Tensor:
        """Route attention mass onto per-relation value vectors."""
        batch, heads, joints, _ = weights.shape
        count = embedding.weight.shape[0]
        pooled = torch.zeros(batch, heads, joints, count, device=weights.device, dtype=weights.dtype)
        pooled = pooled.scatter_add(
            3, index.unsqueeze(1).expand(-1, heads, -1, -1), weights
        )
        return torch.matmul(pooled, self._table(embedding))

"""MoDiffAE: a diffusion autoencoder over motion.

A semantic encoder compresses a clean clip into a per-frame latent, and a
stochastic decoder denoises conditioned on it. Diffusion runs in feature space;
what is learned is a semantic code, which is what makes blending meaningful --
interpolating two codes interpolates two motions.

The reference performs that interpolation inside ``MoDiffAE.forward`` as a
``_mix`` method. Here the model simply accepts a ``z_sem`` in its conditioning
when one is supplied, so the ``LatentMix`` control can blend without the model
knowing that blending exists. No parameter changes; only the signature.
"""

from __future__ import annotations

import torch
from torch import nn

from poseydon.core.batch import Cond, Masks
from poseydon.core.topology import DEFAULT_MAX_PATH, EdgeType
from poseydon.models.anytop import AnyTopLayer
from poseydon.models.base import CLEAN_MOTION, MODELS, Denoiser, Prediction
from poseydon.models.modules.embeddings import sinusoidal_embedding

#: Conditioning key holding a precomputed semantic latent. When present the
#: encoder is skipped, which is how blending and latent caching work.
Z_SEM = "z_sem"


def _extend_topology(
    hops: torch.Tensor, edges: torch.Tensor, extra: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Grow the relation matrices to cover appended virtual joints.

    Virtual joints are pooling tokens rather than bones, so they get their own
    edge type and sit zero hops from everything -- they must see the whole
    skeleton and be seen by it.
    """
    batch, joints, _ = hops.shape
    total = joints + extra
    new_hops = hops.new_zeros((batch, total, total))
    new_edges = edges.new_full((batch, total, total), int(EdgeType.TOKEN_CONNECTION))
    new_hops[:, :joints, :joints] = hops
    new_edges[:, :joints, :joints] = edges
    return new_hops, new_edges


class SemanticEncoder(nn.Module):
    """Clean motion to a per-frame semantic latent.

    Learned virtual joints act as pooling tokens: they attend over the real
    joints and are read out as the summary, so the latent has a fixed width no
    matter how many joints the skeleton has.
    """

    def __init__(
        self,
        feature_dim: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        ff_size: int,
        dropout: float,
        max_path: int,
        n_virtual_joints: int,
        kl_bottleneck: bool,
    ) -> None:
        super().__init__()
        self.n_virtual_joints = n_virtual_joints
        self.kl_bottleneck = kl_bottleneck

        self.embed_root = nn.Linear(feature_dim, d_model)
        self.embed_joint = nn.Linear(feature_dim, d_model)
        self.virtual = nn.Parameter(torch.randn(n_virtual_joints, d_model) * 0.02)

        self.layers = nn.ModuleList(
            AnyTopLayer(d_model, n_heads, ff_size, dropout, max_path, value_bias=False)
            for _ in range(n_layers)
        )
        out_dim = d_model * 2 if kl_bottleneck else d_model
        self.projection = nn.Linear(n_virtual_joints * d_model, out_dim)

    def forward(
        self, x: torch.Tensor, hops: torch.Tensor, edges: torch.Tensor, masks: Masks | None
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """``x`` is ``(B, J, D, T)``; returns ``(B, T, C)`` plus KL terms."""
        batch, _, _, frames = x.shape
        tokens = x.permute(0, 3, 1, 2)
        tokens = torch.cat(
            [self.embed_root(tokens[:, :, :1]), self.embed_joint(tokens[:, :, 1:])], dim=2
        )

        virtual = self.virtual[None, None].expand(batch, frames, -1, -1)
        tokens = torch.cat([tokens, virtual], dim=2)
        hops, edges = _extend_topology(hops, edges, self.n_virtual_joints)

        spatial = None
        if masks is not None:
            valid = torch.cat(
                [
                    masks.joints,
                    torch.ones(
                        batch, self.n_virtual_joints, dtype=torch.bool, device=x.device
                    ),
                ],
                dim=1,
            )
            spatial = (valid[:, :, None] & valid[:, None, :])[:, None]

        timestep = torch.zeros(batch, tokens.shape[-1], device=x.device)
        for layer in self.layers:
            tokens = layer(tokens, timestep, hops, edges, spatial, None)

        summary = tokens[:, :, -self.n_virtual_joints :].flatten(start_dim=2)
        projected = self.projection(summary)

        if not self.kl_bottleneck:
            return projected, {}

        mu, logvar = projected.chunk(2, dim=-1)
        # Reparameterize so the bottleneck is trainable through sampling.
        z = mu + torch.randn_like(mu) * (0.5 * logvar).exp()
        return z, {"mu": mu, "logvar": logvar}


class StochasticDecoder(nn.Module):
    """Denoises conditioned on a semantic latent, by FiLM modulation."""

    def __init__(
        self,
        feature_dim: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        ff_size: int,
        dropout: float,
        max_path: int,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.embed_root = nn.Linear(feature_dim, d_model)
        self.embed_joint = nn.Linear(feature_dim, d_model)
        self.embed_rest_root = nn.Linear(feature_dim, d_model)
        self.embed_rest_joint = nn.Linear(feature_dim, d_model)

        # Per-frame scale and shift, so the latent steers the whole trajectory
        # rather than being averaged into a single global vector.
        self.modulation = nn.Linear(d_model, 2 * d_model)

        self.layers = nn.ModuleList(
            AnyTopLayer(d_model, n_heads, ff_size, dropout, max_path, value_bias=False)
            for _ in range(n_layers)
        )
        self.project_root = nn.Linear(d_model, feature_dim)
        self.project_joint = nn.Linear(d_model, feature_dim)

    def forward(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        z_sem: torch.Tensor,
        rest: torch.Tensor,
        hops: torch.Tensor,
        edges: torch.Tensor,
        masks: Masks | None,
    ) -> torch.Tensor:
        batch, _, _, frames = z_t.shape
        x = z_t.permute(0, 3, 1, 2)
        tokens = torch.cat(
            [self.embed_root(x[:, :, :1]), self.embed_joint(x[:, :, 1:])], dim=2
        )

        rest_tokens = torch.cat(
            [
                self.embed_rest_root(rest[:, None, :1]),
                self.embed_rest_joint(rest[:, None, 1:]),
            ],
            dim=2,
        )
        tokens = torch.cat([rest_tokens, tokens], dim=1)

        # The rest frame is not part of the conditioned trajectory, so it is
        # modulated by the first frame's code rather than one of its own.
        code = torch.cat([z_sem[:, :1], z_sem], dim=1)
        scale, shift = self.modulation(code).chunk(2, dim=-1)
        tokens = tokens * (1.0 + scale[:, :, None]) + shift[:, :, None]

        tokens = tokens + sinusoidal_embedding(
            torch.arange(frames + 1, device=z_t.device)[None].expand(batch, -1), self.d_model
        )[:, :, None, :]

        timestep = sinusoidal_embedding(t.to(z_t.device), self.d_model)
        spatial = masks.spatial().to(z_t.device) if masks is not None else None

        for layer in self.layers:
            tokens = layer(tokens, timestep, hops, edges, spatial, None)

        tokens = tokens[:, 1:]
        out = torch.cat(
            [self.project_root(tokens[:, :, :1]), self.project_joint(tokens[:, :, 1:])], dim=2
        )
        return out.permute(0, 2, 3, 1)


@MODELS.register("modiffae")
class MoDiffAE(Denoiser):
    """Diffusion autoencoder: encode a clean clip, denoise conditioned on it."""

    requires = ("topology", "tpose", CLEAN_MOTION)

    def __init__(
        self,
        feature_dim: int,
        d_model: int = 128,
        n_layers_semantic: int = 4,
        n_layers_stochastic: int = 4,
        n_heads: int = 4,
        ff_size: int = 512,
        dropout: float = 0.1,
        max_path: int = DEFAULT_MAX_PATH,
        n_virtual_joints: int = 3,
        kl_bottleneck: bool = False,
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.d_model = d_model
        self.encoder = SemanticEncoder(
            feature_dim,
            d_model,
            n_layers_semantic,
            n_heads,
            ff_size,
            dropout,
            max_path,
            n_virtual_joints,
            kl_bottleneck,
        )
        self.decoder = StochasticDecoder(
            feature_dim, d_model, n_layers_stochastic, n_heads, ff_size, dropout, max_path
        )

    def encode(
        self, clean: torch.Tensor, cond: Cond, masks: Masks | None = None
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Semantic latent for a clean clip, exposed for blending and caching."""
        topology = cond["topology"]
        return self.encoder(
            clean, topology["hops"].to(clean.device), topology["relations"].to(clean.device), masks
        )

    def forward(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        cond: Cond,
        masks: Masks | None = None,
    ) -> Prediction:
        topology = cond["topology"]
        hops = topology["hops"].to(z_t.device)
        edges = topology["relations"].to(z_t.device)

        supplied = cond.get(Z_SEM)
        if supplied is not None:
            z_sem, aux = supplied.to(z_t.device), {}
        else:
            z_sem, aux = self.encode(cond[CLEAN_MOTION].to(z_t.device), cond, masks)

        out = self.decoder(
            z_t, t, z_sem, cond["tpose"].to(z_t.device), hops, edges, masks
        )
        return Prediction(out=out, aux={**aux, Z_SEM: z_sem})

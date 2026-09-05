"""MoDiffAE: a diffusion autoencoder over motion.

A semantic encoder compresses a clean clip into a per-frame latent; a stochastic
decoder denoises conditioned on it. Diffusion runs in feature space -- what is
learned is a semantic code, and that is what makes blending and cross-skeleton
transfer meaningful: interpolating or re-targeting a code moves the motion, not
the pixels.

Parameter names mirror the reference implementation exactly, so its published
checkpoints load with no remapping. The differences are structural: conditioning
is declared rather than pulled from a fourteen-key dict, masks arrive named, and
the model contains no blending logic. In the reference, interpolation lives
inside ``forward`` as a ``_mix`` method, which is why in-betweening was
unreachable there. Here the model simply accepts a ``z_sem`` when one is
supplied, so a control can blend without the model knowing blending exists.
"""

from __future__ import annotations

import torch
from torch import nn

from poseydon.core.batch import Cond, Masks
from poseydon.core.topology import DEFAULT_MAX_PATH
from poseydon.models.base import CLEAN_MOTION, MODELS, Denoiser, Prediction
from poseydon.models.modules.embeddings import sinusoidal_embedding
from poseydon.models.modules.graph_backbone import (
    GraphMotionDecoder,
    GraphMotionDecoderLayer,
    build_attention_masks,
)

#: Conditioning key holding a precomputed semantic latent. When present the
#: encoder is skipped -- how blending, transfer and latent caching all work.
Z_SEM = "z_sem"
#: Conditioning key holding T5 embeddings of the joint names.
JOINT_NAMES = "joint_names"
#: Conditioning key holding an explicit (B, T+1, T+1) frame-attention mask.
#: The reference restricts temporal attention to a sliding band, which is a
#: dataset property rather than a model one, so it arrives as conditioning.
TEMPORAL_VALID = "temporal_valid"


class InputProcess(nn.Module):
    """Embed motion, prepend the rest pose, add names and frame positions.

    The root is projected separately from the other joints because its slot
    holds a different quantity -- the facing rotation rather than a bone
    rotation. The rest pose enters as an extra leading frame, telling the model
    which skeleton it is animating.
    """

    def __init__(self, feature_dim: int, d_model: int, text_dim: int = 768) -> None:
        super().__init__()
        self.d_model = d_model
        self.root_embedding = nn.Linear(feature_dim, d_model)
        self.joint_embedding = nn.Linear(feature_dim, d_model)
        self.tpos_root_embedding = nn.Linear(feature_dim, d_model)
        self.tpos_joint_embedding = nn.Linear(feature_dim, d_model)
        self.text_embedding = nn.Linear(text_dim, d_model)
        self.joints_names_dropout = nn.Dropout(p=0.1)

    def forward(self, x, tpose, name_embeddings, crop_start) -> torch.Tensor:
        """``x`` is ``(B, J, D, T)``; returns ``(T + 1, B, J, C)``."""
        x = x.permute(3, 0, 1, 2)  # (T, B, J, D)
        tpose = tpose.unsqueeze(0)  # (1, B, J, D)

        motion = torch.cat(
            [self.root_embedding(x[:, :, 0:1]), self.joint_embedding(x[:, :, 1:])], dim=2
        )
        rest = torch.cat(
            [
                self.tpos_root_embedding(tpose[:, :, 0:1]),
                self.tpos_joint_embedding(tpose[:, :, 1:]),
            ],
            dim=2,
        )
        out = torch.cat([rest, motion], dim=0)

        if name_embeddings is not None:
            names = self.text_embedding(self.joints_names_dropout(name_embeddings))
            out = out + names[None]

        # Frame index, offset so a window drawn from mid-animation encodes where
        # it actually sits. The rest frame keeps index 0.
        positions = torch.arange(out.shape[0], device=out.device).view(1, -1).repeat(
            out.shape[1], 1
        )
        positions[:, 1:] = positions[:, 1:] + crop_start.to(out.device).view(-1, 1)
        return out + sinusoidal_embedding(positions, self.d_model)[0][:, None, None, :]


class OutputProcess(nn.Module):
    """Project back to feature space and drop the rest frame."""

    def __init__(self, feature_dim: int, d_model: int) -> None:
        super().__init__()
        self.root_dembedding = nn.Linear(d_model, feature_dim)
        self.joint_dembedding = nn.Linear(d_model, feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        root = self.root_dembedding(x[:, :, 0])
        joints = self.joint_dembedding(x[:, :, 1:])
        out = torch.cat([root.unsqueeze(2), joints], dim=-2)
        return out.permute(1, 2, 3, 0)[..., 1:]


class KLBottleneck(nn.Module):
    """Split a projection into a diagonal Gaussian and sample from it.

    Optional, and absent from the published checkpoints, so enabling it adds
    parameters they do not contain -- a strict load of one of those checkpoints
    into a KL-enabled model correctly fails.
    """

    def forward(self, projected: torch.Tensor):
        mu, logvar = projected.chunk(2, dim=-1)
        z = mu + torch.randn_like(mu) * (0.5 * logvar).exp() if self.training else mu
        return z, {"mu": mu, "logvar": logvar}


class SemanticEncoder(nn.Module):
    """Clean motion to a per-frame semantic latent.

    Two pooling modes. With no virtual joints the real joints are averaged over,
    masked so padding does not dilute the mean. With virtual joints, learned
    pooling tokens are appended to the joint axis, attend over the skeleton, and
    are read out -- giving the latent a fixed width whatever the joint count.
    """

    def __init__(
        self,
        feature_dim: int,
        d_model: int = 128,
        num_layers: int = 4,
        num_heads: int = 4,
        ff_size: int = 1024,
        dropout: float = 0.1,
        max_path_len: int = DEFAULT_MAX_PATH,
        num_virtual_joints: int = 0,
        projection_head_depth: int = 2,
        text_dim: int = 768,
        kl_bottleneck: bool = False,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.max_path_len = max_path_len
        self.num_virtual_joints = num_virtual_joints
        self.kl_bottleneck = kl_bottleneck

        if num_virtual_joints > 0:
            self.v_joint_emb = nn.Parameter(torch.randn(num_virtual_joints, d_model))

        self.input_process = InputProcess(feature_dim, d_model, text_dim)
        self.backbone = GraphMotionDecoder(
            lambda: GraphMotionDecoderLayer(
                d_model=d_model,
                num_heads=num_heads,
                dim_feedforward=ff_size,
                dropout=dropout,
                max_path_len=max_path_len,
                timestep=False,
                cross=False,
                virtual=num_virtual_joints > 0,
            ),
            num_layers,
        )

        in_dim = max(num_virtual_joints, 1) * d_model
        out_dim = d_model * 2 if kl_bottleneck else d_model
        if projection_head_depth > 0:
            dims = [in_dim] + [(in_dim + d_model) // 2] * (projection_head_depth - 1) + [out_dim]
            layers: list[nn.Module] = []
            for i in range(projection_head_depth):
                layers.append(nn.Linear(dims[i], dims[i + 1]))
                if i < projection_head_depth - 1:
                    layers.extend([nn.GELU(), nn.Dropout(dropout)])
            self.projection_head = nn.Sequential(*layers)
        else:
            self.projection_head = nn.Linear(in_dim, out_dim)
        self.kl_layer = KLBottleneck() if kl_bottleneck else None

    def add_virtual_joints(self, x, joint_valid, hops, edges):
        """Append pooling tokens and extend the relation matrices to cover them."""
        if self.num_virtual_joints <= 0:
            return x, joint_valid, hops, edges

        frames, batch, real, _ = x.shape
        extra = self.num_virtual_joints
        total = real + extra

        tokens = self.v_joint_emb[None, None].expand(frames, batch, -1, -1)
        x = torch.cat([x, tokens], dim=2)

        joint_valid = torch.cat(
            [joint_valid, torch.ones(batch, extra, dtype=torch.bool, device=x.device)], dim=1
        )

        topo_rv, topo_vv = self.max_path_len + 1, self.max_path_len + 2
        new_hops = hops.new_full((batch, total, total), topo_vv)
        new_hops[:, :real, :real] = hops
        new_hops[:, :real, real:] = topo_rv
        new_hops[:, real:, :real] = topo_rv

        new_edges = edges.new_full((batch, total, total), 7)
        new_edges[:, :real, :real] = edges
        new_edges[:, :real, real:] = 6
        new_edges[:, real:, :real] = 6

        diagonal = torch.arange(total, device=x.device)
        new_hops[:, diagonal, diagonal] = 0
        new_edges[:, diagonal, diagonal] = 0
        return x, joint_valid, new_hops, new_edges

    def forward(self, x, tpose, names, crop_start, hops, edges, joint_valid, temporal_valid):
        tokens = self.input_process(x, tpose, names, crop_start)
        tokens, joint_valid, hops, edges = self.add_virtual_joints(
            tokens, joint_valid, hops, edges
        )
        spatial, temporal = build_attention_masks(joint_valid, temporal_valid, self.num_heads)
        tokens = self.backbone(tokens, hops, edges, None, None, spatial, temporal)

        if self.num_virtual_joints == 0:
            weight = joint_valid[None, :, :, None].to(tokens.dtype)
            pooled = (tokens * weight).sum(dim=2) / weight.sum(dim=2).clamp(min=1.0)
        else:
            pooled = tokens[:, :, -self.num_virtual_joints :].flatten(start_dim=2)

        projected = self.projection_head(pooled)
        if self.kl_layer is None:
            return projected.unsqueeze(2), {}  # (T+1, B, 1, C)
        latent, terms = self.kl_layer(projected)
        return latent.unsqueeze(2), terms


class StochasticDecoder(nn.Module):
    """Denoises conditioned on the semantic latent."""

    def __init__(
        self,
        feature_dim: int,
        d_model: int = 128,
        num_layers: int = 4,
        num_heads: int = 4,
        ff_size: int = 1024,
        dropout: float = 0.1,
        max_path_len: int = DEFAULT_MAX_PATH,
        text_dim: int = 768,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.input_process = InputProcess(feature_dim, d_model, text_dim)
        self.backbone = GraphMotionDecoder(
            lambda: GraphMotionDecoderLayer(
                d_model=d_model,
                num_heads=num_heads,
                dim_feedforward=ff_size,
                dropout=dropout,
                max_path_len=max_path_len,
                timestep=True,
                cross=True,
                virtual=False,
            ),
            num_layers,
        )

    def forward(
        self, x, tpose, names, crop_start, timestep, z_sem, hops, edges, joint_valid, temporal_valid
    ):
        tokens = self.input_process(x, tpose, names, crop_start)
        spatial, temporal = build_attention_masks(joint_valid, temporal_valid, self.num_heads)
        return self.backbone(tokens, hops, edges, timestep, z_sem, spatial, temporal)


@MODELS.register("modiffae")
class MoDiffAE(Denoiser):
    """Diffusion autoencoder: encode a clean clip, denoise conditioned on it."""

    requires = ("topology", "tpose", CLEAN_MOTION)
    optional = (JOINT_NAMES,)

    def __init__(
        self,
        feature_dim: int,
        d_model: int = 128,
        n_layers_semantic: int = 4,
        n_layers_stochastic: int = 4,
        n_heads: int = 4,
        ff_size: int = 1024,
        dropout: float = 0.1,
        max_path: int = DEFAULT_MAX_PATH,
        n_virtual_joints: int = 0,
        projection_head_depth: int = 2,
        text_dim: int = 768,
        kl_bottleneck: bool = False,
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.d_model = d_model
        self.semantic_encoder = SemanticEncoder(
            feature_dim, d_model, n_layers_semantic, n_heads, ff_size, dropout,
            max_path, n_virtual_joints, projection_head_depth, text_dim, kl_bottleneck,
        )
        self.stochastic_decoder = StochasticDecoder(
            feature_dim, d_model, n_layers_stochastic, n_heads, ff_size, dropout,
            max_path, text_dim,
        )
        self.output_process = OutputProcess(feature_dim, d_model)

    @staticmethod
    def _valid(masks: Masks | None, batch: int, joints: int, frames: int, device):
        """Joint validity and pairwise frame validity, including the rest frame."""
        if masks is None:
            joint_valid = torch.ones(batch, joints, dtype=torch.bool, device=device)
            frame_valid = torch.ones(batch, frames, dtype=torch.bool, device=device)
        else:
            joint_valid = masks.joints.to(device)
            frame_valid = masks.frames.to(device)

        leading = torch.ones(batch, 1, dtype=torch.bool, device=device)
        extended = torch.cat([leading, frame_valid], dim=1)
        pair = extended[:, :, None] & extended[:, None, :]
        # The rest frame conditions everything but attends only to itself
        pair = pair.clone()
        pair[:, 0, :] = False
        pair[:, 0, 0] = True
        return joint_valid, pair

    def encode(self, clean, cond, masks=None, temporal_valid=None):
        """Semantic latent for a clean clip, plus any KL terms."""
        topology = cond["topology"]
        batch, joints, _, frames = clean.shape
        joint_valid, pair = self._valid(masks, batch, joints, frames, clean.device)
        override = temporal_valid if temporal_valid is not None else cond.get(TEMPORAL_VALID)
        if override is not None:
            pair = override.to(clean.device)
        return self.semantic_encoder(
            clean,
            cond["tpose"].to(clean.device),
            cond.get(JOINT_NAMES),
            cond.get("crop_start", torch.zeros(batch, dtype=torch.long)),
            topology["hops"].to(clean.device),
            topology["relations"].to(clean.device),
            joint_valid,
            pair,
        )

    def forward(self, z_t, t, cond: Cond, masks: Masks | None = None) -> Prediction:
        topology = cond["topology"]
        batch, joints, _, frames = z_t.shape
        device = z_t.device
        joint_valid, pair = self._valid(masks, batch, joints, frames, device)
        override = cond.get(TEMPORAL_VALID)
        if override is not None:
            pair = override.to(device)

        supplied = cond.get(Z_SEM)
        if supplied is not None:
            z_sem, aux = supplied.to(device), {}
        else:
            z_sem, aux = self.encode(cond[CLEAN_MOTION].to(device), cond, masks)

        timestep = sinusoidal_embedding(t.to(device), self.d_model)
        hidden = self.stochastic_decoder(
            z_t,
            cond["tpose"].to(device),
            cond.get(JOINT_NAMES),
            cond.get("crop_start", torch.zeros(batch, dtype=torch.long)),
            timestep,
            z_sem,
            topology["hops"].to(device),
            topology["relations"].to(device),
            joint_valid,
            pair,
        )
        return Prediction(out=self.output_process(hidden), aux={**aux, Z_SEM: z_sem})

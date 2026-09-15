"""Masked conditioning: what the model is NOT told, and what stands in for it.

Two siblings, each owning one learned null embedding and one job.
:class:`MotionMasker` hides parts of the clean clip the semantic encoder reads
-- frames, joints, whole kinematic chains -- so the latent cannot carry a copy
of its input and has to describe the motion instead. :class:`ConditionMasker`
hides the latent itself, which is what classifier-free guidance needs.

They are siblings rather than a base and a subclass because they act on
different objects: one replaces `(T, B, J, C)` input tokens and needs the
skeleton to do it, the other replaces a pooled `(T, B, 1, C)` vector and knows
nothing about skeletons. Splitting them also keeps guidance out of
``InputProcess``, which only ever needs the token half.

Both are no-ops until configured, and both are silent outside training unless
a caller asks explicitly -- sampling and retargeting see whole conditioning.
"""

from __future__ import annotations

import torch
from torch import nn

from poseydon.core.batch import Cond
from poseydon.models.base import COND_DROP


def chain_ids(parents: torch.Tensor, joint_valid: torch.Tensor) -> torch.Tensor:
    """``(B, J)`` chain index per joint: maximal runs of single-child joints.

    A chain is a tail, a neck, one toe -- cut at the next branch point, and at
    the root. Joints are in topological order (`core.kinematics`
    `check_topological_order`), so this is a forward scan rather than a
    traversal: a joint continues its parent's chain exactly when that parent
    has no other child.

    Measured against the alternatives over the 73 Truebones rigs, this is the
    bounded one -- max 48.3% of a rig, p99 16.2%. Masking the subtree below a
    joint instead removes more than half the skeleton on 4.9% of draws (p99
    97.9%), because any joint just below the root takes everything with it.

    ``parents`` is ``-1`` for the root AND for padding, which is why
    ``joint_valid`` is needed to count children correctly.
    """
    batch, n_joints = parents.shape
    index = parents.clamp(min=0)
    real_child = ((parents >= 0) & joint_valid).long()
    child_count = torch.zeros_like(real_child).scatter_add_(1, index, real_child)

    ids = torch.arange(n_joints, device=parents.device).expand(batch, n_joints).clone()
    # ponytail: J-step loop (J <= 183 in this corpus, against a measured
    # 0.431 s/step). Move it to a conditioner computed once per item if it
    # ever shows up in a profile.
    for joint in range(1, n_joints):
        parent = index[:, joint]
        # `> 0`, not `>= 0`: a chain never continues through the root, and a
        # padded joint's -1 clamps to 0 and is excluded by the same test.
        continues = (parents[:, joint] > 0) & (child_count.gather(1, parent[:, None])[:, 0] == 1)
        ids[:, joint] = torch.where(continues, ids.gather(1, parent[:, None])[:, 0], ids[:, joint])
    return ids


def one_chain_per_sample(parents: torch.Tensor, joint_valid: torch.Tensor) -> torch.Tensor:
    """``(B, J)`` bool selecting one whole chain per sample, drawn uniformly.

    Uniform over CHAINS, not over joints, which is what the measurement in
    :func:`chain_ids` describes. A chain's id is its own topmost joint, so the
    chains present are exactly the joints that are their own id.
    """
    ids = chain_ids(parents, joint_valid)
    positions = torch.arange(ids.shape[1], device=ids.device)
    # The root is never a chain start here: masking the root slot would remove
    # the facing rotation and the global trajectory, not a limb.
    starts = (ids == positions) & joint_valid & (positions > 0)
    weights = starts.float()
    # A rig whose only valid joint is the root has no chain to draw; fall back
    # to masking nothing rather than letting `multinomial` raise.
    empty = weights.sum(dim=1) == 0
    weights[empty] = 1.0
    picked = torch.multinomial(weights, 1)
    chosen = ids.gather(1, picked)
    return (ids == chosen) & joint_valid & ~empty[:, None]


class MotionMasker(nn.Module):
    """Hides parts of the clip the semantic encoder reads.

    Three schemes, each with its own rate and composed by OR -- the same
    independence ``AugmentPipeline`` gives its entries. Rates are drawn per
    sample from ``U(0, rate)`` rather than applied flat, which is the idiom
    ``DropEndEffector`` already uses and which leaves some samples in every
    batch seen clean, so an always-on mask does not open a gap against a
    sampling path that never masks.

    One null token for all three. A token per scheme would tell the encoder
    WHY a slot is absent, which is exactly what a bottleneck withholds.
    """

    def __init__(
        self,
        d_model: int,
        frames: float = 0.0,
        joints: float = 0.0,
        chains: float = 0.0,
    ) -> None:
        super().__init__()
        for name, value in (("frames", frames), ("joints", joints), ("chains", chains)):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"masking {name} must be in [0, 1], got {value}")
        self.frames = float(frames)
        self.joints = float(joints)
        self.chains = float(chains)
        # Registered only when a rate is actually set, the same idiom
        # `SemanticEncoder.v_joint_emb` and `kl_layer` already use: a
        # default-config model must have no such key, or a strict load of an
        # existing checkpoint (`cli._load_weights`, `strict=True`) fails on a
        # parameter the checkpoint never had a reason to save.
        if self.enabled:
            self.null_token = nn.Parameter(torch.randn(d_model))

    @property
    def enabled(self) -> bool:
        return max(self.frames, self.joints, self.chains) > 0.0

    def mask(
        self, joint_valid: torch.Tensor, parents: torch.Tensor | None, n_frames: int
    ) -> torch.Tensor | None:
        """``(T, B, J)`` bool: which motion tokens the null token replaces.

        ``None`` rather than an all-false mask when nothing is configured or
        the module is not training, so an unconfigured masker costs nothing and
        sampling is untouched.

        ``parents`` is optional because frame and joint masking need nothing
        but shapes; only chain masking needs the skeleton to find a chain to
        hide. A caller that configures ``chains`` but hands in a topology
        payload with no ``parents`` gets a loud error here, not a quietly
        unmasked run -- the alternative is a training run nobody notices
        never actually masked a chain.
        """
        if not self.enabled or not self.training:
            return None
        if self.chains > 0.0 and parents is None:
            raise ValueError(
                "chain masking (masking.chains > 0) needs "
                "cond['topology']['parents'], but this topology payload does "
                "not carry it"
            )

        batch, n_joints = joint_valid.shape
        device = joint_valid.device
        mask = torch.zeros(n_frames, batch, n_joints, dtype=torch.bool, device=device)

        if self.frames > 0.0:
            rate = torch.rand(batch, device=device) * self.frames
            mask |= (torch.rand(n_frames, batch, device=device) < rate)[:, :, None]
        if self.joints > 0.0:
            rate = torch.rand(batch, device=device) * self.joints
            mask |= torch.rand(n_frames, batch, n_joints, device=device) < rate[None, :, None]
        if self.chains > 0.0:
            fires = torch.rand(batch, device=device) < self.chains
            if bool(fires.any()):
                # Across the whole window: a limb hidden in some frames and not
                # others is per-frame joint masking with extra steps.
                chain = one_chain_per_sample(parents, joint_valid) & fires[:, None]
                mask |= chain[None]

        # Padding is already excluded from attention; giving it null tokens
        # would make the number of masked slots depend on the batch's padding.
        return mask & joint_valid[None]

    def apply(self, motion: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        """Substitute the null token into ``(T, B, J, C)`` motion embeddings.

        No guard on ``mask is None`` needing ``null_token`` to exist: `mask()`
        can only return non-``None`` when `enabled` is true, and that is
        exactly when `null_token` was registered. If that coupling is ever
        broken this raises `AttributeError`, not a silent wrong answer.
        """
        if mask is None:
            return motion
        return torch.where(mask[..., None], self.null_token.to(motion.dtype), motion)


class ConditionMasker(nn.Module):
    """Hides the whole semantic latent, which is what CFG trains against.

    The null latent is separate from :class:`MotionMasker`'s null token because
    it lives in a different space: after the encoder's projection head, not
    among its inputs.
    """

    def __init__(self, d_model: int, cond_drop: float = 0.0) -> None:
        super().__init__()
        if not 0.0 <= cond_drop <= 1.0:
            raise ValueError(f"cond_drop must be in [0, 1], got {cond_drop}")
        self.cond_drop = float(cond_drop)
        # Registered only when guidance is actually configured, matching
        # `MotionMasker.null_token`: a default-config model must have no such
        # key, or a strict load of an existing checkpoint fails on a
        # parameter it never had a reason to save.
        if self.cond_drop > 0.0:
            self.null_latent = nn.Parameter(torch.randn(d_model))

    def drop(self, z_sem: torch.Tensor, cond: Cond) -> torch.Tensor:
        """``(T, B, 1, C)`` with dropped samples replaced by the null latent.

        A ``COND_DROP`` in ``cond`` is honoured whatever the training mode, and
        must be applied AFTER a supplied ``Z_SEM`` is read: `Retarget` pins its
        latent through that key, so a guided retarget's unconditional branch
        has to override the pin rather than defer to it. The other way round
        gives two identical branches and guidance that is silently zero.
        """
        batch = z_sem.shape[1]
        forced = cond.get(COND_DROP)
        if forced is not None:
            if not hasattr(self, "null_latent"):
                # `GuidedDenoiser` can force `COND_DROP` at sampling time on
                # any model, whatever it was trained with. A model built with
                # `cond_drop=0` never learned a null latent to guide away
                # from, so there is nothing correct to substitute here --
                # this has to fail loudly rather than crash on a missing
                # attribute or silently return the conditioned latent.
                raise RuntimeError(
                    "classifier-free guidance requires a model trained with "
                    "cond_drop > 0; this ConditionMasker has no null latent "
                    "to guide away from"
                )
            if torch.is_tensor(forced) and forced.ndim > 0:
                drop = forced.to(z_sem.device).bool()
            else:
                drop = torch.full(
                    (batch,), bool(forced), dtype=torch.bool, device=z_sem.device
                )
        elif self.training and self.cond_drop > 0.0:
            drop = torch.rand(batch, device=z_sem.device) < self.cond_drop
        else:
            return z_sem
        # ponytail: the encode is already paid for on dropped samples. Split
        # the batch and skip it if the encoder ever dominates a step.
        return torch.where(
            drop[None, :, None, None], self.null_latent.to(z_sem.dtype), z_sem
        )

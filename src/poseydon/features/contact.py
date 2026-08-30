"""Ground-contact features."""

from __future__ import annotations

import numpy as np

from poseydon.features.base import FEATURES, Feature, FeatureContext


@FEATURES.register("foot_contact")
class FootContact(Feature):
    """Binary per-joint flag: this foot is planted on this frame.

    A foot counts as planted when it moves slower than ``contact.max_speed`` and
    sits lower than ``contact.max_height``. Joints not named as feet are always
    zero, and a rig with no declared feet yields an all-zero block -- which is
    what the reference's auto-detection produces for some rigs, the crab
    included.

    One frame shorter than the animation, since speed needs two frames.
    """

    name = "foot_contact"
    width = 1

    def __call__(self, ctx: FeatureContext) -> np.ndarray:
        contact = ctx.resolved.manifest.contact
        feet = list(ctx.resolved.foot_indices)
        flags = np.zeros((ctx.positions.shape[0] - 1, ctx.n_joints, 1))
        if not feet:
            return flags

        delta = ctx.positions[1:, feet] - ctx.positions[:-1, feet]
        # Compare squared quantities: max_speed is a real speed, so squaring it
        # once here avoids a square root per joint per frame.
        slow = (delta**2).sum(axis=-1) <= contact.max_speed**2
        low = np.abs(ctx.positions[1:, feet, 1]) <= contact.max_height
        flags[:, feet, 0] = (slow & low).astype(np.float64)
        return flags

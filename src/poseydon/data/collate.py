"""Padding items into a MotionBatch."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

from poseydon.conditioners.base import CONDITIONERS
from poseydon.core.batch import Cond, Masks, MotionBatch, WindowInfo
from poseydon.data.dataset import Item


def collate(items: Sequence[Item]) -> MotionBatch:
    """Pad to the batch's own maxima, not to a global constant.

    The reference pads every skeleton to ``MAX_JOINTS = 143``, so a 28-joint
    flamingo spends roughly eighty percent of every attention matrix on joints
    that do not exist. Padding to the batch maximum costs nothing and removes
    that entirely.
    """
    if not items:
        raise ValueError("cannot collate an empty batch")

    spec = items[0].spec
    if any(item.spec != spec for item in items):
        raise ValueError("every item in a batch must share one feature spec")

    batch = len(items)
    max_frames = max(item.features.shape[0] for item in items)
    max_joints = max(item.features.shape[1] for item in items)

    x = np.zeros((batch, max_joints, spec.dim, max_frames), dtype=np.float32)
    frame_mask = torch.zeros(batch, max_frames, dtype=torch.bool)
    joint_mask = torch.zeros(batch, max_joints, dtype=torch.bool)

    for i, item in enumerate(items):
        frames, joints, _ = item.features.shape
        # (T, J, D) -> (J, D, T)
        x[i, :joints, :, :frames] = item.features.transpose(1, 2, 0)
        frame_mask[i, :frames] = True
        joint_mask[i, :joints] = True

    names = list(items[0].cond)
    payloads = {
        name: CONDITIONERS.get(name)().collate([item.cond[name] for item in items], max_joints)
        for name in names
    }

    return MotionBatch(
        x=torch.from_numpy(x),
        spec=spec,
        masks=Masks(frames=frame_mask, joints=joint_mask),
        window=WindowInfo(
            start=torch.tensor([item.start for item in items], dtype=torch.long),
            source_length=torch.tensor(
                [item.source_length for item in items], dtype=torch.long
            ),
        ),
        cond=Cond(payloads),
    )

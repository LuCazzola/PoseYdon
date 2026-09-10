"""Pinning a semantic latent, so a decode follows an instruction from elsewhere."""

from __future__ import annotations

import torch

from poseydon.core.batch import Cond
from poseydon.sampling.base import CONTROLS
from poseydon.sampling.controls.latent_pin import LatentPin


def test_the_pinned_latent_reaches_cond_every_step():
    latent = torch.randn(1, 1, 16)
    control = LatentPin(latent)

    z_t = torch.zeros(1, 4, 3, 8)
    for step in (999, 500, 0):
        _, cond = control.before_step(z_t, torch.tensor([step]), Cond({}))
        assert torch.equal(cond["z_sem"], latent), f"latent not pinned at step {step}"


def test_it_does_not_mutate_the_cond_it_was_given():
    """Controls compose: one that edits its input in place corrupts the next."""
    latent = torch.randn(1, 1, 16)
    original = Cond({"topology": "untouched"})
    _, returned = LatentPin(latent).before_step(torch.zeros(1, 1, 1, 1), torch.tensor([0]), original)

    assert "z_sem" not in original.payloads, "the caller's Cond was mutated"
    assert returned["topology"] == "untouched", "existing payloads must survive"


def test_it_accepts_a_latent_from_a_different_joint_count():
    """The reason this is not `LatentMix`.

    `LatentMix` guards `reference.shape == target.shape`, which is exactly what
    makes cross-rig transfer inexpressible. The semantic latent is pooled over
    joints, so a 40-joint encode is shape-compatible with a 63-joint decode by
    construction -- refusing that would refuse the entire feature.
    """
    latent = torch.randn(1, 1, 16)          # from a 40-joint rig
    z_t = torch.zeros(1, 63, 13, 40)        # decoding onto 63 joints
    _, cond = LatentPin(latent).before_step(z_t, torch.tensor([10]), Cond({}))
    assert cond["z_sem"].shape == latent.shape


def test_it_is_registered_under_its_name():
    assert CONTROLS.get("latent_pin") is LatentPin

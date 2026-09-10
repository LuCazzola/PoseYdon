"""Cross-topology retarget: the test the whole feature rests on."""

from __future__ import annotations

import pytest
import torch

from poseydon.core.batch import Cond
from poseydon.ops.base import OPERATIONS
from poseydon.ops.retarget import Retarget
from poseydon.sampling.controls.latent_pin import LatentPin


class _Latent:
    """A model that encodes to a joint-independent latent, like MoDiffAE."""

    requires = ()

    def __init__(self):
        self.encode_calls = 0
        self.seen_latents = []

    def encode(self, clean, cond, masks=None, temporal_valid=None):
        self.encode_calls += 1
        # Pooled over joints: (T, B, 1, C), independent of the joint count.
        return torch.full((clean.shape[-1], 1, 1, 8), 0.5), {}


class _Recording:
    """A sampler that records the controls it was handed and denoises nothing."""

    def __init__(self):
        self.controls = ()
        self.calls = 0

    def sample(self, model, process, shape, cond, controls=(), **kwargs):
        self.calls += 1
        self.controls = tuple(controls)
        return torch.zeros(shape)


def test_the_content_is_encoded_exactly_once():
    """Encoding per step would be both wrong and 1000x the cost."""
    model, sampler = _Latent(), _Recording()
    content = torch.zeros(1, 40, 3, 60)

    Retarget(content, Cond({})).run(
        model=model,
        process=None,
        sampler=sampler,
        shape=(1, 63, 3, 60),
        cond=Cond({}),
    )

    assert model.encode_calls == 1, (
        f"content encoded {model.encode_calls} times; the latent describes the "
        "content, which does not change as the target is denoised"
    )
    assert sampler.calls == 1
    # The one encode has to actually reach the sampler, or `run` encoded for
    # nothing and the decode is unconditional.
    (pin,) = sampler.controls
    assert isinstance(pin, LatentPin)
    assert torch.equal(pin.latent, torch.full((60, 1, 1, 8), 0.5))


def test_the_output_carries_the_TARGET_joint_count():
    """Encode a 40-joint clip, decode on a 63-joint rig.

    This is the assertion the feature exists for: the semantic latent is pooled
    over joints, so the decode is free to have a different topology.
    """
    model, sampler = _Latent(), _Recording()
    content = torch.zeros(1, 40, 3, 60)  # a 40-joint Flamingo
    target_shape = (1, 63, 3, 60)  # decoded onto a 63-joint Scorpion

    out = Retarget(content, Cond({})).run(
        model=model,
        process=None,
        sampler=sampler,
        shape=target_shape,
        cond=Cond({}),
    )

    assert out.shape == target_shape
    assert out.shape[1] == 63, "the decode must take the TARGET's joint count, not the content's"
    assert out.shape[1] != content.shape[1], "otherwise nothing was retargeted"
    # And the instruction it decoded under came from the 40-joint content: the
    # pinned latent's frame axis is the content's, its joint axis is pooled away.
    (pin,) = sampler.controls
    assert pin.latent.shape == (60, 1, 1, 8)


def test_a_model_without_encode_is_refused():
    """AnyTop has no semantic encoder. Sampling anyway would produce
    unconditional motion that LOOKS like a retarget while ignoring its content
    entirely -- a silent wrong answer, which is worse than a crash.
    """
    with pytest.raises(TypeError, match="encode"):
        Retarget(torch.zeros(1, 4, 3, 8), Cond({})).run(
            model=object(), process=None, sampler=None, shape=(1, 4, 3, 8), cond=Cond({})
        )


def test_it_is_registered_under_its_name():
    assert OPERATIONS.get("retarget") is Retarget

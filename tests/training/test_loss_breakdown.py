"""Per-block and per-quartile loss reporting, and gradient norms.

A single `simple` number averages a position error in metres against a rotation
error in 6D units. It cannot say which block the model is struggling with, and
those cases call for different fixes.
"""

from __future__ import annotations

import pytest
import torch

from poseydon.core.batch import Cond, Masks, MotionBatch, WindowInfo
from poseydon.core.spec import FeatureSpec
from poseydon.losses.base import LOSSES
from poseydon.losses.geodesic import GeodesicLoss

SPEC = FeatureSpec((("ric_pos", 3), ("rot6d", 6), ("local_vel", 3), ("foot_contact", 1)))


def _batch(batch: int = 4, joints: int = 5, frames: int = 7) -> MotionBatch:
    return MotionBatch(
        x=torch.zeros(batch, joints, SPEC.dim, frames),
        spec=SPEC,
        masks=Masks(
            frames=torch.ones(batch, frames, dtype=torch.bool),
            joints=torch.ones(batch, joints, dtype=torch.bool),
        ),
        window=WindowInfo(
            start=torch.zeros(batch, dtype=torch.long),
            source_length=torch.full((batch,), frames, dtype=torch.long),
        ),
        cond=Cond({}),
    )


def test_every_block_is_reported():
    batch = _batch()
    blocks = LOSSES.get("simple")().per_sample_blocks(
        torch.randn_like(batch.x), batch.x, batch, {}
    )
    assert set(blocks) == set(SPEC.names)
    assert all(v.shape == (len(batch),) for v in blocks.values())


def test_the_blocks_sum_to_the_whole():
    """The parts must reconcile with the total, and they SUM to it.

    `element_mask` is (B, J, 1, T), so the divisor counts joints and frames but
    not channels -- `simple` is a sum over channels meaned over real (joint,
    frame) pairs, not a per-element mean. Each block's value is therefore its
    literal contribution to the total. Assuming a channel-weighted mean instead
    is wrong by 68% on this fixture, which is how the docstring got corrected.

    A breakdown that does not reconcile with its own total is worse than none:
    it invites conclusions about a quantity nobody is optimising.
    """
    batch = _batch()
    loss = LOSSES.get("simple")()
    x0_hat = torch.randn_like(batch.x)

    whole = loss.per_sample(x0_hat, batch.x, batch, {})
    blocks = loss.per_sample_blocks(x0_hat, batch.x, batch, {})
    torch.testing.assert_close(sum(blocks.values()), whole)


def test_a_block_only_reflects_its_own_error():
    """Error confined to rot6d must not show up in ric_pos."""
    batch = _batch()
    x0_hat = batch.x.clone()
    x0_hat[:, :, SPEC.slice("rot6d"), :] += 3.0

    blocks = LOSSES.get("simple")().per_sample_blocks(x0_hat, batch.x, batch, {})
    assert float(blocks["rot6d"].mean()) > 8.0
    for quiet in ("ric_pos", "local_vel", "foot_contact"):
        assert float(blocks[quiet].mean()) == 0.0, f"{quiet} picked up rot6d's error"


def test_a_term_that_cannot_separate_says_so():
    """Geodesic measures an angle; there is no per-block decomposition of it."""
    batch = _batch()
    assert GeodesicLoss().per_sample_blocks(
        torch.randn_like(batch.x), batch.x, batch, {}
    ) is None


# --- gradient norms ------------------------------------------------------


class _Recorder:
    """A logger that keeps whatever the module logs."""

    def __init__(self) -> None:
        self.seen: dict[str, float] = {}

    def __call__(self, metrics, *args, **kwargs) -> None:
        self.seen.update({k: float(v) for k, v in metrics.items()})


def _module():
    from poseydon.process.gaussian import GaussianDiffusion
    from poseydon.training.lightning import MotionLitModule

    return MotionLitModule(
        model=_TinyModel(),
        process=GaussianDiffusion(num_steps=10, schedule="cosine", parameterization="x0"),
        losses=[("simple", 1.0, LOSSES.get("simple")())],
    )


class _TinyModel(torch.nn.Module):
    """Two named submodules, so the per-group split has something to split."""

    def __init__(self) -> None:
        super().__init__()
        self.semantic_encoder = torch.nn.Linear(4, 4)
        self.stochastic_decoder = torch.nn.Linear(4, 4)

    def forward(self, z_t, t, cond, masks=None):
        from poseydon.models.base import Prediction

        return Prediction(out=z_t, aux={})


def test_gradient_norms_are_reported_per_module_and_in_total(monkeypatch):
    module = _module()
    for parameter in module.parameters():
        parameter.grad = torch.ones_like(parameter)

    recorder = _Recorder()
    monkeypatch.setattr(module, "log_dict", recorder)
    monkeypatch.setattr(type(module), "trainer", property(lambda self: _Trainer(None)))
    module.on_before_optimizer_step(optimizer=None)

    assert "grad_norm/total" in recorder.seen
    assert {"grad_norm/semantic_encoder", "grad_norm/stochastic_decoder"} <= set(recorder.seen)
    # The total is the norm of the concatenation, not the sum of the parts.
    parts = [recorder.seen[k] for k in recorder.seen if k.startswith("grad_norm/") and
             k not in ("grad_norm/total", "grad_norm/clipped")]
    expected = float(torch.tensor(parts).pow(2).sum().sqrt())
    assert recorder.seen["grad_norm/total"] == pytest.approx(expected, rel=1e-5)


class _Trainer:
    def __init__(self, clip) -> None:
        self.gradient_clip_val = clip


def test_the_clipped_flag_tracks_the_configured_ceiling(monkeypatch):
    """Its running mean is the fraction of steps hitting the clip.

    Reported PRE-clip on purpose: afterwards every norm sits at or below the
    ceiling by construction, so how often clipping engages -- the thing worth
    knowing -- is unrecoverable.
    """
    module = _module()
    for parameter in module.parameters():
        parameter.grad = torch.full_like(parameter, 10.0)

    for clip, expected in ((1.0, 1.0), (1e9, 0.0)):
        recorder = _Recorder()
        monkeypatch.setattr(module, "log_dict", recorder)
        monkeypatch.setattr(type(module), "trainer", property(lambda self, c=clip: _Trainer(c)))
        module.on_before_optimizer_step(optimizer=None)
        assert recorder.seen["grad_norm/clipped"] == expected


def test_nothing_is_logged_before_any_gradient_exists(monkeypatch):
    module = _module()
    recorder = _Recorder()
    monkeypatch.setattr(module, "log_dict", recorder)
    monkeypatch.setattr(type(module), "trainer", property(lambda self: _Trainer(1.0)))
    module.on_before_optimizer_step(optimizer=None)
    assert recorder.seen == {}

"""The crop offset must actually reach the model.

`MotionBatch` has carried `window.start` all along and neither model has ever
seen it: nothing wrote `crop_start` (MoDiffAE's key) or `window_start`
(AnyTop's) into `cond`, so both silently took their defaults and every window
was encoded as if it started at frame 0.
"""

from __future__ import annotations

from pathlib import Path

import torch

from poseydon.core.batch import Cond, Masks, MotionBatch, WindowInfo
from poseydon.core.spec import FeatureSpec
from poseydon.models.base import Denoiser, Prediction

SPEC = FeatureSpec((("ric_pos", 3),))


class _IdentityProcess:
    """The process is not what this test is about: no noise, timestep zero."""

    def sample_t(self, batch_size, device="cpu"):
        return torch.zeros(batch_size, dtype=torch.long, device=device)

    def corrupt(self, z0, t, noise):
        return z0

    def to_z0(self, pred, z_t, t):
        return pred


def _make_batch(starts):
    batch = len(starts)
    joints, frames = 4, 8
    return MotionBatch(
        x=torch.zeros(batch, joints, SPEC.dim, frames),
        spec=SPEC,
        masks=Masks(
            frames=torch.ones(batch, frames, dtype=torch.bool),
            joints=torch.ones(batch, joints, dtype=torch.bool),
        ),
        window=WindowInfo(
            start=torch.tensor(starts, dtype=torch.long),
            source_length=torch.full((batch,), frames, dtype=torch.long),
        ),
        cond=Cond({}),
    )


def _run_one_step(task_module, model, batch):
    """One `compute_losses` call with a spy model and no real loss terms."""
    task = task_module.MotionTask(model=model, process=_IdentityProcess(), losses=[])
    return task.compute_losses(batch)


def test_the_task_injects_crop_start_from_the_window():
    from poseydon.training import task as task_module

    seen = {}

    class _Spy(Denoiser):
        def forward(self, z_t, t, cond, masks=None):
            seen["cond"] = cond
            return Prediction(out=torch.zeros_like(z_t), aux={})

    batch = _make_batch(starts=[7, 13, 0])
    _run_one_step(task_module, _Spy(), batch)

    assert "crop_start" in seen["cond"], "the offset never reached the model"
    assert torch.equal(seen["cond"]["crop_start"], torch.tensor([7, 13, 0]))


def test_anytop_reads_crop_start_not_window_start():
    """The two models must agree on the key, or wiring one silently leaves the
    other on its default.
    """
    from poseydon.models import anytop

    # Strip comments first, then look for the word at all. The earlier form of
    # this test matched the exact string `cond.get("window_start")`, which a
    # reintroduced read could sidestep just by changing the quoting or by
    # coexisting with the correct key (`cond.get("crop_start") or
    # cond.get('window_start')`). Ignoring comments is what lets the check be
    # blunt: the fix leaves a comment naming the dead key it replaced, and that
    # prose is documentation rather than a second lookup.
    code = "\n".join(
        line.split("#")[0] for line in Path(anytop.__file__).read_text().splitlines()
    )
    assert "window_start" not in code, "the dead key is being read again"
    assert 'cond.get("crop_start")' in code

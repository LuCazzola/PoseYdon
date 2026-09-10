"""The retarget-validation callback's contract.

Four of these assert properties that no amount of "it ran and wrote files"
would catch: that the callback does not touch the training RNG stream, that it
restores `train()` mode, that it fires on the declared schedule and nowhere
else, and that a failure inside it does not take the run down with it.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from poseydon.training.validation import RetargetPair, RetargetValidation


class _Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))


def _trainer(step: int = 0, logger=None) -> SimpleNamespace:
    return SimpleNamespace(global_step=step, logger=logger)


def _module(model) -> SimpleNamespace:
    return SimpleNamespace(task=SimpleNamespace(model=model))


class _Logger:
    def __init__(self) -> None:
        self.metrics: list[tuple[dict, int]] = []
        self.experiment = None

    def log_metrics(self, metrics, step=None) -> None:
        self.metrics.append((dict(metrics), step))


PAIRS = [
    {"content": {"rig": "Flamingo", "action": "onelegbent"}, "target": "Scorpion"},
    {"content": {"rig": "Coyote", "action": "attack3"}, "target": "Crab"},
]


def _callback(tmp_path, monkeypatch, results, **kwargs) -> RetargetValidation:
    """A callback whose retarget and metrics are stubbed, so the CONTRACT is what is tested."""
    from poseydon.training import validation as module

    calls: list = []

    def fake_run_retarget(**kw):
        calls.append(kw)
        # Consume the callback's own generator, so a test that asserts the
        # training stream is untouched is actually testing something.
        torch.rand(4, generator=kw["generator"], device=kw["generator"].device)
        if isinstance(results, Exception):
            raise results
        return SimpleNamespace(npz="a.npz", bvh="a.bvh", mp4="a.mp4")

    monkeypatch.setattr(module, "run_retarget", fake_run_retarget)
    monkeypatch.setattr(
        module,
        "retarget_metrics",
        lambda *a, **k: {"foot_skate": 1.0, "ik_residual_mean": 2.0},
    )
    callback = RetargetValidation(
        pairs=PAIRS,
        dataset=object(),
        process=object(),
        sampler=object(),
        out_dir=tmp_path,
        **kwargs,
    )
    callback.calls = calls  # type: ignore[attr-defined]
    return callback


def test_the_training_rng_stream_is_untouched(tmp_path, monkeypatch):
    """The property that makes a resumed run reproduce an uninterrupted one.

    If validation drew from the global stream, the training trajectory after
    step 25000 would depend on whether the run was resumed at 25000 -- a
    divergence that shows up as an unexplained loss discontinuity days later.
    """
    callback = _callback(tmp_path, monkeypatch, results=None)
    model = _Model()

    torch.manual_seed(1234)
    expected = [torch.rand(1).item() for _ in range(3)]

    torch.manual_seed(1234)
    first = torch.rand(1).item()
    callback.run(_trainer(step=25_000), _module(model))
    rest = [torch.rand(1).item() for _ in range(2)]

    assert [first, *rest] == expected


def test_it_restores_train_mode(tmp_path, monkeypatch):
    callback = _callback(tmp_path, monkeypatch, results=None)
    model = _Model().train()
    callback.run(_trainer(), _module(model))
    assert model.training, "the model was left in eval(), silencing dropout for the rest of the run"


def test_a_model_already_in_eval_stays_there(tmp_path, monkeypatch):
    callback = _callback(tmp_path, monkeypatch, results=None)
    model = _Model().eval()
    callback.run(_trainer(), _module(model))
    assert not model.training


def test_a_failure_does_not_take_the_run_down(tmp_path, monkeypatch):
    """Three days of training must not be lost to one transient upload error."""
    callback = _callback(tmp_path, monkeypatch, results=RuntimeError("network went away"))
    model = _Model().train()
    with pytest.warns(RuntimeWarning, match="was skipped"):
        callback.run(_trainer(step=50_000), _module(model))
    assert model.training, "train() must be restored even when the firing failed"


def test_it_fires_on_the_schedule_and_nowhere_else(tmp_path, monkeypatch):
    callback = _callback(tmp_path, monkeypatch, results=None, every_n_steps=10)
    model = _Model()
    module = _module(model)

    fired: list[int] = []
    callback.run = lambda trainer, mod: fired.append(trainer.global_step)  # type: ignore[method-assign]
    for step in range(1, 26):
        callback.on_train_batch_end(_trainer(step), module, None, None, 0)
    assert fired == [10, 20]


def test_run_at_start_fires_once_before_training(tmp_path, monkeypatch):
    callback = _callback(tmp_path, monkeypatch, results=None, run_at_start=True)
    model = _Model()
    fired: list[int] = []
    callback.run = lambda trainer, mod: fired.append(trainer.global_step)  # type: ignore[method-assign]
    callback.on_train_start(_trainer(step=0), _module(model))
    assert fired == [0]


def test_run_at_start_false_stays_quiet(tmp_path, monkeypatch):
    callback = _callback(tmp_path, monkeypatch, results=None, run_at_start=False)
    fired: list[int] = []
    callback.run = lambda trainer, mod: fired.append(trainer.global_step)  # type: ignore[method-assign]
    callback.on_train_start(_trainer(step=0), _module(_Model()))
    assert fired == []


def test_it_logs_every_metric_per_pair_and_their_mean(tmp_path, monkeypatch):
    logger = _Logger()
    callback = _callback(tmp_path, monkeypatch, results=None)
    callback.run(_trainer(step=25_000, logger=logger), _module(_Model()))

    assert len(logger.metrics) == 1
    scalars, step = logger.metrics[0]
    assert step == 25_000
    for slug in ("Flamingo_onelegbent__to__Scorpion", "Coyote_attack3__to__Crab"):
        assert scalars[f"retarget/{slug}/foot_skate"] == 1.0
        assert scalars[f"retarget/{slug}/ik_residual_mean"] == 2.0
    assert scalars["retarget/mean/foot_skate"] == 1.0
    assert scalars["retarget/mean/ik_residual_mean"] == 2.0


def test_the_generator_is_seeded_from_the_step(tmp_path, monkeypatch):
    """Reproducible across a resume: the same step must retarget identically."""
    callback = _callback(tmp_path, monkeypatch, results=None)
    model = _Model()
    callback.run(_trainer(step=25_000), _module(model))
    first = [c["generator"].initial_seed() for c in callback.calls]  # type: ignore[attr-defined]
    callback.calls.clear()  # type: ignore[attr-defined]
    callback.run(_trainer(step=25_000), _module(model))
    second = [c["generator"].initial_seed() for c in callback.calls]  # type: ignore[attr-defined]
    assert first == second == [25_000, 25_000]


def test_every_pair_is_retargeted(tmp_path, monkeypatch):
    callback = _callback(tmp_path, monkeypatch, results=None)
    callback.run(_trainer(), _module(_Model()))
    seen = [(c["content"], c["target_rig"]) for c in callback.calls]  # type: ignore[attr-defined]
    assert seen == [("Flamingo/onelegbent", "Scorpion"), ("Coyote/attack3", "Crab")]


def test_a_pair_parses_from_the_config_shape():
    pair = RetargetPair.parse(
        {"content": {"rig": "Goat", "action": "headbutt"}, "target": "Raptor"}
    )
    assert pair.content == "Goat/headbutt"
    assert pair.slug == "Goat_headbutt__to__Raptor"


def test_a_zero_interval_is_refused():
    with pytest.raises(ValueError, match="every_n_steps"):
        RetargetValidation(
            pairs=PAIRS, dataset=None, process=None, sampler=None, out_dir=".", every_n_steps=0
        )

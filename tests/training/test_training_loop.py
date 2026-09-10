"""The run itself, pinned.

Not a test of whether training works -- the GPU smoke run is that. This file
pins the handful of values that DEFINE the 600 000-step run and that would
otherwise be free to drift silently: the schedule and the interval it is
stepped on, the checkpoint cadence, what the logger is told, and the recipe
carried by `configs/train.yaml`. A wrong value here costs three days of GPU
time and looks exactly like a right one while it is burning them.
"""

from __future__ import annotations

from pathlib import Path

import lightning as L
import pytest
import torch
from hydra import compose, initialize_config_dir
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint

from poseydon.models.base import Denoiser, Prediction
from poseydon.training import build
from poseydon.training.build import build_callbacks, build_datamodule, build_logger, run_name
from poseydon.training.lightning import MotionLitModule

CONFIG_DIR = Path("configs")


@pytest.fixture
def config():
    with initialize_config_dir(config_dir=str(CONFIG_DIR.resolve()), version_base=None):
        return compose(config_name="train")


class _Model(Denoiser):
    """Something with parameters, so AdamW has something to optimize."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)

    def forward(self, z_t, t, cond, masks=None):
        return Prediction(out=z_t, aux={})


def _module() -> MotionLitModule:
    return MotionLitModule(model=_Model(), process=object(), losses=[])


# --------------------------------------------------------------------------
# The schedule
# --------------------------------------------------------------------------


def test_configure_optimizers_pairs_adamw_with_steplr():
    optimizers = _module().configure_optimizers()
    assert isinstance(optimizers["optimizer"], torch.optim.AdamW)
    scheduler = optimizers["lr_scheduler"]["scheduler"]
    assert isinstance(scheduler, torch.optim.lr_scheduler.StepLR)
    assert scheduler.step_size == 10_000
    assert scheduler.gamma == pytest.approx(0.99)


def test_the_schedule_is_stepped_per_optimizer_step_not_per_epoch():
    """The difference is a factor of about a thousand.

    An "epoch" here is one arbitrary pass over a weighted sampler -- roughly
    1900 steps of the 600 000 this run takes. Stepped per epoch, `step_size=10
    000` would fire perhaps thirty times in three days instead of sixty, and
    the learning rate would end the run essentially where it started. Nothing
    about the loss curve would look wrong.
    """
    assert _module().configure_optimizers()["lr_scheduler"]["interval"] == "step"


def test_the_scheduler_is_attached_to_the_optimizer_it_schedules():
    optimizers = _module().configure_optimizers()
    scheduler = optimizers["lr_scheduler"]["scheduler"]
    assert scheduler.optimizer is optimizers["optimizer"]


# --------------------------------------------------------------------------
# Checkpoints and callbacks
# --------------------------------------------------------------------------


def test_the_checkpoint_callback_saves_every_25000_steps_and_keeps_them_all(tmp_path):
    checkpoint = next(c for c in build_callbacks(tmp_path) if isinstance(c, ModelCheckpoint))
    assert checkpoint.save_last is True
    assert checkpoint._every_n_train_steps == 25_000
    assert checkpoint.save_top_k == -1
    # `recipe_from_checkpoint` looks in the checkpoint's directory and one
    # above it, which is what makes `<run>/checkpoints/` the layout, not a
    # preference.
    assert Path(checkpoint.dirpath) == tmp_path / "checkpoints"


def test_the_learning_rate_is_logged_per_step(tmp_path):
    monitor = next(
        c for c in build_callbacks(tmp_path) if isinstance(c, LearningRateMonitor)
    )
    assert monitor.logging_interval == "step"


# --------------------------------------------------------------------------
# The logger
# --------------------------------------------------------------------------


def test_a_missing_api_key_costs_no_run(tmp_path, monkeypatch, config):
    """Unlogged training beats no training."""
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    assert build_logger(config, tmp_path) is None


def test_the_logger_is_told_the_project_the_entity_and_the_run(tmp_path, monkeypatch, config):
    """`entity` comes from the environment because it is account-specific:
    `entity: poseydon` fails against the live API -- `poseydon` is the PROJECT.
    """
    from lightning.pytorch import loggers

    seen = {}

    class _Spy:
        def __init__(self, **kwargs):
            seen.update(kwargs)

    monkeypatch.setattr(loggers, "WandbLogger", _Spy)
    monkeypatch.setenv("WANDB_API_KEY", "not-a-real-key")
    monkeypatch.setenv("WANDB_ENTITY", "some-entity")

    assert isinstance(build_logger(config, tmp_path), _Spy)
    assert seen["project"] == "poseydon"
    assert seen["entity"] == "some-entity"
    assert seen["name"] == run_name(config)
    assert seen["save_dir"] == str(tmp_path)


def test_an_unset_entity_is_none_rather_than_the_empty_string(tmp_path, monkeypatch, config):
    """Compose forwards `WANDB_ENTITY: ${WANDB_ENTITY:-}`, so an unset entity
    arrives as `""`, which wandb would take as a literal entity name.
    """
    from lightning.pytorch import loggers

    seen = {}
    monkeypatch.setattr(loggers, "WandbLogger", lambda **kwargs: seen.update(kwargs))
    monkeypatch.setenv("WANDB_API_KEY", "not-a-real-key")
    monkeypatch.setenv("WANDB_ENTITY", "")

    build_logger(config, tmp_path)
    assert seen["entity"] is None


def test_the_run_name_carries_the_choices_that_distinguish_one_run(config):
    name = run_name(config)
    assert "modiffae" in name
    assert "5" in name  # the attention pool's virtual joints
    assert "16" in name  # batch size
    assert "128" in name  # latent width


# --------------------------------------------------------------------------
# Balanced sampling, from the config
# --------------------------------------------------------------------------


def test_balanced_sampling_is_on_by_default_and_can_be_turned_off(config, monkeypatch):
    """It was unreachable from Hydra: `build_datamodule` never forwarded the
    key, so `balanced=false` composed cleanly and changed nothing.
    """
    monkeypatch.setattr(build, "build_dataset", lambda config, split=None: object())

    assert build_datamodule(config).balanced is True

    config.data.balanced = False
    assert build_datamodule(config).balanced is False


# --------------------------------------------------------------------------
# The run recipe, in the config file that IS the run
# --------------------------------------------------------------------------


def test_the_trainer_config_is_the_measured_run(config):
    trainer = config.trainer
    # 600 000 steps at a measured E[0.431 s/step] = 71.8 h, at batch 16.
    assert trainer.max_steps == 600_000
    # Measured 1.4x over 32-true at 13.2 GB peak. Not fp16: bf16 needs no
    # loss scaling.
    assert trainer.precision == "bf16-mixed"
    assert trainer.devices == 1
    assert trainer.gradient_clip_val == 1.0


def test_the_data_config_is_the_measured_run(config):
    assert config.data.batch_size == 16
    assert config.data.num_workers == 4
    assert config.data.window.length == 40


def test_the_model_is_the_attention_pooled_modiffae_of_the_paper_command(config):
    assert config.model._target_ == "poseydon.models.modiffae.MoDiffAE"
    assert config.model.d_model == 128
    assert config.model.n_layers_semantic == 4
    assert config.model.n_layers_stochastic == 4
    assert config.model.n_heads == 4
    assert config.model.ff_size == 512
    assert config.model.n_virtual_joints == 5
    assert config.model.temporal_window == 31


def test_the_losses_are_the_reference_weights_and_foot_skate_is_not_one(config):
    """`--lambda_geo 1.0` is passed explicitly by the paper's command;
    `--lambda_fs` defaults to 0 and is never passed. Foot skate stays a
    measured diagnostic, not a training term.
    """
    assert dict(config.losses) == {"simple": 1.0, "geodesic": 1.0}


def test_the_learning_rate_is_the_reference_one(config):
    assert float(config.optimizer.learning_rate) == pytest.approx(1e-4)


# --------------------------------------------------------------------------
# `poseydon train`, without training anything
# --------------------------------------------------------------------------


class _Spec:
    dim = 13


class _Dataset:
    spec = _Spec()


class _DataModule:
    train_dataset = _Dataset()
    has_validation = False


def _stub_training(monkeypatch) -> dict:
    """Everything `_train` does except the training."""
    captured: dict = {}

    class _Trainer:
        def __init__(self, **kwargs):
            captured["trainer"] = kwargs

        def fit(self, module, datamodule=None, ckpt_path=None):
            captured["fit"] = {"module": module, "ckpt_path": ckpt_path}

    monkeypatch.setattr(L, "Trainer", _Trainer)
    monkeypatch.setattr(build, "build_datamodule", lambda config: _DataModule())
    monkeypatch.setattr(build, "build_module", lambda config, feature_dim: torch.nn.Linear(2, 2))
    return captured


def test_train_writes_the_recipe_into_the_run_directory_it_creates(tmp_path, monkeypatch):
    """`--out` is a directory OF runs. The recipe belongs to one run, beside
    the checkpoints that run wrote -- which is also the only place
    `recipe_from_checkpoint`'s fallback ever looks.
    """
    from poseydon.cli import main

    captured = _stub_training(monkeypatch)
    assert main(["train", "--out", str(tmp_path), "--quiet", "trainer=debug"]) == 0

    runs = list(tmp_path.iterdir())
    assert len(runs) == 1, runs
    run_dir = runs[0]
    assert (run_dir / "recipe.yaml").is_file()

    checkpoint = next(
        c for c in captured["trainer"]["callbacks"] if isinstance(c, ModelCheckpoint)
    )
    assert Path(checkpoint.dirpath) == run_dir / "checkpoints"
    assert Path(captured["trainer"]["default_root_dir"]) == run_dir


def test_train_passes_resume_through_to_fit(tmp_path, monkeypatch):
    from poseydon.cli import main

    captured = _stub_training(monkeypatch)
    main([
        "train", "--out", str(tmp_path), "--quiet", "--resume", "/tmp/last.ckpt", "trainer=debug"
    ])
    assert captured["fit"]["ckpt_path"] == "/tmp/last.ckpt"


def test_train_hands_the_trainer_the_configured_run(tmp_path, monkeypatch):
    from poseydon.cli import main

    captured = _stub_training(monkeypatch)
    main(["train", "--out", str(tmp_path), "--quiet"])
    trainer = captured["trainer"]
    assert trainer["max_steps"] == 600_000
    assert trainer["precision"] == "bf16-mixed"

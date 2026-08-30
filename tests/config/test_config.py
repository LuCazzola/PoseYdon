"""The shipped configs must compose and build real objects."""

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from poseydon.models.anytop import AnyTop
from poseydon.process import FlowMatching, GaussianDiffusion
from poseydon.training.build import build_losses, build_model, build_process

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"


def load(overrides=()):
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        return compose(config_name="train", overrides=list(overrides))


def test_default_config_composes():
    config = load()
    assert config.features == ["ric_pos", "rot6d", "local_vel", "foot_contact"]
    assert config.conditioners == ["topology", "tpose", "norm_stats"]
    assert set(config.losses) == {"simple", "geodesic", "footskate"}


def test_process_is_swappable_from_the_command_line():
    assert isinstance(build_process(load(["process=gaussian"])), GaussianDiffusion)
    assert isinstance(build_process(load(["process=flow"])), FlowMatching)


def test_features_are_swappable_from_the_command_line():
    # The property the storage design exists for, expressed as one override.
    config = load(["features=[rot6d,local_vel]"])
    assert config.features == ["rot6d", "local_vel"]


def test_model_is_built_for_the_active_feature_width():
    model = build_model(load(), feature_dim=13)
    assert isinstance(model, AnyTop)
    assert model.feature_dim == 13

    narrow = build_model(load(["features=[rot6d]"]), feature_dim=6)
    assert narrow.feature_dim == 6


def test_model_hyperparameters_are_overridable():
    model = build_model(load(["model.d_model=64", "model.n_layers=3"]), feature_dim=13)
    assert model.d_model == 64
    assert len(model.layers) == 3


def test_losses_build_with_their_weights():
    losses = build_losses(load().losses)
    assert {name for name, _, _ in losses} == {"simple", "geodesic", "footskate"}
    weights = {name: weight for name, weight, _ in losses}
    assert weights["geodesic"] == pytest.approx(0.1)


def test_unknown_loss_name_is_rejected():
    with pytest.raises(KeyError, match="unknown loss"):
        build_losses(OmegaConf.create({"nonsense": 1.0}))


def test_trainer_presets_exist():
    assert load(["trainer=debug"]).trainer.max_steps == 5
    assert load(["trainer=default"]).trainer.max_steps > 5

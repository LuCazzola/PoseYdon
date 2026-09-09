"""A checkpoint must carry what it was trained with.

`poseydon sample` recomposes its config from `configs/sample.yaml`, so nothing
tied a checkpoint to the `features:` it was trained under: sampling under a
different schema produced the right shapes with the wrong meaning, silently.
The mismatch test below is the one that matters; the round trip is plumbing.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from poseydon.training.recipe import (
    RECIPE_FILE,
    check_recipe,
    read_recipe,
    recipe_from_checkpoint,
    recipe_of,
    write_recipe,
)

CONFIG_DIR = Path("configs")


@pytest.fixture
def config():
    with initialize_config_dir(config_dir=str(CONFIG_DIR.resolve()), version_base=None):
        return compose(config_name="train")


def test_write_recipe_creates_the_file_and_read_recipe_round_trips(config, tmp_path):
    path = write_recipe(config, tmp_path / "run")
    assert path == tmp_path / "run" / RECIPE_FILE
    assert path.is_file()
    assert read_recipe(path) == recipe_of(config)


def test_the_recorded_features_are_exactly_the_configured_ones(config, tmp_path):
    recorded = read_recipe(write_recipe(config, tmp_path / "run"))
    assert list(recorded.features) == list(config.features)


def test_everything_the_brief_names_is_recorded(config, tmp_path):
    recorded = read_recipe(write_recipe(config, tmp_path / "run"))
    for path in ("features", "conditioners", "losses", "model", "process",
                 "data.window", "dataset.reduce_tolerance", "dataset.normalize",
                 "reconstruct"):
        assert OmegaConf.select(recorded, path, default="__missing__") != "__missing__", path
    assert recorded.reconstruct == "positions_ik"


def test_a_recipe_that_disagrees_with_the_active_config_is_rejected(config, tmp_path):
    """The whole point: a checkpoint may not be sampled under another schema."""
    trained_with = config.copy()
    trained_with.features = ["ric_pos", "rot6d"]
    path = write_recipe(trained_with, tmp_path / "run")

    with pytest.raises(ValueError) as excinfo:
        check_recipe(config, read_recipe(path))

    message = str(excinfo.value)
    assert "features" in message
    assert "ric_pos" in message and "rot6d" in message  # what it was trained with
    assert "local_vel" in message and "foot_contact" in message  # what is configured


def test_an_agreeing_recipe_passes(config, tmp_path):
    check_recipe(config, read_recipe(write_recipe(config, tmp_path / "run")))


def test_the_recipe_travels_in_the_checkpoint(config, tmp_path):
    """`sample` reads the checkpoint it was given, not a directory it guessed."""
    state = {"hyper_parameters": {"recipe": OmegaConf.to_container(recipe_of(config))}}
    recorded = recipe_from_checkpoint(state, tmp_path / "run" / "checkpoints" / "last.ckpt")
    assert recorded is not None
    check_recipe(config, recorded)


def test_a_checkpoint_without_a_recipe_falls_back_to_the_run_directory(config, tmp_path):
    """Checkpoints live in `<run>/checkpoints/`, the recipe in `<run>/`."""
    write_recipe(config, tmp_path / "run")
    recorded = recipe_from_checkpoint({}, tmp_path / "run" / "checkpoints" / "last.ckpt")
    assert recorded is not None
    assert list(recorded.features) == list(config.features)


def test_a_checkpoint_with_no_recipe_anywhere_is_not_an_error(config, tmp_path):
    assert recipe_from_checkpoint({}, tmp_path / "loose.ckpt") is None


def test_the_module_carries_the_recipe_into_its_checkpoints(config):
    """Lightning writes `hparams` to `hyper_parameters`, so a checkpoint moved
    away from its run directory stays self-describing.
    """
    from poseydon.models.base import Denoiser, Prediction
    from poseydon.training.lightning import MotionLitModule

    class _Model(Denoiser):
        def forward(self, z_t, t, cond, masks=None):
            return Prediction(out=z_t, aux={})

    module = MotionLitModule(
        model=_Model(),
        process=object(),
        losses=[],
        recipe=OmegaConf.to_container(recipe_of(config), resolve=True),
    )
    # A path with no `recipe.yaml` anywhere near it: what comes back can only
    # have come from the checkpoint itself.
    recorded = recipe_from_checkpoint(
        {"hyper_parameters": module.hparams}, Path("/nonexistent/last.ckpt")
    )
    assert recorded is not None
    check_recipe(config, recorded)

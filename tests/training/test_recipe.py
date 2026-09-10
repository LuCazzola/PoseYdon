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


# ---------------------------------------------------------------------------
# The `dataset.*` keys, and the message a mismatch prints
# ---------------------------------------------------------------------------


def test_the_dataset_keys_record_the_real_build_values_not_null(config, tmp_path):
    """`configs/dataset/` was composed into neither config, so these two
    recorded `null` on both sides and compared `None != None` forever -- two
    of the six entries in the list that decides whether a checkpoint is safe
    to sample, contributing nothing while looking exactly like the four that
    do.
    """
    recorded = read_recipe(write_recipe(config, tmp_path / "run"))
    assert recorded.dataset.reduce_tolerance == 1e-8
    assert [entry["name"] for entry in recorded.dataset.normalize] == [
        "ric_pos", "rot6d", "local_vel", "foot_contact"
    ]


def test_the_sample_config_composes_the_same_dataset_group(config):
    """One-sided composition is worse than none: `train` would record real
    values, `sample` would recompose `None`, and every legitimate sample would
    be rejected.
    """
    with initialize_config_dir(config_dir=str(CONFIG_DIR.resolve()), version_base=None):
        sampling = compose(config_name="sample")
    assert sampling.dataset.reduce_tolerance == config.dataset.reduce_tolerance
    assert sampling.dataset.normalize == config.dataset.normalize


def test_the_read_path_uses_the_configured_tolerance(config, monkeypatch):
    """Recording a value the dataset then ignores would be a second guard that
    reads as working while checking nothing.
    """
    from poseydon.training import build

    seen = {}
    monkeypatch.setattr(build.CorpusIndex, "load", staticmethod(lambda path: object()))
    monkeypatch.setattr(build, "MotionDataset", lambda **kwargs: seen.update(kwargs))

    config.dataset.reduce_tolerance = 1e-5
    build.build_dataset(config)
    assert seen["reduce_tolerance"] == 1e-5


def test_a_model_mismatch_names_the_override_that_fixes_it(config, tmp_path):
    """`configs/sample.yaml` defaults to `model: anytop` while training runs
    `modiffae`, so this is the message the first person to sample a run
    checkpoint will read. Two nine-key dicts printed in full is not an answer.
    """
    path = write_recipe(config, tmp_path / "run")
    with initialize_config_dir(config_dir=str(CONFIG_DIR.resolve()), version_base=None):
        sampling = compose(config_name="sample")

    with pytest.raises(ValueError) as excinfo:
        check_recipe(sampling, read_recipe(path))

    message = str(excinfo.value)
    assert "model=modiffae" in message, message
    assert "model._target_" in message, message
    # A different group is one fact, not nine differing fields.
    assert "n_layers" not in message, message


def test_a_single_field_difference_is_reported_as_a_single_field(config, tmp_path):
    """A whole-node comparison is the right trade -- an allowlist would go
    stale silently -- but the reader should not have to diff two dicts by eye.
    """
    trained_with = config.copy()
    trained_with.model.dropout = 0.2
    path = write_recipe(trained_with, tmp_path / "run")

    with pytest.raises(ValueError) as excinfo:
        check_recipe(config, read_recipe(path))

    message = str(excinfo.value)
    assert "model.dropout" in message
    assert "d_model" not in message, "fields that agree should not be printed"


# ---------------------------------------------------------------------------
# The two CLI call sites, which nothing pinned
# ---------------------------------------------------------------------------


def test_sample_refuses_a_checkpoint_trained_under_another_schema(config, tmp_path):
    """The whole point of the recipe is that a CLI invocation fails loudly.
    Run as a subprocess: this is the entry point, argument parsing included.
    """
    import subprocess
    import sys

    import torch

    trained_with = config.copy()
    trained_with.features = ["ric_pos", "rot6d"]
    checkpoint = tmp_path / "trained_elsewhere.ckpt"
    torch.save(
        {"hyper_parameters": {"recipe": OmegaConf.to_container(recipe_of(trained_with))}},
        checkpoint,
    )

    result = subprocess.run(
        [sys.executable, "-m", "poseydon.cli", "sample",
         "--checkpoint", str(checkpoint), "--out", str(tmp_path / "samples")],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 1, result.stdout + result.stderr
    assert "features" in result.stderr
    # It must fail before anything is built, and before anything is written.
    assert not (tmp_path / "samples").exists()


def test_train_writes_a_recipe_that_sample_accepts(config, tmp_path, monkeypatch):
    """The `_train` call site, end to end minus the training: what it writes
    must be exactly what `_sample` will later check against.
    """
    from poseydon.cli import main
    from tests.training.test_training_loop import _stub_training

    _stub_training(monkeypatch)
    assert main(["train", "--out", str(tmp_path), "--quiet", "trainer=debug"]) == 0

    written = list(tmp_path.glob("*/" + RECIPE_FILE))
    assert len(written) == 1, written
    check_recipe(config, read_recipe(written[0]))

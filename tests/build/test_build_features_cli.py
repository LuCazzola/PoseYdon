"""The stage-2 runner and its `dataset` config group (task 6).

`build_config` is the one place this task adds: composing
`configs/dataset/truebones.yaml` and turning it into a `BuildConfig`. Its job
is narrow but easy to get silently wrong -- `normalize:` is a list of plain
mappings in YAML, and only `_convert_="all"` makes `hydra.utils.instantiate`
turn each into a real `BlockPolicy` rather than folding it back into a
dict-like `DictConfig` with no error at all (see `build_config`'s docstring in
`scripts/build_features.py`). These tests catch that regression by checking
`isinstance`, not just that the values round-trip.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from scripts.build_features import build_config

from poseydon.build.corpus import PerRigDirectory
from poseydon.build.names import Humanize
from poseydon.build.pipeline import BuildConfig, build_stats
from poseydon.data.normalize import BlockPolicy
from tests.conftest import CORPUS

CONFIG_DIR = Path("configs")
SCHEMA = ("ric_pos", "rot6d", "local_vel", "foot_contact")


def _skip_without_corpus() -> None:
    if not (CORPUS / "clips").is_dir():
        pytest.skip("prepared corpus not present -- run stage 1")


def test_the_dataset_config_group_composes():
    with initialize_config_dir(config_dir=str(CONFIG_DIR.resolve()), version_base=None):
        cfg = compose(config_name="dataset/truebones")

    assert cfg.dataset.root == "data/truebones"
    assert list(cfg.dataset.schema) == list(SCHEMA)
    assert len(cfg.dataset.normalize) == 4


def test_schema_and_normalize_round_trip_into_a_build_config(tmp_path):
    config = build_config("configs", "dataset/truebones", out_root=tmp_path, relabel=False)

    assert isinstance(config, BuildConfig)
    assert config.schema == SCHEMA
    assert isinstance(config.schema, tuple)

    assert len(config.normalize) == 4
    for entry in config.normalize:
        # The regression this guards: without `_convert_="all"`, a frozen
        # dataclass `_target_` result gets folded back into a `DictConfig` by
        # OmegaConf's structured-config boxing -- so `isinstance` is the
        # assertion that actually catches it; equality against a dict-shaped
        # thing would pass either way.
        assert isinstance(entry, BlockPolicy), f"{entry!r} did not instantiate into BlockPolicy"
    # Pinned deliberately: this is the shipped normalization recipe, and a
    # silent change to it changes what every checkpoint means. `scale` moved
    # from "joint_block" to "channel" on measurement -- a joint's quietest
    # channels turned out to be smooth real motion (lag-1 autocorrelation
    # +0.941), not the noise pooling was introduced to suppress. See the
    # comment in configs/dataset/truebones.yaml.
    assert config.normalize[0] == BlockPolicy("ric_pos", center=True, scale="channel")
    assert config.normalize[3] == BlockPolicy("foot_contact", center=False, scale="none")


def test_corpus_and_names_also_instantiate_to_real_types(tmp_path):
    config = build_config("configs", "dataset/truebones", out_root=tmp_path, relabel=False)
    assert isinstance(config.corpus, PerRigDirectory)
    assert isinstance(config.names, Humanize)


def test_stats_only_runs_only_the_stats_pass(tmp_path):
    """No clip/skeleton file appears when the output directory starts empty."""
    _skip_without_corpus()

    result = subprocess.run(
        [
            sys.executable,
            "scripts/build_features.py",
            "--out-root",
            str(tmp_path),
            "--rigs",
            "Goat",
            "--stats-only",
        ],
        capture_output=True,
        text=True,
        cwd=str(Path.cwd()),
        check=False,
    )
    assert result.returncode == 0, result.stderr

    assert (tmp_path / "rigs" / "Goat" / "stats.npz").is_file()
    assert not (tmp_path / "clips").exists()
    assert not (tmp_path / "rigs" / "Goat" / "skeleton.npz").exists()
    assert not (tmp_path / "index.jsonl").exists()


def test_an_unknown_scale_mode_raises_before_any_file_is_written(tmp_path):
    """A config with a typo'd `scale:` must fail loudly, not write a corrupt stats.npz."""
    _skip_without_corpus()

    bad_policy = (BlockPolicy("ric_pos", center=True, scale="not-a-real-mode"),)
    config = BuildConfig(
        root=CORPUS,
        out=tmp_path,
        schema=SCHEMA,
        corpus=PerRigDirectory(),
        names=Humanize(),
        normalize=bad_policy,
    )

    with pytest.raises(ValueError, match="unknown scale mode"):
        build_stats(config, "Goat")

    # `build_stats` makes its output directory before fitting -- that mkdir is
    # not a build artefact -- but must not write `stats.npz` itself.
    written = [p for p in tmp_path.rglob("*") if p.is_file()]
    assert written == [], f"no file may be written once fitting fails, found {written}"


def test_zero_rigs_written_exits_non_zero(tmp_path):
    """A rig name that matches nothing writes zero clips and must fail the run."""
    _skip_without_corpus()

    result = subprocess.run(
        [
            sys.executable,
            "scripts/build_features.py",
            "--out-root",
            str(tmp_path),
            "--rigs",
            "NoSuchRig",
        ],
        capture_output=True,
        text=True,
        cwd=str(Path.cwd()),
        check=False,
    )
    assert result.returncode != 0
    assert "no clips written" in result.stderr

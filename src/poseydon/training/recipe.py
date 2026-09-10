"""What a checkpoint was trained with, recorded beside it.

`poseydon sample` recomposes its config from `configs/sample.yaml`, so a
checkpoint sampled under a different `features:` than it was trained with
produces silent garbage -- the right shapes, the wrong meaning. The recipe
makes that mismatch loud. Plan B's retarget path reads it too, which is why it
lives here and not in the CLI.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

RECIPE_FILE = "recipe.yaml"

# How a generated feature tensor is turned back into a skeleton. Training
# configs carry no `reconstruct:` block -- that choice belongs to sampling --
# so the recipe records the method this representation is meant to be inverted
# with unless the config names one of its own.
DEFAULT_RECONSTRUCT = "positions_ik"

# Dotted paths into the composed config, recorded verbatim. `dataset.*` is the
# stage-2 build group -- `configs/dataset/truebones.yaml`, composed into BOTH
# `train.yaml` and `sample.yaml` so these record the tolerance and the block
# policy the corpus was built under rather than a pair of nulls that compared
# equal forever. What they record is the DECLARED policy, not provenance read
# back out of `rigs/<Rig>/stats.npz`: they catch a policy that changed since
# the checkpoint was trained, not a config edited without a rebuild.
RECORDED: tuple[str, ...] = (
    "features",
    "conditioners",
    "losses",
    "model",
    "process",
    "data.window",
    "dataset.reduce_tolerance",
    "dataset.normalize",
)

# The subset whose disagreement changes what the tensor MEANS: the feature
# schema and its width, the conditioning the model was given, the model and
# process themselves, and the reduction and normalization the values were
# measured under. `losses`, `data.window` and `reconstruct` are recorded for
# the record only -- sampling legitimately configures no losses, a different
# window length and a different reconstruction than training did.
#
# Adding a path here invalidates every recipe written before it: a key the
# recorded side lacks compares `<absent>` against a real value and raises. A
# new entry needs a missing-on-one-side escape, or a migration.
CHECKED: tuple[str, ...] = (
    "features",
    "conditioners",
    "model",
    "process",
    "dataset.reduce_tolerance",
    "dataset.normalize",
)


def _plain(value: Any) -> Any:
    """Containers, not config nodes, so the YAML is self-contained and compares."""
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    return value


def recipe_of(config: DictConfig) -> DictConfig:
    """The mapping recorded for a run, drawn from its composed config."""
    recipe = OmegaConf.create({})
    for path in RECORDED:
        OmegaConf.update(
            recipe, path, _plain(OmegaConf.select(config, path)), force_add=True
        )
    reconstruct = OmegaConf.select(config, "reconstruct.name")
    OmegaConf.update(
        recipe, "reconstruct", str(reconstruct or DEFAULT_RECONSTRUCT), force_add=True
    )
    return recipe


def write_recipe(config: DictConfig, run_dir: str | Path) -> Path:
    """Write `<run_dir>/recipe.yaml` and return its path."""
    path = Path(run_dir) / RECIPE_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(recipe_of(config), path)
    return path


def read_recipe(path: str | Path) -> DictConfig:
    """Load a recipe from a file, or from `recipe.yaml` inside a directory."""
    path = Path(path)
    if path.is_dir():
        path = path / RECIPE_FILE
    recipe = OmegaConf.load(path)
    if not isinstance(recipe, DictConfig):
        raise TypeError(f"{path} is not a recipe mapping")
    return recipe


def recipe_from_checkpoint(
    state: Mapping[str, Any], checkpoint: str | Path
) -> DictConfig | None:
    """The recipe a checkpoint carries, else the one written beside its run.

    Checkpoints live in `<run>/checkpoints/`, the recipe in `<run>/`, so the
    fallback looks one directory up as well. Returns None when neither exists:
    a checkpoint from before recipes were recorded is unverifiable, not wrong.
    """
    recorded = state.get("hyper_parameters", {}).get("recipe")
    if recorded is not None:
        return OmegaConf.create(dict(recorded))

    directory = Path(checkpoint).parent
    for candidate in (directory / RECIPE_FILE, directory.parent / RECIPE_FILE):
        if candidate.is_file():
            return read_recipe(candidate)
    return None


#: Placeholder for a key one side has and the other does not, so a missing key
#: reads as missing rather than as a literal `None` someone configured.
ABSENT = "<absent>"


def _rows(path: str, was: Any, now: Any) -> list[tuple[str, Any, Any]]:
    """One row per field that actually differs.

    `model` and `process` are whole config nodes, and printing two nine-key
    dicts in full leaves the reader to diff them by eye -- for a message whose
    entire purpose is to be acted on. Recursing one level turns that into
    `model.dropout: trained with 0.2, configured 0.1`.
    """
    if isinstance(was, Mapping) and isinstance(now, Mapping):
        # A different `_target_` is a different config GROUP, not nine
        # differing fields: listing the fields of two unrelated models buries
        # the one line the reader can act on.
        if was.get("_target_") != now.get("_target_"):
            return [(f"{path}._target_", was.get("_target_", ABSENT),
                     now.get("_target_", ABSENT))]
        keys = list(was) + [key for key in now if key not in was]
        return [
            (f"{path}.{key}", was.get(key, ABSENT), now.get(key, ABSENT))
            for key in keys
            if was.get(key, ABSENT) != now.get(key, ABSENT)
        ]
    return [(path, was, now)]


def _override_hint(rows: list[tuple[str, Any, Any]]) -> str:
    """The command-line override that would fix a wrong config GROUP.

    `configs/sample.yaml` defaults to `model: anytop` while the run trains
    `modiffae`, so the first person to sample a run checkpoint meets this
    message. Telling them `model=modiffae` is the whole difference between a
    guard that reports a problem and one that answers it.
    """
    hints = []
    for path, was, _ in rows:
        group, _, field = path.rpartition(".")
        if field == "_target_" and isinstance(was, str):
            # `poseydon.models.modiffae.MoDiffAE` -> `configs/model/modiffae.yaml`
            hints.append(f"{group}={was.rsplit('.', 2)[-2]}")
    if not hints:
        return ""
    return "\nThis is a different config group. Re-run with: " + " ".join(hints)


def check_recipe(active: DictConfig, recorded: DictConfig) -> None:
    """Raise unless a recorded recipe agrees with the configuration in force.

    `active` may be a composed config or another recipe: both carry the checked
    keys at the same paths.
    """
    rows: list[tuple[str, Any, Any]] = []
    for path in CHECKED:
        was = _plain(OmegaConf.select(recorded, path, default=ABSENT))
        now = _plain(OmegaConf.select(active, path, default=ABSENT))
        if was != now:
            rows.extend(_rows(path, was, now))
    if not rows:
        return

    lines = "\n".join(
        f"  {path}:\n    trained with: {was!r}\n    configured:   {now!r}"
        for path, was, now in rows
    )
    raise ValueError(
        "this checkpoint was trained under a different configuration, and using "
        "it under this one would produce the right shapes with the wrong "
        f"meaning:\n{lines}{_override_hint(rows)}\n"
        "Override the config to match the recipe, or point at a checkpoint "
        "trained under it."
    )

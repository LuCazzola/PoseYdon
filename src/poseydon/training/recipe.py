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
# stage-2 build group: absent from a composed train config today, recorded as
# null, and picked up automatically if that group is ever composed in.
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


def check_recipe(active: DictConfig, recorded: DictConfig) -> None:
    """Raise unless a recorded recipe agrees with the configuration in force.

    `active` may be a composed config or another recipe: both carry the checked
    keys at the same paths.
    """
    disagreements = [
        (path, _plain(OmegaConf.select(recorded, path)), _plain(OmegaConf.select(active, path)))
        for path in CHECKED
        if _plain(OmegaConf.select(recorded, path)) != _plain(OmegaConf.select(active, path))
    ]
    if not disagreements:
        return

    lines = "\n".join(
        f"  {path}:\n    trained with: {was!r}\n    configured:   {now!r}"
        for path, was, now in disagreements
    )
    raise ValueError(
        "this checkpoint was trained under a different configuration, and using "
        "it under this one would produce the right shapes with the wrong "
        f"meaning:\n{lines}\n"
        "Override the config to match the recipe, or point at a checkpoint "
        "trained under it."
    )

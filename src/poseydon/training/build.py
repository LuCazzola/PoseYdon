"""Turn a composed config into the objects that train.

Every component is a registry name: a config entry names its class by
``type:`` (`poseydon.core.registry.Registry.build`), and list slots whose
entries take no arguments -- features, conditioners, losses -- by the bare
name. A config's ``custom_imports`` are imported first, so a user's own
registered class can be named the same way.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from poseydon.augment.base import AUGMENTATIONS
from poseydon.build.index import CorpusIndex
from poseydon.core.registry import import_custom
from poseydon.data.dataset import MotionDataset
from poseydon.data.window import WINDOWS
from poseydon.losses.base import LOSSES, LossTerm
from poseydon.models.base import MODELS, Denoiser
from poseydon.process.base import PROCESSES, Process
from poseydon.training.lightning import MotionDataModule, MotionLitModule
from poseydon.training.recipe import recipe_of


def build_losses(config: DictConfig) -> list[tuple[str, float, LossTerm]]:
    return [(name, float(weight), LOSSES.get(name)()) for name, weight in config.items()]


def build_dataset(config: DictConfig, split: str | None = None) -> MotionDataset:
    import_custom(config.get("custom_imports"))
    import_custom(config.data.get("custom_imports"))
    index = CorpusIndex.load(Path(config.data.paths.index))
    # Clips this run must never see. Applied here rather than inside
    # `MotionDataset` because this is the one place a config becomes a dataset,
    # and both `train` and `sample` come through it.
    # Gated on the CONTENTS, not on the block: the default block is present and
    # empty, and `without` on two empty sets is a full copy of the index for
    # nothing.
    block = OmegaConf.select(config, "data.filter.blacklist") or {}
    skeletons, clips = block.get("skeletons") or (), block.get("clips") or ()
    if skeletons or clips:
        index = index.without(skeletons=skeletons, clips=clips)
    return MotionDataset(
        index=index,
        root=config.data.paths.root,
        manifest_dir=config.data.paths.rigs,
        features=list(config.features.blocks),
        window=WINDOWS.build(config.data.loader.window),
        conditioners=list(config.conditioners),
        augmentations=AUGMENTATIONS.build_all(config.get("augmentations")),
        split=split,
        seed=config.seed,
        contact=config.data.labels.get("contact"),
    )


def build_datamodule(config: DictConfig) -> MotionDataModule:
    return MotionDataModule(
        train=build_dataset(config, split="train"),
        batch_size=config.data.loader.batch_size,
        num_workers=config.data.loader.num_workers,
        # Balanced sampling is the run's default (parity spec §10) but it was
        # unreachable from Hydra: this key was never forwarded, so
        # `data.loader.balanced=false` composed cleanly and changed nothing.
        balanced=bool(config.data.loader.get("balanced", True)),
    )


def run_name(config: DictConfig) -> str:
    """What distinguishes one run from another: model, pooling, batch, width.

    Used for the wandb run name and for the run's own directory, so a run's
    checkpoints, its recipe and its dashboard all carry the same name.
    """
    parts = [str(OmegaConf.select(config, "model.type") or "model")]
    # Only MoDiffAE has a pooling choice; AnyTop has no latent to pool into,
    # so it gets no pooling segment rather than a misleading one.
    virtual = OmegaConf.select(config, "model.n_virtual_joints")
    if virtual is not None:
        parts.append(f"attn{virtual}" if virtual else "mean")
    parts.append(f"b{config.data.loader.batch_size}")
    parts.append(f"d{OmegaConf.select(config, 'model.d_model')}")
    return "-".join(parts)


def build_logger(config: DictConfig, run_dir: Path):
    """wandb, or nothing at all.

    Absent `WANDB_API_KEY` this returns None and training proceeds unlogged --
    a missing key must never cost a run. The entity comes from the environment
    because it is account-specific: `entity: poseydon` fails against the live
    API ("the provided API key cannot access this resource"); `poseydon` is the
    PROJECT.
    """
    if not os.environ.get("WANDB_API_KEY"):
        return None
    from lightning.pytorch.loggers import WandbLogger

    return WandbLogger(
        project="poseydon",
        # Compose forwards `${WANDB_ENTITY:-}`, so an unset entity arrives as
        # the empty string, which wandb would read as a literal name.
        entity=os.environ.get("WANDB_ENTITY") or None,
        name=run_name(config),
        save_dir=str(run_dir),
    )


def build_validation(config: DictConfig, run_dir: Path, dataset) -> object | None:
    """The retarget-validation callback, or None when the run does not declare one.

    Gated on the `validation:` block, so a config without it behaves exactly as
    a run did before this existed. `DDPM` rather than the config's sampler: a
    validation retarget must solve the whole trajectory, and the sampler a run
    trains under says nothing about how it should be sampled.
    """
    block = config.get("validation")
    if not block or not block.get("pairs"):
        return None
    from poseydon.sampling.diffusion import DDPM
    from poseydon.training.validation import RetargetValidation

    # `is None`, not `or`: an empty list stays empty and is refused downstream.
    configured = block.get("guidance_scales")
    scales = [1.0] if configured is None else [float(g) for g in configured]
    # Any scale but 1 samples the unconditional branch, which needs the null
    # latent only `cond_drop > 0` trains. Asked of a model without one, every
    # such scale raised and took its whole pair down with it, the 1.0 baseline
    # included, so the default config observed nothing. Keep what this model
    # can answer, and say what was dropped.
    if float(config.model.get("cond_drop", 0.0)) == 0.0 and any(g != 1.0 for g in scales):
        warnings.warn(
            f"validation.guidance_scales {scales}: model.cond_drop is 0, so the "
            "model has no unconditional branch to guide with; sampling only 1.0",
            stacklevel=2,
        )
        scales = [1.0]

    return RetargetValidation(
        pairs=OmegaConf.to_container(block.pairs, resolve=True),
        dataset=dataset,
        process=build_process(config),
        sampler=DDPM(),
        out_dir=Path(run_dir) / "retargets",
        every_n_steps=int(block.get("every_n_steps", 25_000)),
        run_at_start=bool(block.get("run_at_start", True)),
        seed=int(config.get("seed", 0)),
        # `[1.0]` unless asked for. Scale 1 is the IDENTITY -- the model
        # sampled as trained -- so this is one unwrapped conditional retarget
        # per pair, which is all a run with no trained null latent can measure
        # and the baseline every other scale is read against.
        #
        # MEASURED on this project's own 25k checkpoint (cond_drop 0.1): a
        # scale of 2.5 made 9 of 10 retarget metrics WORSE and 1.25 made 7 of
        # 10 worse, so guidance is not a free win to be switched on. It is a
        # sweep to be WATCHED -- those numbers come from a checkpoint whose null
        # latent had seen gradient from ~10% of samples, and the question is
        # whether the ordering survives training, which one scalar per firing
        # cannot answer and several can.
        guidance_scales=scales,
    )


def build_callbacks(run_dir: Path, logger: object = None, validation=None) -> list:
    """Checkpoints on a step interval, and the learning rate beside them.

    `every_n_train_steps=25000` matches the reference's `save_interval`;
    `save_top_k=-1` keeps every one of them, because on a 600k-step run with no
    validation metric there is no "top" to select by. `save_last` is what
    `--resume` picks up. The `<run>/checkpoints/` layout is not a preference:
    it is where `recipe_from_checkpoint` looks for the recipe beside a
    checkpoint that carries none.
    """
    from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint

    callbacks = [
        ModelCheckpoint(
            dirpath=str(Path(run_dir) / "checkpoints"),
            save_last=True,
            every_n_train_steps=25_000,
            save_top_k=-1,
        )
    ]
    # LearningRateMonitor has nowhere to write without a logger, and Lightning
    # does not shrug: it raises MisconfigurationException at `on_train_start`.
    # Adding it unconditionally therefore turned "no WANDB_API_KEY" from
    # "train unlogged" into "do not train at all" -- the opposite of what the
    # missing key is supposed to cost. It fails at startup rather than mid-run,
    # but a run that refuses to start is still a run lost.
    if logger:
        callbacks.append(LearningRateMonitor(logging_interval="step"))
    if validation is not None:
        callbacks.append(validation)
    return callbacks


def build_model(config: DictConfig, feature_dim: int) -> Denoiser:
    """The model needs the feature width, which the representation decides."""
    return MODELS.build(config.model, feature_dim=feature_dim)


def build_process(config: DictConfig) -> Process:
    return PROCESSES.build(config.process)


def build_module(config: DictConfig, feature_dim: int) -> MotionLitModule:
    return MotionLitModule(
        model=build_model(config, feature_dim),
        process=build_process(config),
        losses=build_losses(config.losses),
        learning_rate=float(config.optimizer.learning_rate),
        weight_decay=float(config.optimizer.weight_decay),
        recipe=OmegaConf.to_container(recipe_of(config), resolve=True),
    )

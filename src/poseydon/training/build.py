"""Turn a composed config into the objects that train.

Two wiring styles, matched to how each kind of component is used. Singular
choices -- model, process, window -- are Hydra config groups carrying a
``_target_``. List-valued ones -- features, conditioners, losses -- are short
registry names, because a full ``_target_`` block per list entry is noise and
because ``poseydon list`` should be able to enumerate them.
"""

from __future__ import annotations

import os
from pathlib import Path

from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from poseydon.build.index import CorpusIndex
from poseydon.data.dataset import MotionDataset
from poseydon.losses.base import LOSSES, LossTerm
from poseydon.models.base import Denoiser
from poseydon.process.base import Process
from poseydon.training.lightning import MotionDataModule, MotionLitModule
from poseydon.training.recipe import recipe_of


def build_losses(config: DictConfig) -> list[tuple[str, float, LossTerm]]:
    return [(name, float(weight), LOSSES.get(name)()) for name, weight in config.items()]


def build_dataset(config: DictConfig, split: str | None = None) -> MotionDataset:
    index = CorpusIndex.load(Path(config.data.index))
    return MotionDataset(
        index=index,
        root=config.data.root,
        manifest_dir=config.data.manifests,
        features=list(config.features),
        window=instantiate(config.data.window),
        conditioners=list(config.conditioners),
        augmentations=[instantiate(entry) for entry in config.get("augmentations", [])],
        split=split,
        seed=config.seed,
        # The tolerance stage 2 reduced the corpus under. Read from the config
        # rather than left on the dataset's own default so that the value the
        # recipe RECORDS is the value the read path actually applies -- a
        # recorded number the dataset ignores would be a second guard that
        # reads as working while checking nothing.
        # `is None`, not `or`: a configured `reduce_tolerance: 0.0` is falsy, and
        # `or` would silently substitute 1e-8 while the recipe went on recording
        # 0.0 -- exactly the recorded-but-ignored value the comment above rejects.
        reduce_tolerance=float(
            _tolerance if (_tolerance := OmegaConf.select(config, "dataset.reduce_tolerance"))
            is not None else 1e-8
        ),
    )


def build_datamodule(config: DictConfig) -> MotionDataModule:
    return MotionDataModule(
        train=build_dataset(config, split="train"),
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers,
        # Balanced sampling is the run's default (parity spec §10) but it was
        # unreachable from Hydra: this key was never forwarded, so
        # `data.balanced=false` composed cleanly and changed nothing.
        balanced=bool(config.data.get("balanced", True)),
    )


def run_name(config: DictConfig) -> str:
    """What distinguishes one run from another: model, pooling, batch, width.

    Used for the wandb run name and for the run's own directory, so a run's
    checkpoints, its recipe and its dashboard all carry the same name.
    """
    target = str(OmegaConf.select(config, "model._target_") or "model")
    parts = [target.rsplit(".", 1)[-1].lower()]
    # Only MoDiffAE has a pooling choice; AnyTop has no latent to pool into,
    # so it gets no pooling segment rather than a misleading one.
    virtual = OmegaConf.select(config, "model.n_virtual_joints")
    if virtual is not None:
        parts.append(f"attn{virtual}" if virtual else "mean")
    parts.append(f"b{config.data.batch_size}")
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

    return RetargetValidation(
        pairs=OmegaConf.to_container(block.pairs, resolve=True),
        dataset=dataset,
        process=build_process(config),
        sampler=DDPM(),
        out_dir=Path(run_dir) / "retargets",
        every_n_steps=int(block.get("every_n_steps", 25_000)),
        run_at_start=bool(block.get("run_at_start", True)),
        seed=int(config.get("seed", 0)),
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
    return instantiate(config.model, feature_dim=feature_dim)


def build_process(config: DictConfig) -> Process:
    return instantiate(config.process)


def build_module(config: DictConfig, feature_dim: int) -> MotionLitModule:
    return MotionLitModule(
        model=build_model(config, feature_dim),
        process=build_process(config),
        losses=build_losses(config.losses),
        learning_rate=float(config.optimizer.learning_rate),
        weight_decay=float(config.optimizer.weight_decay),
        recipe=OmegaConf.to_container(recipe_of(config), resolve=True),
    )

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
        reduce_tolerance=float(OmegaConf.select(config, "dataset.reduce_tolerance") or 1e-8),
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


def build_callbacks(run_dir: Path) -> list:
    """Checkpoints on a step interval, and the learning rate beside them.

    `every_n_train_steps=25000` matches the reference's `save_interval`;
    `save_top_k=-1` keeps every one of them, because on a 600k-step run with no
    validation metric there is no "top" to select by. `save_last` is what
    `--resume` picks up. The `<run>/checkpoints/` layout is not a preference:
    it is where `recipe_from_checkpoint` looks for the recipe beside a
    checkpoint that carries none.
    """
    from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint

    return [
        ModelCheckpoint(
            dirpath=str(Path(run_dir) / "checkpoints"),
            save_last=True,
            every_n_train_steps=25_000,
            save_top_k=-1,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]


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

"""Turn a composed config into the objects that train.

Two wiring styles, matched to how each kind of component is used. Singular
choices -- model, process, window -- are Hydra config groups carrying a
``_target_``. List-valued ones -- features, conditioners, losses -- are short
registry names, because a full ``_target_`` block per list entry is noise and
because ``poseydon list`` should be able to enumerate them.
"""

from __future__ import annotations

from pathlib import Path

from hydra.utils import instantiate
from omegaconf import DictConfig

from poseydon.data.dataset import MotionDataset
from poseydon.ingest.index import CorpusIndex
from poseydon.losses.base import LOSSES, LossTerm
from poseydon.models.base import Denoiser
from poseydon.process.base import Process
from poseydon.training.lightning import MotionDataModule, MotionLitModule


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
    )


def build_datamodule(config: DictConfig) -> MotionDataModule:
    return MotionDataModule(
        train=build_dataset(config, split="train"),
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers,
    )


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
    )

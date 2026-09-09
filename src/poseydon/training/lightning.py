"""Lightning wrappers.

Everything framework-independent lives in :class:`MotionTask`; this module adds
only optimization, logging and checkpointing. Lightning removes what the
reference hand-rolls: manual step counting, a mixed-precision trainer, distributed
setup, resume-filename regex parsing, and an ``ml_platforms`` module reached via
``eval()`` on a command-line string.
"""

from __future__ import annotations

from collections.abc import Sequence

import lightning as L
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from poseydon.data.collate import collate
from poseydon.data.dataset import MotionDataset
from poseydon.data.sampler import balanced_weights
from poseydon.losses.base import LossTerm
from poseydon.models.base import Denoiser
from poseydon.process.base import Process
from poseydon.training.task import MotionTask


class MotionLitModule(L.LightningModule):
    """Optimization and logging around a :class:`MotionTask`."""

    def __init__(
        self,
        model: Denoiser,
        process: Process,
        losses: Sequence[tuple[str, float, LossTerm]],
        learning_rate: float = 1e-4,
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__()
        self.task = MotionTask(model=model, process=process, losses=losses)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self._checked = False

    def training_step(self, batch, batch_index: int) -> torch.Tensor:
        # Validate the whole configuration against the first real batch, so a
        # mismatch costs a second rather than an epoch.
        if not self._checked:
            self.task.setup_checks(batch)
            self._checked = True

        losses = self.task.compute_losses(batch)
        for name, value in losses.items():
            self.log(f"train/{name}", value, prog_bar=(name == "total"), batch_size=len(batch))
        return losses["total"]

    def validation_step(self, batch, batch_index: int) -> torch.Tensor:
        losses = self.task.compute_losses(batch)
        for name, value in losses.items():
            self.log(f"val/{name}", value, batch_size=len(batch))
        return losses["total"]

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )


class MotionDataModule(L.LightningDataModule):
    """Wraps :class:`MotionDataset` with the padding collate."""

    def __init__(
        self,
        train: MotionDataset,
        val: MotionDataset | None = None,
        batch_size: int = 8,
        num_workers: int = 0,
        balanced: bool = True,
    ) -> None:
        super().__init__()
        self.train_dataset = train
        self.val_dataset = val
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.balanced = balanced

    def _loader(self, dataset: MotionDataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            collate_fn=collate,
            drop_last=False,
        )

    def _worker_init(self, worker_id: int) -> None:
        info = torch.utils.data.get_worker_info()
        if info is not None:
            info.dataset.set_worker_seed(worker_id)

    def train_dataloader(self) -> DataLoader:
        sampler = None
        if self.balanced:
            weights = balanced_weights(self.train_dataset)
            sampler = WeightedRandomSampler(
                weights=torch.as_tensor(weights, dtype=torch.double),
                num_samples=len(self.train_dataset),
                replacement=True,
            )
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            # `shuffle` and `sampler` are mutually exclusive; the sampler
            # already randomizes.
            shuffle=sampler is None,
            sampler=sampler,
            num_workers=self.num_workers,
            collate_fn=collate,
            # A short final batch changes the padded joint count and the loss
            # scale for one step in every epoch, for no benefit.
            drop_last=True,
            worker_init_fn=self._worker_init if self.num_workers else None,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        if self.val_dataset is None:
            raise RuntimeError(
                "no validation dataset was configured; the Trainer should have "
                "been told to skip validation with limit_val_batches=0"
            )
        return self._loader(self.val_dataset, shuffle=False)

    @property
    def has_validation(self) -> bool:
        return self.val_dataset is not None

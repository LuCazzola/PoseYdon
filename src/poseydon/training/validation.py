"""Retargeting, watched during training.

Training loss says the model denoises; it says nothing about whether a Flamingo's
semantics land on a Scorpion. That is the task, and it is only visible by doing
it. Every `every_n_steps` this decodes three fixed pairs onto their target rigs,
writes `.npz` + `.bvh` + `.mp4` for each, and logs six scalars per pair.

Four properties are load-bearing, and each has bitten this kind of callback:

* **It never kills the run.** The whole firing is wrapped. Losing one
  observation is cheap; losing three days to a transient upload error is not.
* **Its randomness is its own.** A dedicated `torch.Generator` seeded from the
  step, so validation is reproducible across a resume and never consumes the
  training RNG stream -- otherwise the training trajectory would depend on
  whether the run was resumed.
* **`train()` is restored.** Leaving the model in `eval()` disables dropout for
  the rest of the run, silently.
* **The artifact is the point.** A video renders but cannot be diffed or opened
  in Blender, and a three-day run is one whose step-25000 output you want next
  week.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lightning as L
import torch

from poseydon.data.dataset import MotionDataset
from poseydon.ops.retarget_run import run_retarget
from poseydon.training.metrics import retarget_metrics

log = logging.getLogger(__name__)

#: Names of the artifact wandb receives, and the type it is filed under.
ARTIFACT_TYPE = "retarget-validation"


@dataclass(frozen=True)
class RetargetPair:
    """One content clip, decoded onto one target rig."""

    content_rig: str
    content_action: str
    target: str

    @property
    def content(self) -> str:
        return f"{self.content_rig}/{self.content_action}"

    @property
    def slug(self) -> str:
        return f"{self.content_rig}_{self.content_action}__to__{self.target}"

    @classmethod
    def parse(cls, entry: Any) -> RetargetPair:
        """From the config's `{content: {rig, action}, target}` shape.

        Accepts a `RetargetPair` unchanged so a caller can build them directly.
        """
        if isinstance(entry, cls):
            return entry
        content = entry["content"]
        return cls(
            content_rig=str(content["rig"]),
            content_action=str(content["action"]),
            target=str(entry["target"]),
        )


class RetargetValidation(L.Callback):
    """Retarget three fixed pairs on a step interval, logged as wandb artifacts."""

    def __init__(
        self,
        pairs: list,
        dataset: MotionDataset,
        process: Any,
        sampler: Any,
        out_dir: str | Path,
        every_n_steps: int = 25_000,
        run_at_start: bool = True,
        seed: int = 0,
    ) -> None:
        if every_n_steps <= 0:
            raise ValueError(f"every_n_steps must be positive, got {every_n_steps}")
        self.pairs = [RetargetPair.parse(p) for p in pairs]
        self.dataset = dataset
        self.process = process
        self.sampler = sampler
        self.out_dir = Path(out_dir)
        self.every_n_steps = every_n_steps
        self.run_at_start = run_at_start
        self.seed = seed

    # -- Lightning hooks ---------------------------------------------------

    def on_train_start(self, trainer: L.Trainer, module: L.LightningModule) -> None:
        if self.run_at_start:
            self.run(trainer, module)

    def on_train_batch_end(self, trainer, module, outputs, batch, batch_index) -> None:
        step = trainer.global_step
        # `global_step` is 1-based at this hook, so step 0's firing is
        # `on_train_start`'s job and this one must not repeat it.
        if step and step % self.every_n_steps == 0:
            self.run(trainer, module)

    # -- The firing --------------------------------------------------------

    def run(self, trainer: L.Trainer, module: L.LightningModule) -> dict[str, float]:
        """Retarget every pair. Returns the scalars it logged, for tests."""
        step = int(trainer.global_step)
        model = module.task.model
        was_training = model.training
        # A generator of its own, seeded from the step: reproducible across a
        # resume, and it never advances the training RNG stream.
        device = next(model.parameters()).device
        generator = torch.Generator(device=device).manual_seed(self.seed + step)

        scalars: dict[str, float] = {}
        try:
            model.eval()
            with torch.no_grad():
                for pair in self.pairs:
                    scalars.update(self._one(pair, model, step, device, generator, trainer))
            self._log(trainer, scalars, step)
        except Exception as error:  # noqa: BLE001 -- see the class docstring
            warnings.warn(
                f"retarget validation failed at step {step} and was skipped: "
                f"{error!r}. Training continues.",
                RuntimeWarning,
                stacklevel=2,
            )
            log.warning("retarget validation failed at step %d: %r", step, error)
        finally:
            if was_training:
                model.train()
        return scalars

    def _one(self, pair, model, step, device, generator, trainer) -> dict[str, float]:
        out = self.out_dir / f"step-{step}"
        out.mkdir(parents=True, exist_ok=True)
        result = run_retarget(
            model=model,
            process=self.process,
            sampler=self.sampler,
            dataset=self.dataset,
            content=pair.content,
            target_rig=pair.target,
            out_dir=out,
            device=device,
            generator=generator,
        )
        metrics = retarget_metrics(
            result,
            model,
            self.dataset,
            content_rig=pair.content_rig,
            target_rig=pair.target,
            device=device,
        )
        self._upload(trainer, pair, result, step)
        return {f"retarget/{pair.slug}/{k}": float(v) for k, v in metrics.items()}

    def _upload(self, trainer, pair, result, step) -> None:
        """One artifact per firing, plus the mp4 as a playable video."""
        logger = getattr(trainer, "logger", None)
        experiment = getattr(logger, "experiment", None)
        if experiment is None or not hasattr(experiment, "log_artifact"):
            return
        import wandb

        artifact = wandb.Artifact(f"retarget-{experiment.id}", type=ARTIFACT_TYPE)
        for path in (result.npz, result.bvh, result.mp4):
            artifact.add_file(str(path), name=f"{pair.slug}/{Path(path).name}")
        experiment.log_artifact(artifact, aliases=[f"step-{step}"])
        experiment.log(
            {f"retarget/{pair.slug}/video": wandb.Video(str(result.mp4))}, step=step
        )

    def _log(self, trainer, scalars: dict[str, float], step: int) -> None:
        """Per-pair scalars, plus the across-pair mean of each metric.

        The mean is what a 600k-step run is actually watched on: three per-pair
        curves times six metrics is eighteen lines, and no one reads eighteen
        lines.
        """
        if not scalars:
            return
        by_metric: dict[str, list[float]] = {}
        for key, value in scalars.items():
            by_metric.setdefault(key.rsplit("/", 1)[1], []).append(value)
        scalars.update(
            {f"retarget/mean/{name}": sum(v) / len(v) for name, v in by_metric.items()}
        )
        logger = getattr(trainer, "logger", None)
        if logger is not None:
            logger.log_metrics(scalars, step=step)

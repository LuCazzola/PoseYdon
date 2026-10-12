# training

`src/poseydon/training/`, library. From a composed config to a trained
checkpoint: object wiring, the single training task, Lightning wrappers, the
recipe recorded beside each checkpoint, and retargeting watched during training.

## Purpose

`training` owns building the objects that train, the training step, logging and
checkpointing, the recipe that makes sampling under a different config fail
loudly, and the retarget-validation callback with its metrics.

It does **not** own the model, process or losses themselves, the read path
([data](data.md)), or the retarget operation it calls
(`src/poseydon/ops/retarget_run.py`, see [ops](ops.md)).

## Key files and types

| file | read it for |
|---|---|
| `src/poseydon/training/build.py` | `build_dataset`, `build_datamodule`, `build_model`, `build_process`, `build_losses`, `build_module`, `build_logger`, `build_callbacks`, `build_validation`, `run_name` |
| `src/poseydon/training/task.py` | `MotionTask`: one task for every model × process × loss combination |
| `src/poseydon/training/lightning.py` | `MotionLitModule` (optimization, logging), `MotionDataModule` (balanced sampler, per-worker RNG, `drop_last`) |
| `src/poseydon/training/recipe.py` | `write_recipe`, `read_recipe`, `recipe_from_checkpoint`, `check_recipe`; `CHECKED` keys; `DEFAULT_RECONSTRUCT` |
| `src/poseydon/training/validation.py` | `RetargetValidation`, `RetargetPair`: fixed pairs retargeted every `every_n_steps` |
| `src/poseydon/training/metrics.py` | `retarget_metrics` and its parts: foot skate, bone-length drift, IK residual, root trajectory error, latent round-trip, all in bone-length units |

## How it connects

Consumes almost every library package: [data](data.md), [models](models.md),
[process](process.md), [losses](losses.md), [features](features.md),
[build](build.md), [io](io.md), [ingest](ingest.md), and [ops](ops.md) plus
[sampling](sampling.md) for validation.

Produces `runs/<run>/` with checkpoints, `recipe.yaml` and, when validation is
declared, retarget `.npz`/`.bvh`/`.mp4` files logged to wandb as one artifact
per firing. The recipe is read back by `poseydon sample`, `poseydon retarget`
and `ops/retarget_run.py`.

Called by: `cli.py` (`train`, and `sample`/`retarget` for recipe and builders),
`ops/retarget_run.py`, and the IK and parity tools.

## Why it is shaped this way

- **One task.** There's no `DiffusionTask` and `LatentDiffusionTask` to keep
  in step, because latent operation lives in the model's class. Source:
  `src/poseydon/training/task.py` docstring.
- **Two wiring styles.** Singular choices (model, process, window) are Hydra
  groups with `_target_`. Lists (features, conditioners, losses) are registry
  names, so `poseydon list` can enumerate them. Source:
  `src/poseydon/training/build.py` docstring.
- **Lightning replaces what the reference hand-rolls**: step counting, mixed
  precision, distributed setup, resume-filename parsing, and an `eval()` on a
  command-line string. Source: `src/poseydon/training/lightning.py` docstring.
- **A recipe beside every checkpoint.** `sample` recomposes its config from
  `configs/sample.yaml`, so a mismatched `features:` produces the right shapes
  with the wrong meaning. `check_recipe` compares `features`, `conditioners`,
  `model`, `process`, `dataset.reduce_tolerance` and `dataset.normalize`.
  Source: `src/poseydon/training/recipe.py` docstring;
  [spec §8](../../superpowers/specs/2026-09-06-data-pipeline-and-training-parity-design.md#8-the-processing-recipe).
- **Retargeting is watched during training** because training loss says
  nothing about whether a Flamingo's semantics land on a Scorpion. Source:
  `src/poseydon/training/validation.py` docstring;
  [training-run spec](../../superpowers/specs/2026-09-08-training-run-and-retarget-validation-design.md);
  [retargeting guide](../../guides/retargeting.md).
- **Bone-length drift is measured before the IK solve.** On solved output it's
  zero by construction and would report the solver, not the model. Source:
  `src/poseydon/training/metrics.py` docstring.
- **The guidance scales on the dashboard are measured.** At 25k steps, scales
  above 1.0 made retargets monotonically worse, so 1.0 is the baseline, and 0.0
  shows the unconditional branch. Source: comment above `guidance_scales:` in
  `configs/train.yaml`; commit `f48a0fe`.

## Gotchas

- **Validation must never kill the run, use its own RNG, and restore `train()`.**
  The firing is wrapped, a dedicated generator is seeded from the step (so
  resumes are reproducible and the training stream is untouched), and leaving
  `eval()` on would silently disable dropout. Source:
  `src/poseydon/training/validation.py` docstring.
- **A missing wandb key costs logging, not the run** (commit `c690da7`).
- **A key the recorded recipe lacks postdates it, and doesn't fail the check**
  (commit `a9642c1`). Adding to `CHECKED` therefore doesn't break old
  checkpoints, but it doesn't protect them either.
- **Both `train.yaml` and `sample.yaml` must compose the `dataset` group.**
  Otherwise the recipe check rejects every legitimate sample.
- **The data pipeline spec says training never imports `build/`.** It does:
  `build.py` reads `CorpusIndex`, and `metrics.py` reads `RigTransform` for the
  prepared scale. Trust the code.

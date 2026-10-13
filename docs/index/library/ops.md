# ops

`src/poseydon/ops/`, library. The user-facing verbs you run with a trained
model: generate, in-between, blend and retarget, plus the end-to-end retarget
run shared by the CLI and training.

## Purpose

An `Operation` assembles inputs, picks the controls that express the task, and
runs a sampler. `retarget_run.py` owns the one retarget code path that both
`poseydon retarget` and the training-time validation callback use.

It does **not** own trajectory solving or step edits ([sampling](sampling.md)),
the network ([models](models.md)), or writing representations back onto a
character ([export](export.md)).

## Key files and types

| file | read it for |
|---|---|
| `src/poseydon/ops/base.py` | `Operation`; the `OPERATIONS` registry |
| `src/poseydon/ops/generate.py` | `Generate` (`generate`): noise to motion, no controls; `guidance_scale` |
| `src/poseydon/ops/inbetween.py` | `Inbetween` (`inbetween`): fill a gap between known frames by imputation |
| `src/poseydon/ops/blend.py` | `Blend` (`blend`): cross-fade two clips through the semantic latent |
| `src/poseydon/ops/retarget.py` | `Retarget` (`retarget`): encode one clip, decode its latent onto another rig with `LatentPin` |
| `src/poseydon/ops/retarget_run.py` | `run_retarget`, `RetargetResult`, `find_clip`, `find_clip_of_rig` |

## How it connects

Consumes [sampling](sampling.md) samplers and controls, [models](models.md),
[process](process.md), [data](data.md) (a real batch supplies the target rig's
conditioning), [features](features.md) (reduction and reconstruction),
[build](build.md) (index lookups), [io](io.md) (BVH, render) and
`training.recipe`.

Produces generated features and, through `run_retarget`, `.npz` (positions
and latent), `.anim.npz` (the generated animation, exactly as generated), `.bvh` and
`.mp4` files plus a `RetargetResult` that `training/metrics.py` scores.
Selected in `configs/sample.yaml` under `operation:`.

Called by: `cli.py` (`sample`, `retarget`), `training/validation.py`,
`training/metrics.py`.

## Why it is shaped this way

- **A new motion task is usually a control plus a thin operation**, not a new
  sampling loop. Source: `src/poseydon/ops/base.py` docstring;
  [architecture guide](../../guides/architecture.md#sampling-separates-three-things-the-reference-conflates).
- **One retarget path for the CLI and the callback.** What a three-day run
  watches has to be the code that ships, or nobody can reproduce what it
  reported. Source: `src/poseydon/ops/retarget_run.py` docstring; commit
  `2532901`.
- **The decode is as long as the content clip**, because the latent is
  per-frame. Source: `run_retarget` docstring.
- **Reconstruction comes from the recipe** (`DEFAULT_RECONSTRUCT`, read through
  the module at call time), so `sample`, `retarget` and validation agree on what
  a generated tensor means. Source: `src/poseydon/ops/retarget_run.py`
  docstring.
- **Retargeting works across topologies because the latent is pooled over
  joints**: a 40-joint encode decodes onto a 63-joint rig with nothing
  reshaped. Source: [retargeting guide](../../guides/retargeting.md).

## Gotchas

- **The target rig's conditioning comes from one of its clips.**
  `find_clip_of_rig` (and `poseydon sample`'s `skeleton=`) picks the
  first-indexed clip, not the rest clip. Because prepared clips carry per-clip
  rest geometry (see [build](build.md)), output can come back rotated or
  distorted when exported with the rig's facing. This is an open issue as of
  2026-10-12; the proposed fix is to template on `rest_action`.
- **`Blend` and `Retarget` need a latent model** (`MoDiffAE`). They drive
  `encode` and `z_sem`.
- **Sampling a training checkpoint needs `model=modiffae` on the command
  line.** See [models](models.md).

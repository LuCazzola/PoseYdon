# Generation

Code: `src/poseydon/process`, `src/poseydon/models`, `src/poseydon/sampling`, `src/poseydon/ops`, `src/poseydon/conditioners`, `src/poseydon/losses`, `src/poseydon/training`

The reference implementation (`external/neural_motion_blending/`, kept read-only as the
parity reference) folds sampling, editing and the user-facing task into the model.
PoseYdon splits them into separate concepts, each with its own registry.

| Concept | Owns | Knows nothing about |
|---|---|---|
| Process (`src/poseydon/process`) | corruption, the prediction parameterization, `to_z0` back to a clean sample | motion, skeletons, features |
| Model (`src/poseydon/models`) | the denoiser, and its own latent if it has one | how a trajectory is solved |
| Sampler (`src/poseydon/sampling`) | solving the reverse trajectory | what is being edited |
| Control (`src/poseydon/sampling/controls`) | changing state or conditioning at each step | the user-facing task |
| Operation (`src/poseydon/ops`) | the verb: assembling inputs, picking controls, running a sampler | the network |

So a new motion task is usually one control plus a thin operation, not a new sampling
loop (`src/poseydon/ops/base.py`). Blending, for example, is the `latent_mix` control
rather than a method on the model.

## Latent operation is a subclass, not a config switch

A model that works on features inherits pass-through `prepare` and `restore` from
`Denoiser`. A latent model subclasses `LatentDenoiser` and overrides them with its own
encoder and decoder (`src/poseydon/models/base.py`). No shipped model does: both AnyTop
and MoDiffAE are plain `Denoiser`s. MoDiffAE diffuses in feature space, and its semantic
encoder reaches the decoder as conditioning (`src/poseydon/models/modiffae.py`). The config never mentions
autoencoding, and a single `MotionTask` (`src/poseydon/training/task.py`) trains every
combination of model, process and loss.

## Losses always see a clean sample, checked against the spec

Whatever the network predicts (x0, epsilon, v or a velocity field), `Process.to_z0`
hands losses a clean sample, so structural losses work under any process. Each loss term
declares the feature blocks it needs and the space (raw or normalized) it needs them in.
`LossTerm.validate` checks the block names against the active `FeatureSpec` on the first
training step; the space is not checked there, so a raw-space need without the
`norm_stats` conditioner fails later, when the loss first reads the block
(`src/poseydon/losses/base.py`).

## Conditioning is declared

A conditioner owns its whole data path: reading, collating and moving to device. Only
the conditioners listed under `conditioners:` are loaded. How a condition is injected is
left to the model, because injection is genuinely model-specific: AnyTop adds the T-pose
as an extra frame (`src/poseydon/conditioners/base.py`). The batch contract
(`src/poseydon/core/batch.py`) mandates only motion, layout, masks and window origin.

## What is watched in training is what ships

`poseydon retarget` and the training-time `RetargetValidation` callback both call the
single function in `src/poseydon/ops/retarget_run.py`. Its docstring gives the reason:
what a multi-day run reports on has to be the code path that ships, or nobody can
reproduce the report afterwards. The two calls still differ in their arguments: the CLI passes
the recipe's `reconstruct` and a `--guidance-scale`, while the callback passes no
`reconstruct` and loops over `validation.guidance_scales`.

A checkpoint carries the recipe it was trained with (`src/poseydon/training/recipe.py`).
Sampling recomposes its config from `configs/sample.yaml`, and the recipe turns a
`features:` mismatch into an error instead of output with the right shapes and the wrong
meaning.

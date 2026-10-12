# process

`src/poseydon/process/`, library. Generative processes: how a clean sample is
corrupted, what the network is asked to predict, and how to recover a clean
sample from that prediction.

## Purpose

A `Process` owns corruption, prediction parameterization and the inverse
(`to_z0`). It knows nothing about motion, skeletons or features.

It does **not** own the network ([models](models.md)), the reverse-time solver
([sampling](sampling.md)) or the loss terms ([losses](losses.md)).

## Key files and types

| file | read it for |
|---|---|
| `src/poseydon/process/base.py` | `Process`: corruption, parameterization, `to_z0`; `expand_to`; the `PROCESSES` registry |
| `src/poseydon/process/gaussian.py` | `GaussianDiffusion` (`gaussian`), variance-preserving; `linear_betas` (rescaled so step count doesn't move the endpoints) and `cosine_betas` |
| `src/poseydon/process/flow.py` | `FlowMatching` (`flow`): rectified flow, a straight-line interpolant between data and noise |

Selected by `configs/process/{gaussian,flow}.yaml`. Gaussian ships
`parameterization: x0`.

## How it connects

Consumes only tensors and `core`.

Produces noised samples and training targets for `training/task.py`, and the
per-step math the samplers in [sampling](sampling.md) and the
in-betweening control use.

Called by: training (`build`, `task`, `lightning`), sampling, ops, `cli.py`,
and the parity tools.

## Why it is shaped this way

- **`to_z0` keeps structural losses process-agnostic.** Whatever the network
  predicts (x0, epsilon, v or a velocity field), a loss that needs rotations
  receives a clean sample. The reference folds its losses into the diffusion
  class, so its geodesic and foot-skate terms can't be reused by any other
  process. Source: `src/poseydon/process/base.py` docstring.
- **Swappable from config.** `poseydon train process=flow` replaces diffusion
  with flow matching and nothing else changes. Source: `README.md`.
- **Parity.** DDPM and DDIM sampling match the reference to ~3e-6 and ~8e-7
  relative over 200 frames × 100 steps. Source: `README.md`.

## Gotchas

- **The noise schedule is accumulated in float64** (commit `2e62369`). Over 100
  steps, a float32 cumprod moved sampling by 2.5e-5 relative against the
  reference, more than the model port's own error. Keep any new schedule in
  float64.
- **No test covers this package.** It was the weak spot of mutation testing
  even before the suite was cut to the round trip: the whole diffusion core
  survived eleven deliberate mutations, signal and noise swapped among them.
  Source: [testing guide](../../guides/testing.md).

# sampling

`src/poseydon/sampling/`, library. Samplers that solve the reverse trajectory,
step-level controls that edit it, and classifier-free guidance as a wrapping
denoiser.

## Purpose

`sampling` owns how a trajectory is solved (`Sampler`) and what is applied at
each step (`Control`).

It does **not** own the user-facing verb ([ops](ops.md)), the corruption math
([process](process.md)) or the network ([models](models.md)).

## Key files and types

| file | read it for |
|---|---|
| `src/poseydon/sampling/base.py` | `Sampler`, `Control` (`before_step`/`after_step` hooks), `apply_before`/`apply_after`; the `SAMPLERS` and `CONTROLS` registries |
| `src/poseydon/sampling/diffusion.py` | `DDPM` (`ddpm`, ancestral); `DDIM` (`ddim`, deterministic, strided, invertible, with `eta` interpolating toward ancestral) |
| `src/poseydon/sampling/flow.py` | `Euler` (`euler`): first-order ODE integration for flow matching |
| `src/poseydon/sampling/guidance.py` | `GuidedDenoiser`: evaluates the inner model conditionally and unconditionally each step and extrapolates `uncond + w·(cond − uncond)` |
| `src/poseydon/sampling/controls/inbetween.py` | `inbetween`: known-frame imputation |
| `src/poseydon/sampling/controls/latent_pin.py` | `latent_pin`: one semantic latent held fixed for the whole trajectory (cross-rig decode) |
| `src/poseydon/sampling/controls/latent_mix.py` | `latent_mix`: blending between two latents on a schedule (`SCHEDULES` registry) |

## How it connects

Consumes a `Denoiser` from [models](models.md), a `Process` from
[process](process.md), and `MotionBatch`/`Cond` from [core](core.md).

Produces generated feature tensors for [ops](ops.md), which then hands them to
[features](features.md) reconstruction. Selected in `configs/sample.yaml` under
`sampler:` (default `DDIM`, 100 steps).

Called by: ops (every operation, plus `retarget_run`), `training/build.py`,
`cli.py`, and the parity tools.

## Why it is shaped this way

- **Sampler, control and operation are separate.** The reference collapses all
  three; its blending lives inside `MoDiffAE.forward`, which made in-betweening
  unreachable without editing the model. Here in-betweening is a control, and
  the model doesn't change. Source: `src/poseydon/sampling/base.py` docstring;
  [architecture guide](../../guides/architecture.md#sampling-separates-three-things-the-reference-conflates).
- **Guidance wraps the denoiser instead of being a control.** Guidance needs
  two model evaluations per step, and `Control.before_step` is never given the
  model. The original `cfg` control wrote a key no model read, so it changed
  nothing. Source: commit `daee602`.
- **One guidance formula across the whole range.** 0.0 is the unconditional
  branch, 1.0 is the model as trained (the identity), and anything above 1
  extrapolates. Source: comments in `configs/sample.yaml` and
  `configs/train.yaml`.
- **Parity.** DDPM and DDIM match the reference to ~3e-6 and ~8e-7 relative.
  DDIM's `eta` is implemented and tested (commit `5c17b40`). Source: `README.md`.
- **Retargeting is `LatentPin`, a control.** The same mechanism therefore serves
  blending and in-betweening. Source: [retargeting guide](../../guides/retargeting.md).

## Gotchas

- **Guidance above 1.0 measured worse for retargeting** at 25k steps (bone
  drift, IK residual and foot skate all rose monotonically with the scale), so
  the default is 1.0. Read the measurements in `configs/train.yaml` before
  raising it.
- **A scale other than 1.0 costs two forward passes per step**, about 1.4x
  wall time per scale once encode, IK and render are included.
- **Retargeting uses DDPM, not DDIM**, because it has no transition zone to
  anchor, so inversion buys nothing. The sampler is still a caller argument.
  See `src/poseydon/ops/retarget_run.py`.

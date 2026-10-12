# models

`src/poseydon/models/`, library. The denoisers: AnyTop (feature space) and
MoDiffAE (a diffusion autoencoder with a semantic latent), plus the masking
that makes classifier-free guidance and a useful latent possible.

## Purpose

`models` owns network architectures, how each one injects its conditioning,
attention masks (padding, the temporal band), and the learned null embeddings
used when conditioning is hidden.

It does **not** own corruption or parameterization ([process](process.md)), the
trajectory solver ([sampling](sampling.md)), what conditioning is loaded
([conditioners](conditioners.md)), or the training loop ([training](training.md)).

## Key files and types

| file | read it for |
|---|---|
| `src/poseydon/models/base.py` | `Denoiser` (features in, `Prediction` out; pass-through `prepare`/`restore`); `LatentDenoiser` (overrides them); mask helpers `padding_pair_mask`, `temporal_pair_mask`, `apply_temporal_override`; the `MODELS` registry |
| `src/poseydon/models/anytop.py` | `AnyTop` (`anytop`): alternating topology-biased joint attention and temporal attention, with a prepended rest-pose frame |
| `src/poseydon/models/modiffae.py` | `MoDiffAE` (`modiffae`): `SemanticEncoder` (clean clip → per-frame latent `z_sem`), `StochasticDecoder` (denoises conditioned on it), `KLBottleneck`, `InputProcess`/`OutputProcess`; the `JOINT_NAMES` key |
| `src/poseydon/models/modules/masking.py` | `MotionMasker` (hides frames, joints, kinematic chains from the encoder); `ConditionMasker` (hides the latent, for CFG); `chain_ids` |
| `src/poseydon/models/modules/graph_attention.py` | attention with a learned bias per joint-pair relation |
| `src/poseydon/models/modules/graph_backbone.py` | the backbone shared by MoDiffAE's encoder and decoder, including `ConditionAttention` (semantic modulation) |
| `src/poseydon/models/modules/embeddings.py` | positional and timestep embeddings |

## How it connects

Consumes `MotionBatch`/`Cond` and `core.topology` relations from [core](core.md).

Produces `Prediction`s for `training/task.py` and the samplers in
[sampling](sampling.md). MoDiffAE's `encode` and its `z_sem` input are what
[ops](ops.md) retargeting and blending drive. Selected by
`configs/model/{anytop,modiffae}.yaml`; training runs `modiffae`.

Called by: training (`build`, `task`, `lightning`, `metrics`), sampling
(`base`, `diffusion`, `flow`, `guidance`), ops (`base`, `retarget`,
`retarget_run`), conditioners (key constant), `cli.py`, and the parity tools.

## Why it is shaped this way

- **Latent operation by subclassing, not an injected autoencoder.** The task
  body is the same for every model, and config never mentions autoencoding.
  Source: `src/poseydon/models/base.py` docstring;
  [architecture guide](../../guides/architecture.md#latent-models-subclass-they-do-not-get-an-injected-autoencoder).
- **Faithful ports with reference parameter names.** AnyTop's architecture is
  the reference's. MoDiffAE's module and parameter names mirror it exactly, so
  published checkpoints load without remapping, and parity is verified against
  them. What changed is around the models: declared conditioning, named masks,
  and no blending inside `forward`. Source: module docstrings; commit `acc2d70`.
- **No blending logic in the model.** The reference's `_mix` inside `forward`
  is why in-betweening was unreachable there. MoDiffAE just accepts a `z_sem`,
  and a sampling control does the blending. Source:
  `src/poseydon/models/modiffae.py` docstring.
- **Masking gives the latent a bottleneck.** Without it, the semantic encoder
  sees the whole clip it's asked to denoise and can carry a near-copy of it,
  which is fatal for retargeting. `MotionMasker` hides frames, joints and whole
  chains. `ConditionMasker` drops the whole latent so classifier-free guidance
  has an unconditional branch. They're siblings because they act on different
  objects. Source: `src/poseydon/models/modules/masking.py` docstring;
  CFG spec (`docs/superpowers/specs/2026-09-15-conditioning-masking-and-cfg-design.md`).
- **Multi-head semantic modulation**, averaged or summed, extends the
  reference's single gated transform. Source: commit `dfeecbc`; CFG spec.
- **The temporal attention band is a model parameter**, not a conditioner.
  Attention locality belongs to the attention module, and an override replaces
  the band but never the padding mask. Source:
  data pipeline spec §9 "The temporal attention band" (`docs/superpowers/specs/2026-09-06-data-pipeline-and-training-parity-design.md`);
  commits `c004f55`, `41d3d52`.

## Gotchas

- **Null embeddings are registered only when their feature is on**
  (commit `a41d60b`). A checkpoint trained with masking off has no
  `null_token`/`null_latent` to load.
- **Sampling a training checkpoint needs `model=modiffae`.** `configs/sample.yaml`
  defaults to `anytop` on purpose, so the recipe check names the override
  instead of silently using the wrong architecture.
- **Joint names are wired differently per model.** AnyTop needs
  `name_embedding_dim: 768`; MoDiffAE raises if given it. See
  [conditioners](conditioners.md).
- **The crop offset reaches the model through `cond`** (commit `63a811b`). Both
  models once ignored it.

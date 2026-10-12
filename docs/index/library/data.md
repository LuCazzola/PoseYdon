# data

`src/poseydon/data/`, library. The read path: a prepared corpus to normalized
feature windows, collated into a `MotionBatch`.

## Purpose

`data` owns everything between the files `build` wrote and the batch a model
sees: reading clips through the index, applying the rig's reduction and any
augmentations, extracting features, normalizing, choosing a window, and padding
a batch. It also owns the normalization statistics' format and policy
(`Normalizer`, `BlockPolicy`), which stage 2 fits and saves.

It does **not** own writing the corpus ([build](build.md)), what a feature is
([features](features.md)), what conditioning means
([conditioners](conditioners.md)), or the DataLoader and worker setup
(`src/poseydon/training/lightning.py`).

## Key files and types

| file | read it for |
|---|---|
| `src/poseydon/data/dataset.py` | `MotionDataset` (index + manifests → `Item`); `ClipView` (what a conditioner sees for one item); `Item` |
| `src/poseydon/data/normalize.py` | `Normalizer` (per-skeleton, invertible statistics for the active `FeatureSpec`); `BlockPolicy` (`center`, `scale` per block); `STD_CONSTANT_THRESHOLD`; `spec_hash` |
| `src/poseydon/data/window.py` | `Window` policies: `RandomCrop` (training), `SlidingWindow` (evaluation), `FullClip` |
| `src/poseydon/data/collate.py` | `collate`: pad to the batch's own maxima and build `Masks` |
| `src/poseydon/data/sampler.py` | `balanced_weights`: every rig gets an equal share of samples |

## How it connects

Consumes `build/index.py` (`CorpusIndex`, `ClipRecord`), the per-rig
`skeleton.npz` and `stats.npz` from [build](build.md), `features.reduce` and
`extract_features` from [features](features.md), `AugmentPipeline` from
[augment](augment.md), and the `CONDITIONERS` registry from
[conditioners](conditioners.md).

Produces a `MotionBatch` (see [core](core.md)) for [training](training.md).
`Normalizer` is also used by `build/pipeline.py` to fit `stats.npz`, and by
retarget validation and `ops/retarget_run.py` to denormalize.

Read path per item ([spec §9](../../superpowers/specs/2026-09-06-data-pipeline-and-training-parity-design.md#9-read-path)):

```
load clips/<Rig>/<action>.npz   prepared RigidBodyAnimation, every joint
apply the rig's reduction       from skeleton.npz
apply augmentations             composed onto the reduction as a JointEdit
extract features                the configured schema
normalize                       stats.npz, transported through that edit
crop                            the window policy
```

Called by: training (`build`, `lightning`, `metrics`, `validation`), ops
(`retarget_run`), `build/pipeline.py`, `scripts/build_features.py`, `cli.py`.

## Why it is shaped this way

- **Windowing at load time.** The reference cut every clip into fixed
  200-frame chunks on disk, so a dance is split where the model never sees
  across it, and the halves can land in different splits. Here window length is
  a config edit. Source: `src/poseydon/data/window.py` docstring.
- **Statistics for the active spec, invertible.** A structural loss can then
  work in raw space. The reference computes geodesic and foot-skate losses on
  z-normalized values, where a per-channel divide is not rotation-preserving.
  Source: `src/poseydon/data/normalize.py` docstring.
- **Normalization is declared per block in config** (`configs/dataset/truebones.yaml`),
  with scale modes `channel`, `joint_block`, `block` and `none`. Source:
  [spec §5](../../superpowers/specs/2026-09-06-data-pipeline-and-training-parity-design.md#5-normalization-policy);
  commit `19d6e99`. The shipped policy centres every block and scales per
  channel. See the config, not the spec's example table.
- **Statistics are loaded, not fitted.** Stage 2 writes `stats.npz` and the
  dataset reads it, which removed the startup cost and the in-memory clip cache.
  Source: spec §9; commit `e6a5f2a`.
- **Rig-balanced sampling.** Truebones spans 70 clips (Trex) to 4 (Chicken),
  17:1 across 73 rigs. The reference's `--balanced` sampler is mirrored. Source:
  `src/poseydon/data/sampler.py` docstring.
- **Pad to the batch's maxima, not a global constant.** Source: `collate`
  docstring.
- **The rest frame comes from the manifest's declared rest pose.** Source:
  commit `a71e507`.

## Gotchas

- **Constant channels are left unscaled (std 1.0).** A bare `+1e-6` epsilon
  is an amplifier: it once pushed features to 1e6 and the loss to ~1e10.
  `STD_CONSTANT_THRESHOLD = 1e-4` sits in an empty gap of the measured
  distribution. This deliberately diverges from the reference. Read the
  comment in `src/poseydon/data/normalize.py` before touching it.
- **Statistics are indexed by the original joint layout.** After a structural
  augmentation they must be *transported* through the `JointEdit`, not
  recomputed. See [augment](augment.md).
- **Worker RNG must be reseeded per worker** (`_worker_init` in
  `src/poseydon/training/lightning.py`). Otherwise every DataLoader worker draws
  the same crops and augmentations.
- **Changing `features:` requires re-running stage 2's stats pass.** `stats.npz`
  is the one schema-dependent artefact.

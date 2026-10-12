# losses

`src/poseydon/losses/`, library. Weighted loss terms, each declaring the
feature blocks and the space (raw or normalized) it needs.

## Purpose

`losses` owns the training objective's terms and the masking helpers they share.

It does **not** own the weighting config (`losses:` in `configs/train.yaml`), how
a clean sample is recovered from a prediction ([process](process.md)), or the
retarget validation metrics (`src/poseydon/training/metrics.py`).

## Key files and types

| file | read it for |
|---|---|
| `src/poseydon/losses/base.py` | `LossTerm` (declares required blocks and space); `element_mask`, `masked_mean`, `masked_mean_per_sample`; `raw_block` (extract and denormalize); the `LOSSES` registry; the `NORM_STATS` key |
| `src/poseydon/losses/simple.py` | `SimpleLoss` (`simple`): masked MSE over the whole feature vector, also reported per block |
| `src/poseydon/losses/geodesic.py` | `GeodesicLoss` (`geodesic`): angular error on the rotation manifold |
| `src/poseydon/losses/footskate.py` | `FootSkateLoss` (`footskate`): movement of joints the target marks as planted |
| `src/poseydon/losses/kl.py` | `KLDivergence` (`kl`): diagonal Gaussian posterior against a standard normal |

## How it connects

Consumes `MotionBatch` and `FeatureSpec` from [core](core.md), plus the
`norm_stats` conditioner's payload, so `raw_block` can undo normalization.

Produces scalar terms that `training/task.py` weights and sums.

Called by: training (`build`, `task`, `lightning`), conditioners (the
`NORM_STATS` key) and `cli.py` (`list`).

## Why it is shaped this way

- **Requirements are checked at fit start.** A term asking for rotations
  against a representation without them fails in the first second with a
  readable message instead of slicing whatever sits in those columns. Source:
  `src/poseydon/losses/base.py` docstring.
- **Structural terms work in raw space.** The reference computes geodesic and
  foot-skate losses on z-normalized values, where a per-channel divide isn't
  rotation-preserving, so it optimizes something other than what its paper
  describes. `raw_block` denormalizes first. Source:
  `src/poseydon/data/normalize.py` docstring.
- **Shipped weights are `simple: 1.0, geodesic: 1.0`**, as in the reference's
  paper command. Foot skate is a measured validation diagnostic, not a
  training term. The earlier `geodesic: 0.1, footskate: 0.5` was neither the
  reference nor a measured choice. Source: comment above `losses:` in
  `configs/train.yaml`.

## Gotchas

- **Threshold the raw contact flag, not the normalized one** (commit `3358df4`).
  A normalized flag isn't 0/1.
- **Use the masked means.** Padding frames and joints must not count. `element_mask`
  is `(B, J, 1, T)`, matching `MotionBatch.x`.
- **Losses are reported per noise quartile and per feature block** (commits
  `8c5dd1f`, `1d1e269`). Keep any new term compatible with that logging.

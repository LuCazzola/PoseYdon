# conditioners

`src/poseydon/conditioners/`, library. Declared conditioning: each extra piece
of data a batch may carry, with its whole read → collate → device path.

## Purpose

A conditioner owns one conditioning signal end to end: it extracts it per
item from a `ClipView`, collates it across a batch (padding joints), and may
ship a default encoder. The shipped conditioners are all skeleton-derived:
`topology`, `tpose`, `norm_stats` and `joint_names`.

It does **not** decide how a model *injects* the signal. That stays in
[models](models.md), because injection is genuinely model-specific. It also
doesn't own the batch type (`core.batch.Cond`, see [core](core.md)).

## Key files and types

| file | read it for |
|---|---|
| `src/poseydon/conditioners/base.py` | `Conditioner`: `name`, `extract(item)`, `collate(payloads, max_joints)`, `encoder(d_model)`; the `CONDITIONERS` registry |
| `src/poseydon/conditioners/skeleton.py` | `Topology` (parents + rest offsets), `TPose` (the rest pose as one frame of features), `NormalizationStats` (so a loss can undo normalization), `JointNames` (T5 embeddings of each joint's name, `(J, 768)`) |

## How it connects

Consumes `ClipView` from [data](data.md) (the record, animation, resolved
skeleton and rig sidecars), `core.topology`, and the payload key constants
`NORM_STATS` (from `losses/base.py`) and `JOINT_NAMES` (from `models/modiffae.py`).

Produces `Cond` entries keyed by conditioner name, collated by
`data/collate.py` into the `MotionBatch`.

Declared in config: `conditioners: [topology, tpose, norm_stats, joint_names]`
in both `configs/train.yaml` and `configs/sample.yaml`. The list is recorded in
the checkpoint recipe and checked at sampling time (see [training](training.md)).

Called by: `data/dataset.py`, `data/collate.py`, `cli.py` (`list`).

## Why it is shaped this way

- **Only declared conditioning is loaded.** The reference packs everything
  into one fourteen-key `cond['y']`. Here a text-conditioned corpus declares
  `[text]` and an unconditional one declares nothing, with no change to the
  dataset, the collate or any model. Source:
  `src/poseydon/conditioners/base.py` docstring;
  [architecture guide](../../guides/architecture.md#only-declared-conditioning-is-loaded).
- **Injection stays in the model.** AnyTop concatenates the T-pose as an extra
  frame and adds name embeddings to per-joint tokens, and no generic mechanism
  captures that honestly. Source: `src/poseydon/conditioners/base.py` docstring.
- **Joint names were a parity gap.** The reference conditions on T5 embeddings
  of joint names, and adding them closed its second parity gap. Source: commit
  `ff64e1d`; `tools/build_joint_name_cache.py` builds the embedding cache.
- **Dropping conditioning is not done here.** Conditioning dropout and the
  null embeddings that classifier-free guidance needs live in the model
  (`MotionMasker`, `ConditionMasker`), not in conditioners. See [models](models.md).

## Gotchas

- **Sampling must declare the same conditioners as training.** `MoDiffAE`
  treats `joint_names` as optional and silently skips it when it's absent, so a
  mismatch produces plausible-looking garbage and no error. The recipe check
  exists for exactly this case. Source: comment in `configs/sample.yaml`.
- **`topology` uses per-clip offsets.** Prepared clips of one rig don't share
  rest geometry (see [build](build.md)), so which clip you template on matters.
- **For AnyTop, `joint_names` needs `name_embedding_dim: 768`**
  (`configs/model/anytop.yaml`). Without it the projection stays `None` and
  the embeddings are silently ignored. `MoDiffAE` wires names differently, and
  passing it the same key raises; see the comment in
  `configs/model/modiffae.yaml`.

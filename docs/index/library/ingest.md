# ingest

`src/poseydon/ingest/`, library. Canonical alignment math (facing, centring,
scale, ground) and the original BVH-corpus ingest pipeline behind
`poseydon ingest`.

## Purpose

`align.py` owns the alignment primitives: facing rotations, rotating a whole
rig, XZ centring, mean-bone-length scaling and grounding. `pipeline.py` owns
skeleton inference from file names (`available_rigs`, `infer_skeleton`) and the
legacy `ingest_corpus`, which writes aligned `RigidBodyAnimation` files plus an
index.

It does **not** own the invertible preparation chain. That's [build](build.md),
which reuses these primitives inside its stages. It also doesn't own the corpus
index format, which moved to `src/poseydon/build/index.py`.

## Key files and types

| file | read it for |
|---|---|
| `src/poseydon/ingest/align.py` | `AlignmentParams`; `facing_quats` / `facing_quat`; `rotate_rig`, `rotate_to_face_axis`; `move_xz_to_origin`, `scale_to_mean_bone_length`, `put_on_ground`; `compute_alignment_params`, `align` |
| `src/poseydon/ingest/pipeline.py` | `available_rigs`, `infer_skeleton`, `ingest_clip`, `ingest_corpus`, `IngestResult` |

## How it connects

Consumes [io](io.md) (`BVH`), `core.skeleton` and `core.animation`, and the index
types in `build/index.py`.

Produces alignment helpers used by `build/prepare.py` (the `FaceAxis` stage and
others), `features/base.py` and `training/metrics.py`, plus rig discovery used
by `scripts/process_dataset_truebones.py`.

Called by: build, features, training, `cli.py` (`poseydon ingest`) and the
dataset scripts.

## Why it is shaped this way

- **Rotate, centre XZ, scale, ground, in that order.** This is the reference's
  `process_anim` order, and it's part of the contract: reordering changes the
  result. Source: `src/poseydon/ingest/align.py` docstring.
- **Alignment constants are per skeleton, not per clip.** Scale and grounding
  come from the rig's reference clip, so every clip of a character shares one
  frame. Source: commit `c8b13e7`.
- **Full-length clips, never chunked.** Windowing happens at load time. Source:
  `ingest_clip` docstring; `src/poseydon/core/animation.py`.
- **A rig is a directory with a `manifest.yaml`.** `available_rigs` returns
  rigs longest first, so `Dog-2` is matched before `Dog` when a file name is
  inferred. This replaced an underscore-prefix convention for shared fragments.
  Source: commits `9686373`, `1bef26f`; `available_rigs` docstring.
- **Failures are collected, not fatal.** `ingest_corpus` keeps going and
  reports every clip that failed. Source: `ingest_corpus` docstring.
- **Why this package still exists separately from build:** **rationale not
  recorded.** The corpus index moved to `build/` (commit `d8a1495`) and
  alignment stayed here.

## Gotchas

- **Nothing consumes `poseydon ingest`'s output.** The model-ready corpus comes
  from `scripts/process_dataset_truebones.py` followed by
  `scripts/build_features.py` (see [build](build.md)). `data/*/aligned/` is not
  read by any library code. Treat `ingest_corpus` as legacy, and don't build on
  it without checking.
- **Skeleton inference relies on file names.** A clip's rig comes from its file
  name prefix matching a rig directory, and the clip slug depends on that
  prefix too. A file whose name doesn't start with its rig's name is misfiled
  or rejected.
- **No up-axis handling.** Alignment assumes Y-up input. A Z-up source would be
  silently mis-prepared.

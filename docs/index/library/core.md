# core

`src/poseydon/core/`, library. The contracts every other package is written
against: animation containers, skeleton manifests, the feature layout, the
batch handed to a model, kinematics and rotations.

## Purpose

`core` owns the types that cross package boundaries and the math they all need.
It imports nothing else from `poseydon`, and every other library package imports
it.

It does **not** own reading files (that's [io](io.md)), preparing or building a
corpus ([build](build.md)), computing features ([features](features.md)), or the
training recipe (`src/poseydon/training/recipe.py`, see [training](training.md)).

## Key files and types

| file | read it for |
|---|---|
| `src/poseydon/core/animation.py` | `Animation` (every joint may translate, which is what a file declares) and `RigidBodyAnimation` (only the root translates, which is what everything downstream asks for) |
| `src/poseydon/core/skeleton.py` | `SkeletonManifest` (per-rig YAML: facing pairs, contact thresholds, foot joints, rest pose, authority), `resolve()` binding names to indices, `ResolvedSkeleton` |
| `src/poseydon/core/spec.py` | `FeatureSpec`, the ordered `(name, width)` block layout, with `slice()`, `take(tensor, name, axis, drop=)` and `rotation_block()` (which of `ROTATION_BLOCKS`, `rot6d_joint` or `rot6d_bone`, it carries) |
| `src/poseydon/core/batch.py` | `MotionBatch` (`x` of shape `(B, J, D, T)`, `spec`, `masks`, `window`, `cond`), plus `Masks`, `WindowInfo` and `Cond` |
| `src/poseydon/core/authority.py` | `AuthorityPolicy` (which source format is authoritative for skeleton, motion and surface), `SourceArtifacts`, and `ArtifactUnavailableError` (the typed refusal an export raises) |
| `src/poseydon/core/rotations.py` | quaternion / Euler / matrix / 6D conversions in numpy, wrapping SciPy |
| `src/poseydon/core/kinematics.py` | numpy forward kinematics; `check_topological_order` |
| `src/poseydon/core/torch_kinematics.py` | differentiable FK in torch, used by the IK solver |
| `src/poseydon/core/topology.py` | `edge_relations` and `hop_distances`, the joint-pair graph relations that let attention span skeletons of different shapes |
| `src/poseydon/core/registry.py` | `Registry`, a name → class lookup for list-valued config slots |

## How it connects

Consumes nothing from the rest of the library.

Produces the types the rest of the library passes around. [io](io.md) parses
files into `Animation`. [build](build.md) and [ingest](ingest.md) turn those into
`RigidBodyAnimation` and resolve manifests. [data](data.md) collates
`MotionBatch`. [models](models.md), [losses](losses.md) and [sampling](sampling.md)
consume it, reading blocks through `FeatureSpec`. [features](features.md),
[solvers](solvers.md) and [export](export.md) run FK and rotation math from here.
`Registry` instances back `FEATURES`, `CONDITIONERS`, `LOSSES` and the other
lookups that `poseydon list` prints.

Called by: every library package, most scripts, and the reference-comparison
tools.

## Why it is shaped this way

- **Two animation types, not one.** BVH lets any joint carry position channels,
  and every 3ds Max Biped rig in Truebones uses them. Parsing into the general
  `Animation` stays lossless, and the rigid assumption becomes an explicit,
  checked conversion (`Animation.as_rigid_body`) instead of something a parser
  silently enforces. Source: `src/poseydon/core/animation.py` docstring; commit
  `fede016`.
- **Joints by name, in YAML.** The reference kept face-joint indices, contact
  thresholds and taxonomy as Python literals. Manifests reference joints by name
  so a re-exported rig fails loudly instead of mirroring the character, and
  `resolve_joint` names the nearest alternatives when a name is wrong. Source:
  `src/poseydon/core/skeleton.py` docstring; commits `3949ef8`, `3ff8b64`;
  [architecture guide](../../guides/architecture.md#skeleton-knowledge-is-data).
- **A named feature layout.** `FeatureSpec` exists so a loss asks for `rot6d_joint` and
  fails loudly against a representation without rotations, instead of slicing
  a magic `[..., 3:9]`. Source: `src/poseydon/core/spec.py` docstring.
- **A four-part batch.** The reference packs T-pose, joint names, topology,
  object type and normalization statistics into one fourteen-key `cond['y']`
  dict. Here only the tensor, its spec, the padding masks and the window
  provenance are mandatory. Everything else is a declared conditioner keyed in
  `Cond`, so a fixed-skeleton text-conditioned corpus drops in without touching
  `batch.py`. Source: `src/poseydon/core/batch.py` docstring;
  [architecture guide](../../guides/architecture.md#only-declared-conditioning-is-loaded).
- **A registry only for lists.** A full Hydra `_target_` block per entry in a
  list (features, conditioners, losses) would be noise. Singular choices
  (model, process, data) stay plain config groups. Source:
  `src/poseydon/core/registry.py` docstring.
- **Animations are never chunked on disk.** Windowing is a load-time sampling
  concern. Source: `src/poseydon/core/animation.py` docstring.
- **Topology as relations, not orderings.** Attention is biased by parent,
  child, sibling and hop distance, so one model spans different skeletons.
  Source: `src/poseydon/core/topology.py` docstring; commit `150ef53`.

## Gotchas

- **Quaternions are scalar-last `(x, y, z, w)`**, matching SciPy and roma. 6D is
  the first two **rows** of the rotation matrix, matching PyTorch3D. Source:
  `src/poseydon/core/rotations.py`.
- **The channel axis is not always last.** Features are `(F, J, D)`, a
  `MotionBatch` is `(B, J, D, T)` (channels at axis 2), and statistics are
  `(J, D)`. `FeatureSpec.take` requires `axis` explicitly for that reason. Use
  `MotionBatch.block(name)`, which passes `axis=2` for you. `drop=True` refuses
  any block wider than 1.
- **Parents must be topologically ordered**, with joint 0 as the root.
  `check_topological_order` enforces it, and FK and later chain computations
  rely on it.
- **`Cond.to(device)` moves nested payloads.** A flat-only move was a real bug
  (commit `113827b`). Keep any new payload type movable.
- **`MotionBatch.__post_init__` validates shapes** against the spec and masks.
  Construct a new batch, with `dataclasses.replace`, instead of mutating
  tensors in place to a different shape.
- **The recorded recipe is not in `core`.** It lives in
  `src/poseydon/training/recipe.py`. Likewise, `chain_ids` lives in
  `src/poseydon/models/modules/masking.py`, not in `core/topology.py`.

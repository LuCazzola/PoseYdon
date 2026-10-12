# features

`src/poseydon/features/`, library. Composable per-joint feature blocks, the
joint reduction the model looks through, and the way back from features to a
skeleton.

## Purpose

`features` owns three directions:

- **Forward:** a `RigidBodyAnimation` to a `(frames, joints, dim)` tensor plus
  its `FeatureSpec` (`extract.py`, from the registered blocks).
- **Reduction:** dropping joints that carry nothing learnable, and putting
  them back (`reduce.py`).
- **Back:** features to positions or an animation (`recover.py`, which is exact,
  and `reconstruct.py`, which is a choice of strategy).

It does **not** own normalization ([data](data.md)), the feature layout type
(`core.spec.FeatureSpec`, see [core](core.md)) or the IK solver itself
([solvers](solvers.md)).

## Key files and types

| file | read it for |
|---|---|
| `src/poseydon/features/base.py` | `Feature` (one named block), `FeatureContext` (shared per-clip work), the `FEATURES` registry |
| `src/poseydon/features/kinematics.py` | `ric_pos` (`RootInvariantPositions`, 3), `rot6d` (`Rotation6D`, 6), `local_vel` (`LocalVelocity`, 3) |
| `src/poseydon/features/contact.py` | `foot_contact` (`FootContact`, 1): binary planted flag |
| `src/poseydon/features/extract.py` | `extract_features(anim, ...)` → array + `FeatureSpec` |
| `src/poseydon/features/reduce.py` | `JointReduction`, `RemovalOp`, `build_reduction`, `apply_reduction`, `invert_reduction`, `DropDegenerateJoints` |
| `src/poseydon/features/recover.py` | `features_to_anim`, `local_rotations`, `root_trajectory`, `positions_from_features`, `contact_flags_from` |
| `src/poseydon/features/reconstruct.py` | `reconstruct(name, ...)` with the `RECONSTRUCTORS` `fk`, `positions`, `positions_ik` |

The default schema `[ric_pos, rot6d, local_vel, foot_contact]` is 13 dims per
joint, the reference's representation.

## How it connects

Consumes `core` (animation, kinematics, rotations, spec), alignment helpers
from [ingest](ingest.md), `augment.JointEdit` (reduction composes with
augmentation edits), and [solvers](solvers.md) for `positions_ik`.

Produces feature tensors for [data](data.md) and `build/pipeline.py` (the stats
pass); reductions for `build`, `data`, [export](export.md) and
`scripts/anim_2_fbx.py`; and reconstruction for the CLI's `sample`,
`ops/retarget_run.py` and `training/metrics.py`.

Called by: build, data, export, ops, training, `cli.py`, the asset scripts,
and the IK and reference-comparison tools.

## Why it is shaped this way

- **Features are pure functions of the canonical animation**, named in a
  registry, so `features: [rot6d, local_vel]` needs no re-preparation. A new
  representation is one class plus a decorator. Source:
  `src/poseydon/features/base.py` docstring;
  [architecture guide](../../guides/architecture.md#features-are-composed-at-read-time).
- **Golden parity.** Every block reproduces the reference's own arrays for
  seven rigs to ~1e-6, and foot contact is bit-exact. Source: commit `1edbcab`;
  [testing guide](../../guides/testing.md).
- **Reduction happens at extraction time and is never baked into the corpus.**
  Skin weights are indexed by the original joint order, so a generated clip must
  come back on the user's rig. The corpus keeps every joint, and reduction is
  the lens the model looks through. Source: `src/poseydon/features/reduce.py`
  docstring; [spec §6](../../superpowers/specs/2026-09-06-data-pipeline-and-training-parity-design.md#6-joint-reduction-and-expansion).
- **Zero-offset leaves are deleted, and zero-offset internal joints are folded
  only when they are an only child.** Folding turns the parent and so every
  sibling, which isn't worth one token. Ops are recorded by *name* so they
  survive reindexing. Source: `src/poseydon/features/reduce.py` docstring.
- **Recovery is exact, not fitted.** Rotations are in the representation, so
  there's nothing to solve. The reference ran 150 IK iterations per clip and
  discarded the rotations it had. Source: `src/poseydon/features/recover.py`
  docstring.
- **Reconstruction is a named choice.** Rotations and positions agree to 1e-9
  on real data, but generated clips can disagree by more than three bone
  lengths. `fk` is rigid and free; `positions` follows the prediction but
  stretches limbs and can't produce a BVH; `positions_ik` (what the reference
  writes) refits positions onto a rigid skeleton. Source:
  `src/poseydon/features/reconstruct.py` docstring; commit `a345a91`;
  [sampling guide](../../guides/sampling.md).

## Gotchas

- **IK can't see twist.** Rotation about a bone's own axis doesn't move its
  children, so `positions_ik` plateaus around 1% of a bone length on average
  and around 15% for the worst joint. It doesn't converge to zero.
- **Reduction drops zero-offset joints even when they are skinned.** Bear's
  forearm-twist bones are an open example: the twist is lost and the skin can
  sit up to 5.5% of the mesh extent off. Fixing that changes the model topology,
  which means retraining.
- **Features are channels-last `(F, J, D)`**, but a `MotionBatch` is
  `(B, J, D, T)`. Pass `axis` to `FeatureSpec.take`.
- **The recorded reduction is the inverse's only input.** `invert_reduction`
  replays it in reverse. Don't recompute a reduction to invert someone else's
  clip.

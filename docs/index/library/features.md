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
| `src/poseydon/features/kinematics.py` | `ric_pos` (`RootInvariantPositions`, 3), `rot6d_joint` (`JointRotation6D`, 6), `heading` (`Heading`, 2), `rot6d_bone` (`BoneRotation6D`, 6), `local_vel` (`LocalVelocity`, 3) |
| `src/poseydon/features/contact.py` | `foot_contact` (`FootContact`, 1): binary planted flag |
| `src/poseydon/features/extract.py` | `extract_features(anim, ...)` → array + `FeatureSpec` |
| `src/poseydon/features/reduce.py` | `JointReduction`, `RemovalOp`, `build_reduction(anim, tolerance, layout)`, `reduction_layout(features)`, `apply_reduction`, `invert_reduction`, `DropDegenerateJoints` |
| `src/poseydon/features/recover.py` | `features_to_anim`, `local_rotations` (either layout), `heading`, `root_trajectory`, `positions_from_features`, `contact_flags_from` |
| `src/poseydon/features/reconstruct.py` | `reconstruct(name, ...)` with the `RECONSTRUCTORS` `fk`, `positions`, `positions_ik` |

The default schema `[ric_pos, rot6d_joint, heading, local_vel, foot_contact]`
is 15 dims per joint.

### Two rotation layouts

The rotation layout is chosen by naming one rotation block in the dataset's
`schema:` (`configs/dataset/<name>.yaml`) and the same blocks in `features:`
(`configs/train.yaml`, `configs/sample.yaml`):

| block | slot `j` holds | root slot | heading | leaves the reduction may drop |
|---|---|---|---|---|
| **`rot6d_joint`** (default) | joint `j`'s own local rotation | the root's rotation, in the heading frame | the `heading` block (cos, sin of the yaw, on the root) | every zero-length leaf |
| **`rot6d_bone`** (HumanML3D) | `parent(j)`'s rotation, orienting the bone that ends at `j` | the heading | in the root slot | a zero-length leaf only if it has a sibling |

Every rotation that orients something is recoverable in both. The reduction
differs because `rot6d_bone` keeps a joint's rotation in its children's slots:
a zero-length End Site that is its parent's only child holds the only copy of
the parent's rotation, so `build_reduction(layout="rot6d_bone")` keeps it.
Measured on Truebones, `rot6d_joint` drops 23% of the corpus's joints and
`rot6d_bone` only 1% (Dog: 53 against 65 of 65).

Switching layouts changes the reduction, which `skeleton.npz` records, so run
`scripts/build_features.py` in full, not `--stats-only`, then pass the same
layout to `RigRecord.load(..., layout=...)` / `poseydon export --layout`.
Checkpoints trained on the old `rot6d` block are the `rot6d_bone` layout:
their recipe names `rot6d`, which no longer exists, so loading one fails with
an "unknown feature" error until the recipe says `rot6d_bone`.

## How it connects

Consumes `core` (animation, kinematics, rotations, spec), alignment helpers
from [ingest](ingest.md), `augment.JointEdit` (reduction composes with
augmentation edits), and [solvers](solvers.md) for `positions_ik`.

Produces feature tensors for [data](data.md) and `build/pipeline.py` (the stats
pass); reductions for `build`, `data` and [export](export.md); and reconstruction for the CLI's `sample`,
`ops/retarget_run.py` and `training/metrics.py`.

Called by: build, data, export, ops, training, `cli.py`, the asset scripts,
and the IK and reference-comparison tools.

## Why it is shaped this way

- **Features are pure functions of the canonical animation**, named in a
  registry, so `features: [rot6d_joint, heading, local_vel]` needs no re-preparation. A new
  representation is one class plus a decorator. Source:
  `src/poseydon/features/base.py` docstring;
  [architecture guide](../../guides/architecture.md#features-are-composed-at-read-time).
- **`rot6d_joint` is the default because `rot6d_bone` loses rotations it
  needn't.** `rot6d_bone` is the HumanML3D layout, where rotations are solved
  from positions by IK, one per bone, so a leaf's own rotation never existed
  to store. Truebones BVH records real rotations for every joint, and under
  `rot6d_bone` plus the old reduction the model never saw an ear's or a tail
  tip's (72 of 73 rigs animate such a joint more than 5 degrees within a
  clip). The project owner chose the default and asked for both to stay
  selectable; `rot6d_bone` is kept for parity with the reference.
- **Golden parity.** Every block reproduced the reference's own arrays for
  seven rigs to ~1e-6, with foot contact bit-exact, when last checked
  (the rotation block as `rot6d_bone`, the reference's layout). No test
  re-checks it since the suite was cut to the round trip. Source: commit
  `1edbcab`; [testing guide](../../guides/testing.md).
- **Reduction happens at extraction time and is never baked into the corpus.**
  Skin weights are indexed by the original joint order, so a generated clip must
  come back on the user's rig. The corpus keeps every joint, and reduction is
  the lens the model looks through. Source: `src/poseydon/features/reduce.py`
  docstring.
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

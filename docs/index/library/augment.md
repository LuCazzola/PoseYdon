# augment

`src/poseydon/augment/`, library. Per-sample data augmentation, structural
(changing the skeleton) or in feature space, composed from config.

## Purpose

`augment` owns the `Augmentation` contract, the `AugmentPipeline` that composes
augmentations, the two structural augmentations ported from the reference
(`DropEndEffector`, `DuplicateJoint`), and `JointEdit`, which carries per-joint
arrays through a structural edit.

It does **not** own where augmentation runs in the read path ([data](data.md))
or the statistics it transports (`data.normalize.Normalizer`).

## Key files and types

| file | read it for |
|---|---|
| `src/poseydon/augment/base.py` | `Augmentation` with two optional hooks, `apply_structural` (before extraction) and `apply_features` (after normalization); `AugmentPipeline` (ordered, each entry gated by its own `p`) |
| `src/poseydon/augment/topology.py` | `DropEndEffector`, `DuplicateJoint`; helpers `drop_joints`, `duplicate_joint`, `leaves`, `child_count` |
| `src/poseydon/augment/joint_edit.py` | `JointEdit`: `source_of[new_index]` and `blend_of`, so stats and the T-pose follow an edit by gathering |

## How it connects

Consumes `core.animation.RigidBodyAnimation` and `data.normalize` (to transport
statistics).

Produces edited animations and a `JointEdit` for [data](data.md)'s
`MotionDataset`. [features](features.md)' reduction composes with the same
`JointEdit` type. The `AUGMENTATIONS` registry feeds `poseydon list`.

Configured in `configs/train.yaml` under `augmentations:`, each entry a full
`_target_` with its own `p`.

Called by: `data/dataset.py`, `features/reduce.py`, `cli.py`.

## Why it is shaped this way

- **Two independent hooks.** Most augmentations need only one, and the base
  class makes the other a no-op. Source: `src/poseydon/augment/base.py`
  docstring.
- **Structural edits are exact mutations of the animation**, so every feature
  block, current or future, is recomputed from a valid skeleton instead of
  patched after extraction. Source: `src/poseydon/augment/topology.py`
  docstring.
- **Statistics are transported, not recomputed.** Normalization statistics and
  the T-pose rest frame are indexed by the original layout, so they're gathered
  through `JointEdit`. An inserted joint averages its two sources. Source:
  `src/poseydon/augment/joint_edit.py` docstring.
- **`_target_` entries, unlike features and conditioners.** Augmentations carry
  their own parameters (a probability, sometimes more), so a bare name list
  doesn't fit. Source: `src/poseydon/augment/__init__.py` docstring.
- **Independent gates, not one exclusive draw.** The reference picks one of
  {none, drop, duplicate} per sample. Here each fires independently with
  p=1/3, which matches the reference's per-type marginal rate but occasionally
  applies both. Composability was the point. Source: the comment above
  `augmentations:` in `configs/train.yaml`.

## Gotchas

- **"The design spec" the module docstrings cite no longer exists.** The
  docstrings themselves and the comments in `configs/train.yaml` carry the
  reasoning.
- **`DropEndEffector` drops only leaves that are neither feet nor facing
  joints**, because dropping those would break contact features and facing.
  `drop_joints` refuses anything that isn't a leaf.
- **An empty pipeline, or one where every gate fails, is the true identity**:
  an identity `JointEdit` and the array unchanged. `augmentations=[]` turns
  augmentation off.

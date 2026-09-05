# PoseYdon — Data Augmentation

Date: 2026-09-05
Status: draft

## 1. Purpose

The reference (`external/neural_motion_blending`) trains AnyTop with a per-sample
structural augmentation: at `__getitem__` time it randomly drops a non-foot
end-effector joint, or duplicates a joint by inserting a midpoint, so a
topology-conditioned model generalizes across joint counts rather than
memorizing the training roster's skeletons. PoseYdon has no augmentation stage
at all yet -- `MotionDataset` reads a window, normalizes it, and hands it
straight to conditioners.

This spec adds a general augmentation subsystem, following the repo's existing
philosophy (§ features, conditioners): a component is a small class, named in a
registry, composed by a config list. It ports the reference's two structural
augmentations as the first concrete implementations, and defines a contract wide
enough that a feature-space augmentation (mirroring, jitter) is a later addition
requiring no redesign.

### Success criteria

1. `augmentations: [...]` is a config list, each entry a `_target_` plus a
   probability, applied in the declared order -- a direct analogue to
   `features:`/`conditioners:`.
2. `DropEndEffector` and `DuplicateJoint` reproduce the reference's selection
   rules (non-foot end-effectors only; duplication only at a single-child,
   non-root-adjacent joint) and are **exact** under forward kinematics, not an
   approximation of one.
3. Conditioners (`Topology`, `TPose`, `NormalizationStats`) require no code
   changes: an augmented window's `ClipView` carries an already-consistent
   `Anim`, `ResolvedSkeleton`, rest frame and `Normalizer`.
4. `augmentations: []` (the default) is exactly today's behaviour -- byte-
   identical output, since the pipeline is empty and every hook is a no-op.

## 2. Scope

### v1 (this spec)

The `Augmentation` contract, `AugmentPipeline`, `JointEdit` transport,
`DropEndEffector`, `DuplicateJoint`, `MotionDataset` integration, config wiring,
tests.

### Deferred

Any concrete feature-space augmentation (mirroring, rotation jitter, additive
noise). The contract supports one (§5, `apply_features`), but none is needed to
match the reference, and adding one without a concrete need would be
speculative. A follow-up adds them as new registry entries with no change to
the pipeline itself.

## 3. Baseline: what the reference does

`data_loaders/truebones/data/dataset.py::MotionDataset.augment` (reference,
not ours) picks exactly one of `{none, remove_joints, add_joint}` per sample via
`random.choice`, weighted implicitly by the 2-or-3-way choice (no add option
once a skeleton is already at `max_joints` or is a Dragon). Both mutations
operate on the **already-extracted, still-unnormalized** `(F, J, 13)` motion
array plus companion per-joint arrays (`parents`, `mean`, `std`,
`joints_names_embs`, `joint_relations`, `joints_graph_dist`) -- not on the BVH or
any rotation representation.

- **Remove**: candidates are kinematic-chain end-effectors (`kinematic_chains[i][-1]`)
  that are never a contact foot (`motion[..., -1] > 0`, i.e. the foot-contact
  channel). A random subset at rate `{0.1, 0.2, 0.3}` is deleted from every
  per-joint array, `parents` renumbered, and any joint that lost its only child
  is promoted to a new end-effector via `joint_relations`.
- **Add**: candidates are joints with exactly one child whose parent is not the
  root. A midpoint joint is spliced in: its motion feature is the parent/child
  average with the child's own rotation and foot-contact copied onto it, the
  child's rotation is reset to identity, and `offsets` are halved between the
  two. `mean`/`std`/`joints_names_embs` for the new joint are averaged from
  parent and child, flagged `# TODO: AUGMENT LIKE MOTION AND TPOS` in the
  reference -- an acknowledged approximation, not something to match exactly.

Two things do not carry over as-is:

- **Feature layout.** The reference's 13-dim vector is fixed
  (`pos3+rot6d6+vel3+foot1`); PoseYdon's is a composed `FeatureSpec` of
  arbitrary blocks. Slicing a *feature array* the way the reference does would
  need per-block special-casing (rotation reset, position averaging) for every
  registered feature -- fragile, and it grows with every new feature type.
- **Rotation representation.** The reference's "reset to identity" and
  "average position" operations are a plausible-looking but not FK-exact
  approximation once you have quaternions and offsets available, as PoseYdon
  does at the point features are extracted.

## 4. Architecture: mutate `Anim`, not the feature array

PoseYdon extracts features from `Anim` at read time (design spec §6.4), so the
natural hook for a structural edit is **before** extraction, on `Anim` and
`ResolvedSkeleton` themselves -- one mutation point, and every feature block
(present or future) is recomputed from a genuinely valid skeleton rather than
patched after the fact.

```
MotionDataset.__getitem__:
  anim, resolved            <- cached, per clip (unchanged)
  anim', resolved', edit    <- AugmentPipeline.apply_structural(anim, resolved, rng)
  raw, spec                 <- extract_features(anim', resolved')      # unchanged call
  normalizer'                <- edit.transport(self._normalizer(skeleton, spec))
  features                  <- normalizer'.normalize(raw)
  features                  <- AugmentPipeline.apply_features(features, spec, resolved', rng)
  window                    <- self.window.bounds(...) on features      # unchanged
  rest_frame'                <- edit.transport_row(self._rest_frame(record, spec, normalizer))
  view = ClipView(anim=anim', resolved=resolved', normalizer=normalizer', rest_frame=rest_frame')
```

`anim'`/`resolved'` are **local to this window's `Item`** -- never written back
into `self._anims`/`self._manifests`, which stay the untouched, shared,
per-clip cache. Every other clip, and every other window of the *same* clip,
sees the original skeleton. Conditioners (`Topology`, `TPose`,
`NormalizationStats`) read `item.anim`, `item.rest_frame`, `item.normalizer`
exactly as they do today -- **zero changes** to `src/poseydon/conditioners/`.

### Removing a leaf is exact

A non-foot end-effector has no children, so dropping it is: remove its row from
`rotations`/`offsets`/`names`, remove it from `parents`, decrement every
surviving parent index greater than the removed one. Forward kinematics over
the remaining joints is untouched -- there is nothing left that referenced the
removed joint.

### Duplicating a joint is exact under FK, not approximate

Original chain: `parent --offset,rot--> j`. World transform of `j`:
`W_parent · T(offset_j) · R(rot_j)`.

Insert `new` between `parent` and `j`, splitting the offset and moving `j`'s own
rotation onto the (renamed) child slot:

```
offset_new = offset_j / 2,  rot_new  = identity      (all frames)
offset_j'  = offset_j / 2,  rot_j'   = rot_j          (unchanged, all frames)
```

Then `W_new = W_parent · T(offset_j/2)`, and since `R(identity)` is the identity
transform, `T(offset_j/2) · R(identity) · T(offset_j/2) = T(offset_j)` exactly,
so `W_j' = W_parent · T(offset_j) · R(rot_j) = W_j` for every frame. The
duplicated joint's own position and everything below it in the hierarchy is
**bit-identical** to before the edit; only the new midpoint joint's position
(the honest midpoint of the split bone) is new information for the model to
learn from -- there is no interpolation artifact to inherit from the reference.

Constraints, ported from the reference: `new` is inserted only at a joint with
exactly one child, whose parent is not the root and which is not the root
itself. In addition (PoseYdon-specific, since indices renumber after either
edit) neither edit may select a joint used in `resolved.facing_indices`
(orientation would silently change) or the root.

## 5. Core contracts

```python
@dataclass(frozen=True)
class JointEdit:
    """New joint count J' plus, for each new index, the ORIGINAL joint its
    feature-space statistics (Normalizer mean/std, rest frame) come from.
    Composes: edit2.compose(edit1) reindexes edit1's sources through edit2.
    """
    source_of: tuple[int, ...]   # len J'; index into the pre-edit joint axis

    @classmethod
    def identity(cls, n_joints: int) -> JointEdit: ...

    def compose(self, prior: JointEdit) -> JointEdit: ...

    def transport(self, normalizer: Normalizer) -> Normalizer:
        """New Normalizer with mean/std rows gathered via source_of."""

    def transport_row(self, row: np.ndarray) -> np.ndarray:
        """Gather a single (J, D) row (e.g. a rest frame) via source_of."""


class Augmentation(ABC):
    def apply_structural(
        self, anim: Anim, resolved: ResolvedSkeleton, rng: np.random.Generator
    ) -> tuple[Anim, ResolvedSkeleton, JointEdit]:
        """Default: no-op (identity edit). Structural augmentations override this."""
        return anim, resolved, JointEdit.identity(anim.n_joints)

    def apply_features(
        self, features: np.ndarray, spec: FeatureSpec,
        resolved: ResolvedSkeleton, rng: np.random.Generator,
    ) -> np.ndarray:
        """Default: no-op. Feature-space augmentations override this instead."""
        return features


@dataclass
class AugmentEntry:
    augmentation: Augmentation
    p: float


class AugmentPipeline:
    """Ordered, independently-gated composition -- config-driven."""

    def __init__(self, entries: Sequence[AugmentEntry]) -> None: ...

    def apply_structural(self, anim, resolved, rng) -> tuple[Anim, ResolvedSkeleton, JointEdit]:
        """Runs each entry's Bernoulli(p) trial in order; composes JointEdits."""

    def apply_features(self, features, spec, resolved, rng) -> np.ndarray:
        """Runs each entry's Bernoulli(p) trial in order on the extracted array."""
```

An empty pipeline (`augmentations: []`) returns `JointEdit.identity(...)` and the
input array unchanged -- this is what makes success criterion 4 hold without a
special case in `MotionDataset`.

Each `AugmentPipeline` call draws its own Bernoulli trials from the dataset's
existing `self._rng`, so augmentation is seeded exactly like windowing already
is (`config.seed`), and reproducible for the same reason `RandomCrop` is.

## 6. `DropEndEffector` and `DuplicateJoint`

```python
@AUGMENTATIONS.register("drop_end_effector")
@dataclass
class DropEndEffector(Augmentation):
    rate: tuple[float, ...] = (0.1, 0.2, 0.3)   # one is drawn uniformly per call

class DuplicateJoint(Augmentation):
    ...  # no parameters; one candidate is drawn uniformly per call
```

Leaves are computed from `parents` directly (a joint index that never appears
in `parents` is a leaf) -- PoseYdon has no separate "kinematic chains" concept
and does not need one; the reference's `kinematic_chains[i][-1]` and "a leaf of
`parents`" are the same set. Candidate filtering:

- `DropEndEffector`: leaves, minus `resolved.foot_indices`, minus any joint
  appearing in `resolved.facing_indices`. If the filtered set is empty, this
  call is a no-op (identity edit) rather than an error -- a 6-joint scorpion
  claw with no spare end-effectors should not crash a training run.
- `DuplicateJoint`: joints with exactly one child, parent index `> 0` (not
  root), not the root, not in `facing_indices`. Same empty-set fallback.

Both are registered in a new `AUGMENTATIONS: Registry[Augmentation]` in
`poseydon/augment/base.py`, mirroring `FEATURES`/`CONDITIONERS`/`WINDOWS`.

## 7. Configuration

```yaml
# configs/train.yaml (new key, default empty -- opt-in like conditioners)
augmentations: []
```

```yaml
# configs/augment/topology.yaml (an example group, not required to use groups)
augmentations:
  - {_target_: poseydon.augment.DropEndEffector, p: 0.15, rate: [0.1, 0.2, 0.3]}
  - {_target_: poseydon.augment.DuplicateJoint,   p: 0.15}
```

Unlike `features`/`conditioners` (bare registry names, because they need no
per-entry parameters beyond a name), augmentation entries carry a probability
and sometimes other fields, so they use full Hydra `_target_` blocks -- the
same pattern already used for `data.window`. `poseydon train
augmentations='[{_target_:poseydon.augment.DropEndEffector,p:0.2}]'` overrides
from the CLI the same way any Hydra list does.

`training/build.py::build_dataset` gains one line:
`augmentations=[instantiate(e) for e in config.get("augmentations", [])]`,
passed to `MotionDataset.__init__` as a new `augmentations: Sequence[Augmentation] = ()`
parameter, wrapped internally in one `AugmentPipeline`.

## 8. Interaction with windowing and normalization statistics

Structural augmentation runs on the **full clip's** `Anim`, before windowing --
topology is a property of the skeleton, not of a time slice, so it must not
vary frame-to-frame within one window. Feature-space augmentation (§5,
`apply_features`, none shipped in v1) runs after normalization and before the
window crop, operating on the same `(frames, joints, dim)` array windowing
already slices.

Normalization statistics are fit once per skeleton over **unaugmented** clips
(`MotionDataset._normalizer`, unchanged) -- augmenting the corpus used to fit
them would make the statistics themselves random per run. `JointEdit.transport`
instead re-derives a per-window `Normalizer` by gathering rows from the
already-fitted one, exactly mirroring how the reference deletes/duplicates rows
of its precomputed `mean`/`std` rather than recomputing them. For a dropped
joint this is exact (its row simply no longer appears); for a duplicated joint
the new row copies the original joint's statistics, a defensible approximation
in the same spirit as -- but simpler than -- the reference's parent/child
average.

The T-pose rest frame (`MotionDataset._rest_frame`, feeding the `TPose`
conditioner) is transported the same way via `JointEdit.transport_row`, since
it is itself one already-normalized feature row.

## 9. Testing

- **Contract**: `AugmentPipeline` with no entries is a no-op end to end
  (`augmentations: []` byte-identical to current `MotionDataset` output) --
  guards success criterion 4 directly.
- **`DropEndEffector`**: on a real fixture (e.g. Goat), dropping a known
  non-foot end-effector shrinks every per-window array by exactly one joint,
  `parents` stays a valid tree (`check_topological_order`), and FK positions of
  every *surviving* joint are bit-identical to the unaugmented window.
- **`DuplicateJoint`**: on a real fixture, the duplicated joint's own FK
  position and all descendants' FK positions are bit-identical to the
  unaugmented window (the exactness argument in §4, made concrete); the new
  joint's position is the midpoint of the split bone.
- **Facing/foot exclusion**: augmenting a fixture many times over many seeds
  never selects a facing or foot joint (property test over `rng` seeds).
- **Empty-candidate fallback**: an augmentation with no legal candidates (e.g.
  `DuplicateJoint` on a 2-joint skeleton) returns the identity edit rather than
  raising.
- **`JointEdit.transport`**: dropping joint `k` from a `Normalizer` removes row
  `k` and nothing else; composing two edits yields the same result as applying
  them one after another.
- **End to end**: `MotionDataset` with `augmentations=[DropEndEffector(p=1.0)]`
  yields a batch whose `Topology`/`TPose`/`NormalizationStats` conditioning is
  self-consistent (same joint count as the feature tensor) without any change
  to the conditioner classes themselves.

## 10. Implementation status

Not yet implemented -- this spec is pending review.

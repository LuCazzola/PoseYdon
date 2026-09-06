# Data pipeline and training parity — design

Date: 2026-09-06

## Problem

PoseYdon cannot train. `poseydon train` fails on the first batch because
`data/truebones/aligned/` does not exist, and the corpus index on disk is stale:
81 clips over 7 skeletons, while the preprocessing scripts have produced 1150
clips over 73. Reaching parity with `external/neural_motion_blending` needs more
than re-running ingest, and the pieces divide into three groups.

**The dataset is assembled in the wrong place.** Per-skeleton statistics are
fitted at runtime (`data/dataset.py:122-130`), re-extracting features for every
clip of a skeleton on first touch, and features are re-extracted again on every
`__getitem__`. At 81 clips this is tolerable; at 1150 it is the difference
between a run and a stall. Nothing about a rig is stored: no statistics, no rest
frame, no joint-name embeddings.

**The feature vector carries joints that hold nothing.** BVH End Sites are read
as ordinary joints (`io/bvh.py:14`), so BrownBear presents 49 joints where the
reference presents 38. Every one of the extra eleven has a zero-length bone.

**The transform is one-way.** `scripts/process_dataset_truebones.py` removes the
bind rotation and rotates each clip to face +Z, recording neither. A generated
clip therefore cannot be returned in the representation a user supplied, which
is the difference between a benchmark and an application.

Beyond the data path, the models are missing conditioning the reference gives
them (T5 joint names, the temporal attention band, the crop offset) and the
training loop is missing the per-rig balanced sampler, the learning-rate
schedule and any checkpoint policy.

## Scope

**In.** The four-stage data path from a raw corpus to a trained checkpoint: the
`prepare` stages and their inverses; `scripts/build_features.py` and the on-disk
layout; joint reduction and expansion; the per-block normalization policy; the
processing recipe bound to a checkpoint; the `MotionDataset` read path and the
`joint_names` / `crop_start` conditioners; the temporal attention band; the
per-rig balanced sampler, `StepLR`, checkpointing and resume; a GPU training
image; six tests.

**Out.** Validation splits and evaluation metrics. Generation during training.
KL weight annealing (`_anneal_lambdas` in the reference), which matters only for
`MoDiffAE` with `kl_bottleneck: true`, off by default. Retargeting.

## 1. Three levels of representation

The pipeline moves motion through three levels. Confusing them is the source of
most of what is wrong today.

| level | where | joints | geometry |
|---|---|---|---|
| **source** | `Truebone_Z-OO/<Species>/*.bvh`, or a user's upload | as authored | as authored |
| **prepared** | `bvh/`, `fbx/` | same as source | bind removed, faced +Z, rigid, scaled, grounded, XZ-centred |
| **reduced** | features | degenerate joints removed | as prepared |

Preparation changes geometry and preserves structure. Reduction changes
structure and preserves geometry. Both are invertible, and the application
round-trip walks the whole chain in both directions:

```
source ──prepare──▶ prepared ──reduce──▶ features ──▶ model ──▶ features
                                                                   │
source ◀─unprepare── prepared ◀──expand────────────────────────────┘
```

## 2. On-disk layout

Organised **by entity**: everything about one character lives under
`rigs/<Rig>/`, everything about one clip shares a basename under
`clips/<Rig>/`. One `ls` answers "what do we know about BrownBear".

```
data/truebones/
├── source/<Rig>/…                raw BVH and FBX, as purchased or uploaded
│
├── rigs/_base.yaml               shared manifest fragment
├── rigs/<Rig>/
│   ├── manifest.yaml   authored  facing, foot_joints, contact, fps, tags
│   ├── prepare.npz     stage 1   rest rotations, scale, ground, XZ, per-clip facing table
│   ├── mesh.npz        stage 1   verts, faces, skin weights, bind pose
│   ├── skeleton.npz    stage 2   offsets, parents, raw and humanized names,
│   │                             reduction map, prepare constants, name embeddings
│   └── stats.npz       stage 2   mean, std, blocks, tpose frame  (schema-dependent)
│
├── clips/<Rig>/
│   ├── <action>.yaml   authored  split, text, notes
│   ├── <action>.bvh    stage 1   prepared motion, openable in a DCC tool
│   ├── <action>.fbx    stage 1   the same motion, with skin
│   └── <action>.npz    stage 2   prepared motion as arrays + this clip's facing quat
│
└── index.jsonl         stage 2   derived join, one flat row per clip
```

A rig is a **directory**, which retires the `_`-prefix convention
`available_skeletons` uses today to tell manifests from shared fragments. The
clip id stays `<Rig>__<action>` as the index identity; on disk the rig is the
directory, so the filename need not repeat it.

Because authored and derived files interleave, `.gitignore` matches by extension
rather than by directory:

```
data/*/source/
data/**/*.npz
data/**/*.bvh
data/**/*.fbx
data/**/index.jsonl
```

**Every file has exactly one writer.** Stage 1 writes `prepare.npz`, `mesh.npz`
and the prepared BVH/FBX; stage 2 writes `skeleton.npz`, `stats.npz`, the clip
`.npz` and the index. Stage 2 reads `prepare.npz` and folds the rig constants
into `skeleton.npz` and each clip's facing quaternion into its own `.npz`, so
the training-facing files are self-contained and stage 2 can be re-run without
Blender.

Three properties follow.

**Only `stats.npz` depends on the feature schema.** Changing `features:` re-runs
one pass over already-prepared motion — seconds, not a corpus rebuild. It records
the block layout it was fitted under, and `MotionDataset` refuses a mismatch
rather than normalizing with the wrong numbers. One schema is stored at a time;
building a second overwrites the first.

**Training never opens a `.bvh`.** The `.bvh` and `.fbx` beside each clip exist
so a human can look at the data; `MotionDataset` reads only `.npz` and `.yaml`.

**Labels are authored, not owned by the build.** A rebuild writes a label file
only when one is absent. `--relabel` refreshes derived fields and preserves every
key it did not write, so a hand-written `text:` survives.

`tags` lives once, in the manifest, and is joined into the index rows at build
time. Today `ingest/pipeline.py:141` copies it into every `ClipRecord`, writing
`[quadruped, mammal]` twenty-two times for BrownBear.

## 3. Stage 1 — prepare

`scripts/process_dataset_truebones.py` and `process_dataset_truebones_fbx.py`
keep their role: turn a raw corpus into a clean, coherent, viewable one. They
already remove the bind rotation and rotate each clip to face +Z. They gain the
three normalizations currently living in `ingest/align.py`, and the obligation
to record their own inverse.

The stages, each implementing `apply` and `invert`:

| stage | fitted from | parameter recorded |
|---|---|---|
| `RestRelative` | the rig's rest pose | rest rotations `(J, 4)`, per rig |
| `FaceAxis(axis="+Z")` | **each clip's** frame 0 | facing quaternion `(4,)`, per clip |
| `EnforceRigid(joint_translation="drop")` | — | the rest offsets; structure inverts, values do not |
| `CentreXZ` | the rig's rest pose | XZ offset, per rig |
| `ScaleToMeanBoneLength(target=0.2092)` | the rig's rest pose | scale factor, per rig |
| `PutOnGround` | the rig's rest pose | ground height, per rig |

Order is contractual and matches the reference's `process_anim`: rotate, centre,
scale, ground. Reordering changes the result.

The per-clip facing is the one that would be lost silently. `rotate_to_face_axis`
derives it from that clip's frame 0, so once the prepared file faces +Z the
original orientation is unrecoverable from it. It is written to
`rigs/<Rig>/prepare.npz` as a clip-id-keyed table.

`EnforceRigid.invert` re-inserts the per-joint position channels filled with the
rest offsets. A source BVH declaring six channels per joint comes back declaring
six channels per joint, with the same hierarchy, names and channel order, and an
FBX keeps its translation curves. Only the values differ — see Limitations.

The reference stores the rig-level constants for the same reason —
`root_pose_init_xz`, `scale_factor`, `ground_height`, `tpos_rots` in `cond.npy`,
commented *"stored so BVH→features conversion at eval time can use exactly the
same values as dataset build time"*. It does not store the per-clip facing,
because it re-derives features from raw BVH rather than round-tripping.

**Cross-source coherence** is a property of this stage: the skeleton extracted
from `bvh/<Rig>` and from `fbx/<Rig>` must agree on joint count, names, parent
array and rest offsets. Currently verified once, by hand, in a commit message.
It becomes a test.

### Rest geometry is not rig-level

`FaceAxis` is clip-scoped and `rotate_rig` turns OFFSETS with the motion, not
just rotations, so **prepared clips of the same rig do not share an OFFSET
block** — each clip's rest geometry is rotated by that clip's own frame-0
facing correction, and those corrections differ clip to clip (up to 0.42 bone
lengths apart, measured on BrownBear). "The skeleton extracted from
`clips/<Rig>/`" is therefore not a single object; there are as many rest
geometries as there are clips, one per facing quaternion.

This matters beyond this phase. Spec §4 has a later phase write one
rig-level `skeleton.npz` carrying `offsets`, and `features/reconstruct.py`
does forward kinematics with them. The canonical rig-level rest geometry has
to be the **T-pose clip's** offsets specifically — using any other clip's
offsets with that clip's own rotations is fine (they are mutually
consistent), but using rig-level offsets from one clip together with a
*different* clip's rotations yields a skeleton rotated by `R_rest · R_clip⁻¹`
relative to the truth. A later phase must take rig-level offsets from the
rest-pose clip, never from an arbitrary one.

The reference implementation avoids this entirely by rotating only the root
joint to face +Z, leaving every other joint's offset untouched and shared
across all of a rig's clips by construction. PoseYdon's choice to rotate the
whole rig (`rotate_rig`, not just the root) is deliberate: it is what lets a
DCC tool draw the rest skeleton and the animation with the same convention,
rather than the root pointing one way and the limbs implicitly assuming
another. That convenience is what produces the per-clip OFFSET divergence
documented here.

## 4. Stage 2 — `scripts/build_features.py`

A thin Hydra runner over a new `poseydon/build/` package, configured by
`configs/dataset/<name>.yaml`. Each pluggable behaviour is a `_target_`; each
value is a plain key.

```yaml
# configs/dataset/truebones.yaml
root:   data/truebones
schema: [ric_pos, rot6d, local_vel, foot_contact]

corpus: {_target_: poseydon.build.corpus.PerRigDirectory, pattern: "*.bvh"}
reduce: {_target_: poseydon.features.reduce.DropDegenerateJoints, tolerance: 1.0e-8}
names:  {_target_: poseydon.build.names.Humanize, lowercase: true, expand_sides: true}
text:   null
```

Passes:

- **per clip** — read `clips/<Rig>/<action>.bvh`, write `<action>.npz` carrying
  the motion arrays and that clip's facing quaternion from `prepare.npz`, and
  write `<action>.yaml` if absent.
- **per rig** — build the reduction map; humanize joint names; optionally encode
  them; fold in `prepare.npz` and `mesh.npz`; write `skeleton.npz`.
- **per rig, schema-dependent** — extract the schema over every clip of the rig
  through the reduction, fit the `Normalizer` under the per-block policy, extract
  and normalize the rest frame, write `stats.npz`.
- **once** — write `index.jsonl`, joining clip labels with rig-level manifest
  fields.

`--stats-only` runs the third pass alone.

**On T5.** `text` is `null` by default and lives behind a `poseydon[text]` extra.
The test image deliberately excludes `transformers` (`Dockerfile.compat`: *"never
depends on … a 900MB T5 download"*), so the suite must never require it and a
corpus must build without it. Embeddings are `(J, 768)` per rig, about 11 MB
across all 73, and go straight into `rigs/*.npz` rather than into a separate
name-keyed cache as the reference does.

## 5. Normalization policy

Declared per block in the dataset config rather than fixed in code:

```yaml
normalize:
  - {name: ric_pos,      center: true,  scale: joint_block}
  - {name: rot6d,        center: false, scale: joint_block}
  - {name: local_vel,    center: true,  scale: joint_block}
  - {name: foot_contact, center: false, scale: none}
```

`scale` modes:

| mode | statistic shape | notes |
|---|---|---|
| `channel` | per `(joint, channel)` | today's `Normalizer.fit` |
| `joint_block` | one scalar per `(joint, block)` | uniform within a block |
| `block` | one scalar per `(root / non-root, block)` | reference-exact |
| `none` | `std = 1` | for flags |

This matters, and not only for parity. Per-channel std makes a barely-moving toe
unit-variance too, so reconstruction loss weights it as heavily as the root.
Pooling preserves relative magnitude. More sharply: dividing a 6D rotation row by
a *uniform* scalar leaves the rotation unchanged after Gram-Schmidt, while
dividing per-channel does not — so pooling is what lets the rotation block
survive normalization at all. `Normalizer`'s existing `1e-6` epsilon stops being
load-bearing, because pooled statistics have no zero-variance channels.

The reference's `get_mean_std` corresponds to `scale: block` on every block plus
`center: true`, with zero-variance contact forced to `1.0`. That combination
remains available and is what parity tests use.

## 6. Joint reduction and expansion

Applied at **feature extraction time**, never baked into the corpus. Structure is
preserved on disk so a user's rig comes back the way they gave it.

The rule has two cases, and conflating them loses real motion:

- **Zero-offset leaves** — drop. The bone reaching them has no length, so the
  parent's rotation moves nothing and has no subtree to propagate to. It is
  unobservable; nothing is lost. Exactly recoverable by re-inserting at the
  recorded index and parent with zero offset and identity rotation. Ten per rig,
  typically every End Site.
- **Zero-offset internals that are an only child** — collapse. Their rotation does
  swing their subtree, so they cannot simply be deleted. Fold it into the parent
  (`rot(p) ← rot(p) · rot(z)`), reparent the children onto `p`, remove `z`. The
  usual case is `Bip01_Pelvis` sitting on the root `Hips`.
- **Zero-offset internals with siblings** — keep. Folding into the parent turns
  the parent, and therefore turns every *other* child of that parent too. The
  fold is exact only when the joint is its parent's sole child, so this case is
  left alone rather than silently displacing its siblings.

Reduction repeats to a fixed point, and the repetition does most of the work:
removing a zero-offset leaf can leave its parent a zero-offset leaf in turn, and
that parent is then droppable by case 1 even though it began as an internal
joint with siblings. The sibling condition is therefore evaluated against the
hierarchy **as it stands at the moment of removal**, never against the original.

Measured on the prepared corpus, this reproduces the reference's stored joint
counts exactly:

| rig | full | removed | reduced | reference `.npy` |
|---|---|---|---|---|
| BrownBear | 49 | 11 | 38 | 38 |
| Flamingo | 53 | 13 | 40 | 40 |
| Goat | 40 | 9 | 31 | 31 |
| Crab | 64 | 10 | 54 | 54 |
| Scorpion | 78 | 15 | 63 | 63 |

Case 3 is real but rarer than a scan of the original hierarchy suggests. A
joint only survives it if its children have genuine offsets, so it never
becomes a leaf — `Tukan/kosi` and `Bear/NPC_Spine1` are the shape. Neither rig
is in the reference's fixture set, so keeping them costs no parity.

The map is a `JointEdit` — the type already in `augment/joint_edit.py`, which
records `source_of[new] = old` and transports normalizer rows and the rest frame
through a relabelling. The reduction is one more edit, computed once per rig,
stored in `rigs/<Rig>/skeleton.npz`, and composed with the per-sample augmentations via
the existing `JointEdit.compose`. Order: reduce (deterministic, rig-level) then
augment (random, per-sample). `resolve()`'s facing and foot indices are remapped
through the same edit by `augment/topology.py::_reindex_resolved`.

`features/reconstruct.py` gains expansion as its final step, so `poseydon sample`
writes a BVH with the user's joint set.

## 7. Joint names

`SkeletonManifest.strip_joint_prefix` exists (`core/skeleton.py:76`), is parsed,
is documented, and is read by nothing. It is replaced by a pluggable stage that
computes the prefix rather than requiring it to be declared:

1. strip the prefix common to every joint of that rig (`Bip01_`, `BN_`,
   `mixamorig:`), computed from the rig's own names;
2. `_` → space, split camelCase;
3. expand isolated `R` / `L` to `right` / `left`;
4. lowercase.

`Bip01_R_Thigh` → `right thigh`.

Both forms are stored. **Raw names remain the canonical identity** — manifests
and `resolve()` match on them, and that is what makes a re-exported rig fail
loudly instead of silently mirroring the character. Humanized names are what the
text encoder sees.

## 8. The processing recipe

Round-tripping needs two things with different lifetimes:

- **the recipe** — which stages ran, in what order, with what settings.
  Rig-independent, frozen at training time, travelling with the checkpoint;
- **the fit** — what those stages resolved to for a given rig or clip: scale
  factor, ground height, reduction map, statistics, this clip's facing
  quaternion. Rig-dependent, living in `rigs/`, `stats/`, `motion/`.

Every stage implements `apply` and `invert`, so the chain reverses by walking it
backwards. A stage that cannot be inverted declares so at build time rather than
at the moment a user asks for their asset back.

```yaml
# runs/<run>/recipe.yaml
poseydon: 0.1.0
dataset: truebones

prepare:
  - {_target_: poseydon.build.prepare.RestRelative}
  - {_target_: poseydon.build.prepare.FaceAxis, axis: "+Z"}
  - {_target_: poseydon.build.prepare.EnforceRigid, joint_translation: drop}
  - {_target_: poseydon.build.prepare.CentreXZ}
  - {_target_: poseydon.build.prepare.ScaleToMeanBoneLength, target: 0.20921428571428569}
  - {_target_: poseydon.build.prepare.PutOnGround}

reduce: {_target_: poseydon.features.reduce.DropDegenerateJoints, tolerance: 1.0e-8}
features: [ric_pos, rot6d, local_vel, foot_contact]
normalize:                    # the per-block policy of §5, verbatim
  - {name: ric_pos,      center: true,  scale: joint_block}
  - {name: rot6d,        center: false, scale: joint_block}
  - {name: local_vel,    center: true,  scale: joint_block}
  - {name: foot_contact, center: false, scale: none}
names: {_target_: poseydon.build.names.Humanize, lowercase: true, expand_sides: true}

fit_root: data/truebones
```

Written into the run directory at train start **and** stashed in the
checkpoint's `hyper_parameters`, so a `.ckpt` moved on its own still works. The
sidecar wins when present; the embedded copy is the fallback.

This closes a live hole: `poseydon sample` recomposes `configs/sample.yaml` from
scratch (`cli.py:83`), so sampling a checkpoint under a different `features:`
than it was trained with produces silent garbage. With a recipe, `sample` reads
the recipe and rejects overrides that touch the data path.

## 9. Read path

`MotionDataset.__getitem__` becomes:

```
load clips/<Rig>/<action>.npz   prepared RigidBodyAnimation, full joint set
apply the rig's reduction       JointEdit from rigs/<Rig>/skeleton.npz
apply augmentations             JointEdit composed onto it
extract features                the schema
normalize                       rigs/<Rig>/stats.npz, transported through the composed edit
crop                            window policy
```

**Statistics are loaded, not fitted.** `_normalizer` becomes a file read. That
removes the startup cost and the reason the dataset caches every animation in
memory (`dataset.py:106-109`).

**`ClipView` gains two sidecars of one type**, since labels and rig data are the
same kind of thing at different scope:

```python
@dataclass(frozen=True)
class ClipView:
    record: ClipRecord
    anim: RigidBodyAnimation
    resolved: ResolvedSkeleton
    start: int
    clip: Annotations   # clips/<Rig>/<action>.yaml plus any per-clip arrays
    rig: Annotations    # rigs/<Rig>/*.npz — names, embeddings, mesh, weights
```

A future `text` conditioner reads `view.clip["text_emb"]`; `joint_names` reads
`view.rig["name_emb"]`. Neither needs a change to the dataset, the collate or
any model.

**Worker RNG.** `self._rng` (`dataset.py:77`) is one `numpy.random.Generator`
copied into every dataloader worker, so with `num_workers > 0` every worker draws
the identical crop and augmentation stream. It is reseeded per worker in
`worker_init_fn` from the base seed and the worker id.

### Conditioners

- **`joint_names`** — `(J, 768)` from `rigs/<Rig>/skeleton.npz`, padded on the
  joint axis.
  `configs/model/anytop.yaml` gains `name_embedding_dim: 768`, without which
  `AnyTop.name_projection` stays `None` and the embeddings are silently ignored
  (`models/anytop.py:139`).
- **`crop_start`** — the window offset, from `ClipView.start`. `AnyTop` reads
  `cond.get("window_start")` (`anytop.py:207`) while `MoDiffAE` reads
  `cond.get("crop_start")` (`modiffae.py:341`); they unify on `crop_start` and
  `AnyTop` is corrected.

### The temporal attention band

The reference restricts temporal attention to a sliding window of 31 frames
(`create_temporal_mask_for_window`). PoseYdon builds only a padding mask, so
attention is global.

`modiffae.py:38-41` calls the band "a dataset property rather than a model one"
and expects it as a `temporal_valid` conditioner. This design disagrees.
Attention locality is a property of the attention module — the dataset does not
decide how far a model may look — and routing it through conditioning would
require `Conditioner.collate` to learn the batch's frame count, which it does not
currently receive. It becomes a model parameter, `temporal_window: 31`,
intersected with the padding mask inside the model. `TEMPORAL_VALID` stays as an
optional override so a sampling-time control can still inject its own mask.

## 10. Training loop

- **Per-rig balanced sampling.** The reference's `TruebonesSampler` is a
  `WeightedRandomSampler` giving each character an equal share, and `--balanced`
  appears in both documented training commands. Without it BrownBear's 22 of 81
  clips dominate. Added as `poseydon.data.sampler.BalancedByRig`, configured in
  `configs/data/*.yaml`, weight `1 / (n_rigs · n_clips_of_rig)`.
- **DataLoader.** `drop_last=True` and `num_workers` from config, matching the
  reference's 8 (`get_data.py:22`).
- **Learning rate.** `StepLR(step_size=10000, gamma=0.99)` stepped per
  optimizer step, from `configure_optimizers`. Currently a bare `AdamW`
  (`training/lightning.py:61-64`).
- **Checkpointing.** `ModelCheckpoint(save_last=True, every_n_train_steps=25000,
  save_top_k=-1)`. There is no checkpoint callback today (`cli.py:55-67`), so the
  `runs/**/last.ckpt` that the README and `poseydon sample` both instruct you to
  use is never written. `poseydon train --resume <ckpt>` maps to
  `trainer.fit(ckpt_path=...)`.
- **GPU image.** `pyproject.toml:23-26` pins torch to the CPU wheel index and
  `docker-compose.yml` has no train service. A `Dockerfile.train` reinstalls
  torch from the CUDA index after `uv sync`, and a `train` service is added.

### One loss correction

`FootSkateLoss` reads the contact flag from **normalized** features and
thresholds at `> 0.5` (`losses/footskate.py:227,231`), while the reference
denormalizes before this loss (`gaussian_diffusion.py:1620-1628`). For a foot
planted most of the time — high mean, low std — the z-scored `1` falls below
`0.5`, every planted frame is discarded and the loss silently goes to zero. It
must read the block through `raw_block`, as the positions beside it already do.

## 11. Tests

Six, chosen to be load-bearing rather than exhaustive.

1. **Round-trip.** `source → prepare → reduce → expand → unprepare → source`
   returns joint positions to BVH's six-decimal write precision, over one biped,
   one quadruped and one milliped. This is the single test the application story
   rests on; if either inverse is wrong it fails.
2. **BVH ↔ FBX skeleton agreement.** Joint count, names, parent array and rest
   offsets agree between the two prepared corpora, per rig. Skips when the FBX
   artefacts are absent, as the reference-dependent tests already do, because the
   test image has no Blender.
3. **Reduction is geometrically exact.** World-space joint positions are
   unchanged by `reduce` on every surviving joint, and the reduced skeleton
   contains no zero-offset leaf. Pins the drop-versus-collapse rule against
   regression far more tightly than a joint count would.
4. **Golden feature parity.** Features extracted by PoseYdon reproduce the
   reference's own `.npy` arrays block by block, with foot contact bit-exact,
   over the joints the two representations share, matched by name. A trimmed
   restoration of the deleted test.
5. **Normalization policy.** Each `scale` mode produces statistics of the
   documented shape, and `joint_block` leaves a 6D row's recovered rotation
   unchanged under Gram-Schmidt. A property, not an example.
6. **Build-then-train smoke.** `build_features` over a tiny fixture corpus, then
   two training steps: the loss is finite, and a schema mismatch between
   `features:` and `stats/` raises before the first step.

## 12. Migration

Each new module lives where its *consumer* is, not where it is produced, so
training imports `features/` and `core/` and never imports `build/`.

| module | role |
|---|---|
| `build/prepare.py` | the invertible geometry stages |
| `build/corpus.py` | clip discovery and label derivation |
| `build/names.py` | humanize, and optionally encode, joint names |
| `build/index.py` | moved from `ingest/`, unchanged |
| `build/pipeline.py` | the four passes |
| `core/recipe.py` | the recorded chain — read by `sample` and the application |
| `features/reduce.py` | reduction and expansion, inverted in `reconstruct.py` |
| `data/normalize.py` | extended with the per-block policy; no new file |

Seven new modules against six deleted, so the package count rises by one:

| path | fate |
|---|---|
| `src/poseydon/ingest/pipeline.py` | deleted; replaced by `build/pipeline.py` |
| `src/poseydon/ingest/align.py` | → `build/prepare.py`, as invertible stages |
| `src/poseydon/ingest/index.py` | → `build/index.py`, unchanged |
| `src/poseydon/ingest/__init__.py` | deleted with the package |
| `src/poseydon/preproc/rest_pose.py` | → `build/prepare.py::RestRelative` |
| `src/poseydon/preproc/__init__.py` | deleted with the package |
| `src/poseydon/datasets/` | deleted; contains only an empty `__init__.py` |
| `SkeletonManifest.strip_joint_prefix` | removed; `Humanize` computes the prefix |
| `poseydon ingest` | → `poseydon build` |
| `data/truebones/skeletons/<Rig>.yaml` | → `rigs/<Rig>/manifest.yaml` |
| `data/truebones/{bvh,fbx}/<Rig>/` | → `clips/<Rig>/` |
| `data/truebones/Truebone_Z-OO/` | → `source/` |
| `data/truebones/index.jsonl` | regenerated; the current file is stale |

`CorpusIndex.load` calls `ClipRecord(**payload)` (`index.py:92`), so any index
written by a newer build fails to load in an older one. The loader is made to
ignore unknown keys.

## Limitations

**Unseen rigs.** A rig the model has never seen has no fitted statistics, and the
recipe's `fit_root` cannot supply them. The three plausible answers — fit from
the user's own clips, borrow the nearest known rig, or refuse until an explicit
stats build runs — behave very differently on a few seconds of input motion.
This design does not choose. The recipe carries an `unseen_rig` field and the
application decides; treat cross-rig generalization at inference as unsupported
until it is settled against real usage.

**Collapsed internal joints are world-exact, not representation-exact.**
`rot(Hips) · rot(Bip01_Pelvis)` can be split infinitely many ways, so expansion
puts the whole product on the parent and re-inserts the joint with identity.
Every joint lands in the right place with the right name, parent and offset; what
differs is which of two coincident joints stores the rotation. Unobservable in
world space, and for generated motion there was never an original split.

**`EnforceRigid` preserves structure but not values.** `invert` re-inserts the
per-joint position channels, so the returned file declares the same channels in
the same order under the same hierarchy and an FBX keeps its translation curves —
the rig is structurally what the user handed over. What is lost is the animated
content of those channels: they come back holding the constant rest offsets,
because dropping per-joint translation discards real motion, measured at roughly
10% of skeleton size on raw Truebones. This is intrinsic to a rotation-based
representation rather than a defect of the implementation, and it is why the
stage is explicit rather than silent.

**A clip whose joints differ from its own rest pose is rejected.**
`RestRelative` requires the clip it prepares to carry exactly the joints its
fitted rest pose declares, in the same order. It has to: a joint the rest pose
never covered has no recorded bind rotation, so no inverse exists for it, and
the whole point of this stage is that the transform can be undone. The earlier
non-invertible implementation fell back to the clip's own frame-0 rotation for
such joints — exact for a childless End Site, an approximation everywhere else.

Measured over the corpus, this rejects **64 clips of 1153 across 8 of 73 rigs**.
Most lose one clip. Four are gutted: Ant keeps 1 of 18, Crab 1 of 11, Deer 1 of
21, Jaguar 1 of 14 — in each case the rig's own T-pose file declares a different
skeleton from its animation clips (Crab's T-pose has 54 joints against the
clips' 64). Those four are effectively unusable until someone supplies a rest
pose that matches their clips, which is a data question rather than a code one.
Recovering them by reinstating the approximate bind would trade this phase's
central guarantee for four characters.

**No validation split.** Out of scope by decision; every index row is
`split: train` and the Trainer runs with validation disabled.

**BVH and FBX clips do not pair by filename for the whole corpus.** Both
stage-1 scripts derive a clip's basename the same way —
`strip_skeleton_prefix(action_slug(stem), rig)` — on the assumption that the
two raw corpora name their files consistently enough for this to converge.
It does not hold everywhere. BrownBear's BVH files are `__<Action>.bvh`,
stripping the manifest's `brownbear_` prefix, while its FBX files are
`BEAR-<Action>.fbx` — `strip_skeleton_prefix` strips the *rig* name, not
`bear_`, so the two sides produce disjoint basenames (0 of 22 clips agree).
Elephant and Fox are worse: their FBX corpora ship `atk 1.fbx` against BVH's
`__Attack1.bvh`, a divergence no filename-derivation rule can close because
the two names share no derivable relationship at all. Closing this needs an
authored per-rig alias map from FBX filename to BVH action, not a cleverer
slug function; BrownBear, Elephant and Fox are the confirmed cases, and the
corpus has not been swept exhaustively for others.

**BVH and FBX still disagree on rest geometry at shared joints, residually.**
`ScaleToMeanBoneLength` averaging over zero-length End Sites (fixed above)
was the dominant cause of an earlier, larger disagreement — up to 8.1e-1 bone
lengths on Flamingo — and fixing it cut the error roughly tenfold. What
remains is real and unexplained: at the rig's own T-pose clip, the worst
per-rig disagreement is Flamingo 6.8e-2, BrownBear 5.5e-2 and Scorpion 3.3e-2
bone lengths, against Crab's 2.0e-5 (which passes; its BVH T-pose has exactly
one zero-length bone, so it was never much affected by the scale bug either).
A coordinate-frame mismatch was ruled out by measuring parent-local and
world-space forms and finding them identical to 1e-7.
`test_bvh_and_fbx_agree_on_rest_geometry` records this as an `xfail` with the
current numbers rather than resolving it — it matters for a later phase,
where `mesh.npz` skin weights are indexed by the FBX joint order and would be
applied to BVH-driven motion, so the disagreement means weights deforming
wrongly. Needs both bind poses inspected side by side in Blender.

## Phasing

Each phase leaves the repository working and is reviewable on its own.

1. **Invertibility.** `build/prepare.py`, the stage contract, the recorded
   constants, `features/reduce.py`, the layout migration, stage-1 script
   changes. Tests 1, 2 and 3. Reduction lands here rather than in phase 2
   because the round-trip test cannot be written without it.
2. **Build and layout.** `build_features.py`, the config group, the artefacts,
   names, the normalization policy. Tests 4 and 5.
3. **Read path.** `MotionDataset`, `Annotations`, the conditioners, the temporal
   band, the recipe, the footskate fix.
4. **Training loop.** Sampler, scheduler, checkpointing, resume, GPU image.
   Test 6.

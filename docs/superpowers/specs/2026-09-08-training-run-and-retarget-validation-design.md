# Training run and retarget validation — design

Date: 2026-09-08

Extends `2026-09-06-data-pipeline-and-training-parity-design.md`, whose Phase 1
landed at `b6184cf`. That spec's phases 2–4 are unbuilt; this one carries them
to a running training job and adds what it deliberately excluded: generation
during training, and retargeting.

## Problem

PoseYdon still cannot train, and the gap is wider than it looks from the last
commit. `poseydon train` fails before the first batch: `configs/data/truebones.yaml`
points `manifests` at `data/truebones/skeletons/`, a directory the Phase 1 layout
migration removed, and `index.jsonl` holds 81 rows addressing `aligned/*.npz`,
a layout that no longer exists. `MotionDataset` reads that old layout, fits
normalization statistics at runtime, and caches every animation in memory.

Beyond the read path, the training loop has no per-rig balanced sampler, no
learning-rate schedule, and no checkpoint callback — so the `runs/**/last.ckpt`
that both the README and `poseydon sample` instruct you to use is never written.
There is no experiment logging of any kind, and `pyproject.toml` pins torch to
the CPU wheel index on a machine with a GB10.

And there is no way to see whether a run is learning anything. The reference's
`evaluate()` is a stub that prints "not implemented"; what it actually does at
save time is `mix_during_training`, generating clips and logging them as video.
PoseYdon has no equivalent, and cannot have one, because it has no operation
that expresses retargeting.

## Scope

**In.** The remaining three phases of the parity spec (build stage, read path,
training loop); a full-corpus stage-1 rebuild; an aarch64 CUDA training image and
a `train` compose service; wandb logging; a `Retarget` operation and the
`LatentPin` control it needs; a `RetargetValidation` callback generating
artifacts and metrics for three fixed pairs; a `poseydon retarget` CLI sharing
the same code path; a root-promotion stage; the test gaps that remain.

**Out.** A held-out validation split and val loss — every index row is still
`split: train`. Text conditioning (T5 stays behind the `poseydon[text]` extra).
Blending and DDIM inversion. The BVH↔FBX basename alias map. KL annealing.
Resolving the residual BVH/FBX rest-geometry disagreement, which is measured
here but not fixed.

## Decisions taken

| decision | value |
|---|---|
| model | `MoDiffAE`, attention pool (`n_virtual_joints: 5`) |
| corpus | all 73 rigs, BVH-derived; FBX run separately as a coherence measurement |
| losses | `simple 1.0`, `geodesic 1.0`; footskate omitted — reference-exact |
| validation cadence | every 25 000 steps, plus once at step 0; one sample per pair |
| validation output | `.npz` + `.bvh` + `.mp4` per pair, plus five scalars |
| reconstruction | `positions_ik` — positions fitted back onto the rigid skeleton |
| rest pose | declared per rig as `rest_pose:` in its manifest; no runtime discovery |
| root promotion | hardcoded per-rig table in stage 1, for the 14 ground-locator rigs |
| wandb | project `poseydon`, entity and key from `.env` |

The three validation pairs, content source → target skeleton:

```
Flamingo / onelegbent  ->  Scorpion   (78 raw, 63 reduced)
Coyote   / attack3     ->  Crab       (64 raw, 54 reduced)
Goat     / headbutt    ->  Raptor     (48 raw, reduction not yet measured)
```

Raw counts include End Sites, which `io/bvh.py` reads as ordinary joints; the
reduced counts are the parity spec's measured figures, which match the
reference's stored arrays exactly. Action slugs are derived at stage 1
(`strip_skeleton_prefix(action_slug(stem), rig)`), so `Coyote/__Attack3.bvh`
becomes `attack3`; Flamingo's `onelegbent` and Goat's `headbutt` are already on
disk under those names, and Coyote's is confirmed at the corpus rebuild.

Coyote stands in for the "Wolf" of the original request: Truebones has no Wolf,
and `Coyote___Attack3_224` is the clip the reference bundles in `assets/`. These
mirror the reference's own retargeting demo, which transfers Flamingo motion onto
the Scorpion skeleton.

## 1 · The corpus is stale, not the code

The four rigs believed gutted are not. `RestRelative` rejects any clip whose
joints differ from its fitted rest pose, and for Ant, Crab, Deer and Jaguar the
file named `__TPOSE.bvh` is a *minority* skeleton — it disagrees with almost every
clip of its own rig. `b2b0151` fixed this by choosing the rest-pose file by modal
joint set rather than by filename. The prepared data on disk predates that commit
by twelve hours.

Re-running stage 1 for those four rigs at HEAD, into a scratch root:

```
Crab 10 clips · Ant 17 · Deer 20 · Jaguar 13   = 60 recovered
warning: Crab: rest-pose file chosen by name (__TPOSE.bvh) is not in the
         modal joint set (10/11 clips); using __Attack1.bvh instead
```

So no new preparation stage is needed, and `RestRelative` is untouched. What is
needed is a full-corpus re-run.

Measured across all 73 rigs (`tools/probe_jointset_spread.py`):

- **65 rigs** have one joint set shared by every clip.
- **8 rigs** carry one outlier file — Ant, Centipede, Crab, Deer, Elephant,
  HermitCrab, Jaguar, Trex. For four of them the outlier is the T-pose (the case
  above); for the other four the T-pose agrees with the majority and one genuine
  clip is odd. Either way exactly one file per rig is lost, correctly.
- **13 rigs have no T-pose file at all** — Anaconda, Bird, Camel, Cricket, Dog,
  Goat, Lion, Monkey, Pteranodon, Rat, SabreToothTiger, Scorpion-2, Trex. Their
  clips agree with each other, so any of them yields a self-consistent fit, but
  *which* one is chosen decides the rig's canonical rest geometry. §1.1 replaces
  today's "first file in the modal set" with a per-rig manifest declaration.

Expected yield: **~1145 clips over 73 rigs**, against 1153 raw and the 81-row
index on disk. The parity spec predicted 64 rejections across 8 rigs; the modal
fix reduces that to 8.

### 1.1 · The rest pose is authored where it cannot be found

**Every rig declares its rest pose in its own manifest**, as
`rest_pose: <filename>` naming the raw clip whose first frame is that rig's rest
pose. Nothing is discovered at runtime, and a rig that declares nothing is a
build error rather than a rig that quietly receives an arbitrary pose.

A rule cannot do this. Matching `idle` as a substring picks Lion's
`__DeathIdle.bvh`, Jaguar's `__LieIdle.bvh` and Trex's `__idle_attack.bvh` — a
dying pose, a lying pose and a crouched attack, which are the exact poses the
choice exists to avoid. And today's final fallback, "first file in the modal
set", is arbitrary: it is what gave Crab a rest geometry fitted from
`__Attack1.bvh`.

The manifest is also where this belongs for reuse. The selection rule currently
exists in three copies — the stage-1 script, `test_roundtrip.py` and
`test_bvh_fbx_agreement.py` — and only the script carries `b2b0151`'s modal-set
fix, which is why the round trip has been silently skipping Crab. One
declaration, read by every consumer, removes the class of bug rather than one
instance of it.

**There is already a manifest key for this, and it is a trap.**
`SkeletonManifest.tpose` is parsed and read by `ingest/pipeline.py:94` and
`data/dataset.py:145`, both guarded by `if manifest.tpose is not None and
manifest.tpose.is_file()` — and **no manifest declares it**. Both consumers have
always taken their silent fallback, so the rest frame the model is shown as a
rig's identity is the first frame of an arbitrary clip, for all 73 rigs. It is
also the wrong shape: it resolves relative to the manifest, while stage 1 needs
a raw source file and everything downstream needs the prepared clip. It is
removed rather than populated, and `rest_pose` holds a filename that each
consumer resolves in its own domain.

56 rigs declare their T-pose. The 17 whose named T-pose is missing or unusable
declare these:

| rig | rest file | why |
|---|---|---|
| Anaconda | `__Idle.bvh` | |
| Ant | `__Idle.bvh` | T-pose is a minority skeleton |
| Bird | `__IdleLoop.bvh` | |
| Camel | `__IdleLoop.bvh` | |
| Crab | `__Walk.bvh` | T-pose is a minority skeleton; no idle clip |
| Cricket | `__Idle.bvh` | |
| Deer | `__Idle.bvh` | T-pose is a minority skeleton |
| Dog | `__Idle.bvh` | |
| Goat | `__Idle.bvh` | |
| Jaguar | `__Idle.bvh` | T-pose is a minority skeleton; `__LieIdle.bvh` is the trap |
| Lion | `__SlowIdle.bvh` | `__DeathIdle.bvh` is the trap |
| Monkey | `__Idle1.bvh` | |
| Pteranodon | `__FlyLoop.bvh` | no idle clip; flying creature |
| Rat | `__Trottle.bvh` | no idle clip; trot is its nearest gait |
| SabreToothTiger | `__Startwalk.bvh` | no idle clip among 44; frame 0 is the standing start |
| Scorpion-2 | `__Idle.bvh` | |
| Trex | `__walk_loop.bvh` | every `idle_*` clip is idle-plus-action; see below |

**The modal-set check stays as the guard over the declaration, and it is not
ceremony.** Trex's natural neutral pick, `__STILL.bvh`, *is* that rig's outlier
file — 66 joints against the modal 78. A declaration trusted on its own would
have silently destroyed the largest rig in the corpus. An authored value is a
judgement about pose content and no test can confirm it; what is confirmed is
that the named file exists and carries the rig's modal joint set, and a
declaration failing that stops the build rather than falling back.

### 1.2 · The root is promoted where it is a ground locator

**14 of 73 rigs root the skeleton at a ground locator rather than at a body
joint.** Measured on the rest pose as the height fraction
`(rootY − minY) / (maxY − minY)`:

```
14 rigs   f ≤ 0.005    Bear Camel Crow Dog Dog-2 Horse Pirrana Pteranodon
                       Raptor3 SabreToothTiger Scorpion-2 Spider Trex Tukan
59 rigs   f ≥ 0.189    everything else
```

Nothing falls between, so the split is a fact about the corpus rather than a
threshold anyone has to defend. `Camel` is the shape: `Hips` and `C_ctrl` both at
Y=0, with `Bip01` — the first real joint — 4.65 bone lengths above. `Tyranno` and
`BrownBear` are the healthy shape: the root is coincident with the pelvis.

**Why this matters, in feature terms.** `ric_pos` is root-relative, so those 14
rigs present every joint with a constant vertical bias of 0.7–6.5 bone lengths;
and the root-trajectory block is a *ground projection* for them and a *body
trajectory* for the other 59. Two conventions for one feature across 19% of the
corpus is exactly the variance a cross-topology model should not have to absorb.

**The operation.** Promote the rig's first branching joint `b` to root: its
translation channel takes the chain's accumulated world offset
(`t_b = t_root + Σ R·offset`), its rotation takes the composed chain rotation
(`R_root · … · R_b`), and the joints above it are removed. World positions of
every surviving joint are unchanged.

**Only the root can do this**, which is what makes it principled rather than an
exception. Non-root joints have no translation channel — `EnforceRigid` drops it —
so an offset has nowhere to go, which is precisely why the reduction rule's
collapse case requires a zero offset. The root is the one joint that can pay for
an offset, so it is the one joint that may absorb one.

The promoted target is a hardcoded table in
`scripts/process_dataset_truebones.py`, alongside §1.1's rest-pose table. This is
Truebones-specific data knowledge, not a general rule, and stage 1 is where such
knowledge already lives.

| rig | promote to | steps | new root height frac |
|---|---|---|---|
| Bear | `NPC_Pelvis` | 1 | 0.679 |
| Camel | `Bip01` | 2 | 0.681 |
| Crow | `_00` | 2 | 0.893 |
| Dog | `Bip01_Pelvis` | 2 | 0.740 |
| Dog-2 | `Bip01_Pelvis` | 2 | 0.740 |
| Horse | `Bip01_Pelvis` | 3 | 0.616 |
| Pirrana | `locator` | 1 | 0.623 |
| Pteranodon | `jt_Cog_C` | 1 | 0.218 |
| Raptor3 | `jt_Cog_C` | 1 | 0.879 |
| SabreToothTiger | `Sabrecat__pelv_` | 1 | 0.126 |
| Scorpion-2 | `jt_Cog_C` | 1 | 0.877 |
| Spider | `_body_` | 1 | 0.275 |
| Trex | `jt_Cog_C` | 1 | 0.997 |
| Tukan | `locator` | — | 0.556 |

Thirteen are the first branching joint along the root's single-child chain.
**Tukan is authored** because its root branches immediately into the real
skeleton (`N_ALL → locator`) and a dead `MESH` subtree of geometry-holder nodes
at zero offset, so no chain rule reaches it.

**The gate must be the locator classification, never the offset.** Lynx and
BrownBear have a correct root on the pelvis and a chain continuing to
`Bip01_Spine` at a real offset (0.79 and 0.89 bone lengths). A rule keyed on
"the chain carries an offset" would promote their root off the pelvis onto the
spine and break two healthy rigs. The hardcoded table makes this structural: a
rig not in it is never touched.

**Ordering is contractual.** `PromoteRoot` runs after `RestRelative` — which is
fitted against the original joint set and would reject a promoted clip — and
**before `ScaleToMeanBoneLength`**. The bones it removes are long: 4.95 on Bear,
6.47 on Pirrana, 2.40 + 3.39 on Horse. They currently enter the mean bone length
and therefore the canonical scale, so promotion changes the fitted scale factor
for these 14 rigs. That is a correction rather than a side effect — a
locator-to-body connector is not an anatomical bone — but it means these rigs'
`prepare.npz` scale differs from what today's code produces, and a test that
restates `ScaleToMeanBoneLength`'s own definition will not notice. This is the
same class of bug as the zero-length End Site scale error that survived all
eleven Phase 1 tasks.

**Ordering, exactly.** `PromoteRoot` is stage 2 of the chain:

```
RestRelative -> PromoteRoot -> FaceAxis -> EnforceRigid
             -> CentreXZ -> ScaleToMeanBoneLength -> PutOnGround
```

After `RestRelative`, whose `_check` requires the clip to carry exactly the joints
its rest pose declares and which a promoted clip would fail. Before
`EnforceRigid`, so that the translation channel `EnforceRigid` preserves is the
*promoted* root's rather than the locator's — otherwise promotion would have to
re-create a channel that was just dropped. And before `ScaleToMeanBoneLength`,
per the note above.

**Structure.** This is the one preparation stage that changes structure rather
than geometry, against the parity spec's *"preparation changes geometry and
preserves structure"*. It earns the exception by being where Truebones-specific
knowledge lives and by making the prepared BVH uniform for a human opening it in
a DCC tool, not only the features. It implements `apply` and `invert` like every
other stage: `invert` re-inserts the removed chain at its recorded offsets with
identity rotations, so a user's rig comes back with the hierarchy they supplied.
§9 states what that does and does not promise.

**`docker-compose.yml` must be fixed first.** It sets `user: "${UID:-1000}:${GID:-1000}"`,
but bash does not export `UID`, so the fallback always wins and every container
has been running as uid 1000 on a machine where the developer is 1001. Files
written into the bind mount by a container are owned by the wrong user and cannot
be removed from the host — the exact failure the file's own comment warns about.
`UID` and `GID` move into `.env`, which compose reads automatically, and which
this design adds anyway for wandb.

## 2 · Stage 2 — `scripts/build_features.py`

As the parity spec §4, unchanged in shape: a thin Hydra runner over
`poseydon/build/pipeline.py`, configured by `configs/dataset/truebones.yaml`,
four passes.

| pass | writes |
|---|---|
| per clip | `clips/<Rig>/<action>.npz` — motion arrays plus that clip's facing quaternion from `prepare.npz`; `<action>.yaml` if absent |
| per rig | `rigs/<Rig>/skeleton.npz` — offsets, parents, raw and humanized names, reduction map, folded prepare constants |
| per rig, schema-dependent | `rigs/<Rig>/stats.npz` — mean, std, block layout |
| once | `index.jsonl` — one flat row per clip, joining rig-level `tags` |

`--stats-only` runs the third pass alone, so changing `features:` costs seconds
rather than a corpus rebuild.

Five amendments, the first four forced by what plan A1 built and measured:

**Rig-level offsets come from the clip the manifest names, via `rest_action()`.**
The parity spec §3 established that prepared clips of one rig do not share an
OFFSET block — each carries its own facing correction — so rig-level `offsets` in
`skeleton.npz` have to come from the rest-pose clip specifically. A stage 2 that
looks for `tpose.bvh` gets a differently-rotated skeleton, silently, and forward
kinematics in `features/reconstruct.py` is then wrong by `R_rest · R_clip⁻¹`.

A1 settled where that choice lives, and it is **not** `prepare.npz`: each rig's
manifest declares `rest_pose:` (§1.1), and
`scripts/process_dataset_truebones.py::rest_action(manifest)` derives the
prepared-clip action slug from it — `strip_skeleton_prefix(action_slug(stem), rig)`,
the same derivation stage 1 used when it wrote the clip. Stage 2 calls
`rest_action`; it must not re-derive the slug and must not match on a filename.
The value is frequently not `tpose`: Crab's is `walk`, Ant/Deer/Jaguar `idle`,
Trex `walk_loop`, and thirteen rigs have no T-pose file at all.

**`mesh.npz` is stale for the 14 promoted rigs and must be regenerated before
stage 2 consumes it.** Its vertex skin weights are indexed by the FBX joint order,
which `PromoteRoot` (§1.2) changes for those rigs by removing the locator chain.
Applying weights indexed against the un-promoted skeleton to promoted motion
deforms the mesh wrongly, and nothing in the current pipeline detects it. The FBX
pass must be re-run for those 14 after promotion, and stage 2 should refuse a
`mesh.npz` whose joint count disagrees with `skeleton.npz` rather than trusting
positional alignment.

**Crab's `mesh.npz` disagrees with its rest geometry by 0.998 bone lengths.**
Measured in A1. Crab is the one rig whose rest pose now resolves to `walk.bvh`
while its `mesh.npz` was built from the FBX T-pose, so the two describe different
poses. `test_bvh_fbx_agree_on_rest_geometry` records this as a per-rig `xfail`
carrying the measured number. Regenerating Crab's FBX artefacts is the fix; until
then Crab's mesh must not be used for anything geometric.

**`configs/data/truebones.yaml` still points `manifests:` at
`data/truebones/skeletons/`**, a directory the Phase 1 layout migration deleted.
It is why `poseydon train` fails at config load today. Stage 2 owns the config
group, so it owns this correction.

**The FBX path does not promote, so the two corpora diverge for 14 rigs.**
`PromoteRoot` (§1.2) lives in the BVH `PrepareChain`;
`scripts/process_dataset_truebones_fbx.py` only rotates and scales a Blender
scene, so for the 14 promoted rigs the prepared BVH has the locator chain removed
while the prepared FBX still has it.
`test_bvh_and_fbx_agree_on_the_skeleton_structure` compares exactly that and
would fail — it passes today only because the five rigs with prepared FBX
artefacts happen to include none of the 14. Agreement by lucky coverage.

Stage 2 must not paper over it. Measured by running the FBX pass on Camel (a
two-step promotion, `Hips -> C_ctrl -> Bip01`): its `mesh.npz` carries 51
joints against the promoted BVH's 49 real joints, and the two extra are
exactly `Hips` and `C_ctrl` -- the removed locator chain, nothing else. The
count is small, but the cost that matters is not the joint count, it is
`PromoteRoot`'s logic: single-child-chain detection, the dead-subtree case
(Tukan's `MESH` branch), and offset absorption onto the promoted root, all of
which took its own task to get right against `poseydon`'s own `Animation`
structure -- and the FBX path has no `invert` to fall back on if a Blender-side
copy gets it wrong. Stage 2 therefore declares the FBX corpus un-promoted:
`build_skeleton` in `poseydon/build/pipeline.py` warns and does not fold a
`mesh.npz` whose joint count disagrees with `skeleton.npz`'s real
(End-Site-free) joint count, naming both counts in the warning --
`skeleton.npz`/`stats.npz` are written regardless, since neither reads
`mesh.npz` (mesh folding does not exist yet, and this is exactly the
condition whoever adds it must refuse on). Task 9's first full-corpus build
found that an earlier version of this guard raised instead of warned, which
withheld Camel's and Goat's training artefacts over a disagreement in a file
training never reads -- corrected in task 9 fix round 1. Teaching the FBX
path the same promotion stays open for whenever a skinned application needs
it, but is not this decision. `test_bvh_and_fbx_agree_on_the_skeleton_structure`
is extended to cover a promoted rig (Camel) so its passing is no longer a
coverage accident.

**`configs/model/anytop.yaml` gains `name_embedding_dim: 768`.** Without it
`AnyTop.name_projection` stays `None` and joint-name embeddings are accepted and
silently ignored (`models/anytop.py:139`). The embeddings themselves are not built
in this design — `text` stays `null`, T5 stays behind the `poseydon[text]` extra,
and the test image deliberately excludes `transformers` — but the config hole is
closed now rather than becoming a silent no-op later.

Measured in plan A2: **`MoDiffAE.__init__` does not accept that keyword at all** — it takes an
unconditional `text_dim` instead, so there is no silent-ignore hole to close there and adding
the key would raise at instantiation. The spec previously said `configs/model/*.yaml`, which was
wrong. This matters for §4's chosen model: the run trains MoDiffAE, so joint-name conditioning
reaches it by a different route than AnyTop's, and A3 must not assume the two share a config
key.

Normalization is the parity spec §5 per-block policy verbatim. `foot_contact`
takes `scale: none`; the reference-exact combination is `scale: block` with
`center: true` on every block, which is what the parity test uses.

## 3 · Read path

`MotionDataset.__getitem__` becomes a read, per parity spec §9:

```
load clips/<Rig>/<action>.npz    prepared animation, full joint set
reduce to rigid body             as_rigid_body(joint_translation="drop") -- MUST match
                                  build_stats (pipeline.py); stats.npz was fit on this
                                  distribution, not on the raw prepared animation
apply the rig's reduction        JointEdit from rigs/<Rig>/skeleton.npz
apply augmentations              JointEdit composed onto it
extract features                 the schema
normalize                        rigs/<Rig>/stats.npz, transported through the edit
crop                             window policy
```

Statistics are loaded, not fitted. That removes the startup cost and the reason
the dataset caches every animation in memory (`dataset.py:106-109`). `ClipView`
gains `clip` and `rig` annotation sidecars, so a later text conditioner needs no
change to the dataset, the collate or any model.

Four corrections ride along, each already located:

- **Worker RNG.** `dataset.py:77` holds one `numpy.random.Generator`, copied into
  every dataloader worker, so with `num_workers > 0` every worker draws the
  identical crop and augmentation stream. Reseeded per worker in `worker_init_fn`
  from the base seed and the worker id.
- **`crop_start` unification.** `AnyTop` reads `cond["window_start"]`
  (`anytop.py:207`); `MoDiffAE` reads `cond["crop_start"]` (`modiffae.py:341`).
  They unify on `crop_start`; `AnyTop` is corrected.
- **The temporal attention band** becomes a model parameter, `temporal_window: 31`,
  intersected with the padding mask inside the model — not a conditioner.
  Attention locality is a property of the attention module; routing it through
  conditioning would require `Conditioner.collate` to know the batch's frame
  count, which it does not receive. `TEMPORAL_VALID` survives as an optional
  sampling-time override.
- **`FootSkateLoss` reads the raw block.** `losses/footskate.py:227,231`
  thresholds the *normalized* contact flag at `> 0.5`; for a foot planted most of
  the time — high mean, low std — the z-scored `1` falls below the threshold,
  every planted frame is discarded, and the loss silently reads zero. It must go
  through `raw_block`, as the positions beside it already do. This run weights the
  term at 0, so the fix removes a latent trap rather than changing training.

The **recipe** (`runs/<run>/recipe.yaml`, also stashed in the checkpoint's
`hyper_parameters`) records the prepare chain, reduction, feature schema,
normalization policy and reconstruction method. It closes a live hole:
`poseydon sample` recomposes `configs/sample.yaml` from scratch (`cli.py:83`), so
sampling a checkpoint under a different `features:` than it was trained with
produces silent garbage. §6's retarget path reads the recipe, so this is
load-bearing for validation, not only for `sample`.

## 4 · Training loop

Per parity spec §10:

- **`BalancedByRig`** weighted sampler, weight `1 / (n_rigs · n_clips_of_rig)`.
  The reference's `--balanced` appears in both documented training commands;
  without it BrownBear's 22 clips dominate.
- **DataLoader** gains `drop_last=True` and `num_workers` from config.
- **`StepLR(step_size=10000, gamma=0.99)`**, stepped per optimizer step from
  `configure_optimizers` — the reference's `training_loop.py:65-69` exactly.
  Currently a bare `AdamW` (`training/lightning.py:61-64`).
- **`ModelCheckpoint(save_last=True, every_n_train_steps=25000, save_top_k=-1)`**,
  matching the reference's `save_interval=25_000`. `poseydon train --resume <ckpt>`
  maps to `trainer.fit(ckpt_path=...)`.

Run recipe, mirroring `docs/TRAIN.md`'s MoDiffAE command with the attention-pool
variant: `n_virtual_joints: 5`, `d_model: 128`, 4 layers, `lr 1e-4`, balanced —
and two deliberate departures from the reference, **`batch_size 16`** (not 10)
and **600 000 steps** (not 450 000), taken on measured evidence rather than by
preference. `precision: bf16-mixed`.

### What a step costs, measured

The real model on the real corpus geometry, on the GB10, bf16 autocast on. The
`s/step` column is an EXPECTED value: step time was measured across
J ∈ {43, 60, 73, 85, 100, 120, 142} and weighted by the empirical distribution
of a batch's padded joint count under the balanced sampler.

| batch | E[s/step] | 25 000 steps | 600 000 steps | peak GB (worst J) |
|---|---|---|---|---|
| 16 | 0.431 | 3.0 h | **71.8 h (3.0 days)** | 13.2 |
| 32 | 1.054 | 7.3 h | 175.7 h (7.3 days) | 26.3 |

Memory is not the constraint — 13 GB against the Spark's 128 GB unified. Time
is, and batch 32 costs 2.4× batch 16 rather than 2×, for the reason below.

**Attention is O(J²) over joints and the balanced sampler mixes rigs, so every
batch pads to its largest member.** The median rig has 43 joints; a batch of 16
pads to a mean of 88 and reaches Dragon's 142 in over 10% of draws. Most of the
step is spent on padding. This is not a divergence from the reference, which
uses a plain `WeightedRandomSampler` over clips and mixes rigs the same way.

Two levers were tested and are NOT taken:

* **`torch.compile`** — +8% at J=85 and 2.3× *worse* at J=142. Dynamic J
  triggers constant recompilation; it is a dead end for this model's shapes.
* **Bucketing batches by rig**, so J is the rig's own joint count rather than
  the batch max, is worth ~2.1× at batch 16 (0.431 → 0.208 s/step, 72 h → 35 h).
  It is not taken because it changes the sampler's semantics: single-rig batches
  remove cross-rig contrast from every gradient step, which for a
  cross-topology model is a research question, not an optimization. If it is
  ever added it goes in as an opt-in flag, never as the default.

For the record: the model runs at roughly 2.5% of the GPU's bf16 throughput,
i.e. it is launch-bound on many small kernels rather than compute-bound. Real
headroom exists but capturing it means reworking the attention implementation,
which is out of scope here.

`losses:` becomes `simple: 1.0, geodesic: 1.0`. The reference's `--lambda_fs`
defaults to 0 and its paper command never passes it, while `--lambda_geo 1.0` is
passed explicitly; PoseYdon currently ships `geodesic: 0.1, footskate: 0.5`,
which is neither. Foot skate is still *measured*, as a validation scalar — the
right place for a diagnostic that is not a training objective.

### Logging

A `WandbLogger`, project `poseydon`, entity from `.env`, run name derived from
the config (model, pooling, batch size, latent dim). `.env` carries
`WANDB_API_KEY`, `WANDB_ENTITY`, `UID` and `GID`; it is gitignored and compose
reads it automatically. The entity is
`lcazzola-fondazione-bruno-kessler` — verified against the live API, which is
also how a first attempt at `entity: poseydon` was found to fail with *"the
provided API key cannot access this resource"*: `poseydon` is the PROJECT, and
the key's only entity is that one. A real run was logged end to end to confirm
the credential before any of this was planned around. The reference reaches its platform through `eval()` on a
command-line string in `utils/ml_platforms.py`; Lightning's logger interface
removes that.

### The aarch64 image — settled by spike

This was the one item with genuine uncertainty. **It is now measured, and the
answer is the cheap route.** Host: GB10 (Blackwell, `sm_121`) on aarch64, driver
580.159.03, CUDA 13.0, CDI registering `nvidia.com/gpu=all`.

`Dockerfile.train` stays on the same `ghcr.io/astral-sh/uv:python3.12-bookworm-slim`
base the test image uses and layers CUDA 13 aarch64/sbsa wheels on top. The NGC
PyTorch aarch64 image was NOT built: route A worked end to end, and pulling a
~20 GB base to compare a route we do not need is not evidence worth buying.

Verified inside the container on the real GPU: `torch 2.14.0+cu130`,
`is_available: True`, device `NVIDIA GB10`, capability `(12, 1)`; fp32 and bf16
2048² matmuls; three `AdamW` + `StepLR` steps; and the full dependency set
(numpy 2.5.3, scipy, lightning 2.6.5) with `poseydon` and `MoDiffAE` importing
clean. Image size 19.4 GB.

One thing to know rather than discover later: the wheel's arch list is
`['sm_80','sm_90','sm_100','sm_110','sm_120']` — **`sm_121` is not in it**. The
GB10 runs anyway through Blackwell minor-version compatibility, which is why the
spike measured real compute rather than stopping at `is_available()`. No source
build is needed.

`pyproject.toml`'s `[tool.uv.sources]` CPU pin is overridden at image-build time
with `uv pip install --index-url https://download.pytorch.org/whl/cu130
--reinstall-package torch torch`, so the shared `pyproject.toml` does not change
and the CPU test image is unaffected. `build-essential` is NOT installed: it is
needed only by `torch.compile`, which this run does not use.

## 5 · FBX coherence sidecheck

FBX is a measurement here, not a training input. `tools/probe_bvh_fbx.py` becomes
a script that runs the Blender stage over the corpus and reports per-rig
rest-geometry disagreement, extending the current numbers — Flamingo 6.8e-2,
BrownBear 5.5e-2, Scorpion 3.3e-2, Crab 2.0e-5 bone lengths — to all 73 rigs. It
stays an `xfail`-backed report rather than a gate, and does not block training.

The basename-pairing gap is reported, not solved. BrownBear ships `BEAR-<Action>.fbx`
against BVH's `__<Action>.bvh`, and Elephant and Fox ship `atk 1.fbx` against
`__Attack1.bvh` — a divergence no filename rule can close. That needs an authored
per-rig alias map, which is out of scope.

Why it matters despite being out of the training path: `mesh.npz` skin weights
are indexed by FBX joint order, so the disagreement means weights deforming
wrongly whenever a later phase drives a skinned mesh from BVH-derived motion.

## 6 · The `Retarget` operation

The mechanism is already latent in the codebase. `MoDiffAE` documents its
`Z_SEM` conditioning key as *"When present the encoder is skipped — how blending,
transfer and latent caching all work"*, and `SemanticEncoder.forward` pools over
joints, returning `(T, B, 1, C)`. The semantic latent is therefore
**joint-independent**: a 40-joint Flamingo latent is shape-compatible with a
63-joint Scorpion decode by construction. Only the frame count must match.

Two pieces:

**`LatentPin(Control)`** — `sampling/controls/latent_pin.py`. Sets
`payloads[Z_SEM] = latent` on every step. It is `LatentMix`'s sibling minus the
schedule, and deliberately without `LatentMix`'s `reference.shape == target.shape`
guard — that guard is exactly what makes `Blend` unable to express cross-rig
transfer today.

**`Retarget(Operation)`** — `ops/retarget.py`. Overrides `run()` to call
`model.encode(target_clean, cond_of_target)` once, build the `LatentPin`, and
delegate to `Operation.run`. It raises on a model with no `encode` — that is,
`AnyTop` — rather than silently producing unconditional motion.

Sampling uses **DDPM**, per `docs/SAMPLE.md`: retargeting has no transition zone
to anchor, so DDIM inversion buys nothing. Output length is the content clip's
own length.

**Reconstruction is `positions_ik`** — the predicted position channels fitted back
onto the rigid target skeleton by `GradientIK`, rather than forward kinematics
from the predicted rotations. The solver optimizes rotations directly, so bone
lengths are exact by construction and a BVH is still writable
(`produces_rotations = True`). `cli.py` already strips `iterations`/`smoothness`
for methods that do not take them. The method is recorded in the recipe, so
`sample`, `retarget` and the validation callback cannot disagree about it.

## 7 · `RetargetValidation`

`training/validation.py`. Fires from `on_train_batch_end` every 25 000 steps, so
it lines up with `ModelCheckpoint`, and once at step 0 — cheap insurance that the
whole path is wired before 25 000 steps are spent discovering it is not.

Pairs are configuration, resolved through `CorpusIndex.query`. This follows the
decision recorded in `2026-08-30-poseydon-design.md`, that retargeting pairs are
a query — *same action, different skeleton* — replacing the reference's
`build_mixamo_retargeting_bench.py` and its materialized txt files. Nothing
downstream parses a filename.

```yaml
validation:
  every_n_steps: 25000
  run_at_start: true
  pairs:
    - {content: {rig: Flamingo, action: onelegbent}, target: Scorpion}
    - {content: {rig: Coyote,   action: attack3},    target: Crab}
    - {content: {rig: Goat,     action: headbutt},   target: Raptor}
```

The callback calls a plain `run_retarget()` that the new `poseydon retarget` CLI
also calls, so what is watched during training is the same code path that ships.
It writes `runs/<run>/validation/step_<N>/<content>__to__<target>.{npz,bvh,mp4}`.

**All three files are logged as a wandb Artifact**, and the MP4 additionally as
`wandb.Video` so it is scrubbable in the run's media panel without a download.
One artifact per validation firing, named `retarget-<run_id>`, typed
`retarget-validation`, versioned by wandb and aliased with the step
(`step-25000`); the three pairs' nine files go in together, since they are one
observation of one checkpoint and are only ever compared as a set.

The artifact is the point, not the video. `wandb.Video` renders but is not
retrievable as data — a `.bvh` you can load into Blender and a `.npz` you can
diff against another step are, and a run that trains for three days is one whose
step-25000 output you will want to open next week. Media panels are for
watching; the artifact is what makes the run auditable after the fact.

Artifact logging must never be able to fail a training run: the whole
log-and-upload block is wrapped, and a failure degrades to a warning on the
logger and a note in the run's console output. Losing a validation upload costs
one observation; losing 72 hours of training to a transient network error costs
the run.

It draws from a **dedicated `torch.Generator` seeded from the step**, so
validation is reproducible across a resume and never consumes the training RNG
stream. It toggles `eval()`/`train()` around itself and runs under `no_grad`.

### Metrics

Five scalars per pair, plus a mean across pairs, logged as
`val/retarget/<pair>/<metric>`.

| metric | definition | catches |
|---|---|---|
| foot skate | horizontal displacement of a foot joint while its contact flag is set, on the solved output | the classic retargeting failure, and the term switched off as a loss |
| bone-length drift | \|‖p_j − p_parent‖ − ‖offset_j‖\| on the **raw predicted positions**, in bone-length units | whether the model predicts a holdable skeleton |
| IK residual | ‖p_predicted − p_solved‖, mean and max, in bone-length units | how far the solver had to travel — whether it is cleaning up or inventing |
| root trajectory error | generated root XZ velocity against the content clip's, each rescaled by its own rig's `scale_factor` from `prepare.npz` | whether content transferred, or only style |
| latent round-trip | distance between `encode(generated)` and the injected `z_sem` | whether the decoder honoured the instruction at all |

Bone-length drift is measured **before** IK deliberately. On the solved output it
would be identically zero by construction, which measures the solver's
parameterization rather than the model — the failure mode Phase 1's retrospective
named as its recurring bug. Measuring it pre-IK, alongside the residual, separates
"the model predicts bad bone lengths" from "the solver had to move joints a long
way".

The target rig's foot joints come from its manifest, remapped through the rig's
reduction `JointEdit` by the same `augment/topology.py::_reindex_resolved` path
augmentation already uses.

## 8 · Tests

**Most of the round trip is already tested.** `test_round_trip_returns_the_source_rig`
walks `source -> prepare -> reduce -> expand -> unprepare` and asserts names,
parents and offsets exactly, `EnforceRigid`'s documented translation loss,
world-exact `expand o reduce`, and that re-preparing the restored asset
reproduces the prepared clip. `tests/build/test_prepare.py` adds five
stage-level inversion tests and the channel-layout recording;
`tests/features/test_reduce.py` covers expansion. What follows adds the gaps,
not a parallel suite.

**Plan A**

1. **Crab stops skipping.** `tests/build/test_roundtrip.py::_rest_path` selects
   the rest file by filename -- `tpos`, then `idle` -- which is the rule
   `b2b0151` fixed in `scripts/process_dataset_truebones.py` and never fixed in
   the test. Crab therefore resolves to its 54-joint `__TPOSE.bvh`, every 64-joint
   clip disagrees, and the round trip has silently skipped for the milliped the
   suite was parametrized to cover. The test shares the production selection
   helper instead of duplicating a stale copy of it.
2. **The features leg.** The round trip is extended through
   `extract_features` and `reconstruct` -- `reduce -> features -> features ->
   expand` with the model stage as identity -- so the loop the application
   actually runs is the loop under test. Today's version stops at
   `reduce -> expand`.
3. **A promoted rig round-trips.** `SAMPLE_RIGS` gains **Camel**, so `PromoteRoot`
   is exercised by every assertion above rather than by a test of its own. None of
   Flamingo, BrownBear, Crab or Scorpion is a ground-locator rig, so promotion
   would otherwise be covered nowhere. Camel specifically because it exercises
   both new tables at once: it has no T-pose file, so §1.1 authors its rest pose
   (`__IdleLoop.bvh`), and it is a two-step promotion (`Hips → C_ctrl → Bip01`)
   rather than the single-step majority.
4. **The round trip returns an FBX.** The same assertions through the FBX arm,
   against the `fbx` compose service, skipped when Blender is absent as
   `test_bvh_fbx_agreement.py` already does.
5. **Root promotion moves nothing, and reaches exactly the right rigs.** Every
   surviving joint's world position is unchanged by promotion, and the new root's
   height fraction clears the locator band. For the 59 others -- Lynx and
   BrownBear named explicitly, since their chains carry a real offset -- promotion
   is a no-op. The world-position half is the assertion that matters: a
   joint-count check would restate the table.
6. **Every rig declares a rest pose that exists and is modal.** Every manifest
   is checked: it declares `rest_pose`, the named file exists, and its joint set
   is the rig's modal one. This is the test that would have caught
   `Trex/__STILL.bvh`, and the one that fails loudly when someone edits a
   manifest by eye.
7. **The recovered rigs stay recovered.** Ant, Crab, Deer and Jaguar keep
   17/10/20/13 clips through stage 1. `b2b0151` is guarded by nothing today, and
   a regression in `find_tpose` would quietly cost 60 clips again.
8. **Golden feature parity.** Features reproduce the reference's `.npy` arrays
   block by block, foot contact bit-exact, over the joints the two
   representations share, matched by name.
9. **Normalization policy.** Each `scale` mode produces statistics of the
   documented shape, and `joint_block` leaves a 6D row's recovered rotation
   unchanged under Gram-Schmidt. A property, not an example.
10. **Build-then-train smoke.** `build_features` over a tiny fixture corpus, then
    two training steps: the loss is finite, and a schema mismatch between
    `features:` and `stats.npz` raises before the first step.

**Plan B**

11. **Cross-topology retarget.** Encode a 40-joint clip, decode on a 63-joint rig;
    the output carries the target's joint count and the target's bone lengths.
    The test the whole feature rests on.
12. **`LatentPin` pins.** The semantic encoder is not invoked when `Z_SEM` is
    present, and the injected latent is what reaches the decoder.
13. **Callback contract.** Fires at the right steps, writes all three artifact
    types, logs five scalars per pair, restores `train()` mode, and leaves the
    training RNG stream untouched.
14. **Metric sanity.** A static clip scores ~zero foot skate; an FK-generated clip
    scores ~zero bone-length drift and ~zero IK residual.
15. **Recipe round-trip.** A checkpoint sampled through its recorded recipe
    reproduces the reconstruction method and feature schema it was trained with,
    and an override touching the data path is rejected.

Phase 1's retrospective is blunt that its plan's tests were its weakest part --
four of eleven tasks shipped a wrong assertion, and in every case the production
code was correct. Test 1 above is that failure mode in its purest form: a test
that skips is a test that passes, and this one skipped for three weeks. Each
test here measures output against an external definition rather than restating
the implementation.

## 9 · The round-trip contract

Non-negotiable, and stated precisely so it can be tested rather than asserted.

```
source .fbx/.bvh -> prepare -> reduce -> features -> model -> features
                                                                 |
source .fbx/.bvh <- unprepare <-------- expand <-----------------+
```

**The guarantee is structure and reference frame, not motion values.** Whatever
leaves the model — a user's own clip passed through, or motion generated from
nothing — converts back into the convention the user supplied. Specifically, the
returned file carries:

- the source's joint count, names, parent array and hierarchy order;
- the source's per-joint channel layout, in the source's declared order;
- the source's scale, world orientation and ground offset;
- every joint in the source's coordinate convention.

**What it does not promise** is that any particular number matches. Four
documented losses sit inside the chain, and all four are value-level:

1. `EnforceRigid` returns per-joint translation channels holding the constant
   rest offsets rather than their animated content — roughly 10% of skeleton size
   on raw Truebones, intrinsic to a rotation-based representation.
2. Reduction's collapse case, and now `PromoteRoot`, put a composed rotation on
   one of several coincident joints and identity on the rest. The world pose is
   identical and the hierarchy is identical; which joint stores the rotation is
   not recovered, and for generated motion there was never an original split to
   recover.
3. `positions_ik` fits rotations to predicted positions with LBFGS, so it lands
   near rather than on. That is a reconstruction-quality choice about generated
   motion and is orthogonal to convertibility — the returned rig is structurally
   the user's either way.
4. **The root's absolute horizontal position does not survive the feature
   representation.** `features/recover.py::root_trajectory` is deliberately
   root-invariant: it stores per-frame velocity, not absolute placement, "so the
   same motion reads identically wherever it happens". Integrating it back gives
   a trajectory that starts at the XZ origin with only its SHAPE preserved.
   Height returns exactly, and so does every joint relative to the root; what is
   gone is where in the world the clip sat. Measured in Task 7 of plan A1 as a
   constant XZ shift with a zero vertical component and a ~1e-14 residual once
   the shift is accounted for.

   This is not a defect, but it is load-bearing for §7: a retargeted clip returns
   at the origin rather than where the content motion was, so the validation
   callback and `poseydon retarget` must either accept that or re-anchor the
   output explicitly against the reference clip's starting position. §7 does not
   currently say which. Decide it when Plan B builds `run_retarget`; the honest
   default is to re-anchor, because a user retargeting their own clip expects it
   to come back where they put it.

This is why promotion needs no per-clip recording of the removed chain's
rotations. Recording them would make a real clip's values recoverable too, but it
would buy nothing the contract asks for, and generated motion — which has no
recorded chain — would still need the identity path. One code path, not two.

**The FBX arm of this contract is not implemented.** Measured while building
plan A1: the FBX stage-1 path is entirely separate from the BVH one.
`scripts/process_dataset_truebones_fbx.py` does `FBX.read -> rotate ->
scale_to_mean_bone_length -> write` directly on a Blender scene; it never
constructs an `Animation`, never runs `PrepareChain`, records no fitted
parameters, and therefore has no `invert`. The diagram above is true of `.bvh`
today and aspirational for `.fbx`.

That is a real gap against the contract, not a documentation nicety: a user who
supplies an FBX cannot currently be handed one back. Closing it means giving the
FBX path the same fit/apply/invert stage contract the BVH path has, with its
parameters recorded per clip — which is a build-stage change, not a test. It
belongs to a later plan and is called out here so it is not discovered by a user.
What A1 does deliver for FBX is narrower and honest: an IO-level round trip
asserting that `FBX.read -> write -> read` preserves joint names, parents and
rest offsets.

**Generated motion has no per-clip facing, and something must supply one.**
`FaceAxis` is `CLIP`-scoped: it records the quaternion that turned *that clip's*
frame 0 to +Z, and `invert` needs a value. A generated clip has no source
orientation. For retargeting, `unprepare` uses **the reference clip's** facing —
the clip that named the target skeleton — so the output returns in the same
orientation as the rig identity the user pointed at. Rig-scoped parameters
(`RestRelative`, `CentreXZ`, `ScaleToMeanBoneLength`, `PutOnGround`) need no such
choice: they come from the target rig's own `prepare.npz`.

## Limitations

**No held-out validation.** Every index row is `split: train`; the five scalars
describe three fixed retarget pairs, not generalization. A rising validation
metric means those three clips got better, which is weaker than it sounds.

**Crab and Raptor are thin.** Crab contributes 10 clips and Raptor 11 out of
~1145, and the balanced sampler equalizes per rig rather than per clip, so both
are seen as often as BrownBear's 22 — with far less variety behind them. Their
retarget outputs will be the least reliable of the three pairs.

**The rest pose is authored for every rig, and for 17 that is load-bearing.**
§1.1's declarations decide canonical rest geometry, and for those 17 it is a
judgement about animation content that no test can confirm — the modal-set guard
proves an entry is *usable*, not that it is *neutral*. Three of the seventeen
(Crab, SabreToothTiger, Trex) rest on a gait clip rather than a still pose, so
their rig identity is a mid-stride stance; Rat rests on a trot. That is what the
`tpose` conditioner shows the model as those characters' identity. It is still
better than the alternative it replaces — Crab was resting on an attack frame —
but it is a choice, not a derivation, and it should be revisited by eye in a DCC
tool rather than trusted because it passes.

Separately, the modal rule itself is a heuristic over data: a rig whose clips
split evenly between two joint sets has no defensible modal answer. None do
today; the closest is Trex at 70/71.

**Root promotion changes the canonical scale of 14 rigs.** The bones it removes
are long — 6.47 on Pirrana, 4.95 on Bear, 2.40 + 3.39 on Horse — and they
currently enter `ScaleToMeanBoneLength`'s mean. Removing a locator-to-body
connector from that average is a correction, not a regression, but it means those
rigs' fitted scale differs from what today's code produces, and any checkpoint or
`prepare.npz` from before the change is incomparable with one after it.

**Promotion returns structure, not the original rotation split.** Every joint of
the removed chain has exactly one child, so their rotations turn the same subtree
and only the product `R_root · … · R_b` is observable. `invert` puts that product
on the promoted joint and re-inserts the chain with identity rotations. Under §9
this satisfies the contract — hierarchy, names, offsets, channels and world pose
all return — but a user who inspects the returned curves will find the motion on
a different joint of that chain than they authored it on. Same limitation the
parity spec records for collapsed internal joints, same reason.

**One preparation stage changes structure.** The parity spec's clean split —
preparation changes geometry, reduction changes structure — has exactly one
exception now. It is deliberate, and the reason is that a human opening
`clips/Camel/walk.bvh` should not see a root five bone lengths under the animal.
But the invariant is no longer free, and a future stage that assumes prepared
clips carry the source joint set is wrong for these 14 rigs.

**BVH and FBX still disagree on rest geometry**, 3–7% of a bone length at the
worst rigs, cause unknown; a coordinate-frame mismatch was ruled out to 1e-7.
This design measures it corpus-wide and does not fix it.

**One outlier file per rig is lost** for Ant, Centipede, Crab, Deer, Elephant,
HermitCrab, Jaguar and Trex — 8 clips of 1153. Recovering them needs a rest pose
matching their clips, a data question rather than a code one.

**`EnforceRigid` preserves structure but not values.** Per-joint translation
channels come back holding constant rest offsets, discarding roughly 10% of
skeleton size in real motion on raw Truebones. Intrinsic to a rotation-based
representation, and the reason the stage is explicit rather than silent.

**Goat's `mesh.npz` disagrees with `skeleton.npz` though Goat is not
promoted.** Task 8's mesh/skeleton joint-count guard (§2) fires on Goat too:
33 FBX joints against 32 real BVH joints. The extra is a `Null` bone, root of
the raw FBX armature and parent of `Hips`, confirmed present straight off
`bpy.ops.import_scene.fbx` on the raw source file — not an artefact of
PoseYdon's own processing. Goat is not in `PROMOTE_ROOT`, so this is a second,
independent instance of the same "un-promoted ground locator" shape the FBX
path already fails to strip for the 14 it does know about, not a bug in the
guard itself (the four other rigs carrying a `mesh.npz` — Flamingo, BrownBear,
Crab, Scorpion — all match their real BVH joint count exactly). Left as data, not fixed here.

**The fix is FBX-side, and adding Goat to `PROMOTE_ROOT` would be wrong.**
`PROMOTE_ROOT` drives the BVH `PrepareChain`, and Goat's BVH needs nothing: its
prepared root is `Hips` at height fraction 0.724, a properly body-rooted
skeleton with no locator to strip. Promoting it would move the root off the
pelvis and onto the spine — exactly the failure the table's ground-locator gate
exists to prevent, and the reason §1.2 keys that gate on the locator
classification rather than on the presence of an offset. The `Null` is an
artefact of the FBX export alone, which is why the two sources disagree at all.
Closing it means either stripping the `Null` in the FBX path or regenerating
`mesh.npz` against the promoted skeleton — both FBX-side work, and both waiting
on the same decision §2 defers about giving that path a real stage contract.

**The FBX all-takes filter drops 13 legitimate clips whose stem ends in "all".**
`scripts/process_dataset_truebones_fbx.py:84` selects an all-takes bundle with
`not stem.lower().endswith("all")` — a suffix check meant to exclude the
whole-rig compilations, but `"camel-fall".endswith("all")` is also `True`. The
rule it actually implements is "the stem ends in the letters a-l-l", which is
wider than "-Fall": `Deer/DEER-WalkCall.fbx` is caught too, and is the clip
that proves the mechanism is the suffix, not the word. Measured over
`data/truebones/source/**/*.fbx`, all 13: `Buffalo-Fall`, `Camel-Fall`,
`DEER-WalkCall`, `Gazelle-Fall`, `PolarBearB-Fall`, and under `Raptor2/` the
four `Raptor-Fall`, `Raptor-FenceClimbFall`, `Raptor-RunFall`,
`Raptor-RunJumpFall`, plus `roach-Fall`, `Stego-Fall`, `Tricera-Fall`,
`Tyranno-Fall`. Since `mesh.npz` is not folded into any training artefact yet
(§2), this has no effect on `skeleton.npz`/`stats.npz`/the clip `.npz`s
today — but the FBX-sourced corpus is missing those takes, and whoever gives
the FBX path a real stage contract must fix the filter, not work around its
current output. The correct test is against the compilation's actual naming
convention (the stem equals the rig's export prefix plus "ALL"), not a suffix
of "all".

**`tests/build/test_roundtrip_fbx.py`'s own filter is broader, and worse: it
skips green instead of failing.** That test (`:61`) uses
`"ALL" not in stem.upper()` to find one per-clip source file per rig — a
SUBSTRING test, so it also excludes every per-clip file of a rig whose EXPORT
PREFIX contains "ALL". Measured over the source tree, seven rigs are left with
no candidate at all and hit the test's own skip branch: Alligator, Anaconda,
Crow, HermitCrab, Lion, SabreToothTiger, Tukan. Of those, only **Tukan** is in
`SAMPLE_RIGS`, so exactly one parametrised case silently skips green today —
the other six are latent, and the blast radius grows with `SAMPLE_RIGS`.
Unlike (a) this produces no wrong-but-visible artefact, only silence, against
this project's own "a test that skips is a test that passes" rule. The fix is
the same: match the compilation's naming convention, not a substring of "ALL"
a legitimate export prefix can contain.

## Phasing

**Plan A — get it training.** aarch64 CUDA spike; `.env` and the compose `UID`
fix; full-corpus stage 1; stage 2 and the build package; the read path; the
training loop, wandb and the train service. Ends at a running job logging loss
curves. Tests 1–10.

**Plan B — validate it.** `LatentPin`, `Retarget`, `poseydon retarget`, the
`RetargetValidation` callback and its five metrics, landed on a run that already
works. Tests 11–15.

Each leaves the repository working and is reviewable on its own, as Phase 1 was.

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
the same code path; a root-promotion stage; eleven tests.

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
  today's "first file in the modal set" with an authored choice.

Expected yield: **~1145 clips over 73 rigs**, against 1153 raw and the 81-row
index on disk. The parity spec predicted 64 rejections across 8 rigs; the modal
fix reduces that to 8.

### 1.1 · The rest pose is authored where it cannot be found

`find_tpose` resolves the rest-pose file in preference order: **T-pose, then
idle, then walk** — or fly, for a flying creature. The first is discovered by
name; the last two are *authored per rig*, because both cheap alternatives fail.

Matching `idle` as a substring picks Lion's `__DeathIdle.bvh`, Jaguar's
`__LieIdle.bvh` and Trex's `__idle_attack.bvh` — a dying pose, a lying pose and a
crouched attack, which are the exact poses this rule exists to avoid. And today's
final fallback, "first file in the modal set", is arbitrary: it is what gave Crab
a rest geometry fitted from `__Attack1.bvh`.

So the 17 rigs whose named T-pose is missing or unusable get an explicit entry:

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

**The modal-set check stays as the outer guard, and it is not ceremony.** Trex's
natural neutral pick, `__STILL.bvh`, *is* that rig's outlier file — 66 joints
against the modal 78. An authored table trusted on its own would have silently
destroyed the largest rig in the corpus. Every entry above is verified to sit in
its rig's modal joint set (`tools/probe_rest_override.py`); an entry that does
not must fail the build loudly rather than fall back.

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

**Structure.** This is the one preparation stage that changes structure rather
than geometry, against the parity spec's *"preparation changes geometry and
preserves structure"*. It earns the exception by being where Truebones-specific
knowledge lives and by making the prepared BVH uniform for a human opening it in
a DCC tool, not only the features. It implements `apply` and `invert` like every
other stage: `invert` re-inserts the removed chain at its recorded offsets with
identity rotations, so a user's rig comes back with the hierarchy they supplied.

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
| per rig, schema-dependent | `rigs/<Rig>/stats.npz` — mean, std, block layout, T-pose frame |
| once | `index.jsonl` — one flat row per clip, joining rig-level `tags` |

`--stats-only` runs the third pass alone, so changing `features:` costs seconds
rather than a corpus rebuild.

Two amendments the modal-set fix forces:

**`prepare.npz` must record which clip it fitted the rest pose from.** The parity
spec §3 established that prepared clips of one rig do not share an OFFSET block —
each carries its own facing correction — so rig-level `offsets` in `skeleton.npz`
have to come from the rest-pose clip specifically. With `find_tpose` now choosing
by modal joint set and then by §1.1's authored table, that file is frequently not
named `tpose`: for Crab it is `walk.bvh`, for Ant, Deer and Jaguar `idle.bvh`,
for Trex `walk_loop.bvh`, and thirteen rigs have no T-pose file at all. A stage 2 that looks for
`tpose.bvh` gets a differently-rotated skeleton, silently, and forward kinematics
in `features/reconstruct.py` is then wrong by `R_rest · R_clip⁻¹`. Stage 1 records
the choice; stage 2 reads it.

**`configs/model/*.yaml` gains `name_embedding_dim: 768`.** Without it
`AnyTop.name_projection` stays `None` and joint-name embeddings are accepted and
silently ignored (`models/anytop.py:139`). The embeddings themselves are not built
in this design — `text` stays `null`, T5 stays behind the `poseydon[text]` extra,
and the test image deliberately excludes `transformers` — but the config hole is
closed now rather than becoming a silent no-op later.

Normalization is the parity spec §5 per-block policy verbatim. `foot_contact`
takes `scale: none`; the reference-exact combination is `scale: block` with
`center: true` on every block, which is what the parity test uses.

## 3 · Read path

`MotionDataset.__getitem__` becomes a read, per parity spec §9:

```
load clips/<Rig>/<action>.npz    prepared animation, full joint set
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
variant: `n_virtual_joints: 5`, `d_model: 128`, 4 layers, `lr 1e-4`,
`batch_size 10`, balanced, 450 000 steps.

`losses:` becomes `simple: 1.0, geodesic: 1.0`. The reference's `--lambda_fs`
defaults to 0 and its paper command never passes it, while `--lambda_geo 1.0` is
passed explicitly; PoseYdon currently ships `geodesic: 0.1, footskate: 0.5`,
which is neither. Foot skate is still *measured*, as a validation scalar — the
right place for a diagnostic that is not a training objective.

### Logging

A `WandbLogger`, project `poseydon`, entity from `.env`, run name derived from
the config (model, pooling, batch size, latent dim). `.env` carries
`WANDB_API_KEY`, `WANDB_ENTITY`, `UID` and `GID`; it is gitignored and compose
reads it automatically. The reference reaches its platform through `eval()` on a
command-line string in `utils/ml_platforms.py`; Lightning's logger interface
removes that.

### The aarch64 image

This is the one item with genuine uncertainty. The host is a GB10 (Blackwell,
`sm_121`) on aarch64 with CUDA 13 and CDI configured at `/var/run/cdi/nvidia.yaml`;
`pyproject.toml` pins the CPU wheel index and the test image carries
`torch 2.13.0+cpu`. Two routes exist — CUDA 13 aarch64/sbsa wheels layered on the
current slim image, or basing `Dockerfile.train` on an NGC PyTorch aarch64 image
(large, but NVIDIA's supported path on DGX Spark).

**Plan A opens with a spike** that gets `torch.cuda.is_available()` true and a
matmul running on `sm_121` inside a container, and reports what worked before any
other work depends on it. The image choice is made on that evidence, not in
advance.

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
It writes `runs/<run>/validation/step_<N>/<content>__to__<target>.{npz,bvh,mp4}`
and logs the MP4 as `wandb.Video`.

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

Eleven, split by plan.

**Plan A**

1. **Golden feature parity.** Features reproduce the reference's `.npy` arrays
   block by block, foot contact bit-exact, over the joints the two
   representations share, matched by name.
2. **Normalization policy.** Each `scale` mode produces statistics of the
   documented shape, and `joint_block` leaves a 6D row's recovered rotation
   unchanged under Gram-Schmidt. A property, not an example.
3. **Build-then-train smoke.** `build_features` over a tiny fixture corpus, then
   two training steps: the loss is finite, and a schema mismatch between
   `features:` and `stats.npz` raises before the first step.
4. **The recovered rigs stay recovered.** Ant, Crab, Deer and Jaguar keep
   17/10/20/13 clips through stage 1. `b2b0151` is guarded by nothing today, and
   a regression in `find_tpose` would quietly cost 60 clips again.
5. **Every authored rest file is real and in its modal set.** §1.1's table is
   checked entry by entry: the file exists, and its joint set is the rig's modal
   one. This is the test that would have caught `Trex/__STILL.bvh`, and it is
   also the test that fails loudly when someone adds a rig to the table by eye.
6. **Root promotion is world-exact, and reaches exactly the right rigs.** For
   each of the 14, every surviving joint's world position is unchanged by
   promotion, and the new root's height fraction clears the locator band. For the
   59 others — Lynx and BrownBear named explicitly, since their chains carry a
   real offset — promotion is a no-op. The world-position half is the assertion
   that matters: a joint-count check would restate the table.

**Plan B**

7. **Cross-topology retarget.** Encode a 40-joint clip, decode on a 63-joint rig;
   the output carries the target's joint count and the target's bone lengths.
   The test the whole feature rests on.
8. **`LatentPin` pins.** The semantic encoder is not invoked when `Z_SEM` is
   present, and the injected latent is what reaches the decoder.
9. **Callback contract.** Fires at the right steps, writes all three artifact
   types, logs five scalars per pair, restores `train()` mode, and leaves the
   training RNG stream untouched.
10. **Metric sanity.** A static clip scores ~zero foot skate; an FK-generated clip
   scores ~zero bone-length drift and ~zero IK residual.
11. **Recipe round-trip.** A checkpoint sampled through its recorded recipe
   reproduces the reconstruction method and feature schema it was trained with,
   and an override touching the data path is rejected.

Phase 1's retrospective is blunt that its plan's tests were its weakest part —
four of eleven tasks shipped a wrong assertion, and in every case the production
code was correct. Each test above measures output against an external definition
rather than restating the implementation.

## Limitations

**No held-out validation.** Every index row is `split: train`; the five scalars
describe three fixed retarget pairs, not generalization. A rising validation
metric means those three clips got better, which is weaker than it sounds.

**Crab and Raptor are thin.** Crab contributes 10 clips and Raptor 11 out of
~1145, and the balanced sampler equalizes per rig rather than per clip, so both
are seen as often as BrownBear's 22 — with far less variety behind them. Their
retarget outputs will be the least reliable of the three pairs.

**The rest pose is authored for 17 of 73 rigs, and that is load-bearing.**
§1.1's table decides canonical rest geometry for those rigs, and it is a
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

**Promotion is world-exact but not representation-exact.** `R_root · … · R_b` can
be split across the chain in infinitely many ways, so `invert` puts the whole
product on the promoted joint and re-inserts the chain with identity rotations.
Every joint returns with the right name, parent, offset and world position; what
differs is which of several coincident joints stores the rotation. This is the
limitation the parity spec already records for collapsed internal joints, and it
applies here for the same reason.

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

## Phasing

**Plan A — get it training.** aarch64 CUDA spike; `.env` and the compose `UID`
fix; full-corpus stage 1; stage 2 and the build package; the read path; the
training loop, wandb and the train service. Ends at a running job logging loss
curves. Tests 1–6.

**Plan B — validate it.** `LatentPin`, `Retarget`, `poseydon retarget`, the
`RetargetValidation` callback and its five metrics, landed on a run that already
works. Tests 7–11.

Each leaves the repository working and is reviewable on its own, as Phase 1 was.

# PoseYdon — Design

Date: 2026-08-30
Status: approved (design), pending implementation plan

## 1. Purpose

PoseYdon is a modular repository for deep learning on motion: synthesis first, then
editing, in-betweening and retargeting. It exists because the MDM lineage — including
`external/neural_motion_blending`, its direct ancestor — hardcodes nearly every choice
it makes. Feature layout, joint counts, skeleton metadata, loss composition and
sampling behaviour are all fixed in Python constants or inlined into forward passes.
That slows debugging, blocks ablation, and makes each new dataset or model a fork
rather than a plugin.

The goal is a repo where components swap, compose and extend through configuration,
and where adding a new component is a structured act that works with everything
already present. Explicitly *not* a goal: maximal generality. Every abstraction here
earns its place by removing a specific, identified coupling.

### Success criteria

1. Changing the motion feature representation is a config edit, with no
   re-preprocessing and no code change.
2. Changing the conditioning set is a config edit; only what is declared is loaded.
3. Adding a dataset means adding data files and manifests, not editing a constants
   module.
4. Adding a model, loss, feature, conditioner, sampler, control or operation means
   adding one class and one config entry, and it inherits the contract test suite.
5. Both reference models (AnyTop, MoDiffAE) run in PoseYdon from converted
   checkpoints and reproduce the reference's samples to the tolerances in §13.

## 2. Scope

### v1 (this spec)

Ingest, feature system, conditioning system, data layer, both models, gaussian
diffusion, flow matching, loss terms, Lightning training, samplers, controls,
operations (generate / blend / in-between), BVH export, IK solver, checkpoint
converter, CLI, test suite.

### Deferred to follow-up specs

Evaluation benchmarks (FID, per-window NN, distance metrics), Blender
visualization, FBX ingest, latent-space models beyond MoDiffAE, retargeting as a
first-class operation.

These must slot in without modifying core contracts. If a follow-up cannot, the
abstraction was wrong and this spec needs revision.

## 3. Baseline findings

Read-only reference: `external/neural_motion_blending`. Never edited. Findings that
directly motivate design decisions below:

| # | Finding | Location | Consequence |
|---|---|---|---|
| F1 | Feature vector hardcoded at 13 dims (pos3+rot6d6+vel3+foot1) | `param_utils.FEATS_LEN`, both models, all losses | Cannot vary representation without editing everything |
| F2 | Per-species knowledge as Python literals (`FACE_JOINTS`, taxonomy lists, thresholds) | `param_utils.py` | New skeleton requires a code edit |
| F3 | Structural losses index raw slices `[...,3:9]`, `[...,12,:-1]` on the **z-normalized** tensor | `gaussian_diffusion.py` | Geodesic distance computed between distorted frames; loss is not the published quantity |
| F4 | Conditioning is a 14-key dict accessed by string | `tensors.py`, both models | Missing key is a runtime failure mid-training |
| F5 | Preprocessing chunks clips at a hardcoded 200/240 frames, names them by a **global counter** | `motion_process.py:380-410` | Ids not reproducible across runs; animations split arbitrarily; transitions never observed |
| F6 | Clip identity parsed from filenames in 8 places, each re-implementing `_m` suffix handling | repo-wide | Fragile; relations between skeletons encoded as naming convention |
| F7 | Blending logic (`_mix`) lives inside `MoDiffAE.forward` | `motion_diffusion_ae.py` | In-betweening and other edits unreachable |
| F8 | Global padding to `MAX_JOINTS=143` | collate, both models | A 28-joint Flamingo wastes ~80% of every attention matrix |
| F9 | `ml_platform_type = eval(args.ml_platform_type)` | `train_anytop.py` | `eval()` on a CLI string |
| F10 | IK is fixed-iteration CPU Jacobian, no joint limits, no customization | `InverseKinematics.animation_from_positions`, used at ingest and export | Determines quality of every exported BVH |
| F11 | `scale()` docstring says "longest armature", code uses mean bone length | `motion_process.py:74-80` | Docs and behaviour disagree |
| F12 | `vel_thresh=0.002` compared against a **squared** displacement | `get_foot_contact` | Threshold is ~0.045 units/frame, not 0.002 |
| F13 | Per-skeleton hack `if object_type == "Anaconda"` inside a generic function | `get_root_quat` | Skeleton data living in code |
| F14 | Preprocessing renders a matplotlib mp4 per chunk inline | `process_object` | Preprocessing is far slower than it needs to be |
| F15 | Install requires conda plus `pip install --no-build-isolation git+...Motion.git` | `README`, `environment.yaml` | Fragile setup |
| F16 | `cond.npy` pickles Holden `Quaternions` objects | `dataset/truebones/.../cond.npy` | The preprocessed artifact cannot be read at all without installing the `Motion` package from git |
| F17 | Alignment constants come from the T-pose and are shared across a skeleton's clips, but nothing says so | `process_object` -> `process_anim(..., root_pose_init_xz, scale_factor, ground_height)` | Easy to reimplement per clip, which grounds flying creatures onto the floor |

## 4. Architecture

Two pipelines over shared contracts:

```
ingest:  BVH/FBX + manifest -> parse -> align -> Anim (.npz) + corpus index
train:   Anim -> features -> conditioners -> MotionBatch -> MotionTask -> ckpt
sample:  ckpt + inputs -> Operation -> Sampler (+ Controls) -> Anim -> solver -> BVH
```

```
src/poseydon/
├── core/          contracts only, no torch modules
│   ├── spec.py        FeatureSpec, Block (name -> slice, raw/normalized space)
│   ├── batch.py       MotionBatch, Cond, Masks, WindowInfo
│   ├── anim.py        Anim: rotations, root_pos, offsets, parents, names, fps
│   ├── skeleton.py    Skeleton, SkeletonManifest
│   ├── rotations.py   quaternion / 6d / matrix conversions, FK
│   └── registry.py    name -> class, for list-valued components
├── io/            bvh.py (vectorized parse/write), fbx.py (deferred)
├── ingest/        pipeline.py, align.py, index.py
├── features/      base.py, kinematics.py, contact.py
├── conditioners/  base.py, skeleton.py, semantic.py
├── data/          dataset.py, window.py, normalize.py, collate.py, datamodule.py
├── models/        base.py, anytop.py, modiffae.py, modules/
├── process/       base.py, gaussian.py, flow.py
├── losses/        base.py, simple.py, geodesic.py, footskate.py, kl.py
├── solvers/       base.py, gradient_ik.py, terms.py
├── sampling/      base.py, ddpm.py, ddim.py, euler.py, controls/
├── ops/           base.py, generate.py, blend.py, inbetween.py
├── training/      task.py, callbacks/
├── compat/        convert.py
└── cli.py
```

## 5. Core contracts

```python
@dataclass(frozen=True)
class Block:
    name: str
    space: Literal["raw", "normalized"] = "normalized"

@dataclass(frozen=True)
class FeatureSpec:
    blocks: Mapping[str, slice]   # ordered; name -> slice into D
    dim: int
    def slice(self, name: str) -> slice          # raises on unknown block

@dataclass(frozen=True)
class MotionBatch:
    x: Tensor                # (B, J, D, T)
    spec: FeatureSpec
    masks: Masks             # temporal / spatial / valid, computed once
    window: WindowInfo       # (start, source_length) per item
    cond: Cond               # only what config declared
    def to(self, device) -> "MotionBatch"
```

`x`, `spec`, `masks` and `window` are the only mandatory members. Everything the
reference puts in `cond['y']` — T-pose, joint names, topology, object type,
normalization stats — is a declared conditioner (§7). Addresses F4.

`Masks` replaces the reference's four overlapping masks and inline
`.repeat(...).reshape(...)` gymnastics with named accessors computed once per batch.

## 6. Data layer

### 6.1 Skeleton manifest

Per-skeleton YAML. Replaces `param_utils.py` (F2, F13). Every joint reference is a
**name** resolved against the BVH hierarchy at ingest; an unknown name fails
immediately with the available names and a nearest-match suggestion.

```yaml
skeleton: Flamingo
tpose: ../tposes/Flamingo.bvh

# Forward axis: normalized sum of (right - left) over the pairs below,
# then forward = cross(world_up, across). Named pairs make a left/right
# swap a readable mistake rather than a silent mirror.
facing:
  hips:      {right: BN_Leg_R_01,  left: BN_Leg_L_01}
  shoulders: {right: BN_Wing_R_01, left: BN_Wing_L_01}
  extra_yaw_deg: 0            # per-skeleton correction (Anaconda: -90)

foot_joints: [BN_Leg_R_03, BN_Leg_R_04, BN_Leg_L_03, BN_Leg_L_04]
fps: 20                       # sources are resampled to this

# Skeleton uniformly scaled so MEAN bone length equals this value.
# All lengths below are in these normalized units.
scale: {mean_bone_length: 0.1387}

contact:
  max_height: 0.30            # normalized length above ground plane
  max_speed:  0.045           # normalized length per frame (NOT squared) -- fixes F12

tags: [biped, bird]           # optional, user metadata only
joint_limits:                 # optional; unconstrained unless declared
  BN_Leg_R_02: {swing_deg: 45, twist_deg: [-20, 20]}
```

Manifests support inheritance via a `base:` key. Mixamo gets one `_base.yaml`
carrying the shared rig's facing/foot joints and `strip_joint_prefix: "mixamorig:"`;
per-character files carry only `tpose:`. Truebones skeletons stay standalone.

### 6.2 Ingest

`poseydon ingest` performs the expensive, deterministic, ambiguous work once:
parse, orient, centre, ground, scale. It writes:

**Alignment constants are skeleton-level, not per-clip** (F17). `root_xz`,
`scale_factor` and `ground_height` are derived once per skeleton -- from the
T-pose when the manifest names one -- and reused for every clip of that
character. Only the facing rotation is recomputed per clip. Computing the
constants per clip would ground a flying creature onto the floor and destroy the
height relationship between a crouch and a stand. Confirmed empirically: the
reference stores `ground_height = -0.0206` for Flamingo while that clip's own
minimum y is `-0.0285`, and the shipped assets have minimum y from `-0.121` to
`+0.026` rather than zero.

**Ingest does not resample.** The fixtures are 24 fps (`Frame Time: 0.041667`)
while the reference hardcodes `FPS = 20` and never resamples. A manifest `fps` is
a target that must be a no-op when it matches the source.


- **One `Anim` per source BVH, full length, never chunked** (fixes F5). Stored as
  `.npz`: rotations (quaternion), root_pos, offsets, parents, names, fps.
- **A corpus index**, one row per clip. Stored as JSONL: append-only, diffable, git-
  friendly, readable without a dependency, and small enough at corpus scale (tens of
  thousands of rows) that a columnar format would be premature.

Ids are deterministic: `{skeleton}__{action_slug}`, where `action_slug` is the source
BVH stem, lowercased, with runs of non-alphanumerics collapsed to `_` and leading and
trailing `_` stripped. A collision appends `__{n}` and is reported. `__` is a reserved
separator;
ingest rejects a skeleton name containing it. Re-ingesting is idempotent, and a clip
id means the same thing on every machine (fixes F5, F6).

No rendering during ingest (fixes F14).

### 6.3 Corpus index

```
clip_id                  skeleton   action      split  n_frames  path
Goblin__Walking          Goblin     Walking     train  213       .../a1b2.npz
Goblin_m__Walking        Goblin_m   Walking     train  213       .../c3d4.npz
Flamingo__OneLegBent     Flamingo   OneLegBent  train  400       .../e5f6.npz
```

Skeletons are distinct entities: `Goblin` and `Goblin_m` are two skeletons, with no
relation field in the core. Relationships and taxonomy live in optional manifest
`tags`. Subsets become index queries (`tags contains flying`) replacing
`OBJECT_SUBSETS_DICT`; retargeting pairs are `same action, different skeleton`,
replacing `build_mixamo_retargeting_bench.py`. Nothing downstream parses a filename
(fixes F6).

Balanced sampling reads the index instead of recomputing weights from name prefixes.

### 6.4 Features

Extractors are pure functions of `Anim`, registry-named, each declaring its
dependencies. Requested blocks are concatenated at load time into `(F, J, D)` plus
the `FeatureSpec` naming every block's slice.

```yaml
features: [ric_pos, rot6d, local_vel, foot_contact]   # D = 13, reference parity
features: [rot6d, local_vel]                          # D = 9, no re-ingest
```

Fixes F1. A new feature type is one class plus a config entry, with no disk touch.

### 6.5 Normalization

`Normalizer` holds per-skeleton statistics **computed for the active feature spec**
and cached by spec hash. It is invertible, so a loss may request `space="raw"` and
receive genuinely unnormalized values. This is what makes F3 unrepresentable.

### 6.6 Windowing

Windowing is a load-time sampling policy, declared per split:

```yaml
data:
  train: {window: {_target_: RandomCrop,    length: 40}}
  val:   {window: {_target_: SlidingWindow, length: 40, stride: 20}}
```

`MotionBatch.window` carries `(start, source_length)`. AnyTop's positional-embedding
offset is therefore relative to the true animation rather than an arbitrary 200-frame
slice. Parameters are unaffected; the equivalence test (§13) pins windows explicitly.

### 6.7 Collate

Padding is **per-batch maximum**, not a global constant (fixes F8).

## 7. Conditioning

A conditioner owns its whole data path and ships a default encoder when it needs
one. The **model decides injection**, because injection is genuinely model-specific:
AnyTop concatenates the T-pose as frame 0 and *adds* name embeddings to joint tokens,
which no generic mechanism captures honestly.

```python
class Conditioner(ABC):
    payload: type
    def extract(self, anim, manifest, labels): ...
    def collate(self, items): ...
    def encoder(self, d_model) -> nn.Module | None: return None
```

```yaml
cond: [tpose, joint_names, topology]   # Truebones / AnyTop
cond: [action]                          # a labelled fixed-skeleton corpus
cond: []                                # unconditional
```

Only declared conditioners are loaded, collated and moved to device. Models declare
`requires` / `optional`, validated once at fit start against the configured set, so a
mismatch fails in the first second with a readable message (fixes F4).

The T5 joint-name cache becomes an ordinary conditioner cache rather than a special
case the dataset constructor checks for and falls back from by instantiating T5 on
`cuda` at import time.

## 8. Models

```python
class Denoiser(nn.Module, ABC):
    requires: tuple[str, ...] = ()
    optional: tuple[str, ...] = ()
    def prepare(self, batch) -> Tensor: return batch.x
    @abstractmethod
    def forward(self, z_t, t, cond) -> Prediction: ...
    def restore(self, z0, batch) -> Tensor: return z0

class LatentDenoiser(Denoiser, ABC):
    @abstractmethod
    def encode(self, x0, cond) -> Tensor: ...
    @abstractmethod
    def decode(self, z0, cond) -> Tensor: ...
    def prepare(self, batch): return self.encode(batch.x, batch.cond)
    def restore(self, z0, batch): return self.decode(z0, batch.cond)

@dataclass
class Prediction:
    out: Tensor
    aux: dict = field(default_factory=dict)   # mu/logvar for KL, activations for probing
```

Latent operation is expressed by subclassing, not by an injected identity component.
Configuration never mentions autoencoding; the model's class decides.

**Ports are parameter-preserving.** Each forward is transcribed faithfully — AnyTop
keeps its behaviour, MoDiffAE keeps the issue-34 mask fix already present in its
`gather_vars`. Moving `_mix` out of `MoDiffAE.forward` into the `LatentMix` control
(fixes F7) changes the signature, not any parameter, so conversion is unaffected.

## 9. Process and losses

```python
class Process(ABC):
    def sample_t(self, batch_size) -> Tensor: ...
    def corrupt(self, z0, t, noise) -> Tensor: ...
    def target(self, z0, noise, t) -> Tensor: ...
    def to_z0(self, pred, z_t, t) -> Tensor: ...
    def solver_class(self) -> type[Sampler]: ...
```

`GaussianDiffusion` (betas linear/cosine, x0/eps/v parameterization, respacing) and
`FlowMatching` (linear interpolant, velocity target). `to_z0` is what makes
structural losses process-agnostic: whatever the network predicts, losses always
receive clean `x0`.

```python
class LossTerm(ABC):
    needs: tuple[Block, ...] = ()          # validated against FeatureSpec at fit start
    def __call__(self, x0_hat, x0, batch, aux) -> Tensor: ...
```

- `simple` — masked L2 on the process target
- `geodesic` — needs `Block("rot6d", space="raw")`; **fixes F3**
- `footskate` — needs `Block("ric_pos", space="raw")` and `Block("foot_contact")`
- `kl` — reads `aux["mu"]`, `aux["logvar"]`; warmup/anneal schedule preserved

Weights are config, not process constructor arguments. A loss requesting a block the
active spec lacks fails at fit start.

## 10. Training

One `MotionTask` LightningModule:

```python
z0   = self.model.prepare(batch)
t    = self.process.sample_t(len(batch))
eps  = torch.randn_like(z0)
z_t  = self.process.corrupt(z0, t, eps)
pred = self.model(z_t, t, batch.cond)
x0h  = self.model.restore(self.process.to_z0(pred, z_t, t), batch)
loss = sum(w * term(x0h, batch.x, batch, pred.aux) for w, term in self.losses)
```

Lightning removes: manual step counting, `MixedPrecisionTrainer`, `dist_util`,
resume-filename regex parsing, and `ml_platforms.py` with its `eval()` call (F9).
EMA becomes a callback. `gen_during_training` becomes a callback that runs a real
`Operation`, so training and sampling share one code path rather than
`training_loop.py` importing sampling scripts at module scope.

## 11. Sampling and operations

```python
class Sampler(ABC):     # how the trajectory is solved
    def __call__(self, model, process, shape, cond, controls) -> Tensor: ...

class Control(ABC):     # what edit is applied, per step
    def before_step(self, z_t, t, cond) -> tuple[Tensor, Cond]: ...
    def after_step(self,  z_t, t, cond) -> Tensor: ...

class Operation(ABC):   # the user-facing verb
    def build_controls(self, batch) -> list[Control]: ...
    def run(self, model, process, batch) -> Anim: ...
```

Samplers: `DDPM`, `DDIM` (with inversion as a mode, not a separate script path),
`Euler`, `Heun`. Controls: `CFG`, `Inbetween` (known-frame imputation), `LatentMix`
(driven by a `Schedule`: static / linear / ease with slope). Operations: `generate`,
`blend`, `inbetween`.

This dissolves `mix.py`'s 705 lines: alpha-schedule maths into `Schedule`; DDIM
inversion and noise alignment into the `DDIM` sampler; control-batch construction
into `Operation.build_controls`; output naming and resume-scanning into a generic run
writer shared by all operations.

## 12. Solver

The IK solver is a seam, not a fixed dependency (fixes F10). Used at ingest
(rebuilding a T-pose consistent with offsets) and at export (generated global
positions to BVH rotations), the latter determining the visual quality of every
exported BVH.

```python
class Solver(ABC):
    def solve(self, targets, skeleton, init=None) -> Tensor: ...

class GradientIK(Solver):      # torch, batched over frames and clips, GPU
    terms: list[IKTerm]        # registry-named, weighted
    optimizer: str             # lbfgs | adam
    iterations: int
```

```yaml
solver:
  _target_: poseydon.solvers.GradientIK
  optimizer: lbfgs
  iterations: 60
  terms:
    position:     {w: 1.0}
    bone_length:  {w: 1.0}
    joint_limits: {w: 0.5}    # swing-twist cones, from the manifest
    smoothness:   {w: 0.1}
    contact_pin:  {w: 2.0}    # pins feet where foot_contact == 1
```

`contact_pin` uses the model's own predicted foot-contact channel, which the
reference exporter discards. This enforces at export what `lambda_fs` only encourages
at training.

### Dependencies

Vendor BVH parse/write (vectorized: the MOTION block is a float matrix, read in one
pass rather than per-line), FK, and the IK solver. Take `roma` for torch rotation
maths and `scipy.spatial.transform.Rotation` for numpy. This drops the git dependency
and conda entirely (fixes F15): install becomes `uv sync`.

Library choice is to be **measured, not asserted**. Milestone 1 includes a benchmark
against the `Motion` baseline on the golden fixtures, with `Motion` retained as an
optional `[compat]` extra for comparison. No off-the-shelf library is known to
provide batched-GPU IK with joint limits over arbitrary BVH hierarchies:
`pytorch_kinematics` is URDF/MJCF-shaped, and `pinocchio`/`placo`/`mink` are
robotics-oriented and CPU-focused. If the benchmark contradicts this, revise.

## 13. Configuration, CLI, conversion

Hydra config groups for singular choices (`model`, `data`, `process`, `sampler`,
`op`, `trainer`), each yaml carrying a `_target_`. A small name registry for
list-valued components (features, conditioners, losses, IK terms) where a full
`_target_` block per entry would be noise and where `poseydon list` should work.
Top-level groups use structured configs, so a typo fails at compose time.

The resolved config is stored in the Lightning checkpoint's hyperparameters.
`poseydon sample ckpt=...` reconstructs the model from the checkpoint itself — no
`args.json` beside `model.pt`, no drift between them, CLI overrides still apply.

CLI: `poseydon ingest | train | sample | convert | list`.

`poseydon convert` maps a reference checkpoint plus its `args.json` into a PoseYdon
checkpoint: the ~90 flags map onto config, state-dict keys remap mechanically.

**Acceptance gate:** same checkpoint, same seed, same explicit window — reference
code and PoseYdon produce the same samples. Tolerance on the output motion tensor:
`atol=1e-4`, `rtol=1e-3` in float32, both models, over at least ten seeds spanning
three skeletons. These are starting values; milestone 3 records the tolerance
actually achieved and tightens them if the gap is smaller. A gate that needs
*loosening* means a real behavioural difference and must be explained, not widened. This is a correctness
proof, not a backward-compatibility requirement; no permanent compat surface is kept.
Loss corrections (F3) are training-only and cannot affect this gate.

## 14. Testing

Three tiers. Real data is primary; it is known-good.

**Tier 1 — real fixtures (primary).** The seven BVH + `.npy` pairs from the reference
`assets/truebones` (31-63 joints, bipeds / quadrupeds / millipeds), each `.npy` an
`(F, J, 13)` float64 feature array for its BVH. Used for ingest parity (features vs
`.npy` at `atol=1e-6` on float64 features, the alignment maths being the part most
likely to drift), and as the substrate for as many behavioural tests as
possible: windowing, normalization, conditioners, collate, export round-trip, solver
benchmark.

**Tier 2 — controlled perturbations of real files.** Ground truth without giving up
real rigs: apply a known 37 degree yaw to a real BVH and assert facing recovers it;
freeze a foot and assert `foot_contact` fires exactly there; drop joints and assert
padding/masking behaviour.

**Tier 3 — synthetic, degenerate cases only.** Only what the seven cannot express: a
one-joint skeleton, a two-frame clip, a manifest naming a nonexistent joint (asserting
the nearest-match error path, not an `IndexError`).

Plus **contract tests parameterized over the registry**: every `Feature`,
`Conditioner`, `Process`, `LossTerm` and `IKTerm` is checked against its ABC's
invariants — for example `to_z0(target(x0, e, t), corrupt(x0, t, e), t) ~= x0`,
features round-trip, the normalizer inverts. New components inherit these for free,
which is the mechanical half of success criterion 4.

**Open decision for the author.** The seven fixtures currently live in the read-only
`external/` checkout, which not every machine or CI runner will have. Making them
primary means copying them into `tests/data/`. They are Truebones assets, already
published in `mmlab-cv/neural_motion_blending`, so this is consistent with existing
practice — but it is a licensing call for the author, not one this spec makes. Until
it is decided, tier 1 skips when `external/` is absent.

## 15. Milestones

| # | Deliverable | Gate |
|---|---|---|
| 1 | core contracts, BVH IO, rotations/FK, ingest, index, fixtures, IO+IK benchmark | tiers 2-3 and contract tests green; benchmark recorded |
| 2 | features, conditioners, normalizer, windowing, collate, datamodule | tier 1 golden parity on all seven fixtures |
| 3 | gaussian + flow processes, losses, MotionTask, AnyTop port, converter | train smoke run on both processes; contract test `to_z0(target(...)) ~= x0` for each; equivalence gate on AnyTop |
| 4 | MoDiffAE port, latent hooks, KL schedule | equivalence gate on MoDiffAE |
| 5 | solvers, samplers, controls, operations, BVH export | blend reproduces reference output |
| 6 | CLI polish, `poseydon list`, docs | — |

## 16. Deliberate deviations from the reference

1. Geodesic and foot-skate losses computed in raw space (F3). Training-only; does not
   affect the equivalence gate.
2. Clips are never chunked at preprocessing (F5).
3. Clip ids are deterministic rather than counter-derived (F5).
4. Padding is per-batch (F8).
5. Contact threshold expressed as speed, squared internally (F12).
6. `scale()` documented as mean bone length, matching its behaviour (F11).
7. Anaconda yaw correction becomes manifest data (F13).

Each is recorded so a numerical difference from published results has a known cause.

## 17. Deferred / open

- FBX ingest (needs Blender or the FBX SDK; kept out of core dependencies).
- Evaluation benchmarks and Blender visualization — follow-up specs, must slot in
  without core changes.
- Licensing decision on vendoring the seven test fixtures (§14).

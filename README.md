<p align="center">
  <img src="assets/media/logo_poseydon.png" alt="PoseYdon" width="620">
</p>

<p align="center">
  <b>Research infrastructure for neural motion.</b><br>
  Generation, editing, in-betweening and cross-topology retargeting —
  on one representation, one training loop, one set of contracts.
</p>

---

MDM is the cornerstone of this field and deserves to be. It is also five years of
accumulated assumptions: feature layout, joint counts, skeleton metadata, loss
composition and sampling behaviour are fixed in Python constants or inlined into
forward passes. Every new dataset is a fork. Every ablation is a patch. Every
follow-up paper reimplements the same preprocessing slightly differently, and the
differences are invisible until the numbers disagree.

PoseYdon is a fresh starting point for that work. It keeps what the lineage got
right — the diffusion formulation, the 13-dimensional per-joint representation,
the graph-attention backbone — and rebuilds the scaffolding around it so the
interesting parts are the parts you change:

```bash
poseydon train                                  # defaults
poseydon train process=flow                     # flow matching instead of diffusion
poseydon train features=[rot6d,local_vel]       # a different representation, no re-ingest
poseydon train model=modiffae losses='{simple:1.0,kl:1e-3}'
poseydon retarget --content Flamingo/onelegbent --target Scorpion
```

Nothing above is a fork or a flag added to a monolith: the process, the feature
set, the model and the loss are separate objects the config composes.

### Verified against the work it builds on

The claim that a rewrite is faithful is worth nothing unless it is measured, so
it is. `external/neural_motion_blending/` is a read-only checkout of the
published implementation, and the parity tools drive BOTH codebases from the same
weights and the same noise:

| what | agreement |
|---|---|
| features, 7 rigs, every block | ~1e-6, foot contact **bit-exact** |
| DDPM sampling, 200 frames x 100 steps | **2.97e-06** relative |
| DDIM sampling, 200 frames x 100 steps | **8.39e-07** relative |

Where PoseYdon deliberately differs — invertible preparation, centred
per-channel normalization, a constant-channel guard — it is documented as a
choice with the measurement behind it, not left as an accident.


## Quick start

The only host requirement is Docker. No Python, pip or uv installation.

```bash
docker compose build test
docker compose run --rm test pytest
docker compose run --rm test python -m poseydon.cli list
```

`tests/build/test_roundtrip_fbx.py` needs Blender, which the `test` image
deliberately does not ship, so it is excluded from the run above and must be
run inside the `fbx` container instead:

```bash
docker compose run --rm fbx blender -b --python-expr \
    "import sys, pytest; sys.exit(pytest.main(['tests/build/test_roundtrip_fbx.py','-v','-rs']))"
```

Ingest a BVH corpus, train, and sample:

```bash
poseydon ingest <bvh_dir> --manifests data/truebones/skeletons --out data/truebones
poseydon train  trainer=debug
poseydon sample --checkpoint runs/**/last.ckpt skeleton=Goat n_samples=4 --render
```

## Sampling

```bash
poseydon sample \
    --checkpoint runs/**/last.ckpt \
    --render --zoom 2.0 --out samples \
    skeleton=Scorpion n_samples=4 n_frames=40 \
    sampler.steps=50 reconstruct.name=positions_ik
```

Everything after the flags is a Hydra override into `configs/sample.yaml`, which
selects the operation (`generate`, `inbetween`, `blend`), the sampler (`ddpm`,
`ddim`, `euler`), and how the skeleton is rebuilt.

The code path, in order:

| step | where |
|---|---|
| parse flags, compose config | `src/poseydon/cli.py::_sample` |
| build dataset, model, process | `src/poseydon/training/build.py` |
| pick the verb | `src/poseydon/ops/` (`generate`, `inbetween`, `blend`) |
| solve the trajectory | `src/poseydon/sampling/` (`ddpm`, `ddim`, `euler`) |
| per-step edits | `src/poseydon/sampling/controls/` |
| features back to a skeleton | `src/poseydon/features/reconstruct.py` |
| write BVH / MP4 | `src/poseydon/io/bvh.py`, `src/poseydon/io/render.py` |

One real batch supplies the skeleton's conditioning -- topology, rest pose,
normalization statistics all describe the rig rather than the clip -- so
`skeleton=` picks which character to animate.

### Choosing how the skeleton is rebuilt

The representation is redundant: it carries rotations AND positions, and a
generated clip need not keep them consistent. On real data they agree to 1e-9;
on model output they differ by more than three bone lengths. So this is a real
choice, not a formality:

```yaml
reconstruct:
  name: fk | positions | positions_ik
  iterations: 150     # positions_ik only
  smoothness: 0.05
```

| method | bone lengths | notes |
|---|---|---|
| `fk` | exact | forward kinematics from rotations. Free, rigid |
| `positions` | **not enforced** | the prediction itself; limbs stretch. No BVH, since a BVH stores rotations |
| `positions_ik` | exact | positions fitted back onto a rigid skeleton by IK |

`positions_ik` is approximate by nature: rotation about a bone's own axis does
not move its children, so twist is invisible to a position-fitting solver. It
plateaus near 1% of a bone length mean error rather than reaching zero.

### Rendering

`--render` writes an MP4 per sample; `--zoom` frames it (values above 1 move the
camera closer). Rendering is an optional extra: `pip install poseydon[render]`.

## How it fits together

```
ingest:  BVH + manifest -> parse -> align -> Anim (.npz) + corpus index
train:   Anim -> features -> conditioners -> MotionBatch -> MotionTask -> ckpt
sample:  ckpt -> Operation -> Sampler (+ Controls) -> features -> Anim -> BVH
```

### Skeleton knowledge is data

A dataset is a folder of BVH files plus one YAML manifest per skeleton. Joints are
referenced by **name**, never by index, so a re-exported rig fails loudly instead
of silently mirroring the character.

```yaml
skeleton: Flamingo
facing:
  hips:      {right: Bip01_R_Thigh,   left: Bip01_L_Thigh}
  shoulders: {right: BN_Forearm_R_02, left: BN_Forearm_L_02}
scale: {mean_bone_length: 0.20921428571428569}
contact:
  max_height: 0.30
  max_speed: 0.044721359549995794
foot_joints: [Bip01_R_Foot, Bip01_R_Toe0, ...]
tags: [biped, bird]
```

Adding a character is adding files. Manifests support `base:` inheritance, so a
corpus sharing one rig — Mixamo, say — declares it once.

### Features are composed at read time

Disk stores the canonical **aligned animation**, not features. Extraction happens
when data is read, so changing the representation never re-runs preprocessing:

```yaml
features: [ric_pos, rot6d, local_vel, foot_contact]   # 13 dims
features: [rot6d, local_vel]                          # 9 dims, same files
```

A `FeatureSpec` owns the layout, so a loss asks for the block by name and fails
loudly against a representation without rotations — rather than a magic
`[..., 3:9]` reading whatever sits there:

```python
spec.take(features, "rot6d")                  # (F, J, 6)
spec.take(batch.x, "rot6d", axis=2)           # (B, J, 6, T) — channels are not last here
spec.take(features, "foot_contact", drop=True)  # (F, J), refuses on a wider block
```

`axis` is explicit because the channel axis genuinely differs by layout, and a
helper that guessed would read frames as channels on the training path.

### Only declared conditioning is loaded

`MotionBatch` carries four mandatory things — the tensor, its spec, the masks
padding makes unavoidable, and where the window came from. Everything else is a
declared conditioner:

```yaml
conditioners: [topology, tpose, norm_stats, joint_names]   # topology-aware
conditioners: []                                           # unconditional
```

Models declare what they need; the mismatch is caught in the first second of a
run, not forty minutes in. `conditioners` is also recorded in the checkpoint's
recipe and checked at sampling time, so a model trained with joint-name
embeddings cannot be sampled without them — a mismatch that produces
plausible-looking garbage and no error otherwise.

### Latent models subclass, they do not get an injected autoencoder

`Denoiser` works in feature space. `LatentDenoiser` overrides `prepare`/`restore`
with its own encoder and decoder. The task body is identical for both, nothing
branches on model kind, and configuration never mentions autoencoding.

### Sampling separates three things the reference conflates

A **Sampler** solves the trajectory (`DDPM`, `DDIM` with inversion and an
`eta` that interpolates to ancestral sampling, `Euler`).
A **Control** applies an edit at each step (guidance, imputation, latent mixing).
An **Operation** is the user-facing verb (`generate`, `inbetween`, `blend`).

In-betweening needs no model change at all — it is imputation applied during
sampling. In the reference, blending lives inside `MoDiffAE.forward`, which is why
in-betweening was unreachable there.

## Retargeting, and watching it during training

Cross-topology transfer is the capability the architecture exists for: a semantic
latent is pooled over joints, so a 40-joint encode decodes onto a 63-joint rig
with nothing reshaped and no per-pair training.

```bash
poseydon retarget --content Flamingo/onelegbent --target Scorpion
```

`LatentPin` holds the content clip's latent fixed while the target rig's
conditioning drives the decode. Because it is a Control rather than model code,
the same mechanism serves blending and in-betweening.

Training can watch it. A `validation:` block retargets fixed pairs on an
interval, writes `.npz` + `.bvh` + `.mp4` for each, and logs six scalars plus one
wandb artifact per firing:

```yaml
validation:
  every_n_steps: 25000
  pairs:
    - {content: {rig: Flamingo, action: onelegbent}, target: Scorpion}
    - {content: {rig: Goat,     action: headbutt},   target: Raptor}
```

Training loss says the model denoises. It says nothing about whether a
Flamingo's semantics land on a Scorpion, and that is the task.

## Layout

```
src/poseydon/
├── core/          contracts: spec, batch, anim, skeleton, topology, rotations
├── io/            BVH read and write
├── ingest/        alignment, corpus index, pipeline
├── features/      extractors, assembly, recovery
├── conditioners/  topology, tpose, normalization statistics
├── data/          windowing, dataset, collate, normalization
├── models/        base, anytop, modiffae
├── process/       gaussian diffusion, flow matching
├── losses/        simple, geodesic, footskate, kl
├── solvers/       batched GPU inverse kinematics
├── sampling/      samplers and step-level controls
├── ops/           generate, inbetween, blend
└── training/      task, Lightning wrappers, config wiring
```

## Testing

Real data is primary. The suite runs against seven Truebones clips spanning
bipeds, quadrupeds and millipeds at 31 to 63 joints.

The decisive test is **golden parity**: features extracted by PoseYdon reproduce
the reference's own `.npy` arrays for all seven skeletons, every block, with foot
contact bit-exact. Tolerance is bounded by the file format — BVH stores six
decimals — not by float64.

Tests requiring those fixtures skip when the reference checkout is absent, so a
fresh clone still runs everything else.

### Passing is not the same as covering

A green suite proves nothing about a test that cannot fail. The parity and unit
tests are periodically checked by **mutation**: deliberately breaking a
behaviour and confirming something goes red.

That has already earned its place twice. The camera tests reimplemented the
framing formula instead of calling it, so they exercised a copy and stayed green
through the exact regression they were written for. And the whole diffusion
core — `corrupt`, the posterior, both sampler loops — survived eleven separate
mutations untouched, including signal and noise swapped in the forward process.
The math was right, verified against the reference to 3e-06; nothing in CI would
have noticed it breaking.

If you add a test here, break the thing it covers and watch it fail first.

## Documents

- `docs/superpowers/specs/2026-08-30-poseydon-design.md` — architecture and the
  findings that motivated each decision
- `docs/superpowers/plans/` — implementation plans

## Reference

`external/neural_motion_blending/` is a **read-only** checkout of the published
work this supersedes, kept for reference and parity checking. Never edit it.

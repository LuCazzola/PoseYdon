# PoseYdon

A modular repository for deep learning on motion — synthesis first, then editing,
in-betweening and retargeting.

It exists because the MDM lineage hardcodes nearly every choice it makes. Feature
layout, joint counts, skeleton metadata, loss composition and sampling behaviour
are fixed in Python constants or inlined into forward passes, so ablation is slow
and each new dataset or model is a fork rather than a plugin.

Here, components swap through configuration:

```bash
poseydon train                                  # defaults
poseydon train process=flow                     # flow matching instead of diffusion
poseydon train features=[rot6d,local_vel]       # a different representation, no re-ingest
poseydon train model=modiffae losses='{simple:1.0,kl:1e-3}'
```

## Quick start

The only host requirement is Docker. No Python, pip or uv installation.

```bash
docker compose build test
docker compose run --rm test pytest              # 636 tests
docker compose run --rm test python -m poseydon.cli list
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

A `FeatureSpec` names every block's slice, so a loss asks for `spec.slice("rot6d")`
and fails loudly against a representation without rotations — rather than a magic
`[..., 3:9]` reading whatever sits there.

### Only declared conditioning is loaded

`MotionBatch` carries four mandatory things — the tensor, its spec, the masks
padding makes unavoidable, and where the window came from. Everything else is a
declared conditioner:

```yaml
conditioners: [topology, tpose, norm_stats]   # a topology-aware model
conditioners: []                              # unconditional
```

Models declare what they need; the mismatch is caught in the first second of a
run, not forty minutes in.

### Latent models subclass, they do not get an injected autoencoder

`Denoiser` works in feature space. `LatentDenoiser` overrides `prepare`/`restore`
with its own encoder and decoder. The task body is identical for both, nothing
branches on model kind, and configuration never mentions autoencoding.

### Sampling separates three things the reference conflates

A **Sampler** solves the trajectory (`DDPM`, `DDIM` with inversion, `Euler`).
A **Control** applies an edit at each step (guidance, imputation, latent mixing).
An **Operation** is the user-facing verb (`generate`, `inbetween`, `blend`).

In-betweening needs no model change at all — it is imputation applied during
sampling. In the reference, blending lives inside `MoDiffAE.forward`, which is why
in-betweening was unreachable there.

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

## Documents

- `docs/superpowers/specs/2026-08-30-poseydon-design.md` — architecture and the
  findings that motivated each decision
- `docs/superpowers/plans/` — implementation plans

## Reference

`external/neural_motion_blending/` is a **read-only** checkout of the published
work this supersedes, kept for reference and parity checking. Never edit it.

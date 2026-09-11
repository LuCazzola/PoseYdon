# How it fits together

```
ingest:  BVH + manifest -> parse -> align -> Anim (.npz) + corpus index
train:   Anim -> features -> conditioners -> MotionBatch -> MotionTask -> ckpt
sample:  ckpt -> Operation -> Sampler (+ Controls) -> features -> Anim -> BVH
```

### Skeleton knowledge is data

A dataset is a folder of BVH files plus one YAML manifest per skeleton. Joints are
referenced by **name**, never by index, so a re-exported rig fails loudly instead
of silently mirroring the character.

`data/truebones/rigs/Flamingo/manifest.yaml`:

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


# Retargeting, and watching it during training

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


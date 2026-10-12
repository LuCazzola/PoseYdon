# Retargeting motion between different skeletons

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
  guidance_scales: [0.0, 1.0, 1.25, 2.5]
  pairs:
    - {content: {rig: Flamingo, action: onelegbent}, target: Scorpion}
    - {content: {rig: Goat,     action: headbutt},   target: Raptor}
```

`guidance_scales` samples each pair once per scale, all from the same starting
noise, so the videos differ by the guidance and nothing else. Metrics are keyed
`retarget/<pair>/g<scale>/<metric>` with a per-scale mean beside them.

The scale is the standard `uncond + w * (cond - uncond)`, so it means one thing
across its whole range: `0.0` is the unconditional branch (the content clip
ignored), `1.0` is the model as trained, `(0, 1)` weakens the conditioning and
above `1.0` extrapolates. **`1.0` is the identity and the default** -- it skips
the wrapper entirely, so it costs one forward pass, and it is all a run without
`model.cond_drop > 0` can measure. `0.0` is not a spelling of "off"; asking for
it gets you output with the conditioning discarded. The content clip is also rendered on its own
rig and pushed **once**, on the first firing, as `retarget/<pair>/source`: every
other video is model output, and without it there is nothing saying what the
retarget was asked to reproduce.

Training loss says the model denoises. It says nothing about whether a
Flamingo's semantics land on a Scorpion, and that is the task.

### Representation-aware export

Exports default to the strict prepared representation. Select `--representation prepared|original` and `--format bvh|fbx`:

    poseydon export samples/Rig__clip.bvh --rig Rig --representation prepared --format bvh --out prepared.bvh

A prepared FBX is a bare skeleton; an original FBX is the clip on the vendor's own skinned asset, and needs the `fbxsdk` container (see [export](../index/library/export.md)).

# Retarget

Take one character's clip and perform it on a different skeleton: a
flamingo's one-legged stand, done by a scorpion. This is the task the
architecture exists for.

It works because the model's **semantic latent** describes *what* a clip is
doing while leaving out which skeleton does it: it is pooled over joints. A
40-joint clip encodes to a latent that decodes onto a 63-joint rig with
nothing reshaped and no training per pair of characters. The task encodes the
content clip once, then holds that latent fixed
([`LatentPin`][poseydon.sampling.controls.LatentPin]) while the target rig's
own conditioning drives the generation. It needs a model with such a latent
(`modiffae`).

```bash
docker compose run --rm test python -m poseydon.cli retarget \
    --checkpoint runs/<run>/checkpoints/last.ckpt \
    --content Flamingo/onelegbent --target Scorpion
```

| flag | means |
|---|---|
| `--content <Rig>/<action>` | the clip to take the motion from |
| `--target <Rig>` | the character to perform it on |
| `--guidance-scale` | how strongly to follow the content clip; see [the guidance scale](overview.md#the-guidance-scale) |
| `--out` | where to write the results (default `retargets/`) |

The output is as long as the content clip, because the latent has one entry
per frame. To put it on the target's real character, mesh and all, export the
`.anim.npz` as in
[Train, generate, export](../tutorials/train-and-generate.md#export-onto-the-real-character).

## Watching it during training

Training loss says the model denoises. It says nothing about whether a
flamingo's motion lands on a scorpion, and that is the task. So training can
retarget fixed pairs on an interval, through the same code as the command:

```yaml
validation:
  every_n_steps: 25000
  guidance_scales: [0.0, 1.0, 1.25, 2.5]
  pairs:
    - {content: {rig: Flamingo, action: onelegbent}, target: Scorpion}
    - {content: {rig: Goat,     action: headbutt},   target: Raptor}
```

Each firing writes `.npz`, `.anim.npz`, `.bvh` and `.mp4` for every pair and
logs six scalars plus one wandb artifact.

- **One sample per guidance scale**, all from the same starting noise, so the
  videos differ by the guidance and nothing else. Metrics are keyed
  `retarget/<pair>/g<scale>/<metric>`, with a per-scale mean beside them.
  A run without `model.cond_drop > 0` can only measure `1.0`.
- **The content clip itself** is rendered on its own rig and logged once, on
  the first firing, as `retarget/<pair>/source`. Every other video is model
  output; without it, nothing shows what the retarget was asked to reproduce.

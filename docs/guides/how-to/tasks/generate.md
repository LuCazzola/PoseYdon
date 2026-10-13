# Generate

New motion for a character, from noise. It is the base case: the model is
sampled as trained, with nothing steering it.

```bash
docker compose run --rm test python -m poseydon.cli sample \
    --checkpoint runs/<run>/checkpoints/last.ckpt \
    --render --zoom 2.0 --out samples \
    model=modiffae skeleton=Scorpion n_samples=4 n_frames=40 \
    sampler.steps=50 reconstruct.name=positions_ik
```

Everything after the flags overrides `configs/sample.yaml`:

| setting | means |
|---|---|
| `skeleton=` | which character to animate. Its topology, rest pose and normalisation come from one of its clips |
| `n_samples=`, `n_frames=` | how many clips, and how long |
| `model=` | must match the checkpoint. The config defaults to `anytop`, so a `modiffae` run needs `model=modiffae`; if they differ, the recipe check names the override to add |
| `operation.guidance_scale=` | see [the guidance scale](overview.md#the-guidance-scale) |
| `sampler.steps=`, `reconstruct.name=` | see [what every task shares](overview.md#what-every-task-shares) |

`--render` adds an MP4 per sample; `--zoom` frames it (above 1 moves the camera
closer).

## The code path

| step | where |
|---|---|
| parse flags, compose config | `src/poseydon/cli.py` (`_sample`) |
| build dataset, model, process | `src/poseydon/training/build.py` |
| the task | [`Generate`][poseydon.ops.Generate], `src/poseydon/ops/generate.py` |
| solve the trajectory | `src/poseydon/sampling/` |
| features back to a skeleton | `src/poseydon/features/reconstruct.py` |
| write BVH and MP4 | `src/poseydon/io/bvh.py`, `src/poseydon/io/render.py` |

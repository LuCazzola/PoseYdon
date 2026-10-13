# Train, generate, export

With the dataset built, this tutorial trains a model, uses it to make motion,
and puts that motion back onto a real character you can open in Blender.

## Train

```bash
docker compose run --rm train python -m poseydon.cli train trainer=debug
```

`trainer=debug` is a short run, enough to check that everything is wired up.
Drop it for a real run. Each run gets its own folder under `runs/`, holding
checkpoints and a **recipe**: a record of the features, conditioning and
dataset settings the model was trained with. Sampling checks the recipe, so a
checkpoint can't silently be used with inputs it never saw.

Everything after `train` is a Hydra override of `configs/train.yaml`, for
example `model=anytop process=flow`. Add `--print-config` to see the composed
config without training. The comments in `configs/train.yaml` say why each
default is what it is.

Runs log to Weights & Biases when `WANDB_API_KEY` is set in a `.env` file at
the repository root.

## Generate

```bash
docker compose run --rm test python -m poseydon.cli sample \
    --checkpoint runs/<run>/checkpoints/last.ckpt skeleton=Goat n_samples=4 --render
```

This makes four new clips for the Goat. Each one is written to `samples/` as:

- `.anim.npz`, the motion exactly as generated. **Export from this one.**
- `.bvh`, for a quick look in any animation tool. It rounds numbers and can't
  hold every joint's rotation, so don't use it as the source for an export.
- `.mp4` (with `--render`), a stick-figure video.

This is the **generate** task. [Motion tasks](../tasks/overview.md) covers the
other three (in-between, blend, retarget), the samplers and how the skeleton
is rebuilt.

## Retarget

```bash
docker compose run --rm test python -m poseydon.cli retarget \
    --checkpoint runs/<run>/checkpoints/last.ckpt --content Flamingo/onelegbent --target Scorpion
```

This takes the Flamingo's *onelegbent* clip and performs it on the Scorpion,
a character with a completely different skeleton. See
[Retargeting](../tasks/retarget.md).

## Export onto the real character

What the model produces lives in the shared, prepared frame and lacks the
joints the model ignores. `export` puts all of it back:

```bash
# as BVH, on the original rig: same joints, units and channels as the vendor file
docker compose run --rm test python -m poseydon.cli export \
    samples/<clip>.anim.npz --rig Goat --representation source --format bvh --out out/goat.bvh

# as FBX, on the vendor's own character, mesh and skin included (needs the FBX SDK)
docker compose run --rm fbxsdk python -m poseydon.cli export \
    samples/<clip>.anim.npz --rig Goat --representation source --format fbx --out out/goat.fbx
```

`--representation prepared` keeps the motion in the shared frame instead.
[The round trip](../../concepts/data/roundtrip.md) explains what each choice gives you and
what is lost on the way.

## Watch it

```bash
docker compose run --rm fbx blender -b --python scripts/render_fbx.py -- \
    out/goat.fbx --out out/goat.mp4 --threads 4
```

This renders the FBX, skinned, to a video. Blender is used only as a renderer
here; it never reads or writes the FBX data itself.

**Where next:** the [Data](../../concepts/data/notation.md) pages if you want to
understand what the model actually sees, the [Design](../../concepts/rationale.md)
pages if you want to change how it works.

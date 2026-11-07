# Building the dataset

A model can't train on raw vendor files. Each character arrives in its own
units, facing its own way, standing at its own height. This tutorial turns the
raw Truebones files into a **corpus**: one consistent dataset the model reads.
It takes two commands, after reading the vendor FBX files once.

## What you start with

The Truebones files are licensed, so they are not in the repository. Put them
under `data/_raw/truebones/source/`, one folder per character:

```
data/_raw/truebones/source/
  Dog/
    __Attack.bvh       the motion, as BVH
    Dog-Attack.fbx     the same motion, as FBX, with the character's mesh
    ...
  Flamingo/
  ...
```

Each character also needs a hand-written description of itself,
`data/truebones/rigs/<Rig>/manifest.yaml`: which joints are the hips and
shoulders (to work out which way it faces), which are feet, which clip holds
its rest pose. The clip list, `data/truebones/fbx_clips.csv`, names the FBX each
clip is read from. These are checked into git. See
[Dataset layout](../../concepts/data/layout.md) for what each field means.

## Step 0: read the FBX files once

Motion and mesh are read from the FBX. The FBX SDK runs only in the `fbxsdk`
container, so these two steps read what the rest needs, once:

```bash
docker compose run --rm -T --entrypoint python -e HOME=/tmp fbxsdk scripts/build_fbx_cache.py --data truebones
docker compose run --rm -T --entrypoint python -e HOME=/tmp fbxsdk tools/build_rest_fbx.py --data truebones
```

The first caches every clip's motion (`data/truebones/cache/fbx/`). The second
builds each character's **rest asset**: its FBX, mesh and bind unchanged, held in
the rest pose its manifest names, which is what the corpus measures rotations from
and what exports are written onto (`data/truebones/cache/rigs/<Rig>/`). Rerun
the second for a character whose `rest_pose:` changes.

## Step 1: prepare every clip

```bash
docker compose run --rm test python scripts/process_dataset_truebones.py
```

This puts every character into one shared frame: facing +Z, standing on the
ground, centred, scaled to a common size, with bones that don't stretch. It
writes the prepared clips to `data/truebones/clips/<Rig>/` and, for each
character, a `preproc.npz` that records exactly how to undo it. That record is
what later lets generated motion go back onto the original character; the
[round trip](../../concepts/data/roundtrip.md) explains how.

Add `--rigs Dog Flamingo` to prepare only some characters.

## Step 2: build what the model reads

```bash
docker compose run --rm test python scripts/build_features.py
```

This reads the prepared clips and writes, per character, the skeleton the
model sees (`skeleton.npz`) and the statistics used to normalise its inputs
(`stats.npz`), plus one index of every clip (`data/truebones/index.jsonl`).
What gets computed is set in `configs/build.yaml`: the dataset
(`configs/data/truebones.yaml`: which joints the model may drop, the labels)
and the feature group the statistics are for (`configs/features/default.yaml`).

If you change only the feature list, add `--stats-only`: the statistics are the
only thing that depends on it. The exception is switching the rotation layout,
which also changes the skeleton, so it needs the full run (see
[Notation](../../concepts/data/notation.md#rotation-layouts)).

## Check it

```bash
docker compose run --rm -T test python -m pytest -q
```

The round-trip test now runs on every clip. It checks that each prepared clip,
pushed through the model's representation and back, matches the raw source
file. If you prepared only some characters, narrow it with `-k Dog`.

To look at the result rather than test it,
`scripts/build_stage_samples.py` (in the `fbxsdk` container) writes one clip
per character at every stage, as BVH and FBX, under `tmp/truebones/<Rig>/`.

**Next:** [Train, generate, export](train-and-generate.md).

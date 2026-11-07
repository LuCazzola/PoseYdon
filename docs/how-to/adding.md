# Adding to PoseYdon

Recipes for adding to the library. Each one lists the files to touch, in
order, and a check that tells you it works. The *why* behind the layout is in
[Design rationale](../user-guide/index.md); this page is only the *what*.

## What are you adding?

| You want to add... | Touch | Recipe |
|---|---|---|
| a character to an existing dataset | a `manifest.yaml`, the vendor files, the clip list | [A rig](#a-rig) |
| a new dataset | `configs/data/<dataset>.yaml` + its rigs | [A dataset](#a-dataset) |
| a preparation stage, reduce rule or clip filter | a registered class + a dataset config entry | [A data step](#a-data-step) |
| a new motion representation | a `Feature` + `configs/features/` | [A feature block](#a-feature-block) |
| a denoising model | a `Denoiser` + `configs/model/` | [A model](#a-model) |
| a generative process | a `Process` + `configs/process/` | [A process](#a-process) |
| a loss term | a `LossTerm` + the run config | [A loss](#a-loss) |
| a motion task for a trained model | a registered operation | [An operation](#an-operation) |

If a recipe has you editing a training loop or a forward pass to support a new
character or dataset, a step was skipped: per-rig facts live in manifests, and
everything else that builds an object is named by `type:` in a config.

Every class goes in through the same door. It registers itself
(`@<REGISTRY>.register("name")`, or just a `name` on a `PrepareStage` or
`ReduceRule` subclass), and a config names it by `type:`. Inside
`src/poseydon/`, make sure its package imports the module. Outside it, list
the module under the config's `custom_imports:`. `poseydon list [kind]` prints
what each registry holds.

Every recipe ends the same way: update the [Map](../reference/map.md) for new
files, and the class diagram on the package's API page for new classes.

## A rig {#a-rig}

A new character in a dataset that already has a config, here Truebones.

1. **Vendor files.** Put them under `data/_raw/truebones/source/<Rig>/`
   (licensed, not in git).
2. **Clip list.** Add a `rig,action,file` row per clip to
   `data/truebones/fbx_clips.csv`.
3. **Manifest.** Write `data/truebones/rigs/<Rig>/manifest.yaml`: facing pairs,
   foot joints, tags, the rest-pose clip. Joints are named, never numbered.
   [The manifest](../reference/data-formats.md#the-manifest) lists every field.
4. **Build it**, with the commands of
   [Building the dataset](tutorials/02-build-a-corpus.md), narrowed with
   `--rigs <Rig>` where a command takes it.

Done when the round trip passes on it:
`docker compose run --rm -T test python -m pytest -q -k <Rig>`.

## A dataset {#a-dataset}

1. **Config.** Copy `configs/data/truebones.yaml` to
   `configs/data/<dataset>.yaml`. It reads top to bottom as the life of one clip:
   `paths`, `source`, `prepare`, `reduce`, `labels`, `corpus`, `filter`, `loader`.
2. **Source.** Name a registered source (`fbx_cache`, `bvh_files`), or add one in
   `src/poseydon/build/sources.py`.
3. **Rigs.** One manifest per character, as above, or generated from the
   dataset's own annotations as `tools/write_unimate_manifests.py` does.
4. **Build**, passing `data=<dataset>` to both build scripts, before any
   `--rigs`.

Done when `docker compose run --rm -T test python -m pytest -q --dataset <dataset>`
passes. It compares against the raw source that `--dataset` selects
(see [Testing](testing.md)).

## A data step {#a-data-step}

A stage of preparation must be invertible, or the round trip can't undo it.

1. **Class.** Subclass `PrepareStage` (`src/poseydon/build/prepare.py`) or
   `ReduceRule` (`src/poseydon/features/reduce.py`) with a `name`, or register a
   clip filter in `src/poseydon/build/clips.py`.
2. **Config.** Add `{type: <name>, ...}` to `prepare:`, `reduce:` or `clips:` in
   the dataset config. Order is the list's order.
3. **Rebuild** the corpus.

Done when the round trip passes on the rebuilt corpus: it is what proves the
step can be undone.

## A feature block {#a-feature-block}

1. **Class.** Subclass `Feature` (`src/poseydon/features/base.py`), set `name` and
   `width`, and implement `__call__(ctx)` under `@FEATURES.register("<name>")`,
   as `src/poseydon/features/kinematics.py` does.
2. **Config.** Add it to `blocks:` in `configs/features/default.yaml`, or
   override `features.blocks=[...]` on the command line.
3. **Statistics.** Rerun `scripts/build_features.py --stats-only`. A change of
   rotation layout also changes the skeleton and needs the full run.

Done when the round trip passes and `poseydon train trainer=debug` runs with the
block.

## A model {#a-model}

1. **Class.** Subclass `Denoiser` (`src/poseydon/models/base.py`), or
   `LatentDenoiser` for a model with an encoder, under
   `@MODELS.register("<name>")`.
2. **Config.** Add `configs/model/<name>.yaml` with `type: <name>` and its
   arguments.

Done when `poseydon train model=<name> trainer=debug` runs.

## A process {#a-process}

1. **Class.** Subclass `Process` (`src/poseydon/process/base.py`) under
   `@PROCESSES.register("<name>")`.
2. **Sampler.** A process needs a sampler that can run it: reuse one, or add one
   under `@SAMPLERS.register` in `src/poseydon/sampling/`.
3. **Config.** Add `configs/process/<name>.yaml`.

Done when `poseydon train process=<name> trainer=debug` runs. No test covers
`process/` or the samplers, so inspect generated motion too.

## A loss {#a-loss}

1. **Class.** Subclass `LossTerm` (`src/poseydon/losses/base.py`) under
   `@LOSSES.register("<name>")`. Ask for blocks by name through the spec, never
   by slice.
2. **Config.** Add it to the losses of `configs/train.yaml`.

Done when `poseydon train trainer=debug` logs the new term.

## An operation {#an-operation}

1. **Class.** Add a module to `src/poseydon/ops/` under
   `@OPERATIONS.register("<name>")`, as `src/poseydon/ops/generate.py` does.
   A step-level control goes in `src/poseydon/sampling/controls/`.
2. **Guide.** Add a page under `docs/how-to/tasks/` and its nav line.

Done when `poseydon sample` runs it from `operation:` in `configs/sample.yaml`.
An operation that takes tensors runs only from Python.

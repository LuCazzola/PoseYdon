# Recipes

How to add each kind of thing this repo grows by. One recipe each; if a step is wrong or
missing, fix it here.

## A component in an existing slot

This covers features, losses, models, processes, samplers, controls, operations,
conditioners, augmentations, windows, IK terms and solvers, clip ops, sources, prepare
stages and reduce rules.

1. Write one class against the slot's base, in the package that owns the slot (see the
   [Map](map.md)).
2. Register it under a short name:
    - **Prepare stages and reduce rules** register themselves when a subclass declares
      `name` (`PrepareStage.__init_subclass__` in `src/poseydon/build/prepare.py`,
      `ReduceRule` in `src/poseydon/features/reduce.py`). A prepare stage also declares
      its `scope` (rig or clip), its `after` ordering and an `invert`.
    - **Everything else** uses the registry's decorator, for example
      `@LOSSES.register("my_loss")`.
3. Make sure the module gets imported. Inside the library, import it from the
   package's `__init__.py`, as `src/poseydon/losses/__init__.py` does for its terms.
   Outside the library, list the module under the config's `custom_imports:`.
4. Name it in config: `type: <name>` in slots that take arguments, the bare name in
   `features:` and `conditioners:`, and `<name>: <weight>` under `losses:` (built with no
   arguments, `src/poseydon/training/build.py`). `poseydon list` shows most registries;
   it does not list sources, clip ops, reconstructors, IK terms, solvers or schedules.
5. A loss also declares the feature blocks it needs; a spec without them fails at the
   first training step (`LossTerm.validate` checks the names, not the space).

## A rig in an existing dataset

1. Put the vendor files under `<paths.raw>/source/<Rig>/`, where `paths.raw` is set in the
   dataset config (all three shipped configs use `data/_raw/truebones`). They are
   licensed: never commit them.
2. Write `data/<dataset>/rigs/<Rig>/manifest.yaml`. Name joints, never index them, and
   copy a similar rig's manifest for the keys (facing pairs, `foot_joints`, `rest_pose`,
   `up_axis`, `promote_root`, `authority`, `artifacts`). UniMate manifests are generated
   by `tools/write_unimate_manifests.py`, not written by hand.
3. For an FBX source (`fbx_cache`), add the rig's clips to the dataset's clip CSV, then
   run `scripts/build_fbx_cache.py --data <dataset> --species <Rig>` and
   `tools/build_rest_fbx.py --data <dataset> <Rig>` in the `fbxsdk` container. The first
   defaults to `--data unimate`, so pass the dataset.
4. Run stage 1, then stage 2, for that rig:
   `docker compose run --rm test python scripts/process_dataset_truebones.py data=<dataset> --rigs <Rig>`,
   then the same with `scripts/build_features.py`. Put the override before `--rigs`:
   `--rigs` takes every value after it, so a trailing `data=` is read as a rig name
   and the default dataset runs.
5. Run the round trip on it:
   `docker compose run --rm -T test python -m pytest -q tests/test_roundtrip.py --dataset <dataset> -k <Rig>`
   (BVH), and in `fbxsdk` for FBX.

## A dataset

1. Add `configs/data/<name>.yaml`. Read `configs/data/truebones.yaml` top to bottom; a
   variant can start from `defaults: [truebones, _self_]`, as
   `configs/data/truebones_bone.yaml` does.
2. Give it `paths`, a `source:` (`bvh_files` or `fbx_cache`, or a new source per the
   first recipe), a `prepare:` chain, a `reduce:` list, and the `labels`, `corpus`,
   `filter` and `loader` blocks (`configs/data/unimate.yaml` copies them from Truebones
   so the config composes). Optional `clips:` filters trim or drop clips in stage 1.
3. Manifests go to `<paths.rigs>`, then follow the rig recipe for every rig. The test
   suite takes `--dataset <name>` and reads `data/<name>/index.jsonl`, so keep
   `paths.root` at `data/<name>`.

## A library package or module

1. Add the code under `src/poseydon/`.
2. Add or extend its API page:
   `docs/parallax/pages/agent-guide/api/poseydon/<pkg>/index.md`, with a `::: poseydon.<pkg>`
   line plus one `:::` line per module, listed in the nav of `docs/parallax/mkdocs.yml`.
3. Add a row to the [Map](map.md). `docs/parallax/tools/check_docs.py` fails on a missing
   stub.

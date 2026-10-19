# Tooling

Everything here uses the library and owns no contract. Deleting a probe in
`tools/` breaks nothing; changing a script's CLI breaks only the person running
it. A contract that a script seems to define (a file layout, a parameter
meaning) actually belongs to a library package, and that package's page is
where it is documented.

The library, tests and scripts never run on host Python; every such command
goes through a compose service. The docs site has a service too (`docs`), but
needs nothing from `poseydon`, so it can also run straight from the host with
`uvx` (see [Containers](#containers)).

## Entry points

There are two kinds, and the split is deliberate:

- **`poseydon` CLI** (`src/poseydon/cli.py`): the user-facing verbs on a corpus
  that already exists. `ingest` aligns a BVH corpus and builds its index (legacy:
  nothing reads its output, so build corpora with the scripts below);
  `train` trains from a composed Hydra config; `sample` generates motion and
  writes each sample as `.anim.npz` (the animation exactly as generated, leaf rotations included: what `export` should be given) and `.bvh` (with `--render` for MP4); `retarget` carries one clip onto another
  rig; `export` writes the prepared or source representation, as BVH or FBX;
  `list` shows the components registered by name. It lives in the library
  because it is installed as the `poseydon` console script.
- **`scripts/`**: corpus construction and verification, one-shot and
  dataset-specific, which is why they are scripts and not CLI verbs. Those
  that write FBX run in the `fbxsdk` container.

Run the CLI inside the test container:

```bash
docker compose run --rm test python -m poseydon.cli list
```

## Scripts

**Corpus construction** (run in this order):

Both read `configs/build.yaml`, which composes a dataset (`configs/data/`)
and the feature group its statistics are measured for (`configs/features/`);
pick another dataset with a Hydra override (`data=truebones_bone`), and one
rig with `--rigs <Rig>`.

- `scripts/process_dataset_truebones.py` (stage 1, `test` container). Reads
  each rig's raw clips through the dataset's `source:` (`type: bvh_files`, the
  BVHs under `paths.raw/source/<Rig>/`; or `type: fbx_cache`, the cache of
  `scripts/build_fbx_cache.py`; `src/poseydon/build/sources.py`), runs them through the
  dataset's `prepare:` chain (a stage that decides from the whole rig, `prune_unskinned`,
  is given every clip of the rig and its skin weights first), and writes `clips/<Rig>/<action>.bvh` plus
  `rigs/<Rig>/preproc.npz`, which holds the fitted transform, its inverse and
  the chain itself. The dataset's optional `clips:` group (registered ops,
  `src/poseydon/build/clips.py`) trims and filters in UniMate's order: `head_trim`
  and `halve_fps` on the source clip before `prepare:`, then `trim_static`,
  `min_frames`, `max_length` (`mode: first` or `windows`, which names clips
  `<action>_w000`...), `low_activity` and `jump` on the prepared clip; `joint_range`
  skips a rig by its prepared joint count. A skip-list file that does not exist is a
  no-op. Each written clip records the source frames it came from as
  `clip/<name>/frames/window` (`start, stop, step`) in `preproc.npz`; a dropped clip
  or rig writes nothing and is named with its reason in `rigs/<Rig>/report.json`
  and `<paths.root>/dropped.json`. Without `clips:` (`truebones`) nothing changes. A chain with `prune_unskinned` writes its BVHs with `leaf_rotations` (a joint left childless keeps its rotation, as a JOINT above a zero-offset `End Site #synthetic` that `BVH.read` drops); `truebones` files are unchanged. A rig grounded on the mesh without a current
  `rest_mesh.npz` (run `tools/build_rest_mesh.py` first) is grounded on its
  lowest joint, with a warning. It then checks each rig's written rest
  clip and warns, without failing, if frame 0 or the rest skeleton does not
  face +Z (after the manifest's `extra_yaw_deg`), if a non-root joint is not
  identity at the rest frame (skipped for a chain without `rest_relative`, i.e.
  `data=unimate`), or if the rest skeleton is not upright (a wrong `up_axis:` or a
  facing pair that is not left/right).
  `--check-against DIR` (`data=unimate`, DIR = `data/_raw/UniML3D/export/truebones`)
  compares the build with UniMate's published sidecars and prints four lines: joint
  sets (the prepared rest skeleton's names against `joint_names.json`, as sets),
  export frames (the cached source clip's frame count, before any filter, against
  `clip_frames.json`), training clips and frames against 1023 / 95491, and rigs whose
  rest tree diameter is 2.0 within 1e-9. Mismatches, each with the stage or filter
  responsible and the dropped clips by filter (the skip list included, so the account adds up to 1097), go to `<paths.root>/replication.json`.
- `scripts/build_features.py` (stage 2, `test` container). Runs the four build
  passes (clips, skeleton, stats, index) with the dataset's `reduce:` and
  `labels:` and the feature group's blocks and normalization. A rig without
  `rest_pose:` (`data=unimate`) has stage 1's one-frame `bind_pose` clip as its rest.

The bone-relative ablation corpus is `data=truebones_bone`: the same chain
with `bone_relative` appended, built into its own root `data/truebones_bone/`
(gitignored, its manifests, clip labels and mesh caches copied from the main
corpus, the checks undoing `bone_relative` first):

```bash
mkdir -p data/truebones_bone && (cd data/truebones && find rigs clips cache \( -name '*.yaml' -o -name rest_mesh.npz \) | cpio -pdmu ../truebones_bone)
ln -sfn truebones data/_raw/truebones_bone   # the test reads data/_raw/<dataset>
docker compose run --rm -T test python scripts/process_dataset_truebones.py data=truebones_bone
docker compose run --rm -T test python scripts/build_features.py data=truebones_bone
```

Train or sample on it with `data=truebones_bone`; check it with
`pytest --dataset truebones_bone`.

**UniMate inputs**:

- `scripts/fetch_unimate_annotations.sh` (host, bash): fetches `clips.csv` and
  the three FBX repair scripts under `scripts/probes/` from the Hugging Face
  dataset `Linzhan/Truebones-ZOO-Annotations`, pinned to one commit in the
  script, into `data/_raw/UniML3D/truebones_annotations/` (gitignored). It also
  links `data/_raw/unimate` to `truebones`. Idempotent: present files are kept.

- `scripts/build_fbx_cache.py` (`fbxsdk` service, about 1 to 10 s a clip here):
  reads the `linked` clips of `clips.csv` once with the FBX SDK and caches, per
  clip, `data/unimate/cache/fbx/<Species>/<Action>.npz`: node names, parents,
  attribute types, the world matrix of every node at every integer frame of the
  key range, bind, fps, up axis and a per-node skin weight summary, so
  everything after runs in `test`. Options: `--species`, `--clips`
  (`Alligator-Big_Mouth`), `--print` (summary, writes nothing), `--force`;
  resumable by source size and mtime. KingCobra is read from repaired copies
  under `data/unimate/cache/repaired/` (`A01` colourway kept; `walk.fbx` gets
  `run.fbx`'s mesh, with the Geometry/Pose declarations the graft probe omits
  added, otherwise the SDK drops the mesh).

There is no FBX preparation pass: preparation runs on the BVH, and FBX enters
only at the two ends, through `poseydon export --format fbx`.

**Export and viewing**:

- `poseydon export <clip> --rig <Rig> --representation source --format fbx [--layout rot6d_joint|rot6d_bone]`
  (`fbxsdk` container): a generated clip onto the user's own FBX asset, with
  its mesh and skin. `--format bvh` runs in `test`.
- `scripts/render_fbx.py` (`fbx` container): any FBX → MP4 with EEVEE. Blender
  is used here as a renderer only; nothing reads or writes FBX through it. The
  camera comes from `poseydon.io.framing`, the same function the stick-figure
  renderer uses, so the two renders of one clip are directly comparable.

**Verification**:

- `scripts/build_stage_samples.py` (`fbxsdk` container): one clip per rig under
  `tmp/truebones/<Rig>/` at every stage, `1_source`, `2_prepared`, `3_model`,
  `4_roundtrip`, each as `.bvh` and `.fbx`, for side-by-side inspection. Every
  FBX is the vendor's skinned asset; the prepared and model ones sit in
  canonical space.

## Tools and probes

`tools/` holds one-off investigations, kept because the question they answered
is part of the design record. Most print a table or a verdict and write
nothing. Files whose docstring says *throwaway* exist to be deleted.

| subject | files |
|---|---|
| UniMate manifests | `tools/write_unimate_manifests.py`: `data/unimate/rigs/<Species>/manifest.yaml` from UniMate's `face_joint_names.json` (raw joint names; Anaconda's body-axis pair gets `extra_yaw_deg: 90`, derivation in its docstring) and the cached clips' `up_axis`; a species not cached yet is reported and skipped, a rerun fills it in. Run in `test` |
| root and hierarchy shape | `tools/probe_floating_root.py`, `tools/probe_promote_table.py`, `tools/probe_root_chain.py`, `tools/probe_root_shape.py`, `tools/probe_tukan.py` |
| rest pose and joint sets | `tools/probe_jointset_spread.py`, `tools/probe_tpose_gap.py`, `tools/probe_rest_override.py`, `tools/write_rest_pose.py`, `tools/probe_resolve_after_reduce.py`, `tools/tpose_author.py` (authored T-poses: `build` in the `fbxsdk` service, then `pose` / `ref` / `paw` / `flips` / `save` in `train`; workflow and lessons in its docstring) |
| inverse kinematics | `tools/ik_grid_search.py`, `tools/ik_contact_compare.py` |
| parity with the reference | `tools/probe_reference_retarget.py`, `tools/probe_wandb_parity.py`, `tools/render.py` |
| rest-pose mesh | `tools/build_rest_mesh.py`: each rig's vendor mesh skinned on its rest pose, cached as `cache/rigs/<Rig>/rest_mesh.npz` for the `ground` stage's `mesh` mode (`fbxsdk` service, about 10 s a rig); rerun for a rig whose `rest_pose:` changes, before stage 1 |
| joint-name embeddings | `tools/build_joint_name_cache.py`: the `joint_names` cache (`cache/joint_names_t5.npz`), from every rig's reduced joint names through `poseydon.conditioners.text` (`compat` image, for the `text` extra); rerun after a reduction change adds names |

`tools/reference/` holds scripts that run the **reference implementation** (the
published code PoseYdon is measured against) inside the `compat` container. They
write raw arrays that PoseYdon's output is compared against: feature parity,
DDPM decomposition, IK and transfer comparisons, and manifest generation.

## Configs

Every slot that builds an object names a registered class by `type:`, with
the rest of the entry as its arguments (`poseydon.core.registry.Registry.build`,
the mmengine pattern); `poseydon list [kind]` prints what each registry holds.
A class of one's own registers itself (a `PrepareStage` or `ReduceRule`
subclass by declaring `name`, anything else with `@<REGISTRY>.register("name")`)
and is named after its module is listed in `custom_imports:`.

| config | holds |
|---|---|
| `configs/data/<dataset>.yaml` | one dataset, in the order a clip lives it: `paths`, `source`, `prepare:` (stage-1 stages), `reduce:` (stage-2 joint rules), `labels:` (split, joint names, foot-contact thresholds), `corpus`, `filter` (blacklist) and `loader` (batch, balancing, window). `truebones_bone` is `truebones` with its own paths and `bone_relative` appended; `unimate` (UniMate's recipe, `fbx_cache` source, manifests from `tools/write_unimate_manifests.py`) has UniMate's `clips:` and `prepare:` (`up_axis`, `prune_unskinned`, `bfs_order`, `enforce_rigid`, `face_axis` with `on_degenerate: nearest`, `centre_xz_clip`, `scale` by diameter 2.0, `ground_clip`) and `reduce: []` |
| `configs/features/` | the model's feature blocks and their normalization (model-side) |
| `configs/build.yaml` | corpus construction: a dataset plus a feature group, for both build scripts |
| `configs/train.yaml`, `configs/sample.yaml` | a run: dataset, features, `configs/model/` (`anytop`, `modiffae`), `configs/process/` (`gaussian`, `flow`), `configs/trainer/` (`default`, `debug`), conditioners, augmentations, losses |

Per-rig facts stay in each rig's `manifest.yaml`: joints, facing pairs, feet,
tags, the rest-pose clip, the vendor files, and the two per-rig settings
`promote_root:` and `ground:`. Training and sampling compose the same
`features` group, because the recipe check compares it between them. The
comments in `configs/train.yaml` record why each default is what it is,
including the measured guidance scales. Read them before changing a default.

## Tests

Two files:

- `tests/test_roundtrip.py`: the data round trip on every prepared clip of
  one dataset, picked with `--dataset <name>` (default `truebones`) and
  narrowed with `-k <Rig>` or `-k <Rig>/<action>`. Its BVH half runs anywhere;
  its FBX half (`test_round_trip_fbx`) needs the `fbxsdk` container and skips
  elsewhere.
- `tests/test_docs_index.py` keeps `docs/index/` structurally in step with
  the tree.

```bash
docker compose run --rm -T test python -m pytest -q                 # BVH, ~2 min
docker compose run --rm -T fbxsdk python -m pytest -q -k Camel      # BVH + FBX, one rig
```

Under QEMU the FBX half costs about 11 s a clip, a few hours for the whole
corpus. What is covered and what is not:
[testing guide](../guides/how-to/testing.md).

## Data

The prepared layout (`data/truebones/rigs/<Rig>/`, `data/truebones/clips/<Rig>/`)
is described file by file in the [layout guide](../guides/concepts/data/layout.md).
Raw licensed vendor files live under `data/_raw/` and never mix into the
prepared layout. Everything derived (`*.npz`, `*.bvh`, `*.fbx`, `index.jsonl`)
is gitignored by extension, because authored and derived files sit side by side
in the same directories.

## Containers

| service | for |
|---|---|
| `test` | the pure suite, the CLI, and the non-Blender scripts. CPU-only torch |
| `train` | training on the GPU, with wandb settings forwarded from `.env` |
| `compat` | running the reference implementation for parity, pinned CPU torch |
| `fbxsdk` | FBX read/write through the Autodesk FBX SDK, and the FBX half of the tests. x86-64 under QEMU on the GB10 |
| `fbx` | headless Blender, for `scripts/render_fbx.py` only |
| `docs` | the docs site at http://127.0.0.1:8019, live-reloading; `restart: unless-stopped`, so after one `docker compose up -d docs` it comes back with the Docker daemon. Python and uv only: mkdocs is fetched by `uvx` and cached in the `docs-cache` volume |

All six bind-mount the repo and run as the host user, so source edits need no
rebuild. Only a `pyproject.toml` or `uv.lock` change needs one.

## Dependencies

Docker provides the OS and system packages (Blender, ffmpeg, a compiler).
Every Python dependency is declared in `pyproject.toml` and pinned in
`uv.lock`, and each image installs it with one `uv sync --locked`, selecting
extras:

| extra | for | images |
|---|---|---|
| `dev` | pytest, ruff, rendering | `test`, `train`, `compat` |
| `cpu` / `cu130` | which torch build; mutually exclusive | `cpu`: `test`, `compat`; `cu130`: `train` |
| `train` | wandb | `train` |
| `text` | transformers, sentencepiece: building the joint-name cache | pulled in by `compat` |
| `compat` | the reference implementation and what its scripts import | `compat` |
| `render` | matplotlib, imageio | pulled in by `dev` and `compat` |

The exception is `fbx`. Blender runs on Debian's own Python, so its numpy,
scipy and PyYAML are apt packages built against that interpreter. Only
`scripts/render_fbx.py` runs there.

**Autodesk FBX SDK** (`fbxsdk`). FBX is read and written through Autodesk's
SDK, because Blender's exporter writes wrong rotations near gimbal lock
(4.76 degrees on Flamingo's root, measured against its own input). The SDK is
proprietary: free to use, but never committed or redistributed, so each user
fetches it and accepts Autodesk's license
(https://aps.autodesk.com/developer/overview/fbx-sdk) themselves. Its Python
wheel is x86-64 and CPython 3.11 only, so on the GB10 the image runs under
QEMU, registered once per boot:

```bash
scripts/fetch_fbx_sdk.sh --accept-license                 # -> third_party/fbxsdk/ (gitignored)
docker run --privileged --rm tonistiigi/binfmt --install amd64,386
docker compose build fbxsdk
```

`fbxsdk` installs the same locked set as `test`, then the SDK's wheel, which
cannot be in `uv.lock`.

To add or upgrade a dependency, run on the host, then rebuild the affected
images:

```bash
uv add <pkg>                      # core dependency
uv add <pkg> --optional train     # into an extra
uv lock --upgrade-package <pkg>   # move one pin
docker compose build test train compat
```

Commit `uv.lock` with the `pyproject.toml` change. `uv sync --locked` refuses
to build an image from a lock that disagrees with `pyproject.toml`.

The docs site runs as the `docs` service (`docker compose up -d docs`), or
without a container: `uvx --from 'mkdocs<2' --with mkdocs-material --with 'mkdocstrings[python]' mkdocs serve`.

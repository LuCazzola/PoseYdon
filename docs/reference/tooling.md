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
  `scripts/build_fbx_cache.py`, which `truebones` and `unimate` both use;
  `src/poseydon/build/sources.py`), runs them through the
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
  and `<paths.root>/dropped.json`. A clip an earlier run wrote and this run does not
  (dropped, skipped, windows of another mode, an action no longer listed) loses its
  `.bvh` and stage 2's `.npz`/`.yaml`. Without `clips:` (`truebones`) the clips are unchanged; `preproc.npz` still gains `clip/<name>/frames/window`, harmlessly. Every BVH is written with `leaf_rotations` (a leaf joint keeps its rotation, as a JOINT above a zero-offset `End Site #synthetic` that `BVH.read` drops): a rig read from FBX has no End Sites, so its wing tips, toes and ear tips are leaves, as are the joints `prune_unskinned` leaves childless. Until 2026-10-21 only a pruned chain did, which on the FBX-read truebones corpus lost every leaf's rotation. A rig grounded on the mesh without a current
  `rest_mesh.npz` (run `tools/build_rest_fbx.py` first) is grounded on its
  lowest joint, with a warning. It then checks each rig's written rest
  clip and warns, without failing, if frame 0 or the rest skeleton does not
  face +Z (after the manifest's `extra_yaw_deg`), if a non-root joint is not
  identity at the rest frame (skipped for a chain without `rest_relative`, i.e.
  `data=unimate`), or if the posed rest frame is not upright (a wrong `up_axis:` or a
  facing pair that is not left/right).
  `--check-against DIR` (`data=unimate`, DIR = `data/_raw/UniML3D/export/truebones`)
  compares the build with UniMate's published sidecars and prints four lines: joint
  sets (the prepared rest skeleton's names against `joint_names.json`, as sets),
  export frames (the cached source clip's frame count, before any filter, against
  `clip_frames.json`), training clips and frames against 1023 / 95491, and rigs whose
  rest tree diameter is 2.0 within 1e-9. Known gap: UniMate's joint order after BFS (spec acceptance 1) is not checked, because the sidecar lacks its hierarchy. Mismatches, each with the stage or filter
  responsible and the dropped clips by filter (the skip list included, so the account adds up to 1097), go to `<paths.root>/replication.json`.
  Export frames differ on 585 of 1097 clips, by design: UniMate's published export
  (`clip_frames.json`, `summary.json`'s 111743 frames) was counted with a plain `ceil`
  of Blender's float32 end time, without `FRAME_RANGE_EPS`, which adds a frame where the
  last key lies just past an integer (`119.0000081`). The build follows the eps rule.
  The extra frame repeats the pose just before it (sampled past the last key), so
  `trim_static` (`clips:`, before any length-dependent op) strips it and the training
  split is the same either way; the training totals (1023 / 95491) match.
- `scripts/build_features.py` (stage 2, `test` container). Runs the four build
  passes (clips, skeleton, stats, index) with the dataset's `reduce:` and
  `labels:` and the feature group's blocks and normalization. A rig without
  `rest_pose:` (`data=unimate`) has stage 1's one-frame `bind_pose` clip as its rest.

The bone-relative ablation corpus is `data=truebones_bone`: the same chain
with `bone_relative` appended, built into its own root `data/truebones_bone/`
(gitignored, its manifests, clip labels and mesh caches copied from the main
corpus, the checks undoing `bone_relative` first):

```bash
mkdir -p data/truebones_bone && (cd data/truebones && find rigs clips cache/rigs \( -name '*.yaml' -o -name 'rest*' \) | cpio -pdmu ../truebones_bone)
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

- `scripts/build_fbx_cache.py --data truebones|unimate` (`fbxsdk` service,
  about 1 to 10 s a clip here): reads the clips of the dataset's clip index
  (`source.index`: `data/truebones/fbx_clips.csv`, or UniMate's `clips.csv`,
  its `linked` rows) once with the FBX SDK and caches, per clip,
  `<source.cache>/<Rig>/<Action>.npz`: node names, parents,
  attribute types, the world matrix of every node at every integer frame of the
  key range, bind, fps, up axis and a per-node skin weight summary, so
  everything after runs in `test`. Options: `--species`, `--clips`
  (`Alligator-Big_Mouth`), `--print` (summary, writes nothing), `--force`;
  resumable by source size and mtime. KingCobra is read from repaired copies
  under `<paths.root>/cache/repaired/` (`A01` colourway kept; `walk.fbx` gets
  `run.fbx`'s mesh, with the Geometry/Pose declarations the graft probe omits
  added, otherwise the SDK drops the mesh).

- `tools/build_rest_fbx.py --data truebones|unimate [Rig ...]`
  (`fbxsdk` service, about 30 s a rig): each rig's rest asset under
  `<paths.root>/cache/rigs/<Rig>/`, before stage 1: `rest.fbx` (the asset FBX
  export writes onto), `rest.npz` (its read, the rest pose `fbx_cache` takes with
  `rest_asset: true`, and `source_fbx`, the vendor file the asset was built from or its repaired copy) and, for truebones, `rest_mesh.npz` (the `ground` stage's
  `mesh` mode). Truebones: the vendor FBX, its mesh, skin and bind unchanged,
  held on one frame (`FBX.pose`) in the manifest's `rest_pose:` BVH, carried onto the FBX nodes through a vendor
  BVH/FBX pair of one clip, checked on every frame; rerun a rig whose
  `rest_pose:` changes. Unimate: the source FBX of the rig's rest clip,
  unchanged.

`fbx_cache` reads FBX at stage 1, through the cache, and export writes onto the
rig's rest asset by node name. A `bvh_files` dataset has no FBX export: its
motion and the vendor mesh are in two files, and nothing maps one onto the other.

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

- `scripts/verify_motion.py` (`fbxsdk` container): judges the FBX files the
  pipeline returns against the rig's vendor file (the `source_fbx` of its
  `rest.npz`; a rig without it must be rebuilt with `tools/build_rest_fbx.py`).
  `build --data D --rig R [--clip C] [--smoke] [--all-frames] [--floor] [--out DIR]` writes
  `<out>/<Rig>__<clip>/` (the `build_stage_samples.py` files plus `case.json`: paths,
  sampled frames, the vendor file's frame for each (through the clip's recorded window), SHA-256 of each file, git commit, container); `file FILE --reference REF`
  judges one FBX; `judge DIR` runs the checks and writes `report.json` and
  `report.txt` beside the stage files (default `--out` is `tmp/verify/`). `judge`
  refuses a file whose hash changed since `build`. Exit status is non-zero unless every check
  is `PASS` or `ALLOWED`. `--floor` replaces the stage files by the source clip alone written onto the rest asset
  (no preparation), to measure the float floor. Checks so far: C1 (inventory: nodes, hierarchy, meshes,
  clusters, binds, materials, textures, units, up axis, fps, frame range) and C2a (each stage file's own
  skin against the reference skin, both driven by that file's joints; `skin_tol`, floor measured beside
  it; takes no allowances) and C2b (the returned `4_roundtrip` file's joints and skin against the source
  clip with each lossy stage's loss law applied, `tools/verify/laws.py`: `C2b-untraced`, file against that
  expected pose, is the verdict, with vertices, joint positions and joint rotations reported apart (vertices and
  positions within `untraced_tol` of rig extent, rotations within `untraced_rot_tol`); `C2b-traced`, source against it,
  is reported and never judged; `C2b-rest` holds the rest inputs the laws read from the rest asset, bone lengths
  and node scales, to the vendor file's bind, and runs even when a law is missing; a stage with neither a law nor a lossless
  declaration makes `C2b-untraced` `NOT RUN`) and C3 (the `2_prepared` and `3_model` files, written in canonical space, against the
  source clip with the same laws, the model stage adding the reduction's: ONE similarity is fitted for the whole clip from the
  expected skin vertices to the file's; `C3-shape` is the residual, judged on vertices, joints and rotations with the C2b-untraced
  tolerances; `C3-placement` (`pipeline` evidence) compares the fit with the `space` recomputed from the record by `RigRecord.placement` (it checks that a canonical file sits where the record says; a wrong record moves `4_roundtrip` off the source, which C2b catches): scale ratio
  `placement_scale_tol`, whole rotation in degrees `placement_rotation_deg`, offset as a fraction of extent `placement_offset_tol`)
  and C4 (the `5_zero` file `build` also writes: identity rotations on `skeleton.npz`'s offsets, the root where the rig's `rest_action`
  clip stands at frame 0 (`build` raises if that clip is missing), placed by `RigRecord.placement`; one pose. C1 and C2a judge it too,
  C1 without clip timing. `C4-mesh` (`pipeline` evidence: its pose reference is `rest.npz:rest`, built by `tools/build_rest_fbx.py` from
  the manifest's rest pose): the file's skin under its joints against the same skin under that rest pose, ONE fitted similarity, judged like
  `C3-shape` (the vendor skin enters through the fit; C2a holds the file's skin to it); a chain without `rest_relative`, i.e. `data=unimate`,
  is `NOT RUN`. `C4-placement` (`pipeline`) compares that fit with the placement recorded for the zero, with C3's `placement_*` bounds.
  `C4-joints` is the independent leg: the zero file's joint positions against the manifest's `rest_pose` BVH read from the raw files,
  after one fitted similarity, within `untraced_tol` of extent (evidence `derived` for an authored T-pose, `independent` for a vendor
  one; `NOT RUN` where the manifest has none). `C4-pose`, `pipeline` evidence only: up +Y (the manifest's `up_axis`, carried by
  `C4-mesh`'s fitted rotation) and facing +Z (the manifest's facing pairs, as `FaceRestAxis` defines it) within `zero_facing_deg`, grounded
  per the rig's ground mode within `zero_ground_tol` of extent; a part the chain does not declare is named and left out. `C4-symmetry` is
  not implemented: it needs a limit format the owner sets in `limits.yaml` (empty today; a rig that gains an entry is still `NOT RUN`, with that
  note), so every run's verdict reads `NOT RUN`).
  A `--floor` run is judged against the source itself, but for the writer's node scale and held joints.
  `file` on a bare FBX has no stages or records, so its C2b is always `NOT RUN`.
  C1 compares a clip's frame count with its recorded window over the vendor keys. Settings (`inventory_tol` among them) and the
  smoke set are in `configs/verify/`; `allowances.yaml` and `limits.yaml` are the owner's.

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
| `configs/verify/` | the motion verifier: `verify.yaml` (frames sampled, tolerances, smoke set), `allowances.yaml` and `limits.yaml` (shipped empty; only the owner edits them) |
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
- `tests/test_docs_index.py` keeps the [Map](map.md) and the API pages
  structurally in step with the tree.

```bash
docker compose run --rm -T test python -m pytest -q                 # BVH, ~2 min
docker compose run --rm -T fbxsdk python -m pytest -q -k Camel      # BVH + FBX, one rig
```

Under QEMU the FBX half costs about 11 s a clip, a few hours for the whole
corpus. What is covered and what is not:
[testing guide](../how-to/testing.md).

## Data

The prepared layout (`data/truebones/rigs/<Rig>/`, `data/truebones/clips/<Rig>/`)
is described file by file in the [data formats](data-formats.md) page.
Raw licensed vendor files live under `data/_raw/` and never mix into the
prepared layout. Everything derived (`*.npz`, `*.bvh`, `*.fbx`, `index.jsonl`)
is gitignored by extension, because authored and derived files sit side by side
in the same directories. `data/unimate/` is gitignored as a whole (its generated
manifests included): `cache/` (FBX and repaired clips), the prepared layout,
`dropped.json` and `replication.json`.

## Containers

| service | for |
|---|---|
| `test` | the pure suite, the CLI, and the non-Blender scripts. CPU-only torch |
| `train` | training on the GPU, with wandb settings forwarded from `.env` |
| `compat` | running the reference implementation for parity, pinned CPU torch |
| `fbxsdk` | FBX read/write through the Autodesk FBX SDK, and the FBX half of the tests. x86-64 only, so under QEMU on an ARM host |
| `fbx` | headless Blender, for `scripts/render_fbx.py` only |
| `docs` | the docs site on port 8019 of the host, live-reloading; `restart: unless-stopped`, so after one `docker compose up -d docs` it comes back with the Docker daemon. Python and uv only: mkdocs is fetched by `uvx` and cached in the `docs-cache` volume |

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
wheel is x86-64 and CPython 3.11 only, so on an ARM host the image runs under
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

## Docs site

The site is published at https://lucazzola.github.io/PoseYdon/.
`.github/workflows/docs.yml` runs the strict build on every PR and deploys to
GitHub Pages on a push to `main`. `.github/workflows/claude.yml` runs Claude
Code on `@claude` mentions in issues and PRs, and needs the repository secret
`CLAUDE_CODE_OAUTH_TOKEN`.

Locally it runs as the `docs` service (`docker compose up -d docs`), or
without a container: `uvx --from 'mkdocs<2' --with mkdocs-material --with 'mkdocstrings[python]' mkdocs serve`.

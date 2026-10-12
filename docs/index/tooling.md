# Tooling

Everything here uses the library and owns no contract. Deleting a probe in
`tools/` breaks nothing; changing a script's CLI breaks only the person running
it. A contract that a script seems to define (a file layout, a parameter
meaning) actually belongs to a library package, and that package's page is
where it is documented.

The library, tests and scripts never run on host Python; every such command
goes through a compose service. The one exception is the docs site (see
[Containers](#containers)).

## Entry points

There are two kinds, and the split is deliberate:

- **`poseydon` CLI** (`src/poseydon/cli.py`): the user-facing verbs on a corpus
  that already exists. `ingest` aligns a BVH corpus and builds its index (legacy:
  nothing reads its output, so build corpora with the scripts below);
  `train` trains from a composed Hydra config; `sample` generates motion and
  writes BVH (with `--render` for MP4); `retarget` carries one clip onto another
  rig; `export` writes the prepared or original representation, as BVH or FBX;
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

- `scripts/process_dataset_truebones.py` (stage 1, `test` container). Reads
  raw BVH clips under `--raw-root/<Rig>/`, runs each through
  `PrepareChain`, and writes `clips/<Rig>/<action>.bvh` plus
  `rigs/<Rig>/preproc.npz`, which holds the fitted transform and its inverse.
- `scripts/build_features.py` (stage 2, `test` container). Runs the four build
  passes (clips, skeleton, stats, index) from `configs/dataset/<name>.yaml`.

There is no FBX preparation pass: preparation runs on the BVH, and FBX enters
only at the two ends, through `poseydon export --format fbx`.

**Export and viewing**:

- `poseydon export <clip> --rig <Rig> --representation original --format fbx`
  (`fbxsdk` container): a generated clip onto the user's own FBX asset, with
  its mesh and skin. `--format bvh` runs in `test`.
- `scripts/render_fbx.py` (`fbx` container): any FBX → MP4 with EEVEE. Blender
  is used here as a renderer only; nothing reads or writes FBX through it. The
  camera comes from `poseydon.io.framing`, the same function the stick-figure
  renderer uses, so the two renders of one clip are directly comparable.

**Verification**:

- `scripts/build_stage_samples.py` (`fbxsdk` container): one clip per rig under
  `tmp/truebones/<Rig>/` at every stage, `1_source`, `2_prepared`, `3_model`,
  `4_roundtrip`, each as `.bvh` and `.fbx`, for side-by-side inspection. The
  prepared and model FBX are bare skeletons; source and roundtrip FBX are the
  vendor's skinned asset.

## Tools and probes

`tools/` holds one-off investigations, kept because the question they answered
is part of the design record. Most print a table or a verdict and write
nothing. Files whose docstring says *throwaway* exist to be deleted.

| subject | files |
|---|---|
| root and hierarchy shape | `tools/probe_floating_root.py`, `tools/probe_promote_table.py`, `tools/probe_root_chain.py`, `tools/probe_root_shape.py`, `tools/probe_tukan.py` |
| rest pose and joint sets | `tools/probe_jointset_spread.py`, `tools/probe_tpose_gap.py`, `tools/probe_rest_override.py`, `tools/write_rest_pose.py`, `tools/probe_resolve_after_reduce.py` |
| inverse kinematics | `tools/ik_grid_search.py`, `tools/ik_contact_compare.py` |
| parity with the reference | `tools/probe_reference_retarget.py`, `tools/probe_wandb_parity.py`, `tools/render.py` |
| joint-name embeddings | `tools/build_joint_name_cache.py` |

`tools/reference/` holds scripts that run the **reference implementation** (the
published code PoseYdon is measured against) inside the `compat` container. They
write raw arrays that PoseYdon's output is compared against: feature parity,
DDPM decomposition, IK and transfer comparisons, and manifest generation.

## Configs

Hydra composes `configs/train.yaml` and `configs/sample.yaml` from these groups:

| group | selects |
|---|---|
| `configs/data/` | where the prepared corpus and index live, batch size, window policy, rig balancing |
| `configs/dataset/` | the stage-2 build: feature schema, corpus discovery, joint-name humanizing, reduction tolerance, normalization policy |
| `configs/model/` | `anytop` or `modiffae` |
| `configs/process/` | `gaussian` (diffusion) or `flow` (flow matching) |
| `configs/trainer/` | `default` or `debug` |

Both top-level configs compose the same `dataset` group, because the recipe
check compares its values between training and sampling. The comments in
`configs/train.yaml` record why each default is what it is, including the
measured guidance scales. Read them before changing a default.

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
[testing guide](../guides/testing.md).

## Data

The prepared layout (`data/truebones/rigs/<Rig>/`, `data/truebones/clips/<Rig>/`)
is described file by file in the [layout guide](../guides/truebones-layout.md).
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

All five bind-mount the repo and run as the host user, so source edits need no
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

The docs site needs no container: `uvx --from 'mkdocs<2' --with mkdocs-material mkdocs serve`.

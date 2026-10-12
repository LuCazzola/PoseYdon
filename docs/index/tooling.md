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
  rig; `export` writes the prepared, presentation or original representation;
  `list` shows the components registered by name. It lives in the library
  because it is installed as the `poseydon` console script.
- **`scripts/`**: corpus construction and verification, one-shot and
  dataset-specific, which is why they are scripts and not CLI verbs. Several
  need Blender and so run only in the `fbx` container.

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
- `scripts/process_dataset_truebones_fbx.py` (`fbx` container). The FBX
  counterpart: same manifests, layout and facing convention, but it carries the
  mesh and skinning through and writes `mesh.npz` per rig.
- `scripts/build_features.py` (stage 2, `test` container). Runs the four build
  passes (clips, skeleton, stats, index) from `configs/dataset/<name>.yaml`.
- `scripts/build_truebones_assets.py` (`fbx` container). Builds the
  mesh-bearing `char-model.fbx` and `char-full.fbx` per rig.

**Export and viewing** (`fbx` container):

- `scripts/anim_2_fbx.py`: a generated `.npz` → FBX on the user's rig. It
  inverts the reduction, then the preparation chain, then writes an armature.
  `--skin` drives the rig's own source asset instead, so the output carries
  mesh, weights and materials.
- `scripts/render_fbx.py`: any FBX → MP4 with EEVEE. The camera comes from
  `poseydon.io.framing`, the same function the stick-figure renderer uses, so
  the two renders of one clip are directly comparable.

**Verification**:

- `scripts/check_fbx_coherence.py`: measures how far the BVH and FBX arms of
  the corpus have drifted, comparing in world space after normalizing scale and
  per-frame placement. It writes nothing.
- `scripts/audit_truebones_assets.py`: counts mesh-bearing assets and complete
  `report.json` files. It has no docstring and is a quick audit, not a pipeline
  step.
- `scripts/build_skeleton_visual_debug.py`, then
  `scripts/build_skeleton_visual_debug_fbx.py` (`fbx` container): one bundle per
  rig under `tmp/skel_visual_debug/<Rig>/` with source, prepared, model and
  roundtrip versions of the *same* clip, for side-by-side inspection in Blender.

## Tools and probes

`tools/` holds one-off investigations, kept because the question they answered
is part of the design record. Most print a table or a verdict and write
nothing. Files whose docstring says *throwaway* exist to be deleted.

| subject | files |
|---|---|
| root and hierarchy shape | `tools/probe_floating_root.py`, `tools/probe_promote_table.py`, `tools/probe_root_chain.py`, `tools/probe_root_shape.py`, `tools/probe_tukan.py` |
| rest pose and joint sets | `tools/probe_jointset_spread.py`, `tools/probe_tpose_gap.py`, `tools/probe_rest_override.py`, `tools/write_rest_pose.py`, `tools/probe_resolve_after_reduce.py` |
| inverse kinematics | `tools/ik_grid_search.py`, `tools/ik_contact_compare.py` |
| scale | `tools/probe_scale_residual.py` |
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

`tests/` mirrors `src/poseydon/` package by package. There are two suites:

- **pure**: `docker compose run --rm -T test python -m pytest -q`
- **Blender**: the FBX tests under `tests/io`, `tests/build` and
  `tests/export`, run inside the `fbx` container (see the
  [quick start](../guides/quickstart.md))

What is covered, golden parity, and why tests are mutation-checked:
[testing guide](../guides/testing.md). `tests/test_docs_index.py` keeps
`docs/index/` structurally in step with the tree.

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
| `fbx` | headless Blender: FBX read/write, Blender tests, rendering |

All four bind-mount the repo and run as the host user, so source edits need no
rebuild. Only a `pyproject.toml` change needs one.

The docs site needs no container: `uvx --from 'mkdocs<2' --with mkdocs-material mkdocs serve`.

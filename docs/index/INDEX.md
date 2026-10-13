# Index

Where things live. Read this first, then the page in the third column.

The repo has two sides, and they change under different rules:

- **Library**: `src/poseydon/`, the importable package. It owns every contract;
  a change here can break every caller.
- **Tooling**: everything built around the library (scripts, probes, configs,
  tests, data, containers). It uses the library and owns no contract.

See [overview](overview.md) for how the pipeline fits together.

## Library

| path | owns | read |
|---|---|---|
| `src/poseydon/augment/` | data augmentation applied to training windows | [augment](../api/augment/index.md) |
| `src/poseydon/build/` | turning a raw corpus into a prepared one, invertibly: preparation, skeleton, statistics, index, reports | [build](../api/build/index.md) |
| `src/poseydon/conditioners/` | declared conditioning: the only extra data a batch carries | [conditioners](../api/conditioners/index.md) |
| `src/poseydon/core/` | the contracts: animation containers, skeleton manifests, `FeatureSpec`, `MotionBatch`, kinematics, rotations, registry | [core](../api/core/index.md) |
| `src/poseydon/data/` | reading a prepared corpus into feature windows, normalization, rig-balanced sampling, collation | [data](../api/data/index.md) |
| `src/poseydon/export/` | `RigRecord`: a rig's recorded reduction, preparation and FBX calibrations; moves model output to the prepared character or the user's source rig, for `BVH`/`FBX.from_animation` to write | [export](../api/export/index.md) |
| `src/poseydon/features/` | composable motion feature representations, extraction and recovery | [features](../api/features/index.md) |
| `src/poseydon/ingest/` | BVH corpus to aligned animations plus a corpus index | [ingest](../api/ingest/index.md) |
| `src/poseydon/io/` | BVH and FBX reading and writing (FBX through the Autodesk FBX SDK), shot framing, stick-figure rendering | [io](../api/io/index.md) |
| `src/poseydon/losses/` | composable loss terms | [losses](../api/losses/index.md) |
| `src/poseydon/models/` | denoising models (AnyTop, MoDiffAE) | [models](../api/models/index.md) |
| `src/poseydon/ops/` | motion tasks run with a trained model: generate, in-between, blend, retarget | [ops](../api/ops/index.md) |
| `src/poseydon/process/` | generative processes: corruption, parameterization and their inverses | [process](../api/process/index.md) |
| `src/poseydon/sampling/` | samplers and step-level controls (guidance) | [sampling](../api/sampling/index.md) |
| `src/poseydon/solvers/` | inverse kinematics: fitting rotations to desired positions | [solvers](../api/solvers/index.md) |
| `src/poseydon/training/` | turning a config into a trained checkpoint: task, Lightning wrappers, recipe, retarget validation | [training](../api/training/index.md) |
| `src/poseydon/cli.py` | the `poseydon` command: `ingest`, `train`, `sample`, `retarget`, `export`, `list` | [tooling](tooling.md#entry-points) |

## Tooling

### Entry-point scripts

| path | does | read |
|---|---|---|
| `scripts/process_dataset_truebones.py` | raw Truebones BVH → canonical, invertible clips under the entity-first layout | [tooling](tooling.md#scripts) |
| `scripts/build_features.py` | prepared BVH → canonical features, per-rig statistics and the clip index | [tooling](tooling.md#scripts) |
| `scripts/render_fbx.py` | an FBX → an MP4 of the character performing it (Blender, as a renderer only) | [tooling](tooling.md#scripts) |
| `scripts/fetch_fbx_sdk.sh` | downloads the Autodesk FBX Python SDK into the gitignored `third_party/fbxsdk/`, after `--accept-license` | [tooling](tooling.md#dependencies) |
| `scripts/build_stage_samples.py` | one clip per rig at every round-trip stage (source, prepared, model, roundtrip), as BVH and FBX, under `tmp/truebones/` | [tooling](tooling.md#scripts) |

### Probes and one-off tools

Investigations kept as a record. Files marked *throwaway* in their docstring are
meant to be deleted once read.

| path | does | read |
|---|---|---|
| `tools/build_joint_name_cache.py` | caches T5 embeddings of every joint name in the corpus | [tooling](tooling.md#tools-and-probes) |
| `tools/ik_grid_search.py` | grid search over `GradientIK` terms, foot-contact term included | [tooling](tooling.md#tools-and-probes) |
| `tools/ik_contact_compare.py` | default vs best-from-grid IK solve, with predicted contacts shown | [tooling](tooling.md#tools-and-probes) |
| `tools/probe_floating_root.py` | which rigs have a root that is a ground locator rather than a body joint | [tooling](tooling.md#tools-and-probes) |
| `tools/probe_promote_table.py` | the promoted-root table: first branching joint per ground-locator rig | [tooling](tooling.md#tools-and-probes) |
| `tools/probe_root_chain.py` | walks the top of the hierarchy to the first branch, with cumulative offset | [tooling](tooling.md#tools-and-probes) |
| `tools/probe_root_shape.py` | per rig: what sits at the top of the hierarchy and how far the first real joint is | [tooling](tooling.md#tools-and-probes) |
| `tools/probe_jointset_spread.py` | per rig: whether clips agree with each other and the T-pose on their joint set | [tooling](tooling.md#tools-and-probes) |
| `tools/probe_tpose_gap.py` | which joints the gutted rigs' T-poses omit | [tooling](tooling.md#tools-and-probes) |
| `tools/probe_rest_override.py` | validates a proposed per-rig rest-pose override against the modal joint set | [tooling](tooling.md#tools-and-probes) |
| `tools/write_rest_pose.py` | emits a `rest_pose:` line per rig for review before applying | [tooling](tooling.md#tools-and-probes) |
| `tools/probe_resolve_after_reduce.py` | whether `resolve()` survives joint reduction, for every rig | [tooling](tooling.md#tools-and-probes) |
| `tools/probe_tukan.py` | inspects the Tukan T-pose: joint count, mean bone, branching | [tooling](tooling.md#tools-and-probes) |
| `tools/probe_reference_retarget.py` | *throwaway*: Flamingo → Scorpion retarget with the reference weights | [tooling](tooling.md#tools-and-probes) |
| `tools/probe_wandb_parity.py` | *throwaway*: pushes a parity retarget run to wandb | [tooling](tooling.md#tools-and-probes) |
| `tools/render.py` | re-export so reference-comparison scripts keep one import path | [tooling](tooling.md#tools-and-probes) |
| `tools/reference/` | scripts that run the reference implementation (`compat` image) to produce ground truth PoseYdon is compared against | [tooling](tooling.md#tools-and-probes) |

### Configs, tests, data, docs, containers

| path | holds | read |
|---|---|---|
| `configs/` | Hydra configs: `configs/train.yaml`, `configs/sample.yaml`, groups `configs/data/`, `configs/dataset/`, `configs/model/`, `configs/process/`, `configs/trainer/` | [tooling](tooling.md#configs) |
| `tests/` | `tests/test_roundtrip.py`, the data round trip on every clip of a dataset (`--dataset`); `tests/test_docs_index.py` keeps this index honest | [testing guide](../guides/how-to/testing.md) |
| `data/truebones/rigs/` | per-character prepared assets: manifest, preprocessing, skeleton, stats, report | [layout guide](../guides/concepts/data/layout.md) |
| `data/truebones/clips/<Rig>/` | prepared motion clips per character (derived; not tracked in git, rebuilt by stage 1) | [layout guide](../guides/concepts/data/layout.md) |
| `data/_raw/` | raw licensed vendor files, kept out of the prepared layout | [tooling](tooling.md#data) |
| `docs/index/` | this index, the overview, tooling and the open-work list; each library package is documented on its API page | [overview](overview.md) |
| `docs/guides/` | the guides: `docs/guides/concepts/` (design rationale, data) to read, `docs/guides/how-to/` (tutorials, how-tos) to follow; the nav is in `mkdocs.yml` | [docs](../README.md) |
| `docker-compose.yml` | services `test`, `train`, `compat`, `fbxsdk` (FBX I/O), `fbx` (Blender, rendering only), `docs` (the docs site, kept running) | [tooling](tooling.md#containers) |
| `Dockerfile` | the multi-stage image behind those services: OS and system packages only | [tooling](tooling.md#containers) |
| `uv.lock` | every Python dependency, pinned; installed with `uv sync --locked` | [tooling](tooling.md#dependencies) |
| `CLAUDE.md` | instructions for agents: start here, and keep `docs/index/` current | [index](INDEX.md) |
| `docs/api/` | one page per library package and subpackage, following the package tree (`docs/api/models/modules/index.md`), plus a page per separate design inside a package (`docs/api/build/preparation.md`, `docs/api/features/reconstruction.md`, `docs/api/features/reduction.md`) in its dropdown: one class diagram, then the reference generated from the docstrings at build time, the package docstring and one `:::` line per module (each module on exactly one page); `docs/api/index.md` is the diagram legend, `docs/api/retargeting.md` the one cross-package overview | [API](../api/index.md) |
| `docs/javascripts/diagrams.js` | draws every Mermaid fence in the page itself (not Material's integration, whose shadow DOM breaks class-diagram markers in dark mode), colours classes by role, and opens a clicked diagram full-screen with pan and zoom; styles in `docs/stylesheets/diagrams.css` | [API](../api/index.md) |
| `mkdocs.yml` | the docs site and its explicit nav (a page left out fails the strict build): `uvx --from 'mkdocs<2' --with mkdocs-material --with 'mkdocstrings[python]' mkdocs serve` | [docs](../README.md) |

## Open work

[todo](todo.md) lists what is known to be open: suspected bugs, leftovers and
decisions still pending. Treat it as leads to check, not as settled fact.

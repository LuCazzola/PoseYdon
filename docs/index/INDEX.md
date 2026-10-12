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
| `src/poseydon/augment/` | data augmentation applied to training windows | [augment](library/augment.md) |
| `src/poseydon/build/` | turning a raw corpus into a prepared one, invertibly: preparation, correspondence, skinning, reports | [build](library/build.md) |
| `src/poseydon/conditioners/` | declared conditioning: the only extra data a batch carries | [conditioners](library/conditioners.md) |
| `src/poseydon/core/` | the contracts: animation containers, skeleton manifests, `FeatureSpec`, `MotionBatch`, kinematics, rotations, registry | [core](library/core.md) |
| `src/poseydon/data/` | reading a prepared corpus into feature windows, normalization, rig-balanced sampling, collation | [data](library/data.md) |
| `src/poseydon/export/` | `CharacterExporter`: model output back onto prepared, presentation or original character space | [export](library/export.md) |
| `src/poseydon/features/` | composable motion feature representations, extraction and recovery | [features](library/features.md) |
| `src/poseydon/ingest/` | BVH corpus to aligned animations plus a corpus index | [ingest](library/ingest.md) |
| `src/poseydon/io/` | BVH and FBX reading and writing, shot framing, stick-figure rendering | [io](library/io.md) |
| `src/poseydon/losses/` | composable loss terms | [losses](library/losses.md) |
| `src/poseydon/models/` | denoising models (AnyTop, MoDiffAE) | [models](library/models.md) |
| `src/poseydon/ops/` | motion tasks run with a trained model: generate, in-between, blend, retarget | [ops](library/ops.md) |
| `src/poseydon/process/` | generative processes: corruption, parameterization and their inverses | [process](library/process.md) |
| `src/poseydon/sampling/` | samplers and step-level controls (guidance) | [sampling](library/sampling.md) |
| `src/poseydon/solvers/` | inverse kinematics: fitting rotations to desired positions | [solvers](library/solvers.md) |
| `src/poseydon/training/` | turning a config into a trained checkpoint: task, Lightning wrappers, recipe, retarget validation | [training](library/training.md) |
| `src/poseydon/cli.py` | the `poseydon` command: `ingest`, `train`, `sample`, `retarget`, `export`, `list` | [tooling](tooling.md#entry-points) |

## Tooling

### Entry-point scripts

| path | does | read |
|---|---|---|
| `scripts/process_dataset_truebones.py` | raw Truebones BVH → canonical, invertible clips under the entity-first layout | [tooling](tooling.md#scripts) |
| `scripts/process_dataset_truebones_fbx.py` | raw Truebones FBX → +Z-facing clips with skin intact, plus one `mesh.npz` per rig | [tooling](tooling.md#scripts) |
| `scripts/build_features.py` | prepared BVH → canonical features, per-rig statistics and the clip index | [tooling](tooling.md#scripts) |
| `scripts/build_truebones_assets.py` | builds every Truebones model/presentation FBX asset from canonical raw data | [tooling](tooling.md#scripts) |
| `scripts/anim_2_fbx.py` | a generated clip → an FBX on the rig the user authored | [tooling](tooling.md#scripts) |
| `scripts/render_fbx.py` | an FBX → an MP4 of the character performing it | [tooling](tooling.md#scripts) |
| `scripts/check_fbx_coherence.py` | checks that the BVH and FBX arms of the corpus describe the same motion | [tooling](tooling.md#scripts) |
| `scripts/audit_truebones_assets.py` | counts mesh-bearing FBX assets and complete `report.json` files across rigs | [tooling](tooling.md#scripts) |
| `scripts/build_skeleton_visual_debug.py` | collects one comparable bundle per rig: source, prepared, model, roundtrip | [tooling](tooling.md#scripts) |
| `scripts/build_skeleton_visual_debug_fbx.py` | generates `CharacterExporter` roundtrip files for the visual-debug bundles | [tooling](tooling.md#scripts) |

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
| `tools/probe_scale_residual.py` | *throwaway* spike probe on scale residuals | [tooling](tooling.md#tools-and-probes) |
| `tools/probe_reference_retarget.py` | *throwaway*: Flamingo → Scorpion retarget with the reference weights | [tooling](tooling.md#tools-and-probes) |
| `tools/probe_wandb_parity.py` | *throwaway*: pushes a parity retarget run to wandb | [tooling](tooling.md#tools-and-probes) |
| `tools/render.py` | re-export so reference-comparison scripts keep one import path | [tooling](tooling.md#tools-and-probes) |
| `tools/reference/` | scripts that run the reference implementation (`compat` image) to produce ground truth PoseYdon is compared against | [tooling](tooling.md#tools-and-probes) |

### Configs, tests, data, docs, containers

| path | holds | read |
|---|---|---|
| `configs/` | Hydra configs: `configs/train.yaml`, `configs/sample.yaml`, groups `configs/data/`, `configs/dataset/`, `configs/model/`, `configs/process/`, `configs/trainer/` | [tooling](tooling.md#configs) |
| `tests/` | pytest suites mirroring `src/poseydon/` package by package; `tests/test_docs_index.py` keeps this index honest | [testing guide](../guides/testing.md) |
| `data/truebones/rigs/` | per-character prepared assets: manifest, preprocessing, skeleton, stats, meshes, report | [layout guide](../guides/truebones-layout.md) |
| `data/truebones/clips/` | prepared motion clips per character | [layout guide](../guides/truebones-layout.md) |
| `data/_raw/` | raw licensed vendor files, kept out of the prepared layout | [tooling](tooling.md#data) |
| `docs/index/` | this index, the overview, tooling and one page per library package | [overview](overview.md) |
| `docs/guides/` | how to run things | [docs](../README.md) |
| `docs/superpowers/specs/` | designs: problem, options, measurements, contract | [docs](../README.md) |
| `docs/superpowers/plans/` | how each spec was sequenced into tasks | [docs](../README.md) |
| `docker-compose.yml` | services `test`, `train`, `compat`, `fbx` | [tooling](tooling.md#containers) |
| `Dockerfile` | the multi-stage image behind those services | [tooling](tooling.md#containers) |
| `CLAUDE.md` | instructions for agents: start here, and keep `docs/index/` current | [index](INDEX.md) |
| `mkdocs.yml` | the docs site: `uvx --from 'mkdocs<2' --with mkdocs-material mkdocs serve` | [docs](../README.md) |

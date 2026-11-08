# Map

Where things live. Find the path, then read the page in the third column.

The repo has two sides, and they change under different rules:

- **Library**: `src/poseydon/`, the importable package. It owns every contract;
  a change here can break every caller.
- **Tooling**: everything built around the library (scripts, probes, configs,
  tests, data, containers). It uses the library and owns no contract.

See the [overview](../user-guide/overview.md) for how the pipeline fits together.

## Library

| path | owns | read |
|---|---|---|
| `src/poseydon/augment/` | data augmentation applied to training windows | [augment](../api/augment/index.md) |
| `src/poseydon/build/` | turning a raw corpus into a prepared one, invertibly: preparation (`prepare.py`; `cleanup.py`: rig cleanup that removes joints), clip and frame filters (`clips.py`: the `clips:` group), sources (`sources.py`: raw BVH or a dataset's FBX cache), skeleton, statistics, index, reports | [build](../api/build/index.md) |
| `src/poseydon/conditioners/` | declared conditioning: the only extra data a batch carries | [conditioners](../api/conditioners/index.md) |
| `src/poseydon/core/` | the contracts: animation containers, skeleton manifests, `FeatureSpec`, `MotionBatch`, kinematics, rotations, registry | [core](../api/core/index.md) |
| `src/poseydon/data/` | reading a prepared corpus into feature windows, normalization, rig-balanced sampling, collation | [data](../api/data/index.md) |
| `src/poseydon/export/` | `RigRecord`: a rig's recorded reduction, preparation and rest asset; moves model output to the prepared character or the user's source rig, for `BVH`/`FBX.from_animation` to write | [export](../api/export/index.md) |
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
| `scripts/fetch_unimate_annotations.sh` | fetches UniMate's Truebones clip index and FBX repair scripts (pinned HF commit) into `data/_raw/UniML3D/truebones_annotations/`, and links `data/_raw/unimate` to `truebones` | [tooling](tooling.md#scripts) |
| `scripts/build_fbx_cache.py` | a dataset's FBX clips (`--data truebones`: `data/truebones/fbx_clips.csv`; `--data unimate`: UniMate's linked clips) read once with the FBX SDK into `<root>/cache/fbx/` (world matrices of every node, bind, skin summary); needs the `fbxsdk` service | [tooling](tooling.md#scripts) |
| `scripts/verify_motion.py` | the motion verifier's entry point: `build` writes a run directory (`tmp/verify/<Rig>__<clip>/`: stage files and `case.json` with SHA-256s), `file` judges one FBX against a vendor reference, `judge` runs the checks and writes `report.json` and `report.txt`; needs the `fbxsdk` service | [tooling](tooling.md#scripts) |

### Probes and one-off tools

Investigations kept as a record. Files marked *throwaway* in their docstring are
meant to be deleted once read.

| path | does | read |
|---|---|---|
| `tools/verify/report.py` | the verifier's `Check` and `Report` (statuses `PASS`/`FAIL`/`ALLOWED`/`NOT RUN`, evidence classes), `apply_allowances` | [tooling](tooling.md#scripts) |
| `tools/verify/case.py` | a verifier `Case` (files under test, the clip's vendor file, the rig's reference skin, sampled frames, recorded window), `loss_record` (the rig's stages and law parameters), `recorded_space` (the placement a stage file was written with), `zero_clip` and `zero_record` (the zero pose, and the stages, placement and manifest definitions C4 judges it by), `load_case`, `sample_frames`, `sha256`, `commit` | [tooling](tooling.md#scripts) |
| `tools/verify/checks.py` | the verifier's checks; `c1_inventory` compares each file's `FBX.inventory` with the reference, `c2a_skin` each file's skin, `c2b_fidelity` the returned file with the source clip under the loss laws, `c3_canonical` the canonical stage files after one fitted similarity, `c4_zero` the identity-rotation `5_zero` file against the rest pose and its declared definition | [io](../api/io/index.md) |
| `tools/verify/laws.py` | the loss laws: what `promote_root`, `enforce_rigid`, `prune_unskinned`, `drop_still`, `drop_end_sites` and `fold_into_root` are defined to do to a clip's world matrices, and `LOSSLESS`, the stages that only move the clip; imports nothing from `poseydon.build`, `poseydon.export` or `poseydon.features` | [tooling](tooling.md#scripts) |
| `tools/verify/skin.py` | the reference skin: `deform` recomputes a vendor FBX's linear blend skinning from `FBX.skin_data`; needs the `fbxsdk` service for its test | [io](../api/io/index.md) |
| `tools/build_rest_fbx.py` | each rig's rest asset under `cache/rigs/<Rig>/`: its vendor FBX held in the rest pose (`rest.fbx`, what FBX export writes onto), its read (`rest.npz`, with the rest pose `fbx_cache` takes with `rest_asset`) and the rest mesh (`rest_mesh.npz`, for `ground: mesh`); needs the `fbxsdk` service | [tooling](tooling.md#scripts) |
| `tools/build_joint_name_cache.py` | caches T5 embeddings of every joint name in the corpus, through `poseydon.conditioners.text` | [tooling](tooling.md#tools-and-probes) |
| `tools/ik_grid_search.py` | grid search over `GradientIK` terms, foot-contact term included | [tooling](tooling.md#tools-and-probes) |
| `tools/ik_contact_compare.py` | default vs best-from-grid IK solve, with predicted contacts shown | [tooling](tooling.md#tools-and-probes) |
| `tools/probe_floating_root.py` | which rigs have a root that is a ground locator rather than a body joint | [tooling](tooling.md#tools-and-probes) |
| `tools/probe_promote_table.py` | the promoted-root table: first branching joint per ground-locator rig | [tooling](tooling.md#tools-and-probes) |
| `tools/probe_root_chain.py` | walks the top of the hierarchy to the first branch, with cumulative offset | [tooling](tooling.md#tools-and-probes) |
| `tools/probe_root_shape.py` | per rig: what sits at the top of the hierarchy and how far the first real joint is | [tooling](tooling.md#tools-and-probes) |
| `tools/probe_jointset_spread.py` | per rig: whether clips agree with each other and the T-pose on their joint set | [tooling](tooling.md#tools-and-probes) |
| `tools/probe_tpose_gap.py` | which joints the gutted rigs' T-poses omit | [tooling](tooling.md#tools-and-probes) |
| `tools/probe_rest_override.py` | validates a proposed per-rig rest-pose override against the modal joint set | [tooling](tooling.md#tools-and-probes) |
| `tools/write_unimate_manifests.py` | per-species manifests for `data=unimate` from UniMate's facing pairs and the FBX cache | [tooling](tooling.md#tools-and-probes) |
| `tools/write_rest_pose.py` | emits a `rest_pose:` line per rig for review before applying | [tooling](tooling.md#tools-and-probes) |
| `tools/probe_resolve_after_reduce.py` | whether `resolve()` survives joint reduction, for every rig | [tooling](tooling.md#tools-and-probes) |
| `tools/probe_tukan.py` | inspects the Tukan T-pose: joint count, mean bone, branching | [tooling](tooling.md#tools-and-probes) |
| `tools/probe_reference_retarget.py` | *throwaway*: Flamingo → Scorpion retarget with the reference weights | [tooling](tooling.md#tools-and-probes) |
| `tools/probe_wandb_parity.py` | *throwaway*: pushes a parity retarget run to wandb | [tooling](tooling.md#tools-and-probes) |
| `tools/tpose_author.py` | authors one-frame T-poses (`__TPOSE_authored.bvh`) for rigs whose rest pose is a clip frame, checked on the skinned vendor mesh; instructions and per-rig configs at its top | [tooling](tooling.md#tools-and-probes) |
| `tools/render.py` | re-export so reference-comparison scripts keep one import path | [tooling](tooling.md#tools-and-probes) |
| `tools/reference/` | scripts that run the reference implementation (`compat` image) to produce ground truth PoseYdon is compared against | [tooling](tooling.md#tools-and-probes) |

### Configs, tests, data, docs, containers

| path | holds | read |
|---|---|---|
| `configs/` | Hydra configs: `configs/train.yaml`, `configs/sample.yaml`, `configs/build.yaml`, groups `configs/data/` (one file per dataset, raw file to training sample), `configs/features/`, `configs/model/`, `configs/process/`, `configs/trainer/` | [tooling](tooling.md#configs) |
| `tests/` | `tests/test_roundtrip.py`, the data round trip on every clip of a dataset (`--dataset`); `tests/test_docs_index.py` keeps this map honest | [testing guide](../how-to/testing.md) |
| `data/truebones/rigs/` | per-character prepared assets: manifest, preprocessing, skeleton, stats, report | [data formats](data-formats.md) |
| `data/truebones/clips/<Rig>/` | prepared motion clips per character (derived; not tracked in git, rebuilt by stage 1) | [data formats](data-formats.md) |
| `data/truebones/fbx_clips.csv` | the Truebones clip index: `rig,action,file` per clip (the vendor FBX of each BVH-era action, `bvh` naming the BVH it replaced), read by the `fbx_cache` source and `scripts/build_fbx_cache.py` | [build](../api/build/index.md) |
| `data/unimate/` | the `data=unimate` corpus, gitignored as a whole (generated manifests included): `data/unimate/cache/`, the prepared layout, `dropped.json`, `replication.json` | [tooling](tooling.md#data) |
| `data/_raw/` | raw licensed vendor files, kept out of the prepared layout | [tooling](tooling.md#data) |
| `docs/reference/` | this map, the glossary, data formats, tooling, the issues & ideas page and its ticket scripts (`docs/reference/open_ticket.py`, `docs/reference/edit_ticket.py`, `docs/reference/close_ticket.py`, run on the host) | [issues & ideas](ideas.md) |
| `docs/user-guide/`, `docs/how-to/` | the guides: `docs/user-guide/` (design rationale, overview, round trip) to read, `docs/how-to/` (tutorials, tasks, adding, testing) to follow; the nav is in `mkdocs.yml` | [docs](../index.md) |
| `docker-compose.yml` | services `test`, `train`, `compat`, `fbxsdk` (FBX I/O), `fbx` (Blender, rendering only), `docs` (the docs site, kept running) | [tooling](tooling.md#containers) |
| `Dockerfile` | the multi-stage image behind those services: OS and system packages only | [tooling](tooling.md#containers) |
| `uv.lock` | every Python dependency, pinned; installed with `uv sync --locked` | [tooling](tooling.md#dependencies) |
| `CLAUDE.md`, `AGENTS.md` (symlink) | rules for agents: start at the docs home, keep the docs true, how to test and behave | [docs](../index.md) |
| `.github/workflows/docs.yml` | builds the site strictly on every PR, and deploys it to GitHub Pages from `main` | [tooling](tooling.md#docs-site) |
| `.github/workflows/claude.yml` | runs Claude Code on `@claude` mentions in issues and PRs | [tooling](tooling.md#docs-site) |
| `.github/ISSUE_TEMPLATE/` | the bug template; ideas are redirected to the ideas table | [issues & ideas](ideas.md) |
| `docs/api/` | one page per library package and subpackage, following the package tree (`docs/api/models/modules/index.md`), plus a page per separate design inside a package (`docs/api/build/preparation.md`, `docs/api/features/reconstruction.md`, `docs/api/features/reduction.md`) in its dropdown: one class diagram, then the reference generated from the docstrings at build time, the package docstring and one `:::` line per module (each module on exactly one page); `docs/api/index.md` is the diagram legend, `docs/api/retargeting.md` the one cross-package overview | [API](../api/index.md) |
| `docs/javascripts/diagrams.js` | draws every Mermaid fence in the page itself (not Material's integration, whose shadow DOM breaks class-diagram markers in dark mode), colours classes by role, and opens a clicked diagram full-screen with pan and zoom; styles in `docs/stylesheets/diagrams.css` | [API](../api/index.md) |
| `docs/stylesheets/extra.css` | site styling: centred tables and images, sidebar group titles, the current page in ochre, underline on hover | [API](../api/index.md) |
| `mkdocs.yml` | the docs site and its explicit nav (a page left out fails the strict build): `uvx --from 'mkdocs<2' --with mkdocs-material --with 'mkdocstrings[python]' mkdocs serve` | [docs](../index.md) |

## Open work

[Issues & Ideas](ideas.md) lists what is known to be open: bugs and loose ends
as GitHub issues, proposed changes and pending decisions as ideas. Treat them as
leads to check, not as settled fact.

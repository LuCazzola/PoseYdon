# Configuration

Code: `src/poseydon/core/registry.py`, `src/poseydon/core/skeleton.py`, `src/poseydon/training/build.py`, `configs`

## Every component is a registered name

Each config slot that builds an object names a registered class by `type:`. The rest of
the entry becomes its arguments, as in mmengine's `build_from_cfg`
(`src/poseydon/core/registry.py`). Slots whose entries take no arguments use the bare name
(`features:`, `conditioners:`), and `losses:` maps each name to its weight. Each slot has
its own registry, so short names only need to be unique within that slot.
`poseydon list` prints most registries; it leaves out sources, clip ops,
reconstructors, IK terms, solvers and latent-mix schedules.

A config's `custom_imports` are imported first, so a user can register a class of their
own and name it the same way, without editing the library (`src/poseydon/training/build.py`).

Configs are composed with Hydra (`configs/train.yaml`, `configs/sample.yaml`,
`configs/build.yaml`). A dataset is one file under `configs/data/`, and it reads top to
bottom as the life of one clip: `paths`, `source`, `prepare` (stage 1), `reduce` and
`labels` (stage 2), `filter`, `loader`.

## Per-character facts live in manifests, named by joint

Facts the reference kept as Python literals (facing joints, foot joints, taxonomy) are
per-rig YAML here: `data/<dataset>/rigs/<Rig>/manifest.yaml`, read by
`SkeletonManifest`. Joints are named, never indexed, so a re-exported rig fails loudly.
Contact thresholds are not per rig: they come from the dataset's `labels.contact`.
A manifest also declares the rest-pose clip, the up axis, an optional `promote_root`, and
which vendor file is authoritative for skeleton, motion and surface (`authority:`,
`artifacts:`).

Adding a character is therefore a data change, which matches the owner's rule in
`CLAUDE.md` to keep what a character or dataset needs in data and configs.

## A blacklist that matches nothing is an error

`filter.blacklist` in a dataset config drops rigs or clips before anything reads the
index. An entry that matches nothing raises an error. The config comment gives the
reason: a misspelled name that quietly matched nothing would train the very clips it was
written to exclude.

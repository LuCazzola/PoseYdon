# todo

What is known to be open: suspected bugs, leftovers from earlier work and
decisions still pending.

!!! warning "Leads, not facts"
    Each row is what was believed when it was written. It may be stale,
    already fixed, wrong about the cause or narrower than it looks. Before
    acting on one, check it against the code and the data, and fix the row if
    it disagrees. Nothing here is a spec.

Keep it current: add a row when you find something you are not fixing now,
and remove or correct a row in the same commit that fixes or disproves it.
Each row says when it was last confirmed and points to the page that explains
it, where one does.

| issue | effect | last confirmed | where |
|---|---|---|---|
| Dog's facing comes from thighs + upper arms (`data/truebones/rigs/Dog/manifest.yaml`) | measured against pelvis-to-head on frame 0 of 41 clips: median 1.8°, max 21°, 4 clips over 10°. Clavicles + hands measured worse (hands swing with the gait: 14 clips over 10°); clavicles alone measured best (median 0.2°, max 16.5°). Awaiting the owner's visual inspection; a change needs `scripts/process_dataset_truebones.py --rigs Dog` | 2026-10-13 | here |
| Model checkpoints trained before 2026-10-13 name the feature `rot6d` | it is now `rot6d_bone`; such a recipe fails to load until edited. No local run is affected (local runs were deleted on 2026-10-13) | 2026-10-13 | [features](../api/features/index.md#two-rotation-layouts) |
| `Blend` sizes its schedule from `reference.shape[0]`, but `LatentMix` counts frames on axis -2 | for a batched `(B, T, C)` latent with `B != T`, `Blend(...)` without `length=` raises; the guide passes `length=` as a workaround | 2026-10-13 | [ops](../api/ops/index.md#limitations); [blend guide](../guides/how-to/tasks/blend.md) |
| `inbetween` and `blend` have no CLI path | `poseydon sample` instantiates `operation:` from the config, and both take tensors as arguments, so they are reachable only from Python. The old sampling guide listed them as selectable there | 2026-10-13 | [ops](../api/ops/index.md#limitations); [tasks overview](../guides/how-to/tasks/overview.md) |
| `Blend` on a model with no semantic latent (AnyTop) is plain generation | `LatentMix` writes `z_sem`, which AnyTop never reads, and `Blend` checks nothing, so it returns unblended motion with no error. `Retarget` raises in the same case; `Blend` could check `model.encode` the same way | 2026-10-13 | [ops](../api/ops/index.md#limitations) |
| No test covers `process/` or the samplers | `tests/` holds only the round trip and the docs checks; a broken schedule or `to_z0` would go unnoticed | 2026-10-13 | [testing guide](../guides/how-to/testing.md) |
| Docstrings and comments cite a "design spec" that is not in the repo | `src/poseydon/augment/__init__.py`, `src/poseydon/augment/topology.py`, `src/poseydon/core/animation.py`, `src/poseydon/ops/retarget_run.py`, `src/poseydon/training/metrics.py`, `src/poseydon/cli.py` point at sections of a document nobody can read; the reasoning should be inlined or the citation dropped | 2026-10-13 | here |
| Crow's `shoulders` facing pair (`_19`, `_16` in `data/truebones/rigs/Crow/manifest.yaml`) points mostly vertically in the rest skeleton | measured: right-to-left 0.29 vertical against 0.16 horizontal, so it is likely not a left/right pair; the facing of every Crow clip is computed partly from it, and stage 1's upright check warns on Crow (right-left axis 0.66 vertical). Awaiting the owner's inspection of which joints are the shoulders | 2026-10-13 | here |

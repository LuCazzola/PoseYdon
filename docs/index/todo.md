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
| Sampling and retargeting take the target rig's conditioning from its first-indexed clip, not its rest clip | output can come back rotated or distorted when exported with the rig's facing; proposed fix: template on `rest_action` | 2026-10-12 | [ops](library/ops.md) gotchas; [round-trip guide](../guides/roundtrip.md#open-issues) |
| A model view written as BVH has no rotation for its leaves (a BVH leaf has no channels) | an FBX made from the `.bvh` holds e.g. Dog's ears and tail tip at rest; export from the `.anim.npz` instead. Fix idea, low priority: write such joints with a zero-length End Site | 2026-10-13 | [export](library/export.md) gotchas |
| Dog's facing comes from thighs + upper arms (`data/truebones/rigs/Dog/manifest.yaml`) | measured against pelvis-to-head on frame 0 of 41 clips: median 1.8°, max 21°, 4 clips over 10°. Clavicles + hands measured worse (hands swing with the gait: 14 clips over 10°); clavicles alone measured best (median 0.2°, max 16.5°). Awaiting the owner's visual inspection; a change needs `scripts/process_dataset_truebones.py --rigs Dog` | 2026-10-13 | here |
| Reduction drops zero-offset joints even when they are skinned | Bear's forearm-twist bones: the twist is lost and the skin can sit up to 5.5% of the mesh extent off; fixing it changes the model topology, so it means retraining | 2026-09-28 | [features](library/features.md) gotchas |
| Model checkpoints trained before 2026-10-13 name the feature `rot6d` | it is now `rot6d_bone`; such a recipe fails to load until edited. No local run is affected (local runs were deleted on 2026-10-13) | 2026-10-13 | [features](library/features.md#two-rotation-layouts) |

# Dataset layout: what each prepared file holds

Each character has one directory under `data/truebones/rigs/<Character>/`.
The raw licensed vendor files are kept separately under `data/_raw/truebones/`
and are not part of the prepared layout.

| File | Meaning |
| --- | --- |
| `manifest.yaml` | Character policy and source declarations: rest pose, facing pairs, authority, and available BVH/FBX artifacts. |
| `preproc.npz` | Stage-1 preprocessing parameters and their inverse: rest-relative transforms, root promotion, rigid-channel metadata, centering, scale, grounding, and per-clip facing rotations. |
| `skeleton.npz` | The model-facing skeleton contract: ordered names, parents, offsets, human-readable names, and reduction mapping. |
| `stats.npz` | Feature-normalization statistics (`mean`, `std`, and block metadata) used by training. |
| `report.json` | Deterministic preparation audit: authorities, source/output checksums, transforms, topology, decisions and warnings. |

The prepared motion clips live separately in
`data/truebones/clips/<Character>/`. `scripts/build_stage_samples.py` writes
one clip per character at every round-trip stage, as BVH and FBX, under
`tmp/truebones/<Character>/`.

There is no prepared FBX asset. Original vendor files remain under
`data/_raw/truebones/source/` and are what the original FBX export writes onto.
A corpus built before 2026-10-12 may still hold `mesh.npz`, `char-model.fbx`,
`char-full.fbx` and `clips/<Character>/*.fbx`; nothing reads them.

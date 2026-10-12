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
| `mesh.npz` | Prepared mesh geometry, faces, skin weights, and the bind skeleton used by the Blender asset pass. |
| `char-model.fbx` | Mesh-bearing reduced model rig. Its topology and bind transforms match `skeleton.npz`; this is the model/prepared FBX representation. |
| `char-full.fbx` | Mesh-bearing full deformation rig. It retains the auxiliary joints needed for richer skin deformation and visual playback. |
| `report.json` | Deterministic preparation audit: authorities, source/output checksums, transforms, topology, decisions, warnings, and artifact availability. |

The prepared motion clips live separately in
`data/truebones/clips/<Character>/`. The visual comparison corpus under
`tmp/skel_visual_debug/<Character>/` contains copied vendor inputs, prepared
representations, and roundtrip outputs for Blender inspection.

`char-full.fbx` is a canonical prepared-space asset, not the untouched vendor
FBX. Original vendor files remain under `data/_raw/truebones/` and are used for
the original-space export/roundtrip path.

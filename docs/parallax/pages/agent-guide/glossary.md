# Glossary and conventions

Terms, units and conventions the code uses without explaining.

| Term | Meaning |
|---|---|
| rig | One character's skeleton, named by its directory under `data/<dataset>/rigs/`. |
| clip / action | One animation of a rig, full length. Its id is `<Rig>__<action>` (for example `Crow__fly1`). |
| vendor files | The dataset's own files under `data/_raw/` (licensed: never committed). |
| stage 1 | Raw source to prepared clip, through the `prepare:` chain (`scripts/process_dataset_truebones.py`). |
| stage 2 | Prepared clip to features, stats and index (`scripts/build_features.py`). |
| prepared | Canonical frame, every joint left after cleanup. |
| model view | Prepared, then reduced: what the model sees. |
| source space | The user's rig as supplied. |
| rig scope / clip scope | Whether a prepare stage fits once per rig (on the rest pose) or once per clip. |
| rest pose | The pose a rig's clips are made relative to: the manifest's `rest_pose:` (a T-pose, an authored one, or a clip's frame). |
| rest asset | The vendor FBX held in the rest pose, written by `tools/build_rest_fbx.py`; FBX export writes onto it by node name. |
| End Site | A BVH leaf with no channels. `drop_end_sites` removes only the zero-length ones; an End Site with a real offset is a limb tip and stays. |
| still joint | A joint, with its subtree, that never rotates more than the configured `degrees` (`drop_still`) over any clip of its rig. |
| promoted root | A rig whose file root is a ground locator; `promote_root:` moves the root onto the body joint. |
| fold into root | Merging a zero-length sole child of the root into the root (Biped `Hips` over `Bip01_Pelvis`). |
| reference | The published implementation in `external/neural_motion_blending/` (read-only), the parity target. Not the same as the verifier's *reference skin*. |

## Conventions

- Quaternions are scalar-last `(x, y, z, w)` (`src/poseydon/core/rotations.py`).
- 6D rotation: the feature blocks and the geodesic loss use the first two **columns** of
  the matrix, the reference's layout (`_REFERENCE_6D_LAYOUT` in
  `src/poseydon/features/kinematics.py`). `matrix_to_rot6d` defaults to rows, and
  `src/poseydon/core/torch_kinematics.py` stacks rows.
- Canonical frame (the Truebones `prepare:` chain; another dataset's chain may differ): +Y up, the rest skeleton faces +Z, and each clip's frame 0 faces +Z.
  The root sits at the XZ origin on the rest pose, which stands on y = 0.
- Scale: Truebones normalizes the mean bone length to `0.20921428571428569`, the
  reference's `HML_AVG_BONELEN` (`configs/data/truebones.yaml`). UniMate scales by
  diameter instead (`configs/data/unimate.yaml`).
- Truebones clips are 30 fps; another rate is reported and skipped.
- Joints are referenced by name, never by index.
- Verifier tolerances: each key's comment in `configs/verify/verify.yaml` states its unit
  (fraction of rig extent, degrees, ratio or matrix-element difference).

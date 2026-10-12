# export

`src/poseydon/export/`, library. `CharacterExporter` takes a model-space
animation and writes it as the prepared or original representation, in BVH or
FBX.

## Purpose

See the [round-trip guide](../../guides/roundtrip.md) for the whole loop this
closes.

`export` owns the last arrow of the pipeline: reduced canonical motion →
a file on a character. It loads a rig's recorded transforms and exposes one
method per representation and format. It also owns `FbxCalibration`, the
measured mapping between a rig's BVH joints and its vendor FBX nodes.

It does **not** own the transforms themselves (`build.prepare`,
`features.reduce`) or file I/O ([io](io.md)).

## Key files and types

| file | read it for |
|---|---|
| `src/poseydon/export/character.py` | `CharacterExporter.load(corpus, raw, rig)`; `prepared_bvh`, `prepared_fbx`, `original_bvh`, `original_fbx` (the last two take `facing_from=<action>`); `calibration()`; `FbxCalibration`; `same_clip_pairs`; `world_matrices` |

Representations:

| representation | joints | frame | BVH | FBX |
|---|---|---|---|---|
| **prepared** | reduced, exactly the model's | canonical: +Z, scaled, centred, grounded | the animation as is | a bare skeleton, no mesh |
| **original** | the source's own, after expanding the reduction | source units, frame, names, channels | the source's `CHANNELS` block | the vendor's own FBX asset: its nodes, mesh, skin and materials |

## How it connects

Consumes `rigs/<Rig>/skeleton.npz`, `preproc.npz` and `manifest.yaml` through
`build.prepare.RigTransform` and `prepare_chain`, `features.reduce`
(`invert_reduction`), `io.bvh.BVH`, `io.fbx.FBX` (lazily, `fbxsdk` container
only), the raw vendor files under `data/_raw/<dataset>/source/<Rig>/`, and
`core.authority.ArtifactUnavailableError`.

Called by: `cli.py` (`poseydon export --representation prepared|original
--format bvh|fbx`), `scripts/build_stage_samples.py` and
`tests/test_roundtrip.py`.

## Why it is shaped this way

- **Expansion belongs to original export only.** A prepared export serializes
  the model character exactly as the model sees it. An original export expands
  the reduction and inverts preparation back to the raw character, as far as
  the contract is invertible.
- **Inversion reads persisted parameters.** The chain only supplies the order
  (`prepare_chain()`, its single definition), so export needs neither the
  channel layout nor the promotion target. Source: comment in
  `CharacterExporter.load`.
- **Facing is explicit.** For original outputs, `facing_from=<action>`
  restores that clip's recorded yaw. With no argument, the rig-level facing
  from `preproc.npz` is restored, which is the rest pose's facing. If that
  facing isn't recorded, export raises instead of substituting an identity.
  Commit `1dad7b5` removed the identity fallback because it displaced
  reinserted roots by 22–167% of rig extent.
- **Original FBX goes through a calibration, not an assumed bind pose.** The
  vendor ships the same clip as BVH and FBX, and the two are the same motion
  under one similarity (units, axes, placement) and a constant per-joint frame
  change. `FbxCalibration.fit` solves both from one such pair, then checks
  them against every frame; the first pair within `CALIBRATION_TOL` (1e-4 of
  the extent) wins, the rest pose's first. All 73 Truebones rigs calibrate,
  worst residual 5.3e-5 (Camel). The previous Blender path instead assumed the
  FBX bind pose was the BVH rest pose and refused the 13 rigs where it is not.
- **The manifest's `artifacts` pair is not used for calibration.** Its BVH and
  FBX are generally different clips (Alligator: a 2-frame `__TPOSE.bvh`
  against `AlligatorALL-BigMouth.fbx`). `same_clip_pairs` matches files by
  action name instead.
- **Hard errors, no silent substitutes.** Mismatched topology, a missing
  stage parameter or no agreeing BVH/FBX pair all raise.

## Gotchas

- **A facing is required only when the chain actually undoes one** (commit
  `e899592`).
- **FBX export runs only in the `fbxsdk` container.** BVH export works
  anywhere.
- **There is no skinned canonical asset.** The prepared FBX is a bare
  skeleton; the mesh exists only on the original representation. The
  `char-model.fbx` / `char-full.fbx` assets and the `presentation`
  representation were removed with the Blender FBX path.
- **Joints only one format has.** FBX nodes with no BVH joint keep the local
  transform they have on the asset's first frame; BVH End Sites have no node.
- **The default facing can mismatch generated clips** that were templated on a
  non-rest clip. See [ops](ops.md).

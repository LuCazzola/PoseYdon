# export

`src/poseydon/export/`, library. `RigRecord` holds everything the corpus
recorded about one rig and moves a clip between the model view, prepared space
and source space. Writing the result is the file format's job:
`BVH.from_animation` or `FBX.from_animation` ([io](io.md)).

## Purpose

See the [round-trip guide](../../guides/roundtrip.md) for the whole loop this
closes.

`export` owns the last arrow of the pipeline: reduced canonical motion →
motion on a character. `RigRecord` loads a rig's reduction, preparation
parameters and FBX calibrations; `expand` and `restore` return plain
animations. It also owns `FbxCalibration`, the measured mapping between a
rig's BVH joints and its vendor FBX nodes.

It does **not** own the transforms themselves (`build.prepare`,
`features.reduce`) or file I/O ([io](io.md)).

## Key files and types

| file | read it for |
|---|---|
| `src/poseydon/export/rig.py` | `RigRecord.load(corpus, raw, rig)`; `expand(model)` (model view → prepared); `restore(anim, facing_from=<action>)` (model view or prepared → source); `source_channels` (the source BVH's CHANNELS layout); `calibration("source" or "prepared")`; `FbxCalibration` (`fit`, `drive`, `inverse`, `to_source`, `placement`); `same_clip_pairs`; `world_matrices` |

Spaces, and how each is written:

| space | joints | frame | get it with | BVH | FBX |
|---|---|---|---|---|---|
| **model view** | reduced, exactly the model's | canonical: +Z, scaled, centred, grounded | the model's output | `BVH.from_animation(anim)` | expand first: the mesh needs every node |
| **prepared** | every joint | canonical | `record.expand(model)`, or a corpus clip | `BVH.from_animation(anim)` | `FBX.from_animation(anim, record.calibration("prepared"))`: the vendor asset, mesh and skin, placed in canonical space |
| **source** | the source's own | source units, frame, names | `record.restore(anim, facing_from=...)` | `BVH.from_animation(anim, channels=record.source_channels)` | `FBX.from_animation(anim, record.calibration("source"))`: the vendor asset in its own space |

## How it connects

Consumes `rigs/<Rig>/skeleton.npz`, `preproc.npz` and `manifest.yaml` through
`build.prepare.RigTransform` and `prepare_chain`, `features.reduce`
(`invert_reduction`), `io.bvh.BVH`, `io.fbx.FBX` (lazily, `fbxsdk` container
only), the raw vendor files under `data/_raw/<dataset>/source/<Rig>/`, and
`core.authority.ArtifactUnavailableError`.

Called by: `cli.py` (`poseydon export --representation prepared|source
--format bvh|fbx`), `scripts/build_stage_samples.py` and
`tests/test_roundtrip.py`.

## Why it is shaped this way

- **Transforming and writing are separate.** `RigRecord` only returns
  animations; a file is written from whatever animation it is given, so the
  space a clip is in decides what the file holds. The rename from
  `CharacterExporter` and the split followed from the project owner asking
  for exactly that: the class writes nothing, it is a record of the rig.
- **A file needs two things a clip doesn't carry.** A source BVH's CHANNELS
  layout (`source_channels`) and an FBX's asset and calibration
  (`calibration(space)`). The space is stated by the caller, not stored on
  the animation: nothing in a clip reliably tells prepared from source
  (their joints nearly coincide), and a stored label that went stale would
  pick the wrong calibration silently. `FbxCalibration.drive` raises on
  missing joints, but a clip in the wrong space with the right joints writes
  at the wrong place and scale.
- **The prepared-space FBX expansion belongs to FBX only.** A prepared BVH of
  the model view is the model character exactly as the model sees it; the FBX
  has to expand it because the mesh needs every node.
- **Inversion reads persisted parameters.** The chain only supplies the order
  (`prepare_chain()`, its single definition), so export needs neither the
  channel layout nor the promotion target. Source: comment in
  `RigRecord.load`.
- **Facing is explicit.** For `restore`, `facing_from=<action>`
  restores that clip's recorded yaw. With no argument, the rig-level facing
  from `preproc.npz` is restored, which is the rest pose's facing. If that
  facing isn't recorded, export raises instead of substituting an identity.
  Commit `1dad7b5` removed the identity fallback because it displaced
  reinserted roots by 22–167% of rig extent.
- **FBX goes through a calibration, not an assumed bind pose.** The
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
- **Every FBX carries the rig's mesh.** The prepared FBX is the vendor asset
  driven through `calibration("prepared")`, the calibration refitted on the
  calibration clip as the corpus prepared it (`clips/<Rig>/<action>.bvh`),
  whose `placement` (`FbxCalibration.inverse()`) moves the whole scene into
  canonical space through `FBX.animate(space=...)`. Its residual is not gated, because
  `EnforceRigid` changes bone lengths by design. There is no bare-skeleton
  FBX (`poseydon export --bare-skeleton` writes BVH only). Reason, as
  the project owner gave it: seeing the mesh on every FBX is what shows the
  whole pipeline is consistent.
- **The model FBX shows the expansion.** Joints the reduction dropped are
  driven by `invert_reduction`'s reconstruction, since the mesh needs every
  node.
- **A model view written as BVH has no rotation for its leaves.** Under the
  default `rot6d_joint` layout the reduction drops zero-length End Sites, so
  e.g. Dog's ear `Bip01_Ponytail21` is a leaf of the model view: the model
  generates its rotation, but a BVH leaf has no channels. Export from the
  `.anim.npz` that `sample` and `retarget` write beside the BVH, which keeps
  it; an FBX made from the BVH holds those nodes, and the mesh they skin, at
  rest. Until 2026-10-13 the model never saw these rotations at all: the only
  layout was `rot6d_bone`, whose slot for them belonged to the dropped End
  Site (72 of 73 rigs animate such a joint more than 5 degrees within a clip).
  See [features](features.md).
- **Joints only one format has.** FBX nodes with no BVH joint keep the local
  transform they have on the asset's first frame; BVH End Sites have no node.
- **The default facing can mismatch generated clips** that were templated on a
  non-rest clip. See [ops](ops.md).

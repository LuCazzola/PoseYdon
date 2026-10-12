# build

`src/poseydon/build/`, library. Turning a raw corpus into a prepared,
model-ready one, invertibly. Stage 1 prepares geometry; stage 2 builds the
skeleton, statistics and index. It also owns the character assets and the
audit reports.

## Purpose

`build` owns every on-disk artefact under `data/truebones/rigs/` and
`data/truebones/clips/`, with exactly one writer per file:

- **Stage 1, prepare** (`prepare.py`): source → prepared geometry, recorded in
  `preproc.npz` together with its inverse.
- **Stage 2, build** (`pipeline.py`): four passes (clip, skeleton, stats,
  index) that write `skeleton.npz`, `stats.npz`, the clip `.npz` and
  `index.jsonl`.
- **Character assets** (`character_assets.py`, `correspondence.py`,
  `skinning.py`): `char-model.fbx` and `char-full.fbx`, plus the validated
  mapping between the model and presentation rigs.
- **Audit** (`report.py`): one deterministic `report.json` per rig.

It does **not** own reading files ([io](io.md)), alignment primitives
([ingest](ingest.md)), the joint reduction itself
(`src/poseydon/features/reduce.py`), or the read path at training time
([data](data.md)).

## Key files and types

| file | read it for |
|---|---|
| `src/poseydon/build/prepare.py` | `prepare_chain()` (the one canonical stage order); `PrepareStage` (`fit` / `apply` / `invert`); `PrepareChain`; the stages `RestRelative`, `PromoteRoot`, `FaceAxis`, `EnforceRigid`, `CentreXZ`, `ScaleToMeanBoneLength`, `PutOnGround`; `RigTransform` (everything fitted for one rig) |
| `src/poseydon/build/pipeline.py` | `BuildConfig`, `build_clips`, `build_skeleton`, `build_stats`, `build_index`, `build_all` |
| `src/poseydon/build/index.py` | `CorpusIndex` and `ClipRecord` (the JSONL index); `clip_id` = `{skeleton}__{action}`; `action_slug` |
| `src/poseydon/build/corpus.py` | `PerRigDirectory` (`clips/<Rig>/<action>.bvh`); `rest_action` / `rest_source` (which clip is the rest pose); label files |
| `src/poseydon/build/names.py` | `Humanize`: `Bip01_R_Thigh` → `right thigh`, for the text encoder only |
| `src/poseydon/build/character_assets.py` | `build_character_assets`: the reduced model FBX and the full presentation FBX |
| `src/poseydon/build/correspondence.py` | `SkeletonCorrespondence`: the validated name and bind mapping between model and presentation rigs |
| `src/poseydon/build/skinning.py` | `plan_skinning`: pure planning of how skin weights move when joints are reduced |
| `src/poseydon/build/report.py` | `PreparationReport`, `AuditSink`: the per-rig `report.json` |
| `src/poseydon/build/source.py` | `SourceMotion`: loads raw motion through the format the manifest declares authoritative |
| `src/poseydon/build/authority.py` | re-exports `core.authority` for build-side callers |

## How it connects

Consumes [io](io.md) (`BVH`, `FBX`), alignment helpers from [ingest](ingest.md),
`core.skeleton`/`core.animation`/`core.authority`, `features.reduce` and
`extract_features` (for stats), and `data.normalize.Normalizer`.

Produces the prepared corpus that [data](data.md) reads (index, clips, skeleton,
stats). It also produces the `RigTransform` that [export](export.md) and
`scripts/anim_2_fbx.py` invert to return a generated clip to source space.

Called by: `scripts/process_dataset_truebones.py` (stage 1),
`scripts/process_dataset_truebones_fbx.py`, `scripts/build_features.py` (stage 2),
`scripts/build_truebones_assets.py`; and in the library by data, export, ingest
(index types), ops (`retarget_run`) and training (`build`, `metrics`).

## Why it is shaped this way

- **Every stage declares `invert` beside `apply`, and fitted parameters are
  persisted.** A user hands over a rig, the model generates on the canonical
  one, and the result must come back on theirs. The facing rotation especially
  can't be recovered once a file already faces +Z. Source:
  `src/poseydon/build/prepare.py` docstring;
  data pipeline spec §1 and §3
  (`docs/superpowers/specs/2026-09-06-data-pipeline-and-training-parity-design.md`).
- **One definition of the stage order.** The order was once written out twice,
  for prepare and for unprepare, and the inverse copy is only exercised when
  inverting, so drift there is silent. `prepare_chain()` is the single source.
  `PromoteRoot` comes after `RestRelative`, before `EnforceRigid` (so the
  preserved translation belongs to the promoted root) and before scaling (a
  locator connector isn't an anatomical bone). Source: `prepare_chain`
  docstring.
- **Rig-scope versus clip-scope stages.** Rig stages fit once on the rest pose;
  fitting per clip would ground a flying creature and erase the height
  difference between a crouch and a stand. `FaceAxis` is clip-scoped. Source:
  `src/poseydon/build/prepare.py` docstring.
- **The whole rig rotates, not just the root.** Unlike the reference,
  `rotate_rig` turns offsets too, so a DCC tool draws the rest skeleton and the
  animation in one convention. The cost is per-clip rest geometry (next
  section). Source: §3 "Rest geometry is not rig-level" (`docs/superpowers/specs/2026-09-06-data-pipeline-and-training-parity-design.md`).
- **Only `stats.npz` depends on the feature schema**, so changing `features:`
  re-runs one pass. Source: `src/poseydon/build/pipeline.py` docstring;
  §4 (`docs/superpowers/specs/2026-09-06-data-pipeline-and-training-parity-design.md`).
- **A JSONL index replaces filename parsing**, which the reference did in eight
  places, along with its filelist trees, taxonomy lists and pair files. It's
  append-only, diffable and needs no dependency. Source:
  `src/poseydon/build/index.py` docstring; commit `95dfff3`.
- **Computed, not declared, joint-name prefixes.** The `strip_joint_prefix`
  manifest key was parsed and read by nothing. Raw names stay the identity;
  humanized names exist only for the text encoder. Source:
  `src/poseydon/build/names.py` docstring; commit `f9ba5ab`.
- **Declared source authority.** Each manifest states whether BVH or FBX is
  authoritative for skeleton, motion and surface. Builds then record what each
  rig can export, instead of failing at export time. Source:
  character export spec (`docs/superpowers/specs/2026-09-19-character-aware-motion-export-design.md`);
  commits `404a5bb`, `53f9d39`.
- **Build failures are fatal per rig and collected.** Source: commit `3229a57`.

## Gotchas

- **Prepared clips of one rig don't share an OFFSET block.** `FaceAxis` rotates
  offsets per clip. Rig-level geometry in `skeleton.npz` must come from the
  **rest-pose clip** (`rest_action`), never an arbitrary one. Pairing one clip's
  offsets with another clip's rotations yields a rotated skeleton. As of
  2026-10-12, `poseydon sample` and the retarget template still pick the
  first-indexed clip rather than the rest clip, which is an open issue.
- **`PromoteRoot`'s target table lives in a script**: `PROMOTE_ROOT` in
  `scripts/process_dataset_truebones.py`, keyed by rig name.
- **Open in committed code: stages after `PromoteRoot` receive source-resolved
  joint indices.** `PrepareChain.fit_rig`/`apply` pass the pre-promotion
  `resolved`, so facing on promoted rigs is computed from the wrong joints
  (13 of 14 promoted rigs in the 2026-09-28 audit). A re-resolving fix exists
  only as uncommitted work as of 2026-10-12. Update this bullet when it
  lands.
- **One rigid geometry per rig is a lossy, deliberate choice.** Per-clip bone
  vectors are replaced by the rest clip's. Up to 25% of extent is lost on some
  attack clips. That loss is by design, not a round-trip bug.
- **A clip name must not contain `/`.** It would corrupt the npz keys (commit
  `b81de29`). Clip slugs also assume the file name starts with the rig name.
- **Rebuild order after a stage-1 change:** stage 1 → stage 2 →
  `scripts/build_truebones_assets.py` → stage 2 again, which refreshes the
  index's availability fields. Checkpoints trained on an older corpus need
  retraining.
- **Blender is imported lazily, inside functions.** `character_assets.py` and
  `source.py` import `poseydon.io.fbx` only inside the functions that need it,
  and `skinning.py` and `correspondence.py` don't import it at all. That keeps
  the BVH-only modality independent of Blender (commit `e15a2a1`). Never move
  those imports to module scope.

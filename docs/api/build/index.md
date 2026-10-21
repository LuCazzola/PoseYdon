# build

Turning a raw corpus into a prepared, model-ready one, invertibly. `build` owns
every artefact under `data/truebones/rigs/` and `data/truebones/clips/`, one
writer per file. Stage 1, [preparation](preparation.md), turns source geometry
into canonical geometry and records the inverse in `preproc.npz`. Stage 2
(`src/poseydon/build/pipeline.py`) runs four passes, clip, skeleton, stats and
index, writing the clip `.npz`, `skeleton.npz`, `stats.npz` and `index.jsonl`.
Only `stats.npz` depends on the feature schema, so changing `features:` re-runs
one pass.

It reads files through [io](../io/index.md), reduces joints through
[features](../features/reduction.md) and computes statistics with
[data](../data/index.md)'s `Normalizer`; [data](../data/index.md) reads what it
writes, and [export](../export/index.md) inverts stage 1 to bring a generated
clip back to the source rig. Where stage 1 reads its raw clips from is a
registered `source:` (`src/poseydon/build/sources.py`): `bvh_files`, BVH files
under `paths.raw/source/<Rig>/`, or `fbx_cache`, FBX clips as
`scripts/build_fbx_cache.py` cached them, listed by a clip index (`fbx_index`)
and with the skeleton rebuilt as Blender's FBX importer would have. Truebones
(`data=truebones`) reads its vendor FBX, one per BVH-era action
(`data/truebones/fbx_clips.csv`), with each rig's rest pose the bind of its rest
asset (`tools/build_rest_fbx.py`); UniMate's recipe (`data=unimate`) reads the
clips UniMate's own index lists. Driven by `scripts/process_dataset_truebones.py`
(stage 1) and `scripts/build_features.py` (stage 2). Between source and
preparation it also trims and filters clips, as the dataset's ``clips:`` group
asks (`src/poseydon/build/clips.py`, a registry of `ClipOp`).

## Limitations

- **A stage-1 change invalidates everything after it.** Nothing detects a
  stale corpus: rebuild stage 1, then stage 2, and retrain checkpoints trained
  on the old corpus.
- **A rest pose that is none of the source's clips is written as one.** Stage 1
  writes the prepared rest skeleton as the one-frame clip `rest_action` names:
  `clips/<Rig>/bind_pose.bvh` for a rig without `rest_pose:` (`data=unimate`), or
  `tpose_authored.bvh` for an authored T-pose, which has no vendor FBX. Like any
  one-frame clip it is left out of the index, and the `rest_frame` conditioner
  is its frame 0.
- **Truebones' clip index is the BVH-era clip set.** `data/truebones/fbx_clips.csv`
  pairs each vendor BVH with the FBX of the same motion (name match, plus 13
  odd names matched by frame count), each pair checked on every shared frame
  (1121 of 1144 within 1e-4 of the extent; the rest within 2e-3 on positions,
  or one frame shifted: SpiderG `attack7`). Left out because no FBX holds them
  outside the all-takes bundles or the pair is a different take: Fox `idle_1`,
  Lynx `idle4`, Elephant `take_001`, Monkey `b1die`, Centipede `take_001`. FBX-only
  clips (KingCobra `run`, `walk`, `walk fast`) are not added. A clip's length is
  the FBX's key range, which differs from its BVH on about 80 clips (Pteranodon's
  BVHs padded to 91 frames with a held pose, Bear's and Dog-2's cut short).
- **The index leaves out clips shorter than two frames.** Velocity features
  need two frames, so such a clip (each authored `__TPOSE_authored.bvh`) has
  no feature frame; it is still written to `clips/`, as the rig's rest pose,
  and stage 2 warns once per clip. Anything that lists clips from disk instead
  of the index sees it.
- **`fbx_cache` approximates Blender's armature.** Bones are `eSkeleton` nodes
  or skin-cluster-linked nodes plus non-bone nodes between them; the largest
  tree is kept. Not modelled: Blender's bone re-orientation and FBX unit scale
  (clips stay in the file's units and axes; a node's own world scale is taken
  off before its local transform is read, so Crow's armature-wide 5 no longer
  shrinks its bones against its root's travel, as it did until 2026-10-21); the joint sets, once
  pruned, equal UniMate's `joint_names.json` on all 74 rigs (`--check-against`).
  A node without a bind takes its frame-0 world as rest, so a bound child under
  an animated unbound node gets offsets that include that motion.
  Where a species' clips differ in joints, the largest group defines the rig
  and a clip missing one of its joints is left out (`FbxCache.dropped`).

- **`clips:` windows describe frames, not content.** A clip's recorded
  `frames/window` maps its prepared frames to source frames only while the
  ops are the ones that made it; in `windows` mode, a window an op drops is not
  recorded (only a clip with none left is). `face_axis` and the other clip-scoped
  stages are fitted once on the whole trimmed clip, not per window.
  `low_activity` and `jump` thresholds are UniMate's, measured on the prepared
  (centred, scaled, grounded) positions; nothing checks they still mean the
  same after a change to `prepare:`.

## Class diagram

```mermaid
classDiagram
  direction LR
  class BuildConfig {
    <<dataclass>>
    root, schema, out
    manifest(rig) SkeletonManifest
    transform(rig) RigTransform
  }
  class CorpusIndex {
    <<dataclass>>
    records: list
    query(...) list
    save(path)
    load(path)$
  }
  class ClipRecord { <<dataclass>> }
  class Source {
    <<abstract>>
    rigs(data) list
    actions(data, manifest) list
    rest(data, manifest) Animation
    rest_channels(data, manifest) tuple
    clip(data, manifest, action) SourceClip
    clips(data, manifest) list
    skin(data, manifest) dict
  }
  class SourceClip { <<dataclass>> }
  class ClipOp {
    <<abstract>>
    when
    rig(rest, rig) str
    windows(anim, clip_id) list
  }
  ClipOp <|-- HeadTrim
  ClipOp <|-- HalveFps
  ClipOp <|-- TrimStatic
  ClipOp <|-- MinFrames
  ClipOp <|-- MaxLength
  ClipOp <|-- LowActivity
  ClipOp <|-- Jump
  ClipOp <|-- JointRange
  Source <|-- BvhFiles
  Source <|-- FbxCache
  Source ..> SourceClip : clip(...)
  BuildConfig o-- PerRigDirectory : corpus
  BuildConfig o-- Humanize : names
  BuildConfig o-- BlockPolicy : normalize
  BuildConfig ..> RigTransform : transform(rig)
  BuildConfig ..> SkeletonManifest : manifest(rig)
  CorpusIndex o-- ClipRecord : one per clip
  cssClass "CorpusIndex,PerRigDirectory,Humanize,BvhFiles,FbxCache,HeadTrim,HalveFps,TrimStatic,MinFrames,MaxLength,LowActivity,Jump,JointRange" impl
  cssClass "BuildConfig,ClipRecord,SourceClip" data
  cssClass "BlockPolicy,SkeletonManifest,RigTransform" ext
```

`BuildConfig` names everything one build needs; `transform(rig)` reads the
`RigTransform` that [preparation](preparation.md) fitted and persisted as
`preproc.npz`. `SkeletonManifest` is in [core](../core/index.md),
`BlockPolicy` in [data](../data/index.md).

## Reference

::: poseydon.build
    options:
      members: false

::: poseydon.build.authority

::: poseydon.build.clips

::: poseydon.build.corpus

::: poseydon.build.index

::: poseydon.build.names

::: poseydon.build.pipeline

::: poseydon.build.sources

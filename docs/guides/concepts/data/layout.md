# Dataset layout

What is on disk, and what each file is for. The words used here are defined
in [Notation](notation.md).

```
data/
  _raw/truebones/source/<Rig>/     the licensed vendor files, untouched
  truebones/
    rigs/<Rig>/                    one folder per character
      manifest.yaml                hand-written: how to read this rig
      preproc.npz                  how the rig was prepared, and how to undo it
      skeleton.npz                 the skeleton the model sees
      stats.npz                    normalisation statistics
      report.json                  what preparation did, and why
    cache/                         derived from vendor files, rebuilt by tools
      joint_names_t5.npz           T5 embeddings of the joint names
      rigs/<Rig>/rest_mesh.npz     the vendor mesh on the rest pose, for grounding
    clips/<Rig>/<action>.npz       the prepared clip
    clips/<Rig>/<action>.bvh       the same, as BVH, to look at
    clips/<Rig>/<action>.yaml      which split the clip belongs to
    index.jsonl                    one line per clip
```

Only the hand-written files are in git: the manifests and the clip `.yaml`
files. What is done to every rig alike, from preparation to loading, is the
dataset config, `configs/data/truebones.yaml`. Everything else is rebuilt by
the two steps in [Building the dataset](../../how-to/tutorials/build-a-corpus.md);
`cache/` comes from `tools/build_joint_name_cache.py` and `tools/build_rest_mesh.py`, the latter needing the FBX SDK.

## The raw files

The Truebones vendor ships each motion twice: as a `.bvh` (the motion alone)
and as an `.fbx` (the motion on the skinned character). Preparation reads
only the BVH. The FBX is used at the very end, when generated motion is
written back onto the vendor's character, mesh and all. The raw files never
mix with the prepared ones.

## The manifest

`manifest.yaml` tells the pipeline what it can't work out alone. Joints are
named, never numbered, so a re-exported rig with reordered joints fails with
an error instead of silently using the wrong ones.

```yaml
skeleton: Dog
rest_pose: __Idle.bvh        # the clip whose first frame is the rest pose
promote_root: Bip01_Pelvis   # only for a rig rooted at a ground locator: the body's root
ground: joints               # only to override the dataset's ground mode for this rig
up_axis: +Y                  # the source's up axis (default +Y; a BVH doesn't record it)
facing:                      # left/right pairs used to find which way it faces
  hips:      {right: Bip01_R_Thigh, left: Bip01_L_Thigh}
  shoulders: {right: Bip01_R_UpperArm, left: Bip01_L_UpperArm}
foot_joints: [Bip01_R_Foot, Bip01_R_Toe0, ...]   # for foot contact
tags: [quadruped, mammal]
authority:                   # which file is trusted for what
  skeleton: bvh
  motion: bvh
  surface: fbx
```

What every Truebones rig shares lives in the dataset config instead: 30 fps
(`source.fps`), the preparation stages and their parameters (`prepare:`, the
target bone length among them), which joints the model may drop (`reduce:`,
with the still-joint cut-off, see
[features: reduction](../../../api/features/reduction.md)) and the
foot-contact thresholds (`labels.contact`).

## The per-rig records

| file | holds | used by |
|---|---|---|
| `preproc.npz` | the preparation chain it was built with and the parameters each stage fitted, including each clip's facing | export, to undo preparation |
| `skeleton.npz` | joint names, parents and offsets of the model view, and which joints were removed and how, the reduce rules, and the still joints with their constant rotation | training, sampling, and export, to put the removed joints back |
| `stats.npz` | per-joint, per-channel mean and standard deviation of the features | training and sampling, to normalise |
| `report.json` | checksums, the transforms applied, the topology, and any decisions and warnings | people, when something looks wrong |

## Leftovers from older builds

A corpus built before 2026-10-12 may still hold `mesh.npz`, `char-model.fbx`,
`char-full.fbx` and `clips/<Rig>/*.fbx`, and its `index.jsonl` may still name
them. Nothing reads them; there is no prepared FBX any more.

## Seeing every stage

`scripts/build_stage_samples.py` (in the `fbxsdk` container) writes one clip
per rig at each stage, `1_source`, `2_prepared`, `3_model` and
`4_roundtrip`, as both BVH and FBX, under `tmp/truebones/<Rig>/`, to open
side by side.

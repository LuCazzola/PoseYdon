# Dataset layout

What is on disk, and what each file is for. The words used here are defined
in [Notation](notation.md).

```
data/
  _raw/truebones/source/<Rig>/     the licensed vendor files, untouched
  truebones/
    rigs/_base.yaml                settings shared by every rig
    rigs/<Rig>/                    one folder per character
      manifest.yaml                hand-written: how to read this rig
      preproc.npz                  how the rig was prepared, and how to undo it
      skeleton.npz                 the skeleton the model sees
      stats.npz                    normalisation statistics
      report.json                  what preparation did, and why
    clips/<Rig>/<action>.npz       the prepared clip
    clips/<Rig>/<action>.bvh       the same, as BVH, to look at
    clips/<Rig>/<action>.yaml      which split the clip belongs to
    index.jsonl                    one line per clip
```

Only the hand-written files are in git: the manifests, `_base.yaml` and the
clip `.yaml` files. Everything else is rebuilt by the two steps in
[Building the dataset](../tutorials/build-a-corpus.md).

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
base: ../_base.yaml          # inherit the shared settings
skeleton: Dog
rest_pose: __Idle.bvh        # the clip whose first frame is the rest pose
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

`_base.yaml` holds what every Truebones rig shares: 30 fps, the target bone
length and the foot-contact thresholds. A rig overrides any of them by
declaring the key itself.

## The per-rig records

| file | holds | used by |
|---|---|---|
| `preproc.npz` | the parameters each preparation stage fitted, including each clip's facing | export, to undo preparation |
| `skeleton.npz` | joint names, parents and offsets of the model view, and which joints were removed and how | training, sampling, and export, to put the removed joints back |
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

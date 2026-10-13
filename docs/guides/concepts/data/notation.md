# Notation and conventions

The words, letters and axes the rest of the docs and the code use, in one
place.

## Words

| word | means |
|---|---|
| **rig** | one character's skeleton, named after the character: `Dog`, `Flamingo`. Also called a *skeleton* or *character* |
| **action** | one motion of a rig, named from its file: `attack`, `onelegbent` |
| **clip** | one action of one rig, written `<Rig>/<action>` (`Dog/attack`); in the clip index its id is `<Rig>__<action>` |
| **joint** | a node of the hierarchy. Joint 0 is the **root**; every other joint has one parent |
| **End Site** | a BVH leaf with an offset but no rotation of its own: the tip of a finger, an ear, a tail |
| **offset** | a joint's position relative to its parent in the rest pose; its length is the bone length |
| **rest pose** | the pose a rig's rotations are measured from, taken from one named clip (`rest_pose:` in the manifest) |
| **facing** | which way a character points, worked out from its left/right hip and shoulder joints |
| **heading** | the facing at one frame, as a rotation about the vertical axis |
| **corpus** | a prepared dataset: clips, per-rig records and an index |
| **manifest** | `manifest.yaml`, the hand-written description of a rig: facing joints, feet, rest pose |
| **recipe** | the record saved beside a checkpoint of what the model was trained with, checked again when sampling |
| **conditioner** | one named kind of extra input a model is given besides the motion, such as the rig's topology or its joint names |

## The three levels

The same motion exists at three levels, and most confusion comes from mixing
them up:

| level | joints | frame | where it lives |
|---|---|---|---|
| **source** | as the vendor authored them | as authored: the vendor's units, axes and placement | `data/_raw/truebones/source/<Rig>/` |
| **prepared** | the same as source | the shared canonical frame (below) | `data/truebones/clips/<Rig>/` |
| **model view** | the prepared joints minus those that carry nothing | canonical | features, `skeleton.npz` |

Preparation changes where things are and keeps the joints. Reduction removes
joints and keeps where things are. [The round trip](roundtrip.md) shows how
each is undone.

## The canonical frame

After preparation every character is in the same frame:

- **Y is up.** The ground is the plane `y = 0`, with the rest pose's lowest
  joint on it.
- **Facing +Z.** Each clip is turned so that on its first frame the character
  faces +Z.
- **Centred.** The root sits at the X/Z origin on the rest pose's first frame.
- **One size.** The rig is scaled so its mean bone length is 0.209, the
  reference implementation's average human bone length (`HML_AVG_BONELEN`,
  set in `data/truebones/rigs/_base.yaml`). Lengths have no physical unit
  after this.
- **Rigid.** Only the root moves through space; every other joint only
  rotates, so bone lengths never change.
- **30 frames per second**, the rate of every Truebones clip.

## Rotations

- **Quaternions are `(x, y, z, w)`**, with the scalar last, as in SciPy.
- **Rotations are local**: each joint's rotation is relative to its parent.
- **6D rotations** are two of the three basis vectors of a rotation matrix.
  The feature blocks take the first two **columns**, which is what the
  reference's stored data uses. The general helper,
  `matrix_to_rot6d` in `src/poseydon/core/rotations.py`, takes rows by
  default (PyTorch3D's convention). The two differ by a transpose, so mixing
  them gives you the inverse rotation without an error.

## Array shapes

| letter | means |
|---|---|
| `F` | frames, in one clip |
| `T` | frames, in one training window |
| `J` | joints |
| `D` | feature channels per joint |
| `B` | clips in a batch |

| array | shape | note |
|---|---|---|
| an animation's rotations | `(F, J, 4)` | quaternions |
| an animation's translations | `(F, J, 3)` | joint 0's is the root's world position |
| offsets | `(J, 3)` | |
| parents | `(J,)` | `-1` for the root; a parent always comes before its children |
| features | `(F, J, D)` | channels last |
| a `MotionBatch`'s `x` | `(B, J, D, T)` | channels at axis 2, **not** last |
| normalisation statistics | `(J, D)` | |

Because the channel axis moves, code reads a feature block by name, through
`FeatureSpec.take(x, "rot6d_joint", axis=...)`, never by slicing
`[..., 3:9]`. On a batch, `batch.block("rot6d_joint")` passes the axis for you.

## Features

The model sees each joint at each frame as a short vector made of named
blocks. The default list is
`[ric_pos, rot6d_joint, heading, local_vel, foot_contact]`, 15 channels per
joint:

| block | width | what it holds |
|---|---|---|
| `ric_pos` | 3 | the joint's position, seen from the root: as if the character stood at the origin facing +Z |
| `rot6d_joint` | 6 | the joint's own local rotation, in 6D; the root's is expressed relative to the heading |
| `heading` | 2 | `(cos, sin)` of the heading angle, on the root only; zero on every other joint |
| `local_vel` | 3 | how far the joint moved since the previous frame, seen from the root |
| `foot_contact` | 1 | 1 if the joint is a foot (listed in the manifest) that is low and nearly still, else 0 |

`local_vel` and `foot_contact` compare two frames, so they are one frame
shorter than the clip.

"Seen from the root" means the walk itself is removed, so the same step
looks the same whichever way and wherever the character walks. The list is
set in `configs/dataset/truebones.yaml` (`schema:`) and in `features:` of
`configs/train.yaml`; the two must agree.

### Rotation layouts

There are two ways to store rotations, picked by which block you name:

- **`rot6d_joint`** (the default): slot `j` holds joint `j`'s own rotation.
- **`rot6d_bone`**: slot `j` holds the rotation of `j`'s parent, the one that
  points the bone ending at `j`. This is the layout of HumanML3D and of the
  reference implementation, and it is kept for comparing against them. It
  replaces `heading` (the root's slot holds the heading instead).

Both recover every rotation that moves something. Switching between them
changes which joints the model view keeps, so it means rebuilding the
skeleton (`scripts/build_features.py` without `--stats-only`) and passing the
same `--layout` to `poseydon export`. Details:
[features](../../index/library/features.md#two-rotation-layouts).

## File extensions

| file | holds |
|---|---|
| `.bvh` | a motion as text: a hierarchy and one row of numbers per frame |
| `.fbx` | a character with mesh, skin and motion; read and written through the Autodesk FBX SDK |
| `.npz` | a prepared clip, as arrays |
| `.anim.npz` | a generated motion exactly as the model produced it; export from this |

The per-rig files (`preproc.npz`, `skeleton.npz`, `stats.npz`,
`report.json`) are described in [Dataset layout](layout.md).

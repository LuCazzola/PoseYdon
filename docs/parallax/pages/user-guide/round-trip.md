# The round trip

Code: `src/poseydon/build`, `src/poseydon/features/reduce.py`, `src/poseydon/export/rig.py`, `src/poseydon/io`

A user hands over a rig, the model generates on a canonical one, and the result has to
come back on the user's rig. Everything in preparation is shaped by that last step.

## Every preparation stage is invertible, and its fit is stored

Each stage in `src/poseydon/build/prepare.py` declares an `invert` beside its `apply`.
The parameters it fitted go to `rigs/<Rig>/preproc.npz` together with the chain itself,
so a corpus is inverted by the chain it was built with rather than by today's config.
The module docstring gives the reason: some fits cannot be recomputed. The facing
rotation comes from a clip's own frame 0, and once the file already faces +Z it is gone.

A stage that must follow another says so in `after`, and a chain in the wrong order is
refused when it is built.

## Rig scope and clip scope

A stage fits either once per **rig**, against the rest pose, with every clip reusing the
result, or once per **clip**. The docstring's reason for fitting most stages per rig is
that fitting per clip would ground a flying creature onto the floor and erase the height
difference between a crouch and a stand. In the Truebones chain
(`configs/data/truebones.yaml`), `face_axis` is the only per-clip stage: each clip's root
is turned so its own frame 0 faces +Z.

## Two stages on disk, each file with one writer

- **Stage 1** (`scripts/process_dataset_truebones.py`) reads raw clips through the
  dataset's `source:`, runs the `prepare:` chain and writes the prepared BVH and
  `preproc.npz`.
- **Stage 2** (`scripts/build_features.py`, passes in `src/poseydon/build/pipeline.py`)
  writes `skeleton.npz`, `stats.npz`, each clip's `.npz` and the index.

The pipeline's docstring says only `stats.npz` depends on the feature schema. In the
code, `skeleton.npz` does too: the reduction is built for the schema's rotation layout
(`rot6d_joint` or `rot6d_bone`). So a change of block list re-runs the stats pass, but a
change of rotation layout needs the skeleton pass as well.

## Three spaces, and two kinds of joint removal

A clip lives in one of three spaces (`src/poseydon/export/rig.py`):

| Space | Joints | Frame |
|---|---|---|
| model view | reduced | canonical |
| prepared | every joint left after cleanup | canonical |
| source | the user's rig as supplied | the user's |

`RigRecord.expand` goes from the model view to prepared, and `RigRecord.restore` goes
from prepared to source.

Joints leave at two different points, and the difference matters:

- **Rig cleanup** (`src/poseydon/build/cleanup.py`, for example `prune_unskinned`) runs in
  stage 1. It needs every clip of the rig plus its skin weights, and what it removes is
  gone from the prepared files. On the way back it is restored at its bind pose only.
  Of the shipped datasets, only `configs/data/unimate.yaml` uses it.
- **Reduction** (`src/poseydon/features/reduce.py`, the `reduce:` list) is a lens applied
  at feature time and never baked into the corpus. It drops End Sites, still subtrees
  and a zero-length sole child of the root, and it puts each of them back on export.

The reduction docstring explains the root fold. Folding into a parent loses that
parent's world rotation, and the root is the one parent whose rotation drives no skin.
The fold also gives every skeleton the same one-joint root (the 46 Biped rigs carry
`Hips` above a coincident `Bip01_Pelvis`). The owner's rationale is quoted there.

## What the trip loses by design

`tests/test_roundtrip.py` lists what it deliberately does not assert:

- per-joint translation, because `enforce_rigid` keeps one rigid geometry per rig;
- the locator chain above a promoted root;
- the root's own rotation where a joint was folded into it;
- the motion of joints that `prune_unskinned` removed.

Everything else is asserted: the structure, the root trajectory, every observable world
rotation, and rigid bones.

## FBX through the Autodesk SDK, not Blender

`src/poseydon/io/fbx.py` reads and writes FBX with the Autodesk FBX SDK. Its docstring
gives the reason. Blender's exporter converts rotations to float32 XYZ Euler with a
"compatible Euler" step that, near gimbal lock, writes a different rotation (4.76° off on
Flamingo's root). The SDK is licensed and fetched by each user
(`scripts/fetch_fbx_sdk.sh`). Its image is x86-64 only, so on an ARM host it runs under
emulation. Blender stays only for rendering (`scripts/render_fbx.py`) and as the
verifier's second reader.

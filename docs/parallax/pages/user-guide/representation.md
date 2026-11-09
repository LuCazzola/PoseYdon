# Motion representation

Code: `src/poseydon/core`, `src/poseydon/features`, `src/poseydon/data`

## Parsing is lossless; rigidity is an explicit, checked step

`src/poseydon/core/animation.py` holds two types. `Animation` is what a motion file
declares, where any joint may translate (every 3ds Max Biped rig in Truebones gives all
of its joints six channels). `RigidBodyAnimation` adds the promise that only the root
translates. Features, IK, augmentation and the trained representation all ask for the
rigid type, so parsing never discards data, and the rigid assumption becomes a
conversion (`Animation.as_rigid_body`) that can fail.

## Features are named blocks with a declared layout

A feature is a pure function of an aligned `RigidBodyAnimation` that produces one named
block per joint (`src/poseydon/features/base.py`). The default blocks are listed in
`configs/features/default.yaml`. `FeatureSpec` (`src/poseydon/core/spec.py`) names each
block's slice, so a loss asks for `spec.slice("rot6d_joint")` and fails loudly on a
representation without rotations, instead of reading whatever sits at `[..., 3:9]`.

The corpus stores the canonical animation, not features, so adding a representation
means one registered class and a stage-2 rebuild: `stats.npz` is checked against the
spec, and a change of rotation layout also changes `skeleton.npz` (see
[The round trip](round-trip.md#two-stages-on-disk-each-file-with-one-writer)).

## Statistics follow the active spec and are invertible

`Normalizer` (`src/poseydon/data/normalize.py`) computes per-skeleton statistics for
whichever blocks are active, and can undo them. Its docstring gives the reason:
structural losses such as geodesic and foot skate have to be measured in raw space,
because a per-channel divide by std does not preserve rotations.

## Windowing happens at load time

Clips stay whole on disk, and a window policy (`src/poseydon/data/window.py`) picks the
frames when a clip is read. The docstring contrasts this with the reference, which cuts
fixed 200-frame chunks on disk. There, a motion is split at boundaries the model never
sees across, and the two halves can land in different splits. Here, window length is a
config edit (`loader.window`).

Sampling across rigs is balanced by default (`loader.balanced`, in
`src/poseydon/data/sampler.py`). Truebones has 4 to 70 clips per rig, so uniform
sampling over clips would be dominated by a few rigs.

## Getting a skeleton back is a choice

The representation is redundant: it carries both rotations and positions. On real data
they agree; on model output they can disagree by more than three bone lengths. So
`src/poseydon/features/reconstruct.py` offers three readings: `fk` (forward kinematics
from rotations), `positions` (raw positions) and `positions_ik` (an IK solve onto the
positions). Each entry point picks differently: `poseydon sample` reads `reconstruct:` in
`configs/sample.yaml` (default `fk`); `poseydon retarget` takes the value recorded in the
checkpoint's recipe (`src/poseydon/cli.py`); `RetargetValidation` passes none, so
`run_retarget` uses its default (`src/poseydon/training/validation.py`).

Recovery from rotations is exact rather than fitted (`src/poseydon/features/recover.py`).
The IK solver (`src/poseydon/solvers`) is for the cases that genuinely need one.

## Conventions

Quaternions are scalar-last `(x, y, z, w)` (`src/poseydon/core/rotations.py`). The 6D
feature blocks use the first two columns of the matrix, the reference's layout
(`src/poseydon/features/kinematics.py`). Joints are referenced by name, never by
index, so a re-exported rig fails loudly instead of silently mirroring the character
(`src/poseydon/core/skeleton.py`).

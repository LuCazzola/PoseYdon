# io

`src/poseydon/io/`, library. File formats in and out: BVH and FBX to and from
`Animation`, plus shot framing and stick-figure rendering.

## Purpose

`io` owns the boundary between files and `core.animation.Animation`. It reads
exactly what a file declares and writes an animation back out.

It does **not** prepare, align or reduce anything. Those are [build](build.md),
[ingest](ingest.md) and [features](features.md). It also doesn't decide which
representation to export; that's [export](export.md).

## Key files and types

| file | read it for |
|---|---|
| `src/poseydon/io/bvh.py` | `BVH`: `read`, `to_animation`, `from_animation`, `write`; `BvhParseError` |
| `src/poseydon/io/fbx.py` | `FBX`: `read`, `names` / `parents` (every node), `world()` (per-frame world matrices, in the file's own units and axes), `from_animation(anim, calibration)` (the clip on the calibration's vendor asset, mesh included; the calibration must match the clip's space), `animate()` (re-key named nodes from world matrices, keeping mesh and skin; `space=` moves the whole scene), `write`. There is no bare-skeleton writer: every FBX is written onto a vendor asset. **Autodesk FBX SDK only** (`fbxsdk` container). |
| `src/poseydon/io/framing.py` | `camera()` and `to_view()`: one framing implementation shared by both renderers |
| `src/poseydon/io/render.py` | `render_skeleton` (MP4) and `render_frame` (PNG) stick figures, via matplotlib |

## How it connects

Consumes `core.animation`, `core.rotations` and `core.kinematics`.

Produces `Animation` objects for [build](build.md) and [ingest](ingest.md), and
files for [export](export.md), the CLI (`sample --render`), training's retarget
validation and `ops/retarget_run.py`. `framing.camera` is also used by
`scripts/render_fbx.py`, the skinned EEVEE renderer, so both renders of a clip
frame the same shot.

Called by: build, ingest, export, ops, training, `cli.py`, most scripts and
nearly every probe in `tools/`.

## Why it is shaped this way

- **Lossless, total BVH reading.** BVH allows position channels on any joint
  and exporters use them, so the parser records whatever the file declares. The
  rigid-bone assumption is applied later, by an explicit
  `Animation.as_rigid_body()`. Source: `src/poseydon/io/bvh.py` docstring;
  commit `fede016`.
- **End Sites are joints.** They carry an offset and a name but no channels,
  which gives every leaf a real direction and lets the writer round-trip End
  Site names. Source: `src/poseydon/io/bvh.py` docstring; commits `bbc6f35`,
  `39c2460`.
- **Vectorized MOTION parsing and direct Euler composition**, because that's
  where a naive parser spends its time (2.3x faster than the reference's
  `Motion`). Source: commit `583ea5c`.
- **FBX goes through the Autodesk FBX SDK, not Blender.** Blender's FBX
  exporter converts rotations to Euler in float32 with a "compatible Euler"
  step that, near gimbal lock, returns a different rotation: 4.76 degrees off
  its own input matrix on Flamingo's root. `animate()` solves Euler angles
  itself in float64 and keys every frame; the SDK only stores them. Round trips
  are exact to the format's float32 keys (5.5e-8 of extent for a skeleton,
  1.9e-7 re-keying a vendor asset). Source: `src/poseydon/io/fbx.py`
  docstring.
- **Native space only.** `FBX` never converts units or axes; mapping a file onto
  a BVH is a calibration owned by [export](export.md).
- **Keys on whole frames.** `FbxTime.SetSecondDouble` is inexact (2.2 s became
  2.1999978 s), and an evaluator a few microseconds off a key interpolates
  instead of hitting it, which cost 1.1e-5 of extent. Times are set with
  `SetFrame`. Source: `_frame` in `src/poseydon/io/fbx.py`.
- **Framing split out of rendering.** `render.py` needs matplotlib at import,
  and the Blender `fbx` container has none. Copying the camera into the Blender
  renderer instead is what let a camera regression pass the whole suite once.
  Source: `src/poseydon/io/framing.py` docstring; commit `82d8e87`.
- **Its own stick-figure renderer.** The reference's plot script calls an API
  removed in matplotlib 3.8. Source: `src/poseydon/io/render.py` docstring.

## Gotchas

- **`poseydon.io.fbx` imports only inside the `fbxsdk` container.** Never
  import it at module scope from code the `test` image runs. The CLI and the
  exporter import it lazily for that reason.
- **Supported FBX transforms are a subset.** XYZ rotation order, no pre/post
  rotation, offsets or pivots: every node of every Truebones file. Anything
  else is refused, not mis-read, because no file here could check that code.
- **Axes in rendering.** Blender is Z-up; Truebones, BVH and PoseYdon are Y-up.
  In `framing.py` the vertical axis is the *third* component (matplotlib's and
  Blender's drawing frame); motion data is never in that frame.
- **Compare BVH and FBX in world space, never by raw rotations.** Each format
  stores a joint's rotation in its own frame, so they differ by a constant
  per-joint offset; `FbxCalibration` in [export](export.md) measures it.
- **The BVH writer follows the hierarchy and the declared channel layout.**
  MOTION columns are written in the depth-first order the HIERARCHY is
  emitted in (`_depth_first`), not joint-index order, and a complete channel
  declaration is kept verbatim, rotations-before-positions interleave
  included (`BVH._layout`). Both were found in the 2026-09-28 audit. An End
  Site gets a `#name:` comment only where `read` would derive a different
  name.
- **An FBX's frame rate must be explicit.** The SDK's "default" time mode
  reads as 30 fps in the SDK and 25 fps in Blender, so a clip written with it
  plays at the wrong speed in Blender. `FBX.animate` writes the explicit mode
  (`_time_mode`) and `FBX.fps` refuses a file that declares none.
- **FBX keys are float32.** That bounds every FBX round trip at about 1e-7 of
  the extent; it is the format, not a bug.

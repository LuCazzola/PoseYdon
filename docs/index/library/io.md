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
| `src/poseydon/io/fbx.py` | `FBX`: `read`, `to_animation`, `from_animation` / `from_bare_animation`, `apply_animation`, `prepare_model_topology`, `write`, `write_mesh_npz`, and the facing and scale helpers the FBX preprocessing pass uses. **Blender only.** |
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
- **FBX goes through Blender.** No usable pure-Python FBX reader exists. The
  module imports `bpy` and nothing else in `io` imports it, so the rest of the
  package works without Blender. Source: `src/poseydon/io/fbx.py` docstring.
- **`automatic_bone_orientation` on import, and not the "connect" options.**
  Pointing bones at their children fixes the fragmented look without moving a
  joint. `force_connect_children` moved joints by up to 66% of the character's
  extent, and `ignore_leaf_bones` deleted 7 joints, so neither is used. Source:
  `src/poseydon/io/fbx.py` docstring.
- **Framing split out of rendering.** `render.py` needs matplotlib at import,
  and the `fbx` container has none. Copying the camera into the Blender
  renderer instead is what let a camera regression pass the whole suite once.
  Source: `src/poseydon/io/framing.py` docstring; commit `82d8e87`.
- **Its own stick-figure renderer.** The reference's plot script calls an API
  removed in matplotlib 3.8. Source: `src/poseydon/io/render.py` docstring.

## Gotchas

- **`poseydon.io.fbx` imports only inside the `fbx` container.** Never import
  it at module scope from code the `test` image runs. The CLI imports it lazily
  for that reason.
- **Axes.** Blender is Z-up; Truebones, BVH and PoseYdon are Y-up. PoseYdon's
  +Z forward is Blender's +Y. In `framing.py` the vertical axis is the *third*
  component (matplotlib's and Blender's drawing frame); motion data is never
  in that frame.
- **Compare BVH and FBX in world space, never by raw quaternions.** Each format
  stores a joint's rotation in its own local bone frame, so they differ by a
  constant per-joint offset. `scripts/check_fbx_coherence.py` and
  `tests/build/test_bvh_fbx_agreement.py` do it that way.
- **FBX bones don't all connect, and leaf bones point oddly.** This is cosmetic,
  because a bone's tail isn't stored in the format. Joint positions are exact.
  Don't file it as a bug.
- **Open in committed code: the BVH writer ignores hierarchy order and channel
  interleave.** It writes MOTION columns in joint-index order rather than the
  hierarchy's DFS order, and ignores a joint's declared
  rotations-before-positions interleave. Both were found in the 2026-09-28
  audit. Fixes exist only as uncommitted work as of 2026-10-12. Update this
  bullet when they land.
- **`FBX.write` settings are load-bearing.** If baking keeps constant curves, the
  exporter writes a keyframed object transform, and the armature then snaps back
  on every frame change. Read `FBX.write` before changing export options.

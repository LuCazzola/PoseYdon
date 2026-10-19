# io

The boundary between files and `Animation`
([core](../core/index.md)): BVH and FBX in and out, plus the shot framing and
stick-figure renderer. A reader records exactly what the file declares: any
joint may carry position channels, and End Sites are joints. Making bones
rigid is a later, explicit `Animation.as_rigid_body()`. Preparing, aligning or
reducing is not done here ([build](../build/index.md),
[ingest](../ingest/index.md), [features](../features/index.md)), and neither
is choosing what to export ([export](../export/index.md)).

`FBX` goes through the Autodesk FBX SDK, so `poseydon.io.fbx` imports only
inside the `fbxsdk` container: import it lazily from anything the `test` image
runs. It works in each file's native units and axes; mapping an FBX onto a BVH
is the calibration [export](../export/index.md) owns. `framing.camera` is
shared with `scripts/render_fbx.py`, so the stick figure and the skinned render
frame the same shot.

## Limitations

- **FBX supports a subset of node transforms.** XYZ rotation order and no
  pre/post rotations, offsets or pivots: what every Truebones file uses.
  Anything else is refused with a `ValueError`, not read.
- **FBX keys are float32.** Every FBX round trip is bounded at about 1e-7 of
  the extent; that is the format.
- **BVH and FBX rotations are not comparable directly.** Each stores a joint's
  rotation in its own frame, so they differ by a constant per-joint offset.
  Compare in world space; `FbxCalibration` ([export](../export/index.md))
  measures the offset.
- **`FBX.skinned_vertices` is plain linear blend skinning of skinned meshes
  only.** Weights are normalised per vertex, unweighted vertices and meshes with
  no skin deformer are left out, and dual-quaternion or blend-shape deformers
  are ignored; none was checked on a Truebones file, which uses neither. Its
  one user is `tools/build_rest_mesh.py`.
- **The read-only methods accept any transform model.** `world`, `bind_matrices`,
  `key_range`, `up_axis`, `node_types` and `skins` go through the SDK's
  evaluator or read the file directly and never call the write path's transform
  check, so a file with `PreRotation` (KingCobra) reads fine; only `animate`
  refuses it. `key_range` is over the first take's curves only. `bind_matrices`
  is NaN for a node that is neither a skin link nor in a bind pose, and a file
  with several bind poses takes the first that names the node. `skins` indexes
  weights by node, so a skin whose link is not a scene node is dropped.
  `scripts/build_fbx_cache.py` is their one user.

## Reference

::: poseydon.io
    options:
      members: false

::: poseydon.io.bvh

::: poseydon.io.fbx

::: poseydon.io.framing

::: poseydon.io.render

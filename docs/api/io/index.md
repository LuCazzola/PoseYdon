# io

The boundary between files and `Animation`
([core](../core/index.md)): BVH and FBX in and out, plus the shot framing and
stick-figure renderer. A reader records exactly what the file declares: any
joint may carry position channels, and End Sites are joints. Making bones
rigid is a later, explicit `Animation.as_rigid_body()`. A BVH leaf has no
channels, so a leaf joint's rotation is lost in a plain file; `BVH.from_animation(...,
leaf_rotations=True)` (stage 1, always: a rig read from FBX has no End Sites, so its tips are real leaves) writes it as a JOINT with a
zero-offset `End Site #synthetic` below, which `BVH.read` drops again. Preparing, aligning or
reducing is not done here ([build](../build/index.md),
[ingest](../ingest/index.md), [features](../features/index.md)), and neither
is choosing what to export ([export](../export/index.md)).

`FBX` goes through the Autodesk FBX SDK, so `poseydon.io.fbx` imports only
inside the `fbxsdk` container: import it lazily from anything the `test` image
runs. It works in each file's native units and axes: a clip read from an FBX
is written back onto it by node name (`FBX.from_animation`), and `FBX.pose`
holds the file in any pose, on its own bind and mesh (the rest asset
[export](../export/index.md) writes onto). `framing.camera` is
shared with `scripts/render_fbx.py`, so the stick figure and the skinned render
frame the same shot. `FBX.skin_data` reads each skin deformer whole (`SkinData`:
every control point, triangulated faces, per-cluster link, bind matrices and
weights), so a skin can be recomputed without the SDK
(`tools/verify/skin.py`), and `FBX.inventory` lists what a file holds (`Inventory`:
nodes, meshes with their skinned flag and counts, clusters, unweighted control
points, materials, textures, units, up axis, fps, frame range, and what
`skinned_vertices` does not model) for `tools/verify/checks.py`.

## Limitations

- **FBX supports a subset of node transforms.** XYZ rotation order and no
  pre/post rotations, offsets or pivots: what every Truebones file uses.
  Anything else is refused with a `ValueError`, not read.
- **FBX keys are float32.** Every FBX round trip is bounded at about 1e-7 of
  the extent; that is the format.
- **BVH and FBX rotations are not comparable directly.** Each stores a joint's
  rotation in its own frame, so a vendor BVH and FBX of one clip differ by a
  constant per-joint offset. Compare in world space; `tools/build_rest_fbx.py`
  measures the offset where it carries a BVH rest pose onto an FBX.
- **`FBX.from_animation` drives nodes by name only.** A joint with no node of its
  name is ignored, and a node with no joint keeps its local transform from the
  asset's first frame; each driven node takes its world scale from the asset's
  first frame, constant per node (the scale an `Animation` cannot carry).
- **`FBX.skinned_vertices` is plain linear blend skinning of skinned meshes
  only.** Weights are normalised per vertex, unweighted vertices and meshes with
  no skin deformer are left out, and dual-quaternion or blend-shape deformers
  are ignored; none was checked on a Truebones file, which uses neither. Its
  user is `tools/build_rest_fbx.py` (the rest mesh). `FBX.skin_data` keeps every
  control point and fans polygons into triangles; `deform` in `tools/verify/skin.py`
  applies the same maths, with the same normalisation, and leaves an unweighted
  vertex at its rest position.
- **`FBX.inventory` reports, it does not refuse.** `unsupported` names
  dual-quaternion or blended skinning, a cluster link mode other than
  normalize, blend shapes and non-zero geometric transforms; none was found on
  a Truebones file. A deformer cut off its geometry is not a skin (`_skins`
  skips it), and a truncated file can still open as an empty scene, so a reader
  of `inventory` sees "no nodes" rather than an error.
- **The read-only methods accept any transform model.** `world`, `bind_matrices`,
  `key_range`, `up_axis`, `node_types` and `skins` go through the SDK's
  evaluator or read the file directly and never call the write path's transform
  check, so a file with `PreRotation` (KingCobra) reads fine; only `animate`
  refuses it. `key_range` is over the first take's curves only. `bind_matrices`
  is NaN for a node that is neither a skin link nor in a bind pose, and a file
  with several bind poses takes the first that names the node. `skins` indexes
  weights by node, so a skin whose link is not a scene node is dropped.
  `scripts/build_fbx_cache.py` is their one user.
- **The SDK's criteria-filtered `GetSrcObject(criteria, n)` is not a strict
  filter.** On 349 Truebones clips it returned objects of other types (an
  `FbxNode` where an `FbxAnimCurveNode` was asked for); cause not established.
  The readers (`key_range`, `skins`, `bind_matrices`, `skinned_vertices`, `skin_data`)
  filter by type, so a new reader of this kind must too.

## Reference

::: poseydon.io
    options:
      members: false

::: poseydon.io.bvh

::: poseydon.io.fbx

::: poseydon.io.framing

::: poseydon.io.render

# Data round trip: from the user's rig and back onto it

A character goes in as the vendor or user authored it. It comes back out, after
the model has generated or retargeted motion for it, **on that same
character**: the same joint names, hierarchy, units, channel layout, bind pose
and, for FBX, mesh. Everything between the two ends is allowed to be canonical
and reduced, because every step that changes the character records how to
undo itself.

That promise is why preparation is a chain of invertible stages rather than a
preprocessing script, and why the corpus keeps every joint even though the
model sees fewer. Without it, PoseYdon would be a benchmark rather than a tool
anyone can hand a rig to.

```mermaid
flowchart LR
  S[source<br/>as authored] -- prepare --> P[prepared<br/>canonical, all joints]
  P -- reduce --> M[model character<br/>degenerate joints removed]
  M -- features --> N((model))
  N -- features --> G[generated motion<br/>model character]
  G -- expand --> P2[prepared]
  P2 -- unprepare --> S2[source<br/>the user's rig]
```

## Three levels

| level | joints | geometry | lives in |
|---|---|---|---|
| **source** | as authored | as authored | `data/_raw/truebones/<Rig>/` |
| **prepared** | same as source | bind removed, root promoted, faced +Z, rigid, centred, scaled, grounded | `data/truebones/clips/<Rig>/` |
| **model** | zero-length bones removed | as prepared | features, `skeleton.npz` |

Preparation changes geometry and keeps structure. Reduction changes structure
and keeps geometry. Keeping them separate is what makes each one exactly
invertible on its own.

## Going in: preparation

`prepare_chain()` in `src/poseydon/build/prepare.py` is the single definition
of the stage order, used both for preparing and for undoing. Each stage
implements `fit`, `apply` and `invert`, and what it fitted is saved to
`rigs/<Rig>/preproc.npz`. Inversion reads those saved parameters and never
refits.

| # | stage | scope | what it does | what undoing it needs |
|---|---|---|---|---|
| 1 | `RestRelative` | rig | removes the per-joint bind rotation Biped exporters bake into every clip, so identity means the rest pose | the rest pose's bind rotations |
| 2 | `PromoteRoot` | rig | moves the root off a ground locator onto the first body joint (14 of 73 Truebones rigs) | the removed chain above the new root |
| 3 | `FaceAxis` | **clip** | turns the clip so its frame-0 facing points along +Z | that clip's yaw, which can't be recovered from the prepared file |
| 4 | `EnforceRigid` | rig | drops non-root translation so bone lengths are constant | the source's channel declaration, which the writer restores |
| 5 | `CentreXZ` | rig | puts the root at the XZ origin on the rest pose's first frame | the offset |
| 6 | `ScaleToMeanBoneLength` | rig | scales so the mean bone length matches the reference's SMPL mean | the scale factor |
| 7 | `PutOnGround` | rig | puts the rest pose's lowest joint at y = 0 | the offset |

**Rig scope versus clip scope** decides what survives. Rig stages fit once on
the rest pose and are shared by every clip of the character. Fitting
grounding per clip would put a flying creature on the floor and erase the
height difference between a crouch and a stand. `FaceAxis` is the only clip
stage, so it is the only parameter recorded per clip.

You run preparation with `scripts/process_dataset_truebones.py`. It runs on
the BVH only; FBX enters at the two ends, at export. Each rig also gets a
`report.json` recording what was done and why. See [build](../index/library/build.md).

## The model's view: reduction

A zero-length bone moves nothing. Truebones rigs carry ten or more per
character: every End Site, plus the odd dummy. `src/poseydon/features/reduce.py`
handles them in two ways:

- a zero-offset **leaf** is deleted outright, losing nothing;
- a zero-offset **internal** joint is folded into its parent only when it is
  that parent's only child. Otherwise folding would turn its siblings, so the
  joint is kept.

Reduction is applied **when features are read, never written into the
corpus**. Skin weights are indexed by the original joint order, so the corpus
has to keep every joint for a generated clip to land back on the rig. The
recorded `JointReduction` is replayed in reverse by `invert_reduction`. See
[features](../index/library/features.md).

## Coming back: export

`CharacterExporter` in `src/poseydon/export/character.py` takes a model-space
animation and writes one of two representations, each as BVH or FBX:

| representation | joints | frame | BVH | FBX |
|---|---|---|---|---|
| **prepared** | exactly the model's | canonical (+Z, scaled, centred, grounded) | the animation | a bare skeleton |
| **original** | the source's own, after expansion | source units, frame, names, channels | the source's `CHANNELS` block | the vendor's own asset: nodes, mesh, skin, materials |

Only the original representation expands the reduction and inverts
preparation. A prepared export is the model character exactly as the model
sees it.

**Original FBX goes through a calibration.** The vendor ships the same clip as
BVH and as FBX, and the two are the same motion under one similarity (units,
axes, placement) and a constant per-joint change of frame. The exporter
measures both from one such pair and checks them on every frame, then writes
the restored clip onto that FBX through the Autodesk FBX SDK. Nothing assumes
the FBX bind pose equals the BVH rest pose, so all 73 Truebones rigs export;
the old Blender path refused 13. See [export](../index/library/export.md).

`poseydon export` takes a generated clip as `.bvh` or `.npz`. BVH output runs
anywhere; FBX output needs the `fbxsdk` container.

```bash
docker compose run --rm test python -m poseydon.cli export samples/Scorpion__clip.bvh \
    --rig Scorpion --representation original --format bvh --out scorpion.bvh
docker compose run --rm fbxsdk python -m poseydon.cli export samples/Scorpion__clip.bvh \
    --rig Scorpion --representation original --format fbx --out scorpion.fbx
```

**Facing.** A generated clip never had a facing of its own. With no
`--facing-from`, the exporter restores the **rig's** facing, the one recorded
for the rest pose. With `--facing-from <action>`, it restores that recorded
clip's yaw instead. If the rig's facing was never recorded, export raises an
error rather than substituting an identity rotation. That identity fallback
was removed because it moved reinserted roots by 22–167% of the rig's extent.

**Refusals.** Export fails loudly, never silently, when the generated topology
doesn't match `skeleton.npz`, when a transform is missing, or when no vendor
BVH/FBX pair of one clip agrees well enough to calibrate.

## What it loses on purpose

The round trip is exact wherever the contract allows, and lossy in a few
places by design. All numbers below come from the 2026-09-28 audit.

- **Per-joint translation** that the source animated is dropped by
  `EnforceRigid`. That is real motion, roughly 10% of skeleton size on raw
  Truebones, and it is the price of a rigid skeleton.
- **One rigid geometry per rig.** Per-clip bone vectors are replaced by the
  rest clip's. On some attack clips joints come back up to 15% (Dog) and 25%
  (Horse) of the extent away from the raw source. About 2.9% of rigid
  joint-clips come back with a different constant bone vector, mostly props
  and locators.
- **Removed locators above the root** come back at a constant pose.
- **Twist on a reduced joint is gone.** For example, Bear's zero-offset but
  skinned forearm-twist bones leave the forearm skin up to 5.5% of the mesh
  extent off.

## How it's verified

| check | what it proves |
|---|---|
| `tests/test_roundtrip.py::test_round_trip` | for every prepared clip of a dataset: features rebuild the model view, and `original_bvh` written to disk matches the **raw source file** in names, hierarchy, `CHANNELS`, frames, offsets, root trajectory and every observable world rotation, with rigid bones |
| `tests/test_roundtrip.py::test_round_trip_fbx` | the rig's vendor FBX and raw BVH calibrate to within 1e-4 of the extent; `original_fbx` written to disk holds exactly the restored clip on the vendor's node hierarchy; prepared and model skeleton FBX read back as themselves (`fbxsdk` container) |
| `scripts/build_stage_samples.py` | one clip per rig at every stage (source, prepared, model, roundtrip), as BVH and FBX, under `tmp/truebones/<Rig>/`, to open side by side |

```bash
docker compose run --rm -T test python -m pytest -q tests/test_roundtrip.py --dataset truebones -k Camel
docker compose run --rm -T fbxsdk python -m pytest -q tests/test_roundtrip.py --dataset truebones -k Camel
```

When checking a round trip by hand, compare against an **independent**
reference: Blender's stock BVH importer, the vendor's own per-action FBX, or
the raw file text. A reference built from the pipeline's own formulas or
manifest facing pairs agrees with the pipeline by construction and proves
nothing.

## Open issues

As of 2026-10-12, this is not fixed. Update this section when it lands.

- `poseydon sample` and retargeting borrow the target rig's conditioning from
  its first-indexed clip, not its rest clip. Because `FaceAxis` rotates
  offsets per clip, exporting with the rig's facing can then come back
  rotated or distorted.

The `PromoteRoot` joint-index fix and the BVH writer's depth-first and
channel-layout fixes landed on 2026-10-12; see [build](../index/library/build.md)
and [io](../index/library/io.md).

## Where to read further

- [build](../index/library/build.md): the stages, `RigTransform`,
  `preproc.npz`
- [features](../index/library/features.md): reduction and expansion
- [export](../index/library/export.md): `CharacterExporter`
- [io](../index/library/io.md): what BVH and FBX can and can't carry

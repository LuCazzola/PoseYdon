# Character-aware motion export and corpus layout — design

Date: 2026-09-19

Extends `2026-09-06-data-pipeline-and-training-parity-design.md`. This design
corrects the unfinished raw-data layout migration and completes the mesh side
of the representation pipeline so a model-facing `Animation` can be exported
either in canonical prepared space or back on the character asset supplied by
the dataset.

## Problem

PoseYdon has an invertible BVH motion path, but its public export surface does
not express the two representations a user needs:

1. the reduced, canonical character representation seen by the model; and
2. the original dataset representation, after joint expansion and inverse
   preparation.

`BVH.from_animation(anim)` writes the first representation because generated
animations carry the reduced model topology in canonical coordinates.
`scripts/anim_2_fbx.py` partially implements the second representation, but it
is a script rather than a reusable character-aware API. Its default FBX path
builds a bare armature, while `--skin` follows a separate route that loads a
raw source asset.

The mesh path is less complete. `data/truebones/mesh/` contains geometry-only
OBJ files. Only six rigs have `rigs/<Rig>/mesh.npz`, those arrays have not been
transported through root promotion or model joint reduction, and they cannot
preserve materials or textures. Consequently there is no model-ready skinned
asset whose hierarchy exactly matches the animation supplied to the model.

The corpus layout also contradicts its intended ownership boundaries:

- `data/_raw/Truebones-Z_OO/` exists but is empty;
- preparation still reads the 2.9 GB raw corpus from
  `data/truebones/source/`;
- `data/truebones/bvh/`, `fbx/`, `mesh/`, and `reference/` are legacy trees
  with overlapping or incompatible representations; and
- `data/truebones/` therefore mixes irreplaceable input with reproducible
  prepared output.

This ambiguity caused the visual-debug bundle to compare a round trip derived
from `source/` against files copied from `bvh/` and `fbx/`. All 73 BVH pairs
share names and counts but differ in their offset blocks; all 73 selected FBX
pairs differ byte-for-byte. The export was internally consistent with the live
pipeline and externally compared against the wrong source representation.

## Authority policy and preparation audit amendment

This section supersedes any narrower authority, prepared-FBX, character-asset,
or build-ownership statement later in this document.

Authority is selected independently for the model skeleton, primary motion, and
surface deformation. A dataset manifest declares:

```yaml
authority:
  skeleton: bvh       # bvh | fbx
  motion: bvh         # bvh | fbx
  surface: fbx
  auxiliary_joints: bake  # exclude | retain | bake
```

A registration command may ask the user for these choices once. Normal
preparation is non-interactive and deterministic. Per-character overrides must
be explicit in the manifest. Truebones defaults to BVH skeleton and motion
authority, FBX surface authority, and `bake` auxiliary handling.

The pipeline distinguishes three skeletons:

- **source skeleton:** the exact authoritative raw structure;
- **model skeleton:** the canonical topology supplied to feature extraction and
  model training; and
- **presentation skeleton:** the full mesh-bearing FBX deformation rig driven
  from model motion.

A strict prepared BVH and strict prepared FBX share the exact model topology.
The strict FBX is a topology diagnostic and may have audited deformation loss.
A separate canonical `char-full.fbx` retains the richer deformation rig and
is the preferred visual/deformation asset. Original FBX drives the untouched
raw presentation rig in its original frame.

Auxiliary joints follow an explicit policy. `retain` includes them in the
model. `exclude` holds or removes them and records the loss. `bake` excludes
them from the model but reconstructs them with a named deterministic driver,
constraint, or secondary-motion rule. Unweighted helpers may be removed.
Rigidly redundant weighted joints may be collapsed only after deformation
validation. Independently animated weighted joints may never be silently
weight-folded; without an explicit retain, exclude, or bake decision,
preparation fails.

Each rig therefore owns:

```text
rigs/<Rig>/
├── char-model.fbx
├── char-full.fbx
└── report.json
```

`char-model.fbx` has exactly the model names, order, parents, and
canonical bind transforms. `char-full.fbx` retains declared auxiliary
deformers and original skin-weight ownership in the canonical frame. Both
preserve mesh, materials, UVs, textures, and valid armature bindings.

### Exact preparation reporting

`report.json` is the authoritative, deterministic audit trail.
Console logging is a readable projection of it and always names its path.
Timestamps and timings, if wanted, live in a separate run envelope.

The report records:

- dataset, rig, pipeline version, resolved authority policy, configuration,
  source paths, sizes, and checksums;
- complete source, model, and presentation skeleton descriptions: ordered
  names, parents, rest/bind transforms, units, axes, and channels;
- every BVH/FBX joint correspondence, its method and residual, including
  renamed, unmatched, ambiguous, and modality-only joints;
- every ordered transform with exact parameters: rest-relative conversion,
  root promotion, facing rotation, rigidity enforcement, centering, scale, and
  grounding;
- for every joint, whether it was retained, renamed, reparented, promoted,
  collapsed, excluded, retained as auxiliary, or assigned a reconstruction
  driver, including its reason and destination;
- every skinning edit: mesh and vertex group, source and destination joints,
  affected vertex count, total transferred weight, maximum per-vertex change,
  normalization correction, and validation outcome;
- every auxiliary reconstruction rule and dependency;
- bind-fit, rest-pose, motion, and deformation residuals with thresholds;
- every warning and explicitly accepted approximation; and
- output paths, topology hashes, mesh/material counts, and final invariants.

Routine progress is `INFO`. Structural and skinning decisions are `DEBUG`
with stable operation IDs shared by the JSON report. Accepted lossy operations
are at least `WARNING`. Ambiguous mappings and undeclared weighted-joint
losses are `ERROR` and abort the rig. A log may not merely say “reduced” or
“converted”: its summary names the affected joints, while the report contains
the full per-joint record.

Corpus acceptance requires all 73 Truebones rigs to have both FBX assets and a
complete audit report. Tests assert that every topology, correspondence,
transform, auxiliary, and skinning change has one stable report entry and that
repeated preparation yields the same semantic report.

### Optional source modalities

A dataset or rig may provide BVH motion only, FBX motion only, or paired BVH
and FBX. Availability is declared explicitly in the manifest and represented
with optional typed artifact references; a missing modality is not represented
by an empty path, fabricated file, or guessed counterpart.

For BVH-only input, BVH may own skeleton and motion. Preparation, model
training, prepared BVH, and original BVH remain fully supported. Surface
authority is `none`; mesh-bearing templates and all FBX exports return a
structured unavailable result whose reason is `source_surface_missing`.
On-disk optional artifact fields are JSON/YAML `null`, and the report records
that mesh and FBX stages were skipped rather than failed.

For FBX-only input, FBX owns every available source role. Blender extraction
produces the source animation, hierarchy, channels needed by the internal
pipeline, mesh, bind data, and motion curves. Preparation, model training, and
both FBX representations remain supported. BVH export is allowed because BVH
is a serializer for the prepared or restored internal animation; the report
must state that no original BVH artifact or source channel declaration existed,
so an original-space BVH uses a deterministic documented channel convention
and is not claimed to reproduce a vendor BVH file.

For paired input, the declared authority policy selects the model skeleton and
motion while FBX normally owns the surface. Correspondence and residual checks
are mandatory. Missing data may reduce capabilities, but it must not make
unrelated preparation or training fail.

The public API exposes availability before export:

```python
@dataclass(frozen=True)
class ArtifactAvailability:
    prepared_bvh: bool
    prepared_fbx: bool
    presentation_fbx: bool
    original_bvh: bool
    original_fbx: bool
    reasons: Mapping[str, str]

exporter.availability -> ArtifactAvailability
```

Export methods for unavailable representations raise a typed
`ArtifactUnavailableError` carrying dataset, rig, representation, format,
and reason. Batch preparation records `null` output paths plus the same reason
and continues when absence is expected by the manifest. It fails only when a
declared required modality or artifact is missing or invalid.

## Representation contract

The terms in this design have one meaning:

```text
raw character
  data/_raw/truebones/<Rig>/...
      |
      | preparation: rest-relative, root promotion, facing, rigidity,
      | centre, scale, grounding
      v
full prepared character
  data/truebones/clips/<Rig>/<action>.bvh
      |
      | build-time/read-time joint reduction
      v
model character
  reduced canonical Animation + model-ready skinned character asset
      |
      | feature extraction / reconstruction
      v
model output Animation
```

A **prepared export** serializes the model character. It remains reduced and
canonical: +Z facing, normalized scale, centered/grounded, with exactly the
joint names and hierarchy used to extract model features.

An **original export** starts from that same model `Animation`, expands the
joint reduction, and inverts preparation. It returns to the raw character's
units, reference frame, names, hierarchy, channel declaration, bind pose, and
mesh-bearing FBX asset as far as the preparation contract is invertible.

Joint expansion is therefore part of original export only. It must not happen
for prepared export.

## Goals

- Export any model-facing `Animation` as prepared BVH.
- Export the same animation as a prepared FBX with a mesh, skin weights,
  materials, and textures matching its reduced canonical topology.
- Export it as original-space BVH with the raw BVH hierarchy, names, offsets,
  and channel declaration restored.
- Export it as original-space FBX by driving the raw mesh-bearing character
  asset rather than synthesizing a bare armature.
- Keep character identity and corpus provenance explicit; do not add dataset
  paths or a rig name to the mathematical `Animation` value type.
- Make raw input and prepared output separate directory trees.
- Remove misleading legacy data and every live code dependency on it.

## Non-goals and inherent limits

- BVH cannot contain a mesh. Mesh requirements apply to both FBX modes only.
- `EnforceRigid` deliberately discards non-root per-joint translation. Original
  export restores the declared channels and rest translations, not discarded
  motion the model never represented.
- A newly generated motion has no historical source-facing direction. Original
  export defaults to the canonical +Z answer unless the caller explicitly
  supplies a recorded action-facing transform.
- BVH and FBX source files are not assumed to have identical bone sets. Each
  original exporter is validated against the source modality it drives.
- The migration does not preserve legacy parity assets under
  `data/truebones/reference/`. Reference probes that still matter must accept
  an explicit external fixture path; otherwise they are removed with the stale
  data rather than silently reading a second corpus representation.

## Corpus layout

The canonical layout becomes:

```text
data/
├── _raw/
│   └── truebones/
│       └── <Rig>/
│           ├── <take>.bvh
│           ├── <take>.fbx
│           └── tex/...
└── truebones/
    ├── clips/
    │   └── <Rig>/
    │       ├── <action>.bvh       full prepared motion, stage 1
    │       ├── <action>.npz       prepared arrays, stage 2
    │       └── <action>.yaml
    ├── rigs/
    │   └── <Rig>/
    │       ├── manifest.yaml
    │       ├── preproc.npz
    │       ├── skeleton.npz
    │       ├── stats.npz
    │       └── character.fbx      reduced canonical bind asset
    └── index.jsonl
```

`data/truebones/source/` moves to `data/_raw/truebones/`. The move is verified
before deletion by recording file count, total byte count, and a deterministic
relative-path/size/checksum manifest, then checking the destination against
that manifest. The empty `data/_raw/Truebones-Z_OO/` placeholder is removed.

After all consumers point at the canonical paths, delete these legacy trees:

```text
data/truebones/bvh/
data/truebones/fbx/
data/truebones/mesh/
data/truebones/reference/
```

The prepared `clips/` and `rigs/` trees remain. Raw data stays ignored by Git.
Prepared artifacts retain their existing ownership rules and are reproducible
from the raw corpus.

No compatibility fallback searches the deleted paths. A stale path must fail
with a message naming the canonical replacement; otherwise the ambiguity that
caused this defect remains.

## Character-aware export API

Low-level format conversion remains deliberately ignorant of datasets:

```python
BVH.from_animation(anim, channels=None) -> BVH
FBX.from_bare_animation(anim, name="Armature") -> FBX
```

The current `FBX.from_animation` bare-armature behavior is renamed to
`from_bare_animation`. A temporary deprecated alias may exist for downstream
callers, but no product export path uses it. Calling it never promises a mesh.

Dataset-aware export lives behind one explicit context:

```python
exporter = CharacterExporter.load(
    corpus=Path("data/truebones"),
    raw=Path("data/_raw/truebones"),
    rig="Flamingo",
)

prepared_bvh = exporter.prepared_bvh(anim)
prepared_fbx = exporter.prepared_fbx(anim)
original_bvh = exporter.original_bvh(anim, facing_from="onelegbent")
original_fbx = exporter.original_fbx(anim, facing_from="onelegbent")
```

Methods return the existing in-memory `BVH` or `FBX` wrappers; callers choose
where to write them. Export methods validate that `anim.names`, parents, and
offsets describe the rig's reduced model skeleton before doing any work.

`CharacterExporter` loads and owns:

- the reduction recorded by `skeleton.npz`;
- rig-scoped inverse-preparation parameters;
- clip-scoped facing rotations from `preproc.npz`;
- the raw BVH channel declaration;
- the reduced canonical `character.fbx`; and
- the manifest-declared raw FBX asset used for original export.

The context, not `Animation`, carries character identity. This keeps
`Animation` reusable for interpolation, augmentation, and generated motion
without embedding filesystem provenance in a mathematical container.

### Prepared BVH

`prepared_bvh(anim)` delegates to `BVH.from_animation(anim)`. It preserves the
reduced model hierarchy. It does not invert reduction or preparation.

### Original BVH

`original_bvh(anim, facing_from=None)`:

1. validates the reduced skeleton;
2. expands the recorded joint reduction;
3. inverts every preparation stage;
4. uses the selected action's recorded facing rotation, or identity for a new
   +Z-facing generated motion; and
5. calls `BVH.from_animation(restored, channels=source_channels)`.

The result restores raw BVH names, hierarchy, offsets, units, bind rotations,
and channel declaration. Tests compare it with `data/_raw/truebones`, never a
legacy duplicate.

### Prepared FBX

`prepared_fbx(anim)` loads `rigs/<Rig>/character.fbx` and applies `anim`
directly. The template already uses the reduced model hierarchy and canonical
bind frame, so there is no similarity fit and no inverse preparation at export
time. The output must contain at least one mesh and preserve the template's
materials, embedded textures, armature modifier, and vertex groups.

Unlike the current bare-armature constructor, the model-ready template
contains a Blender bone for every model joint, including childless joints. A
childless joint receives a cosmetic stub tail and zero skin weight when the raw
asset had only an FBX leaf bone or BVH End Site counterpart. Bone heads, names,
parent indices, and animation transforms remain exact.

### Original FBX

`original_fbx(anim, facing_from=None)`:

1. validates and expands the reduced animation;
2. inverts preparation through every stage after `RestRelative`, leaving the
   motion rest-relative because an FBX bind pose already defines the T-pose;
3. loads the manifest-declared raw FBX asset;
4. applies animation by joint name with the existing bind-fit guard; and
5. exports with mesh, skin, materials, and textures intact.

There is no bare-armature fallback. A missing asset, missing bone, or bind-fit
failure is an error naming the rig and offending path.

## Building the model-ready character asset

Every rig receives one `rigs/<Rig>/character.fbx` during corpus preparation.
It is a bind asset, not a motion clip. The builder starts from the
manifest-declared raw FBX so material slots, UVs, textures, mesh partitions,
and skinning survive.

The asset follows the same semantic edits as the BVH path:

1. align the raw FBX bind skeleton to the rig's prepared rest geometry;
2. apply canonical facing, centering, scale, and grounding to the armature,
   meshes, and bind matrices together;
3. apply `PromoteRoot` to the armature and mesh skinning;
4. apply the recorded `JointReduction` to the hierarchy; and
5. remove source animation curves, leaving a clean canonical bind template.

Structural edits operate by joint name, never positional index.

When reduction removes a coincident joint, its skin-weight column is added to
the surviving joint that receives its transform. Weight rows are renormalized
after all edits. Removed bones must have no remaining vertex group, constraint,
or armature reference. The resulting FBX bone names and parent array must equal
the model animation's skeleton exactly.

This replaces `mesh.npz` as the export asset. NPZ remains suitable for numeric
training data but cannot preserve materials, UV/image relationships, or Blender
skinning semantics. No new geometry-only OBJ tree is produced.

The manifest gains an explicit raw FBX template filename. Selection must not
depend on alphabetical order or on whichever action happened to be processed
first. The declared asset must exist, contain a mesh and armature, and pass the
BVH/FBX bind-fit threshold before a character artifact is written.

## Build ownership

Stage 1 reads only `data/_raw/truebones` and writes full prepared motion plus
invertibility metadata:

- `clips/<Rig>/<action>.bvh`
- `rigs/<Rig>/preproc.npz`

Stage 2 derives the exact model representation and writes:

- `clips/<Rig>/<action>.npz`
- `rigs/<Rig>/skeleton.npz`
- `rigs/<Rig>/stats.npz`
- `rigs/<Rig>/character.fbx`
- `index.jsonl`

`character.fbx` belongs to stage 2 because it must use the same reduction as
`skeleton.npz` and the dataset read path. Re-running feature normalization does
not rebuild it; changing preparation, root promotion, reduction tolerance, or
the raw character asset does.

The Blender-dependent character-asset pass may run in the `fbx` container, but
its inputs and outputs are stage-2 contracts and its failures are reported per
rig. A corpus is not export-complete unless all indexed rigs have a validated
`character.fbx`.

## Facing semantics

Prepared outputs always face +Z.

For original outputs:

- `facing_from=<action>` restores that real clip's recorded source yaw;
- no `facing_from` leaves generated motion facing +Z; and
- an unknown action is an error listing valid recorded actions.

The API does not claim to infer a historical direction that generated motion
never had.

## Error handling and invariants

The following are hard errors:

- reduced animation names, parents, or offsets differ from `skeleton.npz`;
- the raw root, prepared corpus, transform, or character template is missing;
- the prepared FBX contains no mesh or no armature modifier;
- its bone names or hierarchy differ from the reduced model topology;
- any removed joint retains skin weights after reduction;
- any vertex has invalid, negative, or non-normalized skin weights;
- the raw FBX template does not carry every bone required for original export;
- bind-fit residual exceeds the documented threshold; or
- source-channel count disagrees with the expanded original BVH skeleton.

No exporter silently substitutes a bare armature, a first alphabetical asset,
an identity facing for a requested real action, or a legacy data path.

## CLI integration

Generation and retargeting write prepared outputs by default because they are
the model's native representation. CLI export options select representation
explicitly:

```text
--representation prepared|original
--format bvh|fbx
--facing-from <action>
```

Requesting FBX always means a mesh-bearing character export. A separate
developer/debug option may request a bare skeleton, but it is named
`--bare-skeleton` and routes through `FBX.from_bare_animation`; it is never the
default meaning of FBX.

`scripts/anim_2_fbx.py` becomes a thin wrapper over `CharacterExporter` or is
replaced by the corresponding CLI command. It must not retain a second copy of
inverse-preparation logic.

## Migration and deletion sequence

The data cleanup is destructive and therefore ordered so the irreplaceable raw
corpus is proven intact first:

1. inventory `data/truebones/source/` by relative path, byte size, and checksum;
2. verify `data/_raw/Truebones-Z_OO/` is empty and contains no user data;
3. move `source/` to `data/_raw/truebones/` without crossing filesystems when
   possible;
4. verify the destination against the inventory;
5. update preparation, tests, documentation, ignore rules, and compose-facing
   examples to the canonical raw path;
6. run stage-1 and round-trip tests against the new location;
7. remove `data/truebones/bvh/`, `fbx/`, `mesh/`, and `reference/`;
8. remove or migrate every code reference to those trees; and
9. assert in tests that a prepared corpus contains only the documented output
   roots.

The raw move is recoverable by moving the verified directory back until the
legacy deletions occur. The four explicitly deleted legacy trees are not
recoverable from the working tree after deletion; they are treated as
reproducible or obsolete data, per the requested cleanup.

## Verification

### Unit tests

- `BVH.from_animation` preserves reduced prepared topology.
- original BVH export expands reduction and restores source names, parents,
  offsets, channels, units, and recorded facing.
- skin-weight folding maps every removed joint to the reduction recipient and
  preserves row sums.
- character contexts reject cross-rig animations and altered skeletons.
- missing facing, raw asset, template, and transform errors name the rig and
  path/action.

### Blender integration tests

- prepared FBX has a mesh, armature modifier, materials, and exactly the model
  skeleton's names and hierarchy;
- its bind joint heads match the reduced prepared rest animation;
- posing it reproduces prepared-animation joint positions over first, middle,
  and final frames;
- original FBX preserves the raw asset's mesh/material counts and bone
  hierarchy;
- its posed joints agree with original BVH export in world space after
  modality-specific End Site handling; and
- both prepared and original exports deform vertices on a moving clip.

### Corpus acceptance

- all 73 indexed rigs have `character.fbx`;
- no indexed rig silently skips character export;
- `data/_raw/truebones` matches the pre-move inventory;
- deleted legacy directories are absent;
- no source code, configuration, test, or maintained tool references the
  deleted paths;
- full compatibility and Blender test suites pass; and
- a regenerated `tmp/skel_visual_debug` bundle contains, per rig, raw source
  BVH/FBX plus prepared and original BVH/FBX, with both FBXs mesh-bearing and
  metadata naming every source artifact and transformation mode.

## Success criteria

A caller holding a model-facing animation and explicit character context can
produce four honest artifacts:

| export | topology/frame | mesh |
|---|---|---|
| prepared BVH | reduced, canonical | unsupported by format |
| prepared FBX | reduced, canonical | required |
| original BVH | expanded, raw BVH frame | unsupported by format |
| original FBX | raw FBX character frame | required |

The two prepared artifacts describe the same model character. The two original
artifacts restore their respective raw modalities without pretending that BVH
and FBX source skeletons are universally identical. No export called “FBX” is
silently a bare armature, and no prepared-corpus path contains raw or legacy
copies.

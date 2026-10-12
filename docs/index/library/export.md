# export

`src/poseydon/export/`, library. `CharacterExporter` takes a model-space
animation and writes it as the prepared, presentation or original
representation, in BVH or FBX.

## Purpose

`export` owns the last arrow of the pipeline: reduced canonical motion →
a file on a character. It loads a rig's recorded transforms and assets and
exposes one method per representation and format.

It does **not** own the transforms themselves (`build.prepare`,
`features.reduce`), file I/O ([io](io.md)), or building the character assets
(`src/poseydon/build/character_assets.py`).

## Key files and types

| file | read it for |
|---|---|
| `src/poseydon/export/character.py` | `CharacterExporter.load(corpus, raw, rig)`; `prepared_bvh`, `original_bvh`, `prepared_fbx`, `presentation_fbx`, `original_fbx`, each with an optional `facing_from=<action>` |

Representations
([character export spec](../../superpowers/specs/2026-09-19-character-aware-motion-export-design.md#representation-contract)):

| representation | joints | frame | asset |
|---|---|---|---|
| **prepared** | reduced, exactly the model's | canonical: +Z, scaled, centred, grounded | `char-model.fbx` |
| **presentation** | full deformation rig | canonical | `char-full.fbx` |
| **original** | the source's own, after expanding the reduction | source units, frame, names, channels, bind | the raw vendor FBX |

## How it connects

Consumes `rigs/<Rig>/skeleton.npz` and `preproc.npz` through
`build.prepare.RigTransform` and `prepare_chain`, `features.reduce`
(`invert_reduction`), `io.bvh.BVH`, `io.fbx.FBX` (lazily), and
`core.authority.ArtifactUnavailableError`.

Produces BVH or FBX files. Availability per rig is recorded in the index and
`report.json` (see [build](build.md)).

Called by: `cli.py` (`poseydon export --representation prepared|presentation|original --format bvh|fbx`),
`scripts/anim_2_fbx.py`, `scripts/build_skeleton_visual_debug_fbx.py`.

## Why it is shaped this way

- **Expansion belongs to original export only.** A prepared export serializes
  the model character exactly as the model sees it. An original export expands
  the reduction and inverts preparation back to the raw character, as far as
  the contract is invertible. Source: character export spec, "Representation
  contract".
- **Inversion reads persisted parameters.** The chain only supplies the order
  (`prepare_chain()`, its single definition), so export needs neither the
  channel layout nor the promotion target. Source: comment in
  `CharacterExporter.load`.
- **Facing is explicit.** Prepared outputs always face +Z. For original
  outputs, `facing_from=<action>` restores that clip's recorded yaw, and no
  argument leaves generated motion facing +Z. The exporter never invents a
  historical direction. Source:
  [spec "Facing semantics"](../../superpowers/specs/2026-09-19-character-aware-motion-export-design.md#facing-semantics).
- **Hard errors, no silent substitutes.** Mismatched topology, missing assets,
  stray skin weights or bind-fit residuals over threshold all raise. No bare
  armature, first-alphabetical asset or identity facing is substituted. Source:
  [spec "Error handling and invariants"](../../superpowers/specs/2026-09-19-character-aware-motion-export-design.md#error-handling-and-invariants);
  commits `1e93973`, `debe9be`.
- **The source FBX is chosen by bind geometry**, not by taking the first vendor
  file, because several Truebones files have locator roots or action-specific
  bind variants. Source: comment in `CharacterExporter.load`.

## Gotchas

- **A facing is required only when the chain actually undoes one** (commit
  `e899592`).
- **BVH-only rigs can't produce FBX representations.** They report
  `source_surface_missing`. Check a rig's `report.json` or the index
  availability before requesting one.
- **FBX output runs only in the `fbx` container.** BVH output works anywhere.
- **Held leading or trailing frames are dropped from FBX keys**
  (`bake_anim_force_startend_keying=False`). The take span stays correct, but
  key-range readers see a shorter clip. This is open.
- **The default facing can mismatch generated clips** that were templated on a
  non-rest clip. See [ops](ops.md).

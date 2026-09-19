# Character-Aware Motion Export Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a character-aware, auditable preparation/export pipeline that supports BVH-only, FBX-only, and paired datasets; produces strict model and full presentation representations; preserves mesh deformation where available; and round-trips model animations into original source space.

**Architecture:** Dataset manifests resolve independent skeleton, motion, surface, and auxiliary-joint policies. Pure-Python preparation owns motion transforms, topology contracts, availability, and deterministic audit records; Blender-backed code owns FBX extraction, canonical mesh-bearing assets, skinning edits, and FBX export. A `CharacterExporter` joins those artifacts without putting dataset identity into `Animation`.

**Tech Stack:** Python 3.11+, NumPy, PyYAML, Blender/`bpy`, pytest, Docker Compose services `test`, `compat`, and `fbx`.

**Spec:** `docs/superpowers/specs/2026-09-19-character-aware-motion-export-design.md`

## Global Constraints

- Run every command inside Docker; use `docker compose run --rm test ...` for normal tests and the `fbx` service for Blender-dependent tests.
- Inspect executable code and actual artifacts before changing behavior. Tests and comments are evidence, not authority; stale tests/comments must be corrected when they contradict observed behavior.
- Characterize current behavior with focused tests before refactoring it.
- Keep `Animation` a mathematical value type with no dataset paths or rig identity.
- Missing modalities are typed absence: manifest/output fields are `null`, availability supplies a reason, and no empty or fabricated asset is written.
- BVH-only preparation and training must not require Blender or mesh data.
- FBX-only preparation extracts motion and skeleton through Blender and must not require a vendor BVH.
- Strict prepared BVH and strict prepared FBX use exactly the model joint names, order, parents, rest offsets, and canonical frame.
- Presentation FBX retains the declared full deformation rig; independently animated weighted joints are never silently folded.
- Every topology, correspondence, transform, auxiliary-joint, and skinning decision has a stable operation ID in `preparation-report.json`.
- Accepted lossy operations log at least `WARNING`; ambiguous or undeclared weighted-joint loss aborts that rig.
- Do not add compatibility searches for `data/truebones/{source,bvh,fbx,mesh,reference}`.
- Preserve unrelated pre-existing worktree changes and commit only task-scoped files.
- Do not move or delete data until the checksum inventory and destination verification steps in Task 10 pass.

## Review Focus

- BVH-only rig with no FBX: training succeeds, FBX availability is false with `source_surface_missing`, and no Blender artifact is expected.
- FBX-only rig with no BVH: motion extraction and training succeed; synthesized BVH uses documented channels and does not claim vendor-BVH fidelity.
- Paired files with matching names but incompatible bind geometry: preparation fails before an asset or misleading report is published.
- Weighted auxiliary joint removed without explicit policy: preparation fails rather than folding weights implicitly.
- Re-running identical preparation: semantic report, topology hashes, mappings, and output availability are deterministic.

---

### Task 1: Characterize Existing Motion and FBX Behavior

**Files:**
- Create: `tests/build/test_current_export_contract.py`
- Create: `tests/io/test_fbx_current_contract.py`
- Modify: `docs/superpowers/plans/2026-09-19-character-aware-motion-export.md` only if measured behavior invalidates a later assumption

**Interfaces:**
- Consumes: `BVH.from_animation`, `FBX.from_animation`, `FBX.apply_animation`, `build_reduction`, `PrepareChain.invert`
- Produces: executable characterization of current behavior; no production API

- [ ] **Step 1: Read the implementations and construct tiny in-memory fixtures**

Record observations from method bodies, not docstrings: BVH preserves childless joints; bare FBX omits them; `apply_animation` rejects armature bones absent from the animation; reduction is applied during feature extraction rather than corpus preparation.

- [ ] **Step 2: Write the BVH/reduction characterization tests**

```python
def test_bvh_writer_preserves_names_parents_offsets_and_childless_joint(tiny_animation):
    reread = BVH.from_animation(tiny_animation).to_animation()
    assert reread.names == tiny_animation.names
    np.testing.assert_array_equal(reread.parents, tiny_animation.parents)
    np.testing.assert_allclose(reread.offsets, tiny_animation.offsets)

def test_reduction_is_invertible_in_world_space(tiny_rigid_animation):
    reduction = build_reduction(tiny_rigid_animation)
    reduced = apply_reduction(tiny_rigid_animation, reduction)
    restored = invert_reduction(reduced, reduction)
    np.testing.assert_allclose(
        restored.global_positions(), tiny_rigid_animation.global_positions(), atol=1e-8
    )
```

- [ ] **Step 3: Run the pure-Python characterization**

Run: `docker compose run --rm test pytest tests/build/test_current_export_contract.py -v`

Expected: PASS against current executable behavior.

- [ ] **Step 4: Write Blender characterization tests**

```python
def test_bare_fbx_has_no_mesh_and_omits_bvh_end_sites(tiny_animation):
    scene = FBX.from_animation(tiny_animation)
    assert scene.meshes == []
    assert set(scene.joint_names) == {"Root", "Leaf"}

def test_apply_animation_rejects_undriven_armature_bone(skinned_fixture):
    with pytest.raises(ValueError, match="animation does not drive"):
        skinned_fixture.apply_animation(animation_missing_auxiliary_bone())
```

- [ ] **Step 5: Run the Blender characterization**

Run: `docker compose run --rm fbx blender -b --python-expr "import pytest; raise SystemExit(pytest.main(['tests/io/test_fbx_current_contract.py','-v']))"`

Expected: PASS, proving the current bare/skinned distinction before it is refactored.

- [ ] **Step 6: Commit**

```bash
git add tests/build/test_current_export_contract.py tests/io/test_fbx_current_contract.py
git commit -m "test: characterize motion and fbx export behavior"
```

### Task 2: Model Authority, Modalities, and Artifact Availability

**Files:**
- Create: `src/poseydon/build/authority.py`
- Test: `tests/build/test_authority.py`
- Modify: `src/poseydon/core/skeleton.py`
- Modify: `tests/build/test_rest_selection.py`

**Interfaces:**
- Consumes: YAML loaded by `SkeletonManifest.load(path)`
- Produces: `AuthorityPolicy`, `SourceArtifacts`, `ArtifactAvailability`, `ArtifactUnavailableError`, and new `SkeletonManifest.authority`/`artifacts`

- [ ] **Step 1: Write failing manifest and availability tests**

```python
def test_bvh_only_manifest_has_no_surface_outputs(tmp_path):
    manifest = load_manifest(tmp_path, artifacts={"bvh": "__Idle.bvh", "fbx": None})
    availability = ArtifactAvailability.from_manifest(manifest)
    assert availability.prepared_bvh
    assert not availability.prepared_fbx
    assert availability.reasons["prepared_fbx"] == "source_surface_missing"

def test_fbx_only_manifest_accepts_null_bvh(tmp_path):
    manifest = load_manifest(
        tmp_path,
        authority={"skeleton": "fbx", "motion": "fbx", "surface": "fbx"},
        artifacts={"bvh": None, "fbx": "__Idle.fbx"},
    )
    assert manifest.artifacts.bvh is None
    assert manifest.artifacts.fbx == Path("__Idle.fbx")

def test_declared_authority_requires_its_modality(tmp_path):
    with pytest.raises(ManifestError, match="motion authority.*bvh"):
        load_manifest(tmp_path, authority={"motion": "bvh"}, artifacts={"bvh": None})
```

- [ ] **Step 2: Verify failure**

Run: `docker compose run --rm test pytest tests/build/test_authority.py tests/build/test_rest_selection.py -v`

Expected: FAIL because authority/artifact fields are unknown.

- [ ] **Step 3: Implement immutable policy types**

```python
@dataclass(frozen=True)
class AuthorityPolicy:
    skeleton: Literal["bvh", "fbx"]
    motion: Literal["bvh", "fbx"]
    surface: Literal["fbx", "none"]
    auxiliary_joints: Literal["exclude", "retain", "bake"]

@dataclass(frozen=True)
class SourceArtifacts:
    bvh: Path | None
    fbx: Path | None
    textures: Path | None = None

@dataclass(frozen=True)
class ArtifactAvailability:
    prepared_bvh: bool
    prepared_fbx: bool
    presentation_fbx: bool
    original_bvh: bool
    original_fbx: bool
    reasons: Mapping[str, str]

class ArtifactUnavailableError(RuntimeError):
    def __init__(self, dataset: str, rig: str, representation: str,
                 format: str, reason: str): ...
```

Extend manifest validation so authority never depends on filename discovery. Preserve backwards readability only long enough to migrate checked-in manifests in Task 9; emit an explicit deprecation error rather than guessing in the final state.

- [ ] **Step 4: Run focused tests**

Run: `docker compose run --rm test pytest tests/build/test_authority.py tests/build/test_rest_selection.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/poseydon/build/authority.py src/poseydon/core/skeleton.py tests/build/test_authority.py tests/build/test_rest_selection.py
git commit -m "feat: declare dataset source authority and availability"
```

### Task 3: Add Deterministic Preparation Audit Records

**Files:**
- Create: `src/poseydon/build/report.py`
- Create: `tests/build/test_preparation_report.py`
- Modify: `src/poseydon/build/prepare.py`
- Modify: `tests/build/test_prepare.py`

**Interfaces:**
- Consumes: stage names and fitted parameter dictionaries from `PrepareChain`
- Produces: `PreparationReport`, `OperationRecord`, `JointDecision`, `SkinningDecision`; optional `audit: AuditSink | None` arguments on preparation methods

- [ ] **Step 1: Write report determinism and completeness tests**

```python
def test_report_semantics_are_deterministic(tmp_path, prepared_fixture):
    first = run_and_report(prepared_fixture)
    second = run_and_report(prepared_fixture)
    assert first.semantic_dict() == second.semantic_dict()
    assert first.semantic_sha256 == second.semantic_sha256

def test_prepare_chain_records_every_stage_in_execution_order(prepared_fixture):
    report = PreparationReport.start(dataset="fixture", rig="Rig", policy=POLICY)
    CHAIN.apply(prepared_fixture.anim, prepared_fixture.resolved,
                prepared_fixture.rig_params, audit=report)
    assert [op.kind for op in report.operations] == [
        "rest_relative", "promote_root", "face_axis", "enforce_rigid",
        "centre_xz", "scale_to_mean_bone_length", "put_on_ground",
    ]
    assert all(op.operation_id for op in report.operations)
```

- [ ] **Step 2: Verify failure**

Run: `docker compose run --rm test pytest tests/build/test_preparation_report.py tests/build/test_prepare.py -v`

Expected: FAIL because audit types and hooks do not exist.

- [ ] **Step 3: Implement canonical JSON records**

Use sorted keys, finite numeric validation, arrays converted to lists, SHA-256 source/output hashes, and stable IDs derived from `rig/kind/ordinal`. Keep runtime timestamps outside `semantic_dict()`.

```python
class AuditSink(Protocol):
    def record(self, operation: OperationRecord) -> None: ...

@dataclass
class PreparationReport:
    def record_transform(self, *, stage: str, scope: str,
                         parameters: Mapping[str, JsonValue]) -> str: ...
    def write(self, path: Path) -> None: ...
```

- [ ] **Step 4: Instrument actual stage execution**

Add audit calls immediately beside each `fit/apply/invert` invocation so the report describes executed operations, not a separately reconstructed story. Record input/output topology hashes and exact numeric parameters.

- [ ] **Step 5: Run tests**

Run: `docker compose run --rm test pytest tests/build/test_preparation_report.py tests/build/test_prepare.py -v`

Expected: PASS with existing transform numerical tests unchanged.

- [ ] **Step 6: Commit**

```bash
git add src/poseydon/build/report.py src/poseydon/build/prepare.py tests/build/test_preparation_report.py tests/build/test_prepare.py
git commit -m "feat: audit every preparation transform"
```

### Task 4: Extract FBX-Only Motion into the Internal Animation Model

**Files:**
- Modify: `src/poseydon/io/fbx.py`
- Create: `tests/io/test_fbx_to_animation.py`
- Create: `src/poseydon/build/source.py`
- Create: `tests/build/test_source_motion.py`

**Interfaces:**
- Consumes: `FBX.read(path)`, manifest authority, frame range and FPS
- Produces: `FBX.to_animation() -> Animation`; `SourceMotion.load(manifest, raw_rig_dir) -> Animation`

- [ ] **Step 1: Write a Blender test against a tiny keyed FBX fixture**

```python
def test_fbx_to_animation_preserves_hierarchy_rest_and_motion(keyed_skinned_fbx):
    source = FBX.read(keyed_skinned_fbx)
    anim = source.to_animation()
    assert anim.names == tuple(source.joint_names)
    np.testing.assert_array_equal(anim.parents, source.joint_parents)
    np.testing.assert_allclose(anim.offsets, source.joint_offsets)
    assert anim.n_frames == source.frame_range()[1] - source.frame_range()[0] + 1
```

Also assert reconstructed global joint positions against `joint_position()` on first, middle, and last frames.

- [ ] **Step 2: Verify failure in Blender**

Run: `docker compose run --rm fbx blender -b --python-expr "import pytest; raise SystemExit(pytest.main(['tests/io/test_fbx_to_animation.py','-v']))"`

Expected: FAIL because `FBX.to_animation` does not exist.

- [ ] **Step 3: Implement FBX extraction from evaluated pose matrices**

Sample the declared frame range, convert Blender Z-up matrices to PoseYdon Y-up world transforms, derive parent-local quaternions/translations, and use bind heads for offsets. Do not infer behavior from FBX comments; compare the extracted animation's global transforms directly with Blender-evaluated bones.

- [ ] **Step 4: Add source dispatch tests**

```python
@pytest.mark.parametrize("authority,loader", [("bvh", "load_bvh"), ("fbx", "load_fbx")])
def test_source_motion_dispatches_only_to_declared_authority(authority, loader, spies):
    SourceMotion.load(manifest_for(authority), RAW)
    assert spies[loader].call_count == 1
```

- [ ] **Step 5: Implement `SourceMotion` dispatch and run tests**

Run:
`docker compose run --rm test pytest tests/build/test_source_motion.py -v`

Then:
`docker compose run --rm fbx blender -b --python-expr "import pytest; raise SystemExit(pytest.main(['tests/io/test_fbx_to_animation.py','-v']))"`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/poseydon/io/fbx.py src/poseydon/build/source.py tests/io/test_fbx_to_animation.py tests/build/test_source_motion.py
git commit -m "feat: extract internal animations from fbx motion"
```

### Task 5: Validate Skeleton Correspondence and Skinning Policy

**Files:**
- Create: `src/poseydon/build/correspondence.py`
- Create: `src/poseydon/build/skinning.py`
- Create: `tests/build/test_correspondence.py`
- Create: `tests/build/test_skinning_policy.py`
- Modify: `src/poseydon/features/reduce.py`

**Interfaces:**
- Consumes: source/model/presentation names, parents, bind positions, `JointReduction.ops`, mesh weights, auxiliary policy
- Produces: `SkeletonCorrespondence`, `JointDisposition`, `SkinningPlan`; report records for every joint and weight edit

- [ ] **Step 1: Write mapping and ambiguity tests**

```python
def test_correspondence_matches_by_declared_name_and_checks_bind_residual():
    match = build_correspondence(MODEL, PRESENTATION, aliases={"mixamorig:Hips": "Hips"})
    assert match.presentation_for("Hips") == "mixamorig:Hips"
    assert match.max_bind_residual < 0.05

def test_duplicate_alias_is_rejected():
    with pytest.raises(CorrespondenceError, match="ambiguous"):
        build_correspondence(MODEL, PRESENTATION, aliases={"A": "Hip", "B": "Hip"})
```

- [ ] **Step 2: Write weighted-joint policy tests**

```python
def test_weighted_animated_auxiliary_without_policy_fails():
    with pytest.raises(SkinningPolicyError, match="weighted.*animated"):
        plan_skinning(AUX_WEIGHTED_ANIMATED, policy=None)

def test_unweighted_leaf_can_drop_without_changing_weights():
    plan = plan_skinning(UNWEIGHTED_LEAF, policy="exclude")
    assert plan.decisions[0].operation == "drop"
    np.testing.assert_allclose(plan.output_weights.sum(axis=1), 1.0)

def test_collapse_reports_exact_weight_transfer_metrics():
    plan = plan_skinning(COLLAPSIBLE_WEIGHTED, policy="exclude")
    assert plan.decisions[0].destination == "Parent"
    assert plan.decisions[0].affected_vertices == 2
```

- [ ] **Step 3: Verify failure**

Run: `docker compose run --rm test pytest tests/build/test_correspondence.py tests/build/test_skinning_policy.py -v`

Expected: FAIL because mapping/policy modules do not exist.

- [ ] **Step 4: Implement pure topology and weight planning**

Do not mutate Blender data here. Classify `DROP` reduction operations as unweighted helpers unless explicitly overridden; classify `COLLAPSE` by motion independence, hierarchy, and influence. Require a named bake driver for `bake`. Normalize weights only after all transfers and report maximum correction.

- [ ] **Step 5: Extend reduction metadata without changing motion math**

Add stable per-operation IDs and explicit recipient/reason fields while keeping `apply_reduction`/`invert_reduction` world-space behavior identical. Update characterization tests if field construction changes, never their numerical invariants.

- [ ] **Step 6: Run tests**

Run: `docker compose run --rm test pytest tests/build/test_correspondence.py tests/build/test_skinning_policy.py tests/features/test_reduce.py tests/build/test_current_export_contract.py -v`

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/poseydon/build/correspondence.py src/poseydon/build/skinning.py src/poseydon/features/reduce.py tests/build/test_correspondence.py tests/build/test_skinning_policy.py tests/features/test_reduce.py
git commit -m "feat: validate skeleton mapping and skinning policy"
```

### Task 6: Build Canonical Model and Presentation FBX Assets

**Files:**
- Modify: `src/poseydon/io/fbx.py`
- Create: `src/poseydon/build/character_assets.py`
- Create: `tests/io/test_character_assets.py`
- Replace: `scripts/process_dataset_truebones_fbx.py`

**Interfaces:**
- Consumes: raw FBX, prepared model rest animation, `SkeletonCorrespondence`, `SkinningPlan`, preparation transforms, audit sink
- Produces: `build_character_assets(...) -> CharacterAssets(model: Path | None, presentation: Path | None)`

- [ ] **Step 1: Write Blender tests for both assets**

```python
def test_model_asset_exactly_matches_model_topology(built_assets, model_rest):
    model = FBX.read(built_assets.model)
    assert tuple(model.joint_names) == model_rest.names_without_end_sites
    assert tuple(model.joint_parents) == model_rest.parents_without_end_sites
    assert model.meshes

def test_presentation_asset_retains_auxiliary_deformers(built_assets):
    presentation = FBX.read(built_assets.presentation)
    assert "ClothSecondary" in presentation.joint_names
    assert presentation.vertex_group_weight("ClothSecondary") > 0

def test_bvh_only_returns_typed_absence(tmp_path):
    assets = build_character_assets(BVH_ONLY_CONTEXT, tmp_path)
    assert assets.model is None and assets.presentation is None
    assert assets.reason == "source_surface_missing"
```

- [ ] **Step 2: Verify failure in Blender**

Run: `docker compose run --rm fbx blender -b --python-expr "import pytest; raise SystemExit(pytest.main(['tests/io/test_character_assets.py','-v']))"`

Expected: FAIL because the builder does not exist.

- [ ] **Step 3: Add explicit FBX scene operations**

Implement narrowly scoped methods for canonical whole-scene transforms, bone removal/reparenting, vertex-group transfer, action clearing, topology hashing, and invariant validation. Rename current `FBX.from_animation` to `FBX.from_bare_animation`; keep a deprecation alias only if an existing external caller requires it.

- [ ] **Step 4: Implement the asset builder**

Build `presentation.fbx` first from raw FBX with full declared deformers. Derive `model_character.fbx` from a fresh raw import using the validated skinning plan. Never derive the presentation rig from the reduced rig. Write temporary outputs and atomically rename only after all invariants and the report pass.

- [ ] **Step 5: Run Blender tests and inspect one real rig**

Run:
`docker compose run --rm fbx blender -b --python-expr "import pytest; raise SystemExit(pytest.main(['tests/io/test_character_assets.py','-v']))"`

Then:
`docker compose run --rm fbx blender -b --python scripts/process_dataset_truebones_fbx.py -- --rigs Flamingo --raw-root data/_raw/truebones --out-root /tmp/poseydon-character-assets`

Expected: tests PASS; the command writes both FBXs and a report for Flamingo, with nonzero meshes and matching topology hashes.

- [ ] **Step 6: Commit**

```bash
git add src/poseydon/io/fbx.py src/poseydon/build/character_assets.py tests/io/test_character_assets.py scripts/process_dataset_truebones_fbx.py
git commit -m "feat: build model and presentation character assets"
```

### Task 7: Implement the Character-Aware Export API

**Files:**
- Create: `src/poseydon/export/__init__.py`
- Create: `src/poseydon/export/character.py`
- Create: `tests/export/test_character_exporter.py`
- Create: `tests/export/test_character_exporter_fbx.py`
- Modify: `scripts/anim_2_fbx.py`

**Interfaces:**
- Consumes: reduced model `Animation`, rig artifacts, availability, reduction, preparation parameters
- Produces: `CharacterExporter.load(corpus, raw, rig)`; `prepared_bvh`, `prepared_fbx`, `presentation_fbx`, `original_bvh`, `original_fbx`

- [ ] **Step 1: Write pure exporter contract tests**

```python
def test_prepared_bvh_keeps_reduced_model_topology(exporter, model_anim):
    out = exporter.prepared_bvh(model_anim).to_animation()
    assert out.names == model_anim.names
    np.testing.assert_array_equal(out.parents, model_anim.parents)

def test_original_bvh_expands_then_inverts_preparation(exporter, model_anim):
    out = exporter.original_bvh(model_anim, facing_from="walk").to_animation()
    assert out.names == exporter.source_bvh_names
    assert out.channels == exporter.source_bvh_channels

def test_unavailable_fbx_raises_typed_error(bvh_only_exporter, model_anim):
    with pytest.raises(ArtifactUnavailableError) as error:
        bvh_only_exporter.prepared_fbx(model_anim)
    assert error.value.reason == "source_surface_missing"
```

Include the Review Focus FBX-only test: `original_bvh` succeeds with deterministic default channels and report provenance `source_bvh: null`.

- [ ] **Step 2: Verify failure**

Run: `docker compose run --rm test pytest tests/export/test_character_exporter.py -v`

Expected: FAIL because the API does not exist.

- [ ] **Step 3: Implement validation, prepared BVH, and original BVH**

Load reduction metadata once. Validate names, parents, and offsets before exporting. Original BVH performs `invert_reduction` then inverse preparation; it uses recorded channels only when a vendor BVH exists.

- [ ] **Step 4: Write and run Blender exporter tests**

```python
def test_prepared_and_presentation_fbx_keep_mesh(exporter, model_anim, tmp_path):
    for scene in (exporter.prepared_fbx(model_anim),
                  exporter.presentation_fbx(model_anim)):
        assert scene.meshes
```

Run: `docker compose run --rm fbx blender -b --python-expr "import pytest; raise SystemExit(pytest.main(['tests/export/test_character_exporter_fbx.py','-v']))"`

Expected initially: FAIL.

- [ ] **Step 5: Implement the three FBX modes**

Prepared FBX drives `model_character.fbx`; presentation FBX drives canonical `presentation.fbx` including auxiliary rules; original FBX expands/inverts through the stage before `RestRelative` and drives the raw FBX. No mode calls the bare-armature constructor.

- [ ] **Step 6: Replace script-local inverse logic**

Make `scripts/anim_2_fbx.py` a thin argument adapter around `CharacterExporter`. Remove `source_asset`, `rig_transform`, and `unprepare` duplicates after parity tests pass.

- [ ] **Step 7: Run exporter suites**

Run:
`docker compose run --rm test pytest tests/export/test_character_exporter.py tests/build/test_anim_2_fbx.py tests/build/test_roundtrip.py -v`

Then:
`docker compose run --rm fbx blender -b --python-expr "import pytest; raise SystemExit(pytest.main(['tests/export/test_character_exporter_fbx.py','tests/build/test_roundtrip_fbx.py','-v']))"`

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add src/poseydon/export scripts/anim_2_fbx.py tests/export tests/build/test_anim_2_fbx.py tests/build/test_roundtrip.py tests/build/test_roundtrip_fbx.py
git commit -m "feat: export character-aware motion representations"
```

### Task 8: Integrate Multi-Modality Preparation and Reporting

**Files:**
- Modify: `src/poseydon/build/pipeline.py`
- Modify: `src/poseydon/build/corpus.py`
- Modify: `scripts/process_dataset_truebones.py`
- Modify: `scripts/build_features.py`
- Create: `tests/build/test_optional_modalities.py`
- Modify: `tests/build/test_pipeline.py`
- Modify: `tests/build/test_corpus_yield.py`

**Interfaces:**
- Consumes: manifest policy, source loader, character asset builder, report types
- Produces: per-rig prepared clips, `skeleton.npz`, optional FBXs, `preparation-report.json`, and index availability metadata

- [ ] **Step 1: Write the three modality matrix tests**

```python
@pytest.mark.parametrize(
    "case,expected",
    [
        ("bvh_only", {"clips": 1, "model_fbx": None, "presentation_fbx": None}),
        ("fbx_only", {"clips": 1, "model_fbx": "model_character.fbx",
                      "presentation_fbx": "presentation.fbx"}),
        ("paired", {"clips": 1, "model_fbx": "model_character.fbx",
                    "presentation_fbx": "presentation.fbx"}),
    ],
)
def test_build_matrix(case, expected, corpus_fixture):
    result = build_all(corpus_fixture.config(case))
    assert result.failures == []
    assert corpus_fixture.outputs(case) == expected
```

Assert BVH-only report stages are `skipped` with reason rather than failures. Assert a declared-but-missing file is a failure.

- [ ] **Step 2: Verify failure**

Run: `docker compose run --rm test pytest tests/build/test_optional_modalities.py -v`

Expected: FAIL because build passes assume prepared BVH and legacy mesh conventions.

- [ ] **Step 3: Generalize clip discovery and stage ownership**

Replace `PerRigDirectory(pattern="*.bvh")` as the source-of-truth with index records produced by the selected motion adapter. Keep prepared stage-1 clips representable as BVH/NPZ regardless of source modality. Remove `mesh.npz` mismatch logic only after the new character-asset validation is active.

- [ ] **Step 4: Persist availability and reports**

Write optional asset fields as JSON/YAML `null`. Merge pure-Python and Blender audit fragments by stable operation ID, reject duplicates/conflicts, and atomically publish `preparation-report.json`.

- [ ] **Step 5: Run build tests**

Run: `docker compose run --rm test pytest tests/build/test_optional_modalities.py tests/build/test_pipeline.py tests/build/test_corpus.py tests/build/test_corpus_yield.py tests/data/test_dataset_read_path.py -v`

Expected: PASS.

- [ ] **Step 6: Run one BVH-only and one paired end-to-end fixture**

Run:
`docker compose run --rm test python scripts/build_features.py --config tests/fixtures/corpora/bvh_only.yaml`

Then run the paired fixture's asset pass in `fbx` followed by feature build in `test`. Expected: both produce trainable indices; only paired produces mesh-bearing assets.

- [ ] **Step 7: Commit**

```bash
git add src/poseydon/build scripts/process_dataset_truebones.py scripts/build_features.py tests/build/test_optional_modalities.py tests/build/test_pipeline.py tests/build/test_corpus.py tests/build/test_corpus_yield.py tests/data/test_dataset_read_path.py
git commit -m "feat: prepare bvh-only fbx-only and paired corpora"
```

### Task 9: Expose Representation-Aware CLI Export

**Files:**
- Modify: `src/poseydon/cli.py`
- Modify: `configs/sample.yaml`
- Modify: `docs/guides/quickstart.md`
- Modify: `docs/guides/retargeting.md`
- Create: `tests/ops/test_export_cli.py`

**Interfaces:**
- Consumes: `CharacterExporter`
- Produces: `--representation prepared|presentation|original`, `--format bvh|fbx`, `--facing-from`, and explicit `--bare-skeleton`

- [ ] **Step 1: Write CLI routing and unavailable-artifact tests**

```python
def test_export_cli_routes_prepared_fbx_to_character_exporter(cli, spy_exporter):
    result = cli("export", INPUT, "--rig", "Flamingo",
                 "--representation", "prepared", "--format", "fbx")
    assert result.exit_code == 0
    spy_exporter.prepared_fbx.assert_called_once()

def test_export_cli_reports_missing_surface_without_traceback(cli, bvh_only_corpus):
    result = cli("export", INPUT, "--rig", "Rig", "--format", "fbx")
    assert result.exit_code == 2
    assert "source_surface_missing" in result.stderr
```

- [ ] **Step 2: Verify failure**

Run: `docker compose run --rm test pytest tests/ops/test_export_cli.py -v`

Expected: FAIL because no representation-aware export verb exists.

- [ ] **Step 3: Implement parser and routing**

Default generated outputs to prepared representation. Require `--facing-from` only when historical yaw is requested. Keep bare skeleton export visibly separate and never use it as an FBX fallback.

- [ ] **Step 4: Update guides with modality matrix**

Document which outputs exist for BVH-only, FBX-only, and paired corpora and show how to inspect `preparation-report.json`.

- [ ] **Step 5: Run CLI tests**

Run: `docker compose run --rm test pytest tests/ops/test_export_cli.py tests/ops/test_retarget_run.py -v`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/poseydon/cli.py configs/sample.yaml docs/guides/quickstart.md docs/guides/retargeting.md tests/ops/test_export_cli.py
git commit -m "feat: expose prepared presentation and original exports"
```

### Task 10: Migrate Raw Truebones Data and Remove Legacy Trees

**Files:**
- Modify: `.gitignore`
- Modify: `configs/data/truebones.yaml`
- Modify: all maintained code/tests/docs found by the legacy-path scan
- Create temporarily: `tmp/truebones-source-inventory.sha256`
- Move: `data/truebones/source/` to `data/_raw/truebones/`
- Delete after verification: `data/_raw/Truebones-Z_OO/`
- Delete after consumer migration and tests: `data/truebones/bvh/`, `data/truebones/fbx/`, `data/truebones/mesh/`, `data/truebones/reference/`

**Interfaces:**
- Consumes: verified current raw corpus
- Produces: canonical raw/prepared separation with no fallback path

- [ ] **Step 1: Inventory the exact source tree without modifying it**

Run:
`docker compose run --rm compat sh -lc 'find data/truebones/source -type f -print0 | sort -z | xargs -0 sha256sum > tmp/truebones-source-inventory.sha256'`

Then record file count and bytes in a sibling metadata file. Expected: nonzero counts and no read errors.

- [ ] **Step 2: Verify the placeholder is truly empty**

Run: `docker compose run --rm compat sh -lc 'find data/_raw/Truebones-Z_OO -mindepth 1 -print -quit'`

Expected: no output. If anything is printed, stop; do not delete or move data.

- [ ] **Step 3: Move the raw corpus to the canonical path**

Use an explicit directory rename inside the bind-mounted workspace, then regenerate checksums with paths normalized from each root and compare them. Do not proceed unless file count, byte count, and every checksum match.

- [ ] **Step 4: Update consumers and assert no fallback remains**

Run:
`docker compose run --rm compat sh -lc 'grep -RInE "data/truebones/(source|bvh|fbx|mesh|reference)|truebones/source" src scripts tests configs docs || true'`

Edit each maintained reference to `data/_raw/truebones` or the new rig asset/report path. Delete obsolete reference probes rather than pointing them at a second corpus copy.

- [ ] **Step 5: Run path and stage-1 tests before deletion**

Run: `docker compose run --rm test pytest tests/build/test_layout.py tests/build/test_rest_selection.py tests/build/test_pipeline.py tests/features/test_reference_parity.py -v`

Expected: PASS or an intentional removal of a legacy-reference test whose product invariant is now covered by preparation-report/round-trip tests.

- [ ] **Step 6: Remove only the explicitly approved legacy directories**

Delete exactly:
`data/_raw/Truebones-Z_OO`,
`data/truebones/bvh`,
`data/truebones/fbx`,
`data/truebones/mesh`,
and `data/truebones/reference`.
Immediately verify those paths are absent and `data/_raw/truebones` still matches the inventory. Report that the four legacy data trees are not recoverable from the working tree.

- [ ] **Step 7: Add a layout assertion**

```python
def test_truebones_layout_has_one_raw_root_and_no_legacy_roots():
    assert Path("data/_raw/truebones").is_dir()
    for name in ("source", "bvh", "fbx", "mesh", "reference"):
        assert not (Path("data/truebones") / name).exists()
```

- [ ] **Step 8: Commit code/config/docs separately from untracked raw data**

```bash
git add .gitignore configs src scripts tests docs
git commit -m "refactor: separate raw and prepared truebones data"
```

Do not force-add ignored raw data or generated FBXs.

### Task 11: Regenerate the Blender Visual-Comparison Corpus

**Files:**
- Create: `scripts/build_skeleton_visual_debug.py`
- Create: `tests/build/test_skeleton_visual_debug.py`
- Generated, untracked: `tmp/skel_visual_debug/<Rig>/...`

**Interfaces:**
- Consumes: `CharacterExporter`, corpus index, preparation reports, raw artifacts
- Produces: one auditable folder per rig with only the artifacts available for that rig

- [ ] **Step 1: Write bundle-contract tests**

```python
def test_bundle_uses_manifest_sources_and_records_absence(tmp_path, contexts):
    build_bundle(contexts, tmp_path)
    paired = json.loads((tmp_path / "Paired" / "metadata.json").read_text())
    assert paired["original_bvh"]["sha256"] == contexts.paired.source_bvh_sha256
    bvh_only = json.loads((tmp_path / "BvhOnly" / "metadata.json").read_text())
    assert bvh_only["prepared_fbx"] is None
    assert bvh_only["unavailable"]["prepared_fbx"] == "source_surface_missing"
```

- [ ] **Step 2: Verify failure**

Run: `docker compose run --rm test pytest tests/build/test_skeleton_visual_debug.py -v`

Expected: FAIL because the builder does not exist.

- [ ] **Step 3: Implement deterministic sample selection and copying**

Choose the first sorted action for which all modalities required by that rig's manifest exist. Copy only manifest-declared originals. Include model animation NPZ, prepared BVH, strict prepared FBX, presentation FBX, original BVH/FBX, metadata, source/report hashes, and `null` entries for unavailable artifacts.

- [ ] **Step 4: Run tests and generate all Truebones bundles**

Run:
`docker compose run --rm test pytest tests/build/test_skeleton_visual_debug.py -v`

Then:
`docker compose run --rm fbx blender -b --python scripts/build_skeleton_visual_debug.py -- --corpus data/truebones --raw data/_raw/truebones --out tmp/skel_visual_debug`

Expected: 73 rig folders; every paired rig has mesh-bearing strict/presentation/original FBXs; metadata hashes match the preparation report.

- [ ] **Step 5: Run a non-visual structural audit**

Check every BVH/FBX topology, mesh count, bind/rest residual, frame count, facing selection, and availability reason. Fail the script on any discrepancy instead of merely printing warnings.

- [ ] **Step 6: Commit**

```bash
git add scripts/build_skeleton_visual_debug.py tests/build/test_skeleton_visual_debug.py
git commit -m "feat: build auditable skeleton visual comparison bundles"
```

### Task 12: Full Verification and Documentation Consistency

**Files:**
- Modify as evidence requires: stale tests/comments/docs touched by Tasks 1–11
- No new production interface

**Interfaces:**
- Consumes: complete implementation
- Produces: verified corpus/export behavior and a clean maintained-path scan

- [ ] **Step 1: Re-read production code against the spec**

Trace one BVH-only, one FBX-only, and one paired rig from manifest through preparation, model reduction, exporter, and report. Do not infer correctness from passing tests; compare actual inputs, transformations, and persisted outputs.

- [ ] **Step 2: Run the pure-Python suite**

Run: `docker compose run --rm test pytest -v`

Expected: PASS with no unexpected skips for pure-Python functionality.

- [ ] **Step 3: Run the Blender suite**

Run: `docker compose run --rm fbx blender -b --python-expr "import pytest; raise SystemExit(pytest.main(['tests/io','tests/export','tests/build/test_roundtrip_fbx.py','tests/build/test_fbx_coherence.py','-v']))"`

Expected: PASS; modality-declared absences are assertions, not skips.

- [ ] **Step 4: Run corpus acceptance**

Verify 73 reports, expected optional assets, deterministic report hashes after a second preparation, no undeclared lossy operations, and no legacy-path references.

- [ ] **Step 5: Inspect representative outputs in Blender**

Open at least one simple biped, one promoted-root rig, one rig with auxiliary weighted joints, and each previously discrepant rig (Camel, Crab, Tukan). Compare rest pose, facing, hierarchy, frame motion, and mesh deformation across strict prepared, presentation, and original artifacts.

- [ ] **Step 6: Correct stale tests and comments found by inspection**

Any change must cite observed executable behavior or an explicit new contract. Re-run the owning focused test before continuing.

- [ ] **Step 7: Commit final consistency corrections**

```bash
git add src scripts tests configs docs
git commit -m "test: verify character export across source modalities"
```

# PoseYdon — Raw-Export Preprocessing Package

Date: 2026-09-05
Status: draft

## 1. Purpose

PoseYdon's ingest pipeline (`poseydon.ingest.pipeline.ingest_corpus`) turns a
directory of *clean* BVH files plus per-skeleton manifests into the aligned,
windowed corpus training reads. It has never been run against
`data/truebones/Truebone_Z-OO/`, the vendored raw dump exported directly from
3ds Max Biped rigs, because that dump isn't in a shape `poseydon.io.bvh.load_bvh`
can read: every joint (not just the root) declares position channels, and
many skeletons wrap their true root in a redundant zero-offset child joint.

This spec covers a **dataset-agnostic preprocessing package** — functions that
turn a raw rig export into something PoseYdon's existing ingest can consume —
plus a script that composes them for Truebones specifically. The functions are
factored so a future Mixamo or arbitrary-BVH/FBX script reuses them without
reimplementing the same rig-export quirks.

### Prior attempt

A first version of this (`poseydon.datasets.raw_bvh`, `scripts/create_truebones_dataset.py`)
was implemented, tested, and run end-to-end, then reverted uncommitted: manual
inspection in Blender showed the cleaned animations were unrecognizable
against the reference's own curated output for the same clips. Its tests
passed because they validated function-level behavior on synthetic data and
printed (rather than asserted on) an end-to-end comparison report — nothing
ever forced a human or a test suite to confront a large numeric disagreement.

Before redesigning, each piece of that attempt was checked against the
**actual reference source**, not against its own docstrings, using the
`compat` Docker image (`Dockerfile.compat`, `docker compose build compat`)
which pip-installs the real `Animation`/`BVH`/`Quaternions`/`InverseKinematics`
modules (`git+https://github.com/inbar-2344/Motion.git`) the reference
(`external/neural_motion_blending`) depends on but does not vendor:

- `merge_redundant_root`'s rotation-composition order (offset kept from the
  old root, rotations composed root-then-child, translations summed) matches
  the real `BVH.load`'s merge branch exactly, including which of its two
  branches applies to Truebones-shaped data.
- `freeze_non_root_translation` is not a PoseYdon-only simplification: the
  curated fixture BVHs (`external/neural_motion_blending/assets/truebones/*.bvh`)
  themselves declare `CHANNELS 3` (rotation-only) on every non-root joint, and
  the reference's own `get_hml_aligned_anim` pins every non-root joint's
  position to its offset before writing its final BVH — the same discard.
- `remove_bind_pose`'s quaternion formula, checked term-by-term against the
  real `compute_rots_from_tpos` and the real `Quaternions.__mul__`/`__neg__`
  source, uses the same composition order and the same negation-as-inverse.
- **Not verified, and the standing risk**: T-pose geometry recovery
  (`establish_rest_pose`) reimplemented gradient-descent IK from scratch
  (a bespoke Kabsch initialization plus an inline Adam loop) instead of using
  PoseYdon's own `poseydon.solvers.gradient_ik.GradientIK`, which already
  exists, is registered, and is tested for "reduces position error" but has
  never been benchmarked for *accuracy* against the reference's actual CPU
  solver (`InverseKinematics.animation_from_positions`, a dedicated
  Jacobian-based solver). This redesign closes that gap (§5).

The empirical premise both versions share — that a raw file's declared
`OFFSET` values do not match true bone geometry, so a T-pose file's geometry
must be recovered from its position channels — was independently reverified:
on `BrownBear/__Tpose.bvh`, `Bip01_R_Thigh`'s declared offset length is 21.2
units against a frame-0 parent-to-child position distance of 31.7; several
other joints disagree similarly. This part of the old design's reasoning
holds.

### Success criteria

1. `scripts/preproc_truebones_dataset.py` produces `data/truebones/aligned/*.npz`
   plus `data/truebones/index.jsonl` from the raw `.bvh` files of the seven
   species that already have manifests, using those manifests unchanged —
   the same layout `configs/data/truebones.yaml` already expects.
2. The cleanup logic lives in `src/poseydon/preproc/`, a dataset-agnostic
   package with no Truebones-specific assumptions in its function
   signatures, so a future dataset script reuses it.
3. `poseydon.io.bvh`'s public contract is untouched: `load_bvh` still rejects
   non-root translation.
4. Every stage that has a ground truth to check against is checked against
   it as a **hard, asserting test**, not a printed report:
   - the raw-cleanup functions against the reference's real algorithms
     (via the `compat` image), not just against ported docstrings;
   - `GradientIK`'s T-pose fit against the reference's real IK solver on
     the same target, answering whether the GPU solver is accurate enough
     for this use before anything depends on it;
   - the full raw→clean→align→extract path against the curated fixtures'
     extracted features, for the seven clips that exist under a
     recognizably shared name in both the raw dump and the curated set.
5. `poseydon.io.bvh.load_bvh`/`save_bvh` are cross-checked against the real
   reference reader/writer directly (not only indirectly, through feature
   extraction that happens to depend on them being correct).

## 2. Scope

### This spec

- Raw BVH cleanup: redundant-root merge, non-root-translation freeze,
  bind-pose removal, T-pose geometry recovery via `GradientIK`.
- One composition script, targeting the seven species with existing
  manifests (`BrownBear`, `Coyote`, `Crab`, `Flamingo`, `Goat`, `Scorpion`,
  `Skunk`).
- Reference-parity tests for `io.bvh`, the raw-cleanup functions, and the
  IK-based rest-pose recovery, run against the real reference implementation
  inside the `compat` image.
- A golden-parity test comparing the composed pipeline's output features
  against the curated fixtures, as a hard regression gate.

### Explicitly deferred

- **FBX parsing.** Every pilot species has full raw BVH coverage. The
  package's functions take array data
  (`names, parents, offsets, rotations, positions`), not BVH text, wherever
  that's practical, specifically so an FBX-sourced dataset script could feed
  the same functions later without change — but no FBX parser is written now.
- **Mixamo-specific handling and manifest auto-generation** for the ~66
  Truebones species that don't yet have a manifest. Separate, larger work.
- **Relaxing the fixed-bone-length assumption** in `Anim` (see §6).

## 3. Baseline: what the raw files contain

Unchanged from the prior attempt's inspection, since these are file-format
facts, not algorithm choices:

- Every joint in a raw file declares `CHANNELS 6 Xposition Yposition
  Zposition Zrotation Xrotation Yrotation` — a 3ds Max Biped export
  artifact.
- Some skeletons wrap their true root in a redundant, zero-offset,
  single-child joint (`ROOT Hips` → `JOINT Bip01_Pelvis` with
  `OFFSET 0 0 0`). Confirmed per-species: `BrownBear`, `Coyote`, `Flamingo`,
  `Goat`, `Skunk` have this shape; `Crab`'s root has a single child at a
  non-zero offset (no merge), `Scorpion`'s root has two children (no merge).
- Non-root joints' position channels carry real per-frame variation, not
  redundant noise — confirmed on `BrownBear/__RiseSwat.bvh`. PoseYdon's
  `Anim` has no field to carry per-joint translation, so this is discarded,
  matching the curated fixtures' own final convention (§1).
- Raw files declare 9–15 more joints than the curated fixtures per species,
  entirely accounted for by `End Site` leaves that `poseydon.io.bvh.load_bvh`
  treats as ordinary joints (its own documented convention) while the
  fixtures omit. Comparisons in this spec match by joint **name**
  intersection, never by joint count.
- Raw files bake a large, constant, per-joint rotation into every clip of a
  skeleton — an exporter axis-convention artifact, not real animation. This
  is what bind-pose removal exists to undo.

## 4. Architecture

### `src/poseydon/preproc/raw_bvh.py`

```python
def merge_redundant_root(
    names: tuple[str, ...],
    parents: np.ndarray,
    offsets: np.ndarray,
    rotations: np.ndarray,   # (F, J, 4) quats, one per joint
    positions: np.ndarray,   # (F, J, 3), one per joint
) -> tuple[tuple[str, ...], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Merge a zero-offset single child of the root into the root.

    Applies only the one branch of the reference's redundant-root handling
    that occurs in Truebones data (root has exactly one child, that child's
    offset is ~zero): new offset is the old root's; new rotation is
    root-then-child composed; new translation is the two summed; the old
    root is dropped. Raises if the shape doesn't match -- callers should
    treat that as "nothing to merge".
    """

def freeze_non_root_translation(positions: np.ndarray) -> np.ndarray:
    """Discard every non-root joint's animated translation.

    Returns the root's own (F, 3) trajectory unchanged. See §1/§3 for why
    this matches the reference's own eventual output, not a PoseYdon-only
    shortcut.
    """

def load_raw_biped_bvh(path: str | Path) -> Anim:
    """Read a raw multi-channel Biped BVH export into a valid Anim.

    Applies merge_redundant_root when the shape matches, then
    freeze_non_root_translation. Reuses poseydon.io.bvh's hierarchy/motion
    parsing via the shared _channels_to_arrays split (§4.3) -- only
    load_bvh's "only the root may translate" restriction differs.
    """
```

### `src/poseydon/preproc/rest_pose.py`

```python
def recover_raw_global_pose(
    rotations: np.ndarray,  # (F, J, 4), post-merge
    positions: np.ndarray,  # (F, J, 3), post-merge, every joint's raw translation
    parents: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Global positions/rotations treating every joint's raw (rotation,
    position) pair as its own local transform -- the only way to see a raw
    T-pose file's true geometry, since these files encode it across both
    channels together (see §1's OFFSET-vs-position discrepancy).
    """

def establish_rest_pose(path: str | Path, iterations: int = 150) -> Anim:
    """Recover a raw T-pose file's true geometry via GradientIK.

    Parses and merges the raw file (raw_bvh._parse_and_merge), recovers
    frame 0's true global positions (recover_raw_global_pose), then fits
    PoseYdon's rotation-only, rigid-bone-length representation to those
    positions using the existing poseydon.solvers.gradient_ik.GradientIK
    with the registered "position" IKTerm -- no new solver code. Final
    offsets are the fitted pose's own bone vectors, expressed in each
    joint's parent-local frame.
    """

def remove_bind_pose(anim: Anim, rest_anim: Anim) -> Anim:
    """Re-express anim so zero rotation on every joint reproduces rest_anim.

    Ports compute_rots_from_tpos, verified term-by-term against the real
    reference source (§1). Matched by joint NAME -- rest_anim need not cover
    every joint anim does; a joint missing from rest_anim falls back to
    anim's own frame-0 rotation as its bind (exact for a childless End Site,
    an approximation elsewhere).
    """
```

### `src/poseydon/io/bvh.py` (internal split, no contract change)

Split `_channels_to_local` into a reusable `_channels_to_arrays` (per-joint
rotations AND positions, no root-only enforcement) plus a thin
`_channels_to_local` wrapper that raises `BvhParseError` for non-root
position channels — exactly as the prior attempt did. `load_bvh`'s public
behaviour is unchanged; `raw_bvh.py` imports `_channels_to_arrays` to avoid
duplicating the euler-batching parse logic.

### `scripts/preproc_truebones_dataset.py`

Same shape as the prior attempt's pilot script: for each of the seven
species, resolve a rest reference (a raw `*Tpose*.bvh` file via
`establish_rest_pose` when present, else the first frame of the
alphabetically-first raw clip — the same fallback
`MotionDataset._rest_frame` already uses), clean every raw clip
(`load_raw_biped_bvh` → `remove_bind_pose`), write each to a scratch BVH via
`poseydon.io.bvh.save_bvh`, and hand the results to the unmodified
`poseydon.ingest.pipeline.ingest_corpus`.

## 5. Validation strategy

Three tiers, each catching a different class of bug:

### 5.1 Structural unit tests (main test image)

`merge_redundant_root` and `freeze_non_root_translation` against synthetic
hierarchies (redundant child present / absent / non-zero-offset child
raises; root translation untouched, non-root replaced). Cheap, fast, run in
every CI invocation. Necessary but — per the prior attempt — explicitly not
sufficient on their own.

### 5.2 Reference-parity tests (new; `compat` image only)

The `compat` image already exists to "verify PoseYdon against the published
implementation" (its own Dockerfile comment) but nothing uses it yet. Every
test module in this tier opens with `pytest.importorskip("BVH")` so the main
suite (`docker compose run --rm test pytest`) is entirely unaffected; these
run via `docker compose run --rm compat pytest tests/io/test_reference_parity.py
tests/preproc/test_reference_parity.py`.

A small shared helper (`tests/reference_compat.py`) converts between the
reference's `Animation` (scalar-first quaternions, `(w, x, y, z)`) and
PoseYdon's `Anim` (scalar-last, `(x, y, z, w)`) arrays, so each test module
isn't reimplementing that reorder.

- **`tests/io/test_reference_parity.py`**: for each curated fixture,
  `poseydon.io.bvh.load_bvh` vs. the real `BVH.load` — global positions
  agree to a tight tolerance. This is a direct read check, one hop closer
  to ground truth than the existing golden-parity test (which only checks
  that features extracted from PoseYdon's read reproduce a `.npy` that was
  itself produced by the reference's *own* pipeline, an indirect check).
  Then a round trip: an `Anim` written via `poseydon.io.bvh.save_bvh`, read
  back by the real `BVH.load`, confirming PoseYdon's writer output is
  genuinely standards-agreeing BVH, not merely self-consistent with
  PoseYdon's own reader.
- **`tests/preproc/test_reference_parity.py`**:
  - `merge_redundant_root` vs. the real `BVH.load`'s merge branch, on the
    actual raw files for the species where it applies.
  - The `GradientIK`-based fit in `establish_rest_pose` vs. the reference's
    real `InverseKinematics.animation_from_positions`
    (`BasicInverseKinematics`), on the same T-pose target positions for
    each of the seven species — the accuracy benchmark. If `GradientIK`
    disagrees materially, that is a finding to fix (more iterations, a
    warm start, an added `IKTerm`) with a concrete number to close, not a
    plausibility argument.

### 5.3 Golden-parity end-to-end gate (main test image)

`tests/preproc/test_golden_parity.py`, modeled on
`tests/features/test_golden_parity.py`: for the seven (raw clip, curated
fixture) pairs already identified by name
(e.g. `BrownBear/__RiseSwat.bvh` ↔ `BrownBear___RiseSwat_132`), run the full
`load_raw_biped_bvh → remove_bind_pose → align → extract_features` path and
**assert** each feature block's error against the fixture's own extracted
features is under a tolerance wide enough for the known approximation
(discarded secondary motion from frozen non-root translation) but tight
enough that a structural or sign bug fails the suite outright — unlike the
prior attempt's version of this comparison, which only printed numbers.

## 6. Interaction with future work

Unchanged from the prior attempt's reasoning, reconfirmed as still true:

- **Relaxing the fixed-bone-length assumption.** `freeze_non_root_translation`
  stays a single, separately-named function; a future `Anim` able to carry
  per-joint translation replaces its call site, not the surrounding
  structure. `merge_redundant_root` has no dependency on it.
- **FBX / Mixamo.** Reusing `establish_rest_pose`/`remove_bind_pose` for a
  different raw format means writing a new parser that produces the same
  `(names, parents, offsets, rotations, positions)` shape `raw_bvh._parse_and_merge`
  does — the rest of `preproc/` needs no change.

## 7. Testing plan (summary)

- `tests/preproc/test_raw_bvh.py`: structural unit tests, §5.1.
- `tests/io/test_reference_parity.py`, `tests/preproc/test_reference_parity.py`:
  compat-only, §5.2. Skip via `pytest.importorskip` when `Animation`/`BVH`/
  `Quaternions`/`InverseKinematics` aren't importable.
- `tests/preproc/test_golden_parity.py`: end-to-end hard gate, §5.3.
- `scripts/preproc_truebones_dataset.py`: an integration test running it
  against a small subset (one clip per species) and asserting the output
  index + npz files are readable by `MotionDataset`.

## 8. Implementation status

Not yet implemented. This spec supersedes the reverted prior attempt
(originally `docs/superpowers/specs/2026-09-05-truebones-preprocessing-design.md`,
deleted uncommitted along with its implementation once the output was found
to be unrecognizable in visual review).

# PoseYdon — Truebones Raw Preprocessing (Pilot)

Date: 2026-09-05
Status: draft

## 1. Purpose

`data/truebones/Truebone_Z-OO/` is the *original* vendored dump: BVH and FBX
files exported directly from 3ds Max Biped rigs, with no preprocessing at all.
PoseYdon's ingest pipeline (`poseydon.ingest.pipeline.ingest_corpus`) already
turns a directory of *clean* BVH files plus per-skeleton manifests into the
aligned, windowed corpus training reads -- but it has never been run against
the raw dump, because the raw dump isn't in a shape it can read.

Seven skeletons (BrownBear, Coyote, Crab, Flamingo, Goat, Scorpion, Skunk)
already have hand-authored manifests (`data/truebones/skeletons/*.yaml`) and
hand-curated *clean* BVH fixtures (`external/neural_motion_blending/assets/`,
`data/truebones/reference/*.npz`) used only by tests today. Nothing in the
repo yet turns the raw dump itself into training data.

This spec covers a **pilot**: prove out raw-BVH cleanup end to end for those
same seven skeletons, reusing their existing manifests unchanged, so the
approach is validated before it's pointed at the other ~66 species. Manifest
auto-generation and FBX support are explicitly out of scope here (see §2).

### Success criteria

1. A new script produces `data/truebones/aligned/*.npz` +
   `data/truebones/index.jsonl` from the raw `.bvh` files of the seven pilot
   species, using their existing manifests -- exactly the layout
   `configs/data/truebones.yaml` already expects, so `poseydon train` works
   against it with no config changes.
2. The cleanup logic lives in a new, dataset-agnostic module so a future
   dataset script hitting the same rig-export quirk (redundant root,
   per-joint translation) reuses it rather than reimplementing it.
3. `poseydon.io.bvh` is untouched in its public contract: `load_bvh` still
   rejects non-root translation. The pilot does not weaken that invariant.
4. For at least one clip per pilot species that exists in both the raw dump
   and the curated fixture set under a recognizably shared name, features
   extracted from (raw → new cleanup → existing `ingest_clip`) are compared
   against features extracted from (fixture BVH → existing `ingest_clip`).
   Root trajectory and major-joint rotation should agree closely; the report
   is a per-joint error table, not a single pass/fail assertion, since the
   two paths are not expected to be bit-identical (see §3).

## 2. Scope

### Pilot (this spec)

- Raw BVH cleanup: redundant-root merge + non-root translation freeze.
- One script, hardcoded to the seven pilot species and their existing
  manifests.
- Validation report comparing raw-cleaned vs. fixture-cleaned features for a
  shared clip per species.

### Explicitly deferred

- **Raw FBX parsing.** Every pilot species already has full raw BVH coverage.
  FBX is a materially larger, separate parser with nothing in this repo to
  build on. A future pass.
- **Manifest auto-generation** (facing pairs from the reference's
  `FACE_JOINTS` table, T-pose/foot-joint auto-detection) for the ~66
  species that don't yet have a manifest. The pilot's seven species already
  have manifests; extending coverage is separate, larger work.
- **Relaxing the fixed-bone-length assumption.** See §4.

## 3. Baseline: what the raw files actually contain

Confirmed by inspection of the raw dump and the reference's own BVH loader
(`external/neural_motion_blending`'s vendored `BVH.py`/`Animation.py`):

- Every joint declares `CHANNELS 6 Xposition Yposition Zposition Zrotation
  Xrotation Yrotation` -- not just the root. This is a 3ds Max Biped export
  artifact, not a PoseYdon-specific format choice.
- The two joints under the root are redundant: `ROOT Hips` (the true
  world-space rest offset) followed immediately by `JOINT Bip01_Pelvis` with
  `OFFSET 0 0 0`. The reference's loader special-cases exactly this shape:
  when the root has a single, zero-offset child, it merges them into one
  joint -- offset from the old root, rotation composed
  (`old_root_rot * old_child_rot`), translation summed
  (`old_root_pos + old_child_pos`) -- and drops the old root. This is the
  **only** one of the reference's two merge branches that applies to
  Truebones data (the other handles a joint-1-is-a-leaf case that doesn't
  occur here).
- Below that merged root, non-root joints' position channels are **not**
  redundant noise -- they carry real per-frame variation (verified
  numerically on `BrownBear/__RiseSwat.bvh`: a mid-spine joint's declared
  position moves frame to frame, it isn't pinned to its `OFFSET`). The
  reference's `Animation.transforms_local` uses `positions` as the
  translation component of *every* joint's local transform, not just the
  root's -- their skeleton model allows every joint to translate.

PoseYdon's `Anim`/`ResolvedSkeleton` model does not: bone lengths
(`offsets`) are fixed per skeleton, and only the root translates. This
contract is load-bearing elsewhere (the topology-augmentation FK-exactness
proofs, `poseydon.io.bvh.load_bvh`'s explicit rejection of non-root
position channels). So "cleaning" a raw clip is not just a channel-count
fix: for every non-root joint, the animated translation is **frozen** to
that joint's declared rest offset, and only its rotation is kept as
animated signal. This discards whatever secondary motion (rig
squash/stretch, jiggle) the raw translation channel carried on non-root
joints.

This is a deliberate simplification matching how the rest of this codebase
already treats skeletons, not an oversight. See §5 for how this is kept
easy to relax later.

Two things are *not* the cleanup step's concern, because they're already
handled downstream: raw files are in different units/orientation than the
curated fixtures, but `poseydon.ingest.align` (rotate to face +Z, scale to
mean bone length, ground) already normalizes exactly that, per clip, using
the manifest that's unchanged from today. The cleanup step's only job is to
produce a *structurally valid* `Anim` -- correct hierarchy, single-root
translation, plausible per-frame rotations -- not one that's numerically
pre-aligned.

## 4. Architecture

### `src/poseydon/datasets/raw_bvh.py` (new)

A dataset-agnostic module for raw BVH hygiene -- named for the class of
problem (raw Biped-rig exports), not for Truebones, so a future dataset
script hitting the same export quirk reuses it.

```python
def merge_redundant_root(
    names: tuple[str, ...],
    parents: np.ndarray,
    offsets: np.ndarray,
    rotations: np.ndarray,   # (F, J, 4) quats, one per joint
    positions: np.ndarray,   # (F, J, 3), one per joint
) -> tuple[tuple[str, ...], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Merge a zero-offset single child of the root into the root.

    Ports the one branch of the reference's redundant-root handling that
    applies to Truebones data: root offset kept, rotations composed
    (root then child), translations summed, old root dropped. Raises if the
    root's single child does not have a near-zero offset -- callers should
    treat that as "nothing to merge", not call this function.
    """

def freeze_non_root_translation(
    offsets: np.ndarray,       # (J, 3), post-merge
    positions: np.ndarray,     # (F, J, 3), post-merge
) -> np.ndarray:
    """Replace every non-root joint's animated position with its offset.

    Returns new (F, 3) root_pos (positions[:, 0] unchanged) -- non-root
    translation is discarded entirely, not averaged or sampled, since
    PoseYdon's Anim has no field to carry it.
    """

def load_raw_biped_bvh(path: str | Path) -> Anim:
    """Read a raw multi-channel Biped BVH export into a valid Anim.

    Applies merge_redundant_root when the shape matches, then
    freeze_non_root_translation. Reuses poseydon.io.bvh's hierarchy/motion
    grammar parsing -- only the "only the root may translate" restriction
    differs from load_bvh.
    """
```

### `src/poseydon/io/bvh.py` (small refactor, no contract change)

`_channels_to_local` currently does two things in one pass: convert MOTION
values into per-joint rotations, and enforce "only joint 0 may carry
position channels" while building `root_pos`. Split the conversion out so
`raw_bvh.py` can reuse it without duplicating the euler-batching logic:

```python
def _channels_to_arrays(values, channels, n_joints) -> tuple[np.ndarray, np.ndarray]:
    """Per-joint rotations (F, J, 4) and per-joint positions (F, J, 3).

    Joints with no position channels get zeros. No contract enforcement --
    callers decide what a non-root position column means.
    """
```

`_channels_to_local` becomes a thin wrapper: call `_channels_to_arrays`,
then raise `BvhParseError` if any joint beyond index 0 has a non-zero
position column, then return `rotations, positions[:, 0]`. `load_bvh`'s
public behaviour is unchanged -- this is an internal split, not a new
public function, so it needs no test beyond the existing `test_bvh.py`
suite continuing to pass.

### `scripts/create_truebones_dataset.py` (new)

Pilot-scoped, not general:

```python
PILOT_SKELETONS = ("BrownBear", "Coyote", "Crab", "Flamingo", "Goat", "Scorpion", "Skunk")

def main() -> None:
    manifest_dir = Path("data/truebones/skeletons")
    raw_root = Path("data/truebones/Truebone_Z-OO")
    out_dir = Path("data/truebones")

    index = CorpusIndex()
    for skeleton in PILOT_SKELETONS:
        manifest = SkeletonManifest.load(manifest_dir / f"{skeleton}.yaml")
        raw_paths = sorted((raw_root / skeleton).glob("*.bvh"))
        # Clean each raw file into a temp Anim, write it where ingest_clip
        # expects a loadable BVH, then defer to the existing pipeline --
        # ingest_clip calls poseydon.io.bvh.load_bvh internally, so cleaned
        # output must round-trip through save_bvh first.
        ...
    index.save(out_dir / "index.jsonl")
```

Open implementation question the plan must settle: `ingest_clip` calls
`poseydon.io.bvh.load_bvh(bvh_path)` internally rather than accepting an
`Anim` directly. The pilot script therefore either (a) writes each cleaned
`Anim` out via `poseydon.io.bvh.save_bvh` to a scratch location and calls
`ingest_clip` on that path, or (b) inlines a version of `ingest_clip`'s body
that takes an `Anim` instead of a path. (a) reuses more existing, tested
code and keeps `ingest_clip`'s signature untouched; the plan should adopt it
unless it proves awkward (e.g. `save_bvh`'s lossy `.6f` formatting mattering
at this stage -- it shouldn't, since alignment downstream tolerates far
larger differences than 1e-6).

### Validation report

A script (or a test) that, per pilot species, finds one clip present under
recognizably the same name in both the raw dump and the curated fixture set,
runs each through `ingest_clip` + `extract_features`, and prints/asserts a
per-block error summary (e.g. max/mean abs error for `ric_pos`, `rot6d`)
rather than exact equality. This is diagnostic output for a human to read,
not a strict regression gate -- the two paths are expected to diverge
somewhat wherever a joint's raw motion depended on the translation channel
this pilot freezes.

## 5. Interaction with future work

The user has flagged two follow-ups this design should not foreclose:

- **Relaxing the fixed-bone-length assumption.** `freeze_non_root_translation`
  is a single, separately-named function with one job -- discard non-root
  translation -- rather than logic inlined into the parser. A future version
  of `Anim` that *can* carry per-joint translation would replace this one
  function's call site, not restructure `raw_bvh.py` or the ingest
  pipeline. `merge_redundant_root` has no dependency on the freeze step and
  survives such a change unchanged.
- **Per-frame mesh recovery.** Out of scope for this spec entirely; noted
  here only so a future mesh-recovery pass knows raw FBX parsing (deferred
  in §2) is likely the more direct route to skinning data than the BVH path
  this spec builds.

## 6. Testing plan

- `tests/datasets/test_raw_bvh.py`: `merge_redundant_root` on synthetic
  hierarchies (redundant child present / absent / non-zero-offset child
  should raise); `freeze_non_root_translation` (root position untouched,
  non-root positions replaced by input offsets); `load_raw_biped_bvh`
  against each of the seven pilot species' raw T-pose file -- assert it
  loads without error, joint count matches the manifest's expectations, and
  round-trips through `poseydon.core.anim.Anim.check_topological_order`.
- `tests/io/test_bvh.py`: extend to cover the `_channels_to_arrays` split
  (existing `load_bvh` tests continue to pass unmodified -- this is the
  regression gate for "no contract change").
- Pilot script: an integration test running it against a tiny subset (one
  clip per species, not the full raw directory, to keep test runtime
  reasonable) and asserting the output index + npz files are readable by
  `MotionDataset`.
- Validation report: run manually (or as a slow/marked test) across the
  seven species' full clip sets once the above pass; its output is reviewed
  by a human, not asserted against a fixed tolerance, per §4.

## 7. Implementation status

Not started. This spec is written to hand off to `writing-plans`.

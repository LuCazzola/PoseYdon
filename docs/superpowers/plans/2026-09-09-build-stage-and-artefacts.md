# Build Stage and Artefacts (A2) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn 1145 prepared clips into the artefacts a training run reads — `skeleton.npz`, `stats.npz`, per-clip `.npz`, and a regenerated `index.jsonl`.

**Architecture:** A new `poseydon/build/` package holds four passes over the prepared corpus, driven by a thin Hydra runner `scripts/build_features.py` and a `configs/dataset/truebones.yaml` config group. Normalization becomes a declared per-block policy rather than a fixed `Normalizer.fit`. Joint names gain a computed `Humanize` stage, retiring the parsed-but-unread `strip_joint_prefix`. Nothing in `build/` is imported by training: training reads `features/` and `core/`.

**Tech Stack:** Python 3.12, NumPy, Hydra, pytest, Docker Compose. No new runtime dependencies (T5 stays behind the `poseydon[text]` extra and is not built here).

**Spec:** `docs/superpowers/specs/2026-09-08-training-run-and-retarget-validation-design.md` §2, which amends `docs/superpowers/specs/2026-09-06-data-pipeline-and-training-parity-design.md` §2, §4, §5, §6, §7, §8. Both are binding; where they differ the 2026-09-08 spec wins.

This is plan **A2** of three. A1 (stage-1 corrections and the corpus rebuild) is merged at `8baaeb9`. A3 covers the read path, training loop, wandb and the aarch64 GPU image.

## Global Constraints

- **Everything runs in Docker.** `docker compose run --rm test <cmd>`. Never invoke `pytest` or the build script on the host.
- **`ruff` must be clean** before every commit: `docker compose run --rm test ruff check .`
- `data/truebones/source/` is a **paid, irreplaceable dataset**. Read-only, always. `data/truebones/clips/*.bvh` and `rigs/*/prepare.npz` are A1's output — also read-only in this plan; nothing here re-runs stage 1.
- **One writer per file** (parity spec §2). Stage 1 owns `prepare.npz`, `mesh.npz` and the prepared BVH/FBX. Stage 2 owns `skeleton.npz`, `stats.npz`, `clips/<Rig>/<action>.npz` and `index.jsonl`. A task that writes outside stage 2's set is wrong.
- **Nothing in `build/` may be imported by training code.** `build/` is consumed by `scripts/`; `features/` and `core/` are what training imports. A `from poseydon.build...` inside `data/`, `models/` or `training/` is a defect.
- The canonical mean bone length is `HML_MEAN_BONE_LENGTH = 0.20921428571428569` (`poseydon.core.skeleton`).
- Joint arrays are hierarchy-ordered (a parent always precedes its children); quaternions are scalar-last.
- **Assertions must measure, not restate.** A1's reviews caught a tautological assertion, a test that could check zero rigs, and a tolerance eight orders above its measured error. Every assertion here compares against an independently-derived quantity.
- **A test that skips is a test that passes.** Any new `pytest.skip` must be justified by absent data or absent Blender, never by a rig the code cannot handle.

## What A1 left you

Read these before Task 1; they are the interfaces this plan builds on.

| thing | where | note |
|---|---|---|
| `SkeletonManifest.rest_pose: str \| None` | `core/skeleton.py` | a raw filename, e.g. `"__IdleLoop.bvh"`; all 73 rigs declare one |
| `rest_action(manifest) -> str` | `scripts/process_dataset_truebones.py` | the prepared-clip slug for that rig's rest pose. **Call this; never re-derive it and never match on a filename.** Crab's is `walk`, Trex's `walk_loop` |
| `rest_source(manifest, clip_paths) -> Path` | same | raises rather than falling back |
| `PROMOTE_ROOT: dict[str, str]` | same | 14 rigs whose root was moved off a ground locator |
| `RigTransform.load(path)` | `build/prepare.py` | `.rig_params` and `.clip_params[clip]`; per-clip `face_axis/rotation` is the quaternion each clip needs folded into its `.npz` |
| prepared corpus | `data/truebones/clips/<Rig>/*.bvh` | 1145 clips, 73 rigs |
| `mesh.npz` | `rigs/<Rig>/mesh.npz` | **only 5 rigs have one, and it is stale for any promoted rig** — see Task 8 |

Three rigs (Tukan 0.7579, Trex 0.8207, Crow 0.9433) have prepared clips that do not face +Z, xfailed with measured values in `tests/build/test_corpus_facing.py`. Out of scope here; do not try to fix them, and do not remove their xfails.

---

### Task 1: The per-block normalization policy

Spec: parity §5. `Normalizer.fit` currently takes per-channel mean and std over everything. Per-channel std makes a barely-moving toe unit-variance, so reconstruction loss weights it as heavily as the root; and dividing a 6D rotation row per-channel changes the rotation it recovers under Gram-Schmidt, while dividing by a uniform scalar does not. Pooling is what lets the rotation block survive normalization at all.

**Files:**
- Modify: `src/poseydon/data/normalize.py`
- Test: `tests/data/test_normalize.py` (create the directory if absent, with `__init__.py`)

**Interfaces:**
- Consumes: `FeatureSpec` (`core/spec.py`) with `.blocks: tuple[tuple[str, int], ...]`, `.dim`, `.names`, `.slice(name)`; `Normalizer(mean, std, spec)` with `.normalize`, `.denormalize`, `.save`, `.load`.
- Produces:
  ```python
  @dataclass(frozen=True)
  class BlockPolicy:
      name: str
      center: bool = True
      scale: str = "channel"   # channel | joint_block | block | none

  Normalizer.fit(arrays, spec, policy: Sequence[BlockPolicy] | None = None) -> Normalizer
  ```
  `policy=None` keeps today's behaviour exactly, so every existing caller is unaffected.

- [ ] **Step 1: Write the failing tests**

Create `tests/data/test_normalize.py`:

```python
"""The per-block normalization policy (parity spec §5).

Each `scale` mode must produce statistics of a documented SHAPE, and
`joint_block` must leave a 6D rotation row's recovered rotation unchanged under
Gram-Schmidt -- which is the property that makes the rotation block survivable
at all, and the reason pooling exists rather than per-channel scaling.
"""

from __future__ import annotations

import numpy as np
import pytest

from poseydon.core.spec import FeatureSpec
from poseydon.data.normalize import BlockPolicy, Normalizer

SPEC = FeatureSpec((("ric_pos", 3), ("rot6d", 6), ("foot_contact", 1)))


def _clips(n_clips: int = 3, frames: int = 20, joints: int = 5) -> list[np.ndarray]:
    rng = np.random.default_rng(0)
    out = []
    for _ in range(n_clips):
        a = rng.normal(size=(frames, joints, SPEC.dim))
        # Give channels within a block deliberately different scales, so a
        # pooled statistic is provably not the per-channel one.
        a[..., SPEC.slice("ric_pos")] *= np.array([1.0, 10.0, 100.0])
        a[..., SPEC.slice("foot_contact")] = 1.0     # zero variance, on purpose
        out.append(a)
    return out


def _gram_schmidt(row: np.ndarray) -> np.ndarray:
    """The 6D-to-rotation recovery: two vectors, orthonormalized."""
    a, b = row[:3], row[3:]
    e1 = a / np.linalg.norm(a)
    rest = b - (e1 @ b) * e1
    e2 = rest / np.linalg.norm(rest)
    return np.stack([e1, e2, np.cross(e1, e2)])


def test_channel_mode_keeps_a_statistic_per_joint_and_channel():
    fit = Normalizer.fit(_clips(), SPEC, [BlockPolicy("ric_pos", scale="channel")])
    block = fit.std[:, SPEC.slice("ric_pos")]
    # The three channels were scaled 1/10/100, so per-channel stds must differ.
    assert not np.allclose(block[:, 0], block[:, 1])


def test_joint_block_mode_gives_one_scalar_per_joint_and_block():
    fit = Normalizer.fit(_clips(), SPEC, [BlockPolicy("ric_pos", scale="joint_block")])
    block = fit.std[:, SPEC.slice("ric_pos")]
    for joint in range(block.shape[0]):
        assert np.allclose(block[joint], block[joint, 0]), (
            "joint_block must be uniform across the block's channels"
        )
    # Different joints may still differ; if they did not, this would be `block`.
    assert block.shape == (5, 3)


def test_block_mode_gives_one_scalar_for_root_and_one_for_the_rest():
    fit = Normalizer.fit(_clips(), SPEC, [BlockPolicy("ric_pos", scale="block")])
    block = fit.std[:, SPEC.slice("ric_pos")]
    assert np.allclose(block[0], block[0, 0])
    assert np.allclose(block[1:], block[1, 0]), "non-root joints must share one scalar"


def test_none_mode_leaves_std_at_one():
    fit = Normalizer.fit(
        _clips(), SPEC, [BlockPolicy("foot_contact", center=False, scale="none")]
    )
    np.testing.assert_allclose(fit.std[:, SPEC.slice("foot_contact")], 1.0)
    np.testing.assert_allclose(fit.mean[:, SPEC.slice("foot_contact")], 0.0)


def test_joint_block_preserves_the_recovered_rotation():
    """The property, not an example: a uniform divisor commutes with Gram-Schmidt.

    Per-channel scaling does not, which is why the rotation block is pooled.
    """
    fit = Normalizer.fit(_clips(), SPEC, [BlockPolicy("rot6d", center=False,
                                                     scale="joint_block")])
    rng = np.random.default_rng(1)
    raw = rng.normal(size=(1, 5, SPEC.dim))
    normalized = fit.normalize(raw)

    for joint in range(5):
        before = _gram_schmidt(raw[0, joint, SPEC.slice("rot6d")])
        after = _gram_schmidt(normalized[0, joint, SPEC.slice("rot6d")])
        np.testing.assert_allclose(after, before, atol=1e-12)


def test_per_channel_scaling_does_NOT_preserve_the_recovered_rotation():
    """The contrast that makes the previous test meaningful."""
    fit = Normalizer.fit(_clips(), SPEC, [BlockPolicy("rot6d", center=False,
                                                      scale="channel")])
    rng = np.random.default_rng(1)
    raw = rng.normal(size=(1, 5, SPEC.dim))
    normalized = fit.normalize(raw)

    before = _gram_schmidt(raw[0, 0, SPEC.slice("rot6d")])
    after = _gram_schmidt(normalized[0, 0, SPEC.slice("rot6d")])
    assert not np.allclose(after, before, atol=1e-6)


def test_no_policy_reproduces_the_previous_behaviour():
    """Every existing caller must be unaffected."""
    clips = _clips()
    np.testing.assert_allclose(
        Normalizer.fit(clips, SPEC).std, Normalizer.fit(clips, SPEC, None).std
    )


def test_an_unknown_scale_mode_is_refused():
    with pytest.raises(ValueError, match="nonsense"):
        Normalizer.fit(_clips(), SPEC, [BlockPolicy("ric_pos", scale="nonsense")])


def test_a_policy_naming_an_absent_block_is_refused():
    with pytest.raises(ValueError, match="no_such_block"):
        Normalizer.fit(_clips(), SPEC, [BlockPolicy("no_such_block")])


def test_the_policy_survives_save_and_load(tmp_path):
    policy = [BlockPolicy("ric_pos", scale="joint_block"),
              BlockPolicy("rot6d", center=False, scale="joint_block"),
              BlockPolicy("foot_contact", center=False, scale="none")]
    fit = Normalizer.fit(_clips(), SPEC, policy)
    path = tmp_path / "stats.npz"
    fit.save(path)
    back = Normalizer.load(path)
    np.testing.assert_allclose(back.mean, fit.mean)
    np.testing.assert_allclose(back.std, fit.std)
    assert back.spec.blocks == fit.spec.blocks
```

Create `tests/data/__init__.py` (empty) if the directory does not exist.

- [ ] **Step 2: Run to verify failure**

Run: `docker compose run --rm test pytest tests/data/test_normalize.py -v`
Expected: FAIL — `ImportError: cannot import name 'BlockPolicy'`.

- [ ] **Step 3: Implement**

In `src/poseydon/data/normalize.py`, add above `Normalizer`:

```python
@dataclass(frozen=True)
class BlockPolicy:
    """How one feature block is normalized.

    Declared per block in the dataset config rather than fixed in code, because
    the right answer differs by block. Per-channel std makes a barely-moving toe
    unit-variance, so reconstruction loss weights it as heavily as the root;
    pooling preserves relative magnitude. More sharply: dividing a 6D rotation
    row by a UNIFORM scalar leaves the rotation unchanged after Gram-Schmidt,
    while dividing per-channel does not -- so pooling is what lets the rotation
    block survive normalization at all.
    """

    name: str
    center: bool = True
    #: ``channel`` per (joint, channel); ``joint_block`` one scalar per (joint,
    #: block); ``block`` one scalar per (root / non-root, block); ``none`` leaves
    #: std at 1, for flags.
    scale: str = "channel"


_SCALE_MODES = ("channel", "joint_block", "block", "none")
```

Replace `Normalizer.fit` with:

```python
    @classmethod
    def fit(
        cls,
        arrays: Iterable[np.ndarray],
        spec: FeatureSpec,
        policy: Sequence[BlockPolicy] | None = None,
    ) -> Normalizer:
        """Fit over clips of one skeleton, each ``(frames, joints, dim)``.

        ``policy`` declares per-block centring and pooling. ``None`` keeps the
        per-channel behaviour every existing caller expects.
        """
        stacked = np.concatenate([np.asarray(a, dtype=np.float64) for a in arrays], axis=0)
        if stacked.ndim != 3:
            raise ValueError(f"expected (frames, joints, dim) arrays, got {stacked.ndim} dims")

        mean = stacked.mean(axis=0)
        std = stacked.std(axis=0) + STD_EPSILON
        if policy is None:
            return cls(mean=mean, std=std, spec=spec)

        known = set(spec.names)
        for entry in policy:
            if entry.name not in known:
                raise ValueError(
                    f"policy names block `{entry.name}`, which the spec does not "
                    f"declare. Available: {', '.join(sorted(known))}"
                )
            if entry.scale not in _SCALE_MODES:
                raise ValueError(
                    f"block `{entry.name}` has unknown scale mode `{entry.scale}`. "
                    f"Available: {', '.join(_SCALE_MODES)}"
                )

        for entry in policy:
            block = spec.slice(entry.name)
            if not entry.center:
                mean[:, block] = 0.0

            if entry.scale == "channel":
                continue
            if entry.scale == "none":
                std[:, block] = 1.0
                continue

            # Pool the VARIANCE, not the std: pooling std would average
            # magnitudes rather than energies and is not the same statistic.
            variance = stacked[:, :, block].var(axis=0)  # (J, W)
            if entry.scale == "joint_block":
                pooled = np.sqrt(variance.mean(axis=-1, keepdims=True))  # (J, 1)
                std[:, block] = pooled + STD_EPSILON
            else:  # "block": one scalar for the root, one for everything else
                root = np.sqrt(variance[0].mean())
                rest = np.sqrt(variance[1:].mean()) if variance.shape[0] > 1 else root
                std[0, block] = root + STD_EPSILON
                std[1:, block] = rest + STD_EPSILON

        return cls(mean=mean, std=std, spec=spec)
```

Add `Sequence` to the `collections.abc` import and `dataclass` if not present.

- [ ] **Step 4: Run the tests**

Run: `docker compose run --rm test pytest tests/data/test_normalize.py -v`
Expected: all 10 PASS. If `test_block_mode...` fails on the root/non-root split, check you pooled variance across BOTH the joint and channel axes for the non-root group.

- [ ] **Step 5: Confirm no regression**

Run: `docker compose run --rm test pytest -q`
Expected: unchanged from before this task (195 passed, 11 skipped, 7 xfailed) plus the 10 new.

- [ ] **Step 6: `ruff` and commit**

```bash
docker compose run --rm test ruff check .
git add src/poseydon/data/normalize.py tests/data/
git commit -m "feat(normalize): declare normalization per block

Per-channel std makes a barely-moving toe unit-variance, so reconstruction loss
weights it as heavily as the root. More sharply: dividing a 6D rotation row by a
uniform scalar leaves the rotation unchanged after Gram-Schmidt while dividing
per-channel does not, so pooling is what lets the rotation block survive
normalization at all. The tests assert that property and its contrast.

policy=None reproduces the previous behaviour exactly, so no existing caller
changes."
```

---

### Task 2: `Humanize` joint names, and retire `strip_joint_prefix`

Spec: parity §7. `SkeletonManifest.strip_joint_prefix` is parsed, documented, and read by nothing — the same dead-key pattern A1 removed from `tpose`. It is replaced by a stage that COMPUTES the prefix from the rig's own names rather than requiring it to be declared.

**Files:**
- Create: `src/poseydon/build/__init__.py` (if absent), `src/poseydon/build/names.py`
- Modify: `src/poseydon/core/skeleton.py` — remove `strip_joint_prefix`
- Test: `tests/build/test_names.py`

**Interfaces:**
- Produces:
  ```python
  @dataclass(frozen=True)
  class Humanize:
      lowercase: bool = True
      expand_sides: bool = True
      def __call__(self, names: Sequence[str]) -> tuple[str, ...]: ...
  ```
  `Bip01_R_Thigh` → `right thigh`. Task 5 stores both raw and humanized names in `skeleton.npz`.

- [ ] **Step 1: Write the failing tests**

Create `tests/build/test_names.py`:

```python
"""Humanized joint names (parity spec §7).

RAW names remain the canonical identity -- manifests and `resolve()` match on
them, and that is what makes a re-exported rig fail loudly instead of silently
mirroring the character. Humanized names are only what a text encoder sees.
"""

from __future__ import annotations

import pytest

from poseydon.build.names import Humanize


def test_the_common_prefix_is_computed_not_declared():
    """Bip01_ is stripped because every joint carries it, not because a
    manifest said so -- which is the whole point of replacing
    strip_joint_prefix."""
    out = Humanize()(["Bip01_Pelvis", "Bip01_Spine", "Bip01_R_Thigh"])
    assert out == ("pelvis", "spine", "right thigh")


def test_a_prefix_shared_by_only_some_joints_is_kept():
    out = Humanize()(["Bip01_Pelvis", "Spine", "Bip01_Head"])
    assert out == ("bip01 pelvis", "spine", "bip01 head")


def test_camel_case_is_split():
    assert Humanize()(["LeftForeArm"]) == ("left fore arm",)


def test_isolated_side_letters_expand_and_embedded_ones_do_not():
    """`R` alone is a side; the R in `Arm` is not."""
    out = Humanize()(["R_Thigh", "L_Foot", "Arm"])
    assert out == ("right thigh", "left foot", "arm")


def test_expand_sides_can_be_disabled():
    assert Humanize(expand_sides=False)(["R_Thigh"]) == ("r thigh",)


def test_lowercase_can_be_disabled():
    assert Humanize(lowercase=False)(["Bip01_R_Thigh"]) == ("right Thigh",)


def test_real_truebones_names():
    """Names taken verbatim from the corpus, not invented."""
    out = Humanize()(["BN_Leg_R_11", "BN_Leg_L_11", "BN_Arm_R_02"])
    assert out == ("leg right 11", "leg left 11", "arm right 02")


def test_output_length_always_matches_input():
    names = ["Hips", "Bip01_Pelvis", "jt_Cog_C", "_00", "Sabrecat__pelv_"]
    assert len(Humanize()(names)) == len(names)


def test_an_empty_name_list_is_refused():
    with pytest.raises(ValueError, match="no joint names"):
        Humanize()([])
```

- [ ] **Step 2: Run to verify failure**

Run: `docker compose run --rm test pytest tests/build/test_names.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'poseydon.build.names'`.

- [ ] **Step 3: Implement**

Create `src/poseydon/build/names.py`:

```python
"""Joint names a text encoder can read.

`SkeletonManifest.strip_joint_prefix` existed, was parsed, was documented, and
was read by nothing -- the same dead-key pattern `tpose` carried before plan A1
removed it. The replacement COMPUTES the prefix from the rig's own names, so a
new rig needs no declaration and cannot declare a wrong one.

RAW names stay the canonical identity: manifests and `resolve()` match on them,
and that is what makes a re-exported rig fail loudly rather than silently
mirroring the character. These names exist only for the text encoder.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from os.path import commonprefix

_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_SEPARATORS = re.compile(r"[_\-.:]+")
_SIDES = {"r": "right", "l": "left"}


@dataclass(frozen=True)
class Humanize:
    """`Bip01_R_Thigh` -> `right thigh`."""

    lowercase: bool = True
    expand_sides: bool = True

    def __call__(self, names: Sequence[str]) -> tuple[str, ...]:
        names = list(names)
        if not names:
            raise ValueError("cannot humanize: no joint names given")

        prefix = self._shared_prefix(names)
        return tuple(self._one(name[len(prefix) :] or name) for name in names)

    @staticmethod
    def _shared_prefix(names: Sequence[str]) -> str:
        """The separator-terminated prefix EVERY joint carries.

        Truncated at a separator so `Bip01_Pelvis`/`Bip01_Spine` yields
        `Bip01_` rather than `Bip01_`+`S`-style partial-word garbage, and so a
        rig whose names merely happen to share letters loses nothing.
        """
        if len(names) < 2:
            return ""
        shared = commonprefix(list(names))
        cut = max(shared.rfind(c) for c in "_-.:")
        return shared[: cut + 1] if cut >= 0 else ""

    def _one(self, name: str) -> str:
        parts: list[str] = []
        for chunk in _SEPARATORS.split(name):
            if chunk:
                parts.extend(p for p in _CAMEL.split(chunk) if p)

        if self.expand_sides:
            parts = [_SIDES.get(p.lower(), p) if len(p) == 1 else p for p in parts]

        text = " ".join(parts)
        return text.lower() if self.lowercase else text
```

Create `src/poseydon/build/__init__.py` if it does not exist (empty file).

- [ ] **Step 4: Run the tests**

Run: `docker compose run --rm test pytest tests/build/test_names.py -v`
Expected: all 9 PASS.

If `test_lowercase_can_be_disabled` fails, note the expectation: side expansion produces the lowercase word `right` even when `lowercase=False`, because the expansion is a substitution rather than a case transform. That is intended and the test encodes it.

- [ ] **Step 5: Remove `strip_joint_prefix`**

In `src/poseydon/core/skeleton.py`, remove the `strip_joint_prefix` field from `SkeletonManifest`, its entry in `_KNOWN_KEYS`, and its handling in `_build`. Confirm nothing reads it:

```bash
docker compose run --rm test sh -c "grep -rn 'strip_joint_prefix' --include='*.py' --include='*.yaml' . || echo 'no references remain'"
```

`tests/build/test_layout.py::test_strip_joint_prefix_is_gone` already exists and asserts this — confirm it passes rather than writing a new one.

- [ ] **Step 6: Full suite, `ruff`, commit**

```bash
docker compose run --rm test pytest -q
docker compose run --rm test ruff check .
git add src/poseydon/build/ src/poseydon/core/skeleton.py tests/build/test_names.py
git commit -m "feat(build): compute humanized joint names, retire strip_joint_prefix

strip_joint_prefix was parsed, documented and read by nothing -- the same dead
key pattern tpose carried before A1 removed it. Humanize computes the shared
prefix from the rig's own names instead, so a new rig needs no declaration and
cannot declare a wrong one.

Raw names stay the canonical identity; these are only what a text encoder sees."
```

---

### Task 3: Move the index into `build/`, and make it forward-compatible

Spec: parity §12. `ingest/index.py` moves to `build/index.py` unchanged in behaviour, plus one fix: `CorpusIndex.load` calls `ClipRecord(**payload)`, so an index written by a newer build cannot be read by an older one. Unknown keys are ignored instead.

**Files:**
- Create: `src/poseydon/build/index.py` (moved from `src/poseydon/ingest/index.py`)
- Modify: every importer of `poseydon.ingest.index`
- Test: `tests/build/test_index.py`

**Interfaces:**
- Produces: `poseydon.build.index` exporting `ClipRecord`, `CorpusIndex`, `action_slug`, `strip_skeleton_prefix`, `clip_id` — same signatures as today.

- [ ] **Step 1: Find every importer**

```bash
docker compose run --rm test sh -c "grep -rn 'ingest.index\|ingest import index' --include='*.py' ."
```

Record the list; you will update each. Expect `scripts/process_dataset_truebones.py`, `scripts/process_dataset_truebones_fbx.py`, `src/poseydon/data/dataset.py`, `src/poseydon/ingest/pipeline.py`, and tests.

- [ ] **Step 2: Write the failing test**

Create `tests/build/test_index.py`:

```python
"""The corpus index, moved to build/ and made forward-compatible."""

from __future__ import annotations

import json

from poseydon.build.index import ClipRecord, CorpusIndex, action_slug, clip_id


def _record(**over) -> ClipRecord:
    base = dict(clip_id="Goat__walk", skeleton="Goat", action="walk", split="train",
                n_frames=40, fps=30.0, path="clips/Goat/walk.npz", tags=("quadruped",))
    return ClipRecord(**{**base, **over})


def test_an_index_written_by_a_newer_build_still_loads(tmp_path):
    """A row carrying a field this version does not know must not break the
    loader -- otherwise a corpus built by a colleague on a newer branch is
    unreadable, and the failure is a TypeError about an unexpected keyword."""
    path = tmp_path / "index.jsonl"
    payload = {
        "clip_id": "Goat__walk", "skeleton": "Goat", "action": "walk",
        "split": "train", "n_frames": 40, "fps": 30.0,
        "path": "clips/Goat/walk.npz", "tags": ["quadruped"],
        "a_field_from_the_future": {"nested": True},
    }
    path.write_text(json.dumps(payload) + "\n")

    index = CorpusIndex.load(path)
    assert len(index) == 1
    assert index.records[0].clip_id == "Goat__walk"


def test_round_trip_through_a_file(tmp_path):
    index = CorpusIndex()
    index.add(_record())
    index.add(_record(clip_id="Goat__run", action="run"))
    path = tmp_path / "index.jsonl"
    index.save(path)

    back = CorpusIndex.load(path)
    assert [r.clip_id for r in back.records] == ["Goat__walk", "Goat__run"]
    assert back.records[0].tags == ("quadruped",)


def test_helpers_moved_with_it():
    assert action_slug("__Attack3") == "attack3"
    assert clip_id("Goat", "walk") == "Goat__walk"
```

- [ ] **Step 3: Run to verify failure**

Run: `docker compose run --rm test pytest tests/build/test_index.py -v`
Expected: FAIL — no module `poseydon.build.index`.

- [ ] **Step 4: Move the module and fix the loader**

```bash
git mv src/poseydon/ingest/index.py src/poseydon/build/index.py
```

In `CorpusIndex.load`, replace `ClipRecord(**payload)` with a filtered construction:

```python
    @classmethod
    def load(cls, path: str | Path) -> CorpusIndex:
        index = cls()
        fields = {f.name for f in dataclasses.fields(ClipRecord)}
        for line in Path(path).read_text().splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            # Ignore keys this version does not know: an index written by a
            # newer build must stay readable, or a colleague's corpus becomes
            # a TypeError about an unexpected keyword argument.
            known = {k: v for k, v in payload.items() if k in fields}
            if "tags" in known:
                known["tags"] = tuple(known["tags"])
            index.add(ClipRecord(**known))
        return index
```

Add `import dataclasses` at the top.

- [ ] **Step 5: Update every importer**

Change `from poseydon.ingest.index import ...` to `from poseydon.build.index import ...` in every file Step 1 listed. Then confirm:

```bash
docker compose run --rm test sh -c "grep -rn 'ingest.index' --include='*.py' . || echo 'no stale imports'"
```

- [ ] **Step 6: Full suite, `ruff`, commit**

```bash
docker compose run --rm test pytest -q
docker compose run --rm test ruff check .
git add -A src/poseydon/ scripts/ tests/
git commit -m "refactor(build): move the corpus index into build/, ignore unknown keys

Each module lives where its consumer is: stage 2 writes the index, so it owns
it. CorpusIndex.load previously called ClipRecord(**payload), so an index
written by a newer build failed to load in an older one with a TypeError about
an unexpected keyword. Unknown keys are now ignored."
```

---

### Task 4: `build/corpus.py` — clip discovery and labels

Spec: parity §4 (per-clip pass) and §2 ("labels are authored, not owned by the build").

**Files:**
- Create: `src/poseydon/build/corpus.py`
- Test: `tests/build/test_corpus.py`

**Interfaces:**
- Produces:
  ```python
  @dataclass(frozen=True)
  class PerRigDirectory:
      pattern: str = "*.bvh"
      def rigs(self, root: Path) -> list[str]: ...
      def clips(self, root: Path, rig: str) -> list[Path]: ...

  def read_labels(path: Path) -> dict
  def write_labels(path: Path, derived: dict, relabel: bool = False) -> None
  ```
  `write_labels` writes only when the file is absent, unless `relabel=True`, which refreshes derived keys and preserves every key it did not write.

- [ ] **Step 1: Write the failing tests**

Create `tests/build/test_corpus.py`:

```python
"""Clip discovery and label files (parity spec §2, §4).

Labels are AUTHORED, not owned by the build: a rebuild must never destroy a
hand-written `text:` or a changed `split:`.
"""

from __future__ import annotations

import pytest
import yaml

from poseydon.build.corpus import PerRigDirectory, read_labels, write_labels


def _corpus(tmp_path):
    for rig, actions in (("Goat", ("walk", "idle")), ("Crab", ("walk",))):
        d = tmp_path / "clips" / rig
        d.mkdir(parents=True)
        for action in actions:
            (d / f"{action}.bvh").write_text("HIERARCHY\n")
            (d / f"{action}.fbx").write_text("not a bvh")
    return tmp_path


def test_rigs_are_directories_under_clips(tmp_path):
    root = _corpus(tmp_path)
    assert PerRigDirectory().rigs(root) == ["Crab", "Goat"]


def test_clips_match_the_pattern_only(tmp_path):
    root = _corpus(tmp_path)
    found = PerRigDirectory().clips(root, "Goat")
    assert [p.stem for p in found] == ["idle", "walk"]
    assert all(p.suffix == ".bvh" for p in found), "the .fbx must not be picked up"


def test_a_rig_with_no_matching_clip_yields_an_empty_list(tmp_path):
    root = _corpus(tmp_path)
    (root / "clips" / "Empty").mkdir()
    assert PerRigDirectory().clips(root, "Empty") == []


def test_labels_are_written_when_absent(tmp_path):
    path = tmp_path / "walk.yaml"
    write_labels(path, {"split": "train", "action": "walk"})
    assert yaml.safe_load(path.read_text()) == {"split": "train", "action": "walk"}


def test_an_existing_label_file_is_left_alone(tmp_path):
    path = tmp_path / "walk.yaml"
    path.write_text("split: val\ntext: a goat walking uphill\n")
    write_labels(path, {"split": "train", "action": "walk"})
    assert read_labels(path) == {"split": "val", "text": "a goat walking uphill"}


def test_relabel_refreshes_derived_keys_and_preserves_the_rest(tmp_path):
    path = tmp_path / "walk.yaml"
    path.write_text("split: val\ntext: a goat walking uphill\naction: stale\n")
    write_labels(path, {"action": "walk"}, relabel=True)

    back = read_labels(path)
    assert back["action"] == "walk", "a derived key must be refreshed"
    assert back["text"] == "a goat walking uphill", "an authored key must survive"
    assert back["split"] == "val", "a key the build did not write must survive"


def test_reading_an_absent_label_file_gives_an_empty_mapping(tmp_path):
    assert read_labels(tmp_path / "nope.yaml") == {}


def test_a_malformed_label_file_is_refused_rather_than_ignored(tmp_path):
    path = tmp_path / "walk.yaml"
    path.write_text("- this is a list, not a mapping\n")
    with pytest.raises(ValueError, match="mapping"):
        read_labels(path)
```

- [ ] **Step 2: Run to verify failure**

Run: `docker compose run --rm test pytest tests/build/test_corpus.py -v`
Expected: FAIL — no module `poseydon.build.corpus`.

- [ ] **Step 3: Implement**

Create `src/poseydon/build/corpus.py`:

```python
"""Finding the prepared clips, and the label files beside them."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class PerRigDirectory:
    """The entity-first layout: `clips/<Rig>/<action>.bvh`."""

    pattern: str = "*.bvh"

    def rigs(self, root: str | Path) -> list[str]:
        clips = Path(root) / "clips"
        if not clips.is_dir():
            return []
        return sorted(p.name for p in clips.iterdir() if p.is_dir())

    def clips(self, root: str | Path, rig: str) -> list[Path]:
        return sorted((Path(root) / "clips" / rig).glob(self.pattern))


def read_labels(path: str | Path) -> dict:
    """A clip's authored label file, or an empty mapping when absent."""
    path = Path(path)
    if not path.is_file():
        return {}
    loaded = yaml.safe_load(path.read_text()) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"{path}: a label file must be a YAML mapping")
    return loaded


def write_labels(path: str | Path, derived: dict, relabel: bool = False) -> None:
    """Write a label file, without ever destroying authored content.

    A rebuild writes only when the file is absent. `--relabel` refreshes the
    keys the build derives and preserves every key it did not write, so a
    hand-written `text:` or a changed `split:` survives.
    """
    path = Path(path)
    if path.is_file() and not relabel:
        return

    merged = {**read_labels(path), **derived} if relabel else dict(derived)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(merged, sort_keys=True))
```

- [ ] **Step 4: Run the tests, `ruff`, commit**

```bash
docker compose run --rm test pytest tests/build/test_corpus.py -v
docker compose run --rm test ruff check .
git add src/poseydon/build/corpus.py tests/build/test_corpus.py
git commit -m "feat(build): clip discovery and authored label files

A rebuild writes a label file only when one is absent; --relabel refreshes
derived keys and preserves every key it did not write, so a hand-written text:
survives a corpus rebuild."
```

---

### Task 5: `build/pipeline.py` — the four passes

Spec: parity §4, §2; 2026-09-08 spec §2. This is the heart of the plan.

**Files:**
- Create: `src/poseydon/build/pipeline.py`
- Test: `tests/build/test_pipeline.py`

**Interfaces:**
- Consumes: `PerRigDirectory`, `read_labels`/`write_labels`, `Humanize`, `BlockPolicy`, `Normalizer`, `CorpusIndex`/`ClipRecord`, `RigTransform.load`, `rest_action`, `build_reduction`/`apply_reduction`, `extract_features`, `resolve`, `SkeletonManifest.load`, `BVH.read`.
- Produces:
  ```python
  @dataclass(frozen=True)
  class BuildConfig:
      root: Path
      schema: tuple[str, ...]
      corpus: PerRigDirectory
      names: Humanize
      normalize: tuple[BlockPolicy, ...]
      reduce_tolerance: float = 1e-8
      relabel: bool = False

  def build_clips(config, rig) -> int
  def build_skeleton(config, rig) -> None
  def build_stats(config, rig) -> None
  def build_index(config) -> CorpusIndex
  def build_all(config, rigs=None, stats_only=False) -> BuildResult
  ```

- [ ] **Step 1: Write the failing tests**

Create `tests/build/test_pipeline.py`. These run against the REAL prepared corpus and skip cleanly without it:

```python
"""The four build passes (parity spec §4).

Run against the real prepared corpus -- these artefacts are what training reads,
and a synthetic fixture would not exercise the rig-shaped edge cases (a promoted
rig, a rig whose rest clip is not named tpose) that motivate the design.
"""

from __future__ import annotations

import numpy as np
import pytest

from poseydon.build.corpus import PerRigDirectory
from poseydon.build.names import Humanize
from poseydon.build.pipeline import (
    BuildConfig,
    build_clips,
    build_index,
    build_skeleton,
    build_stats,
)
from poseydon.core.skeleton import SkeletonManifest
from poseydon.data.normalize import BlockPolicy, Normalizer
from tests.conftest import CORPUS

SCHEMA = ("ric_pos", "rot6d", "local_vel", "foot_contact")
POLICY = (
    BlockPolicy("ric_pos", center=True, scale="joint_block"),
    BlockPolicy("rot6d", center=False, scale="joint_block"),
    BlockPolicy("local_vel", center=True, scale="joint_block"),
    BlockPolicy("foot_contact", center=False, scale="none"),
)


@pytest.fixture
def config(tmp_path):
    """Reads the real corpus, writes into tmp_path -- so a test never mutates
    the corpus other tests read."""
    if not (CORPUS / "clips").is_dir():
        pytest.skip("prepared corpus not present -- run stage 1")
    return BuildConfig(
        root=CORPUS,
        out=tmp_path,
        schema=SCHEMA,
        corpus=PerRigDirectory(),
        names=Humanize(),
        normalize=POLICY,
    )


def test_the_clip_pass_writes_one_npz_per_prepared_clip(config):
    written = build_clips(config, "Goat")
    produced = sorted((config.out / "clips" / "Goat").glob("*.npz"))
    source = sorted((CORPUS / "clips" / "Goat").glob("*.bvh"))
    assert written == len(source)
    assert [p.stem for p in produced] == [p.stem for p in source]


def test_each_clip_npz_carries_its_own_facing_quaternion(config):
    """FaceAxis is clip-scoped; folding the wrong clip's rotation in would be
    invisible until someone tried to un-prepare a generated clip."""
    from poseydon.build.prepare import RigTransform

    build_clips(config, "Goat")
    recorded = RigTransform.load(CORPUS / "rigs" / "Goat" / "prepare.npz")
    for path in sorted((config.out / "clips" / "Goat").glob("*.npz")):
        with np.load(path) as data:
            np.testing.assert_allclose(
                data["facing"],
                recorded.clip_params[path.stem]["face_axis"]["rotation"],
            )


def test_the_skeleton_pass_takes_offsets_from_the_rest_clip(config):
    """Prepared clips of one rig do NOT share an OFFSET block -- each carries
    its own facing correction -- so rig-level offsets must come from the rest
    clip specifically. Crab's is `walk`, which is exactly why matching on a
    filename would be wrong."""
    from scripts.process_dataset_truebones import rest_action

    from poseydon.io.bvh import BVH

    rig = "Crab"
    build_skeleton(config, rig)
    manifest = SkeletonManifest.load(CORPUS / "rigs" / rig / "manifest.yaml")
    expected = BVH.read(
        CORPUS / "clips" / rig / f"{rest_action(manifest)}.bvh"
    ).to_animation()

    with np.load(config.out / "rigs" / rig / "skeleton.npz", allow_pickle=True) as data:
        np.testing.assert_allclose(data["offsets"], expected.offsets, atol=1e-9)
        assert list(data["names"]) == list(expected.names)


def test_the_skeleton_pass_stores_both_raw_and_humanized_names(config):
    build_skeleton(config, "Goat")
    with np.load(config.out / "rigs" / "Goat" / "skeleton.npz", allow_pickle=True) as d:
        raw, human = list(d["names"]), list(d["humanized"])
    assert len(raw) == len(human)
    assert raw != human, "humanizing must actually change something"
    assert all(n.islower() or not n.isalpha() for n in human if n)


def test_stats_are_fitted_under_the_declared_policy(config):
    """joint_block pooling must be visible in the stored std, not just claimed."""
    build_clips(config, "Goat")
    build_skeleton(config, "Goat")
    build_stats(config, "Goat")

    stats = Normalizer.load(config.out / "rigs" / "Goat" / "stats.npz")
    block = stats.std[:, stats.spec.slice("rot6d")]
    for joint in range(block.shape[0]):
        np.testing.assert_allclose(block[joint], block[joint, 0], atol=1e-12)

    contact = stats.std[:, stats.spec.slice("foot_contact")]
    np.testing.assert_allclose(contact, 1.0)


def test_stats_record_the_schema_they_were_fitted_under(config):
    build_clips(config, "Goat")
    build_skeleton(config, "Goat")
    build_stats(config, "Goat")
    stats = Normalizer.load(config.out / "rigs" / "Goat" / "stats.npz")
    assert stats.spec.names == SCHEMA


def test_the_index_has_one_row_per_clip_and_joins_rig_tags(config):
    for rig in ("Goat", "Crab"):
        build_clips(config, rig)
    index = build_index(config)

    rows = [r for r in index.records if r.skeleton == "Goat"]
    on_disk = sorted((CORPUS / "clips" / "Goat").glob("*.bvh"))
    assert len(rows) == len(on_disk)

    manifest = SkeletonManifest.load(CORPUS / "rigs" / "Goat" / "manifest.yaml")
    assert rows[0].tags == manifest.tags, "tags live once, in the manifest"


def test_the_index_path_points_at_the_npz_not_the_bvh(config):
    """Training never opens a .bvh."""
    build_clips(config, "Goat")
    index = build_index(config)
    assert all(r.path.endswith(".npz") for r in index.records)
```

- [ ] **Step 2: Run to verify failure**

Run: `docker compose run --rm test pytest tests/build/test_pipeline.py -v`
Expected: FAIL — no module `poseydon.build.pipeline`.

- [ ] **Step 3: Implement**

Create `src/poseydon/build/pipeline.py`:

```python
"""The four build passes: clip, skeleton, stats, index.

Every file here has exactly one writer. Stage 1 owns prepare.npz, mesh.npz and
the prepared BVH; this stage owns skeleton.npz, stats.npz, the clip .npz and the
index. Only stats.npz depends on the feature schema, so changing `features:`
re-runs one pass over already-prepared motion rather than rebuilding a corpus.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from poseydon.build.corpus import PerRigDirectory, write_labels
from poseydon.build.index import ClipRecord, CorpusIndex, clip_id
from poseydon.build.names import Humanize
from poseydon.build.prepare import RigTransform
from poseydon.core.skeleton import SkeletonManifest, resolve
from poseydon.data.normalize import BlockPolicy, Normalizer
from poseydon.features import extract_features
from poseydon.features.reduce import apply_reduction, build_reduction
from poseydon.io.bvh import BVH


@dataclass(frozen=True)
class BuildConfig:
    root: Path                      # the prepared corpus this reads
    schema: tuple[str, ...]
    corpus: PerRigDirectory
    names: Humanize
    normalize: tuple[BlockPolicy, ...]
    #: Where artefacts are written. Defaults to `root`; tests point it at a
    #: tmp_path so a test never mutates the corpus another test reads.
    out: Path | None = None
    reduce_tolerance: float = 1e-8
    relabel: bool = False

    def __post_init__(self) -> None:
        if self.out is None:
            object.__setattr__(self, "out", self.root)
        object.__setattr__(self, "root", Path(self.root))
        object.__setattr__(self, "out", Path(self.out))

    def manifest(self, rig: str) -> SkeletonManifest:
        return SkeletonManifest.load(self.root / "rigs" / rig / "manifest.yaml")

    def transform(self, rig: str) -> RigTransform:
        return RigTransform.load(self.root / "rigs" / rig / "prepare.npz")


@dataclass
class BuildResult:
    clips: int = 0
    rigs: int = 0
    warnings: list[str] = field(default_factory=list)


def build_clips(config: BuildConfig, rig: str) -> int:
    """Prepared BVH -> arrays, plus THIS clip's facing quaternion.

    The facing is clip-scoped and unrecoverable from a file that already faces
    +Z, so folding the wrong clip's rotation in would be invisible until someone
    tried to un-prepare a generated clip.
    """
    recorded = config.transform(rig)
    out_dir = config.out / "clips" / rig
    out_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    for path in config.corpus.clips(config.root, rig):
        action = path.stem
        anim = BVH.read(path).to_animation()
        np.savez_compressed(
            out_dir / f"{action}.npz",
            rotations=anim.rotations,
            translations=anim.translations,
            offsets=anim.offsets,
            parents=anim.parents,
            names=np.array(anim.names),
            fps=np.float64(anim.fps),
            facing=recorded.clip_params[action]["face_axis"]["rotation"],
        )
        write_labels(
            out_dir / f"{action}.yaml",
            {"split": "train", "action": action},
            relabel=config.relabel,
        )
        written += 1
    return written


def build_skeleton(config: BuildConfig, rig: str) -> None:
    """Rig-level geometry, names and the reduction map.

    Offsets come from the REST clip specifically: FaceAxis is clip-scoped and
    rotate_rig turns OFFSETS with the motion, so prepared clips of one rig do
    not share an OFFSET block. Which clip that is comes from the manifest via
    rest_action -- never from a filename match, since Crab's is `walk`.
    """
    from scripts.process_dataset_truebones import rest_action

    manifest = config.manifest(rig)
    rest = BVH.read(
        config.root / "clips" / rig / f"{rest_action(manifest)}.bvh"
    ).to_animation()

    reduction = build_reduction(rest, tolerance=config.reduce_tolerance)
    rig_params = config.transform(rig).rig_params

    payload: dict[str, np.ndarray] = {
        "offsets": rest.offsets,
        "parents": rest.parents,
        "names": np.array(rest.names),
        "humanized": np.array(config.names(rest.names)),
        "reduction_source_of": np.array(reduction.source_of),
        "reduction_source_names": np.array(reduction.source_names),
        "rest_action": np.array(rest_action(manifest)),
    }
    # Fold stage 1's rig constants in, so the training-facing file is
    # self-contained and stage 2 can be re-run without Blender.
    for stage, params in rig_params.items():
        for key, value in params.items():
            payload[f"prepare/{stage}/{key}"] = np.asarray(value)

    out = config.out / "rigs" / rig
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "skeleton.npz", **payload)


def build_stats(config: BuildConfig, rig: str) -> None:
    """The one schema-dependent artefact."""
    manifest = config.manifest(rig)
    arrays, spec = [], None
    for path in config.corpus.clips(config.root, rig):
        anim = BVH.read(path).to_animation().as_rigid_body(joint_translation="drop")
        reduction = build_reduction(anim, tolerance=config.reduce_tolerance)
        reduced = apply_reduction(anim, reduction)
        features, spec = extract_features(
            reduced, resolve(manifest, reduced.names), config.schema
        )
        arrays.append(features)

    if spec is None:
        raise ValueError(f"{rig}: no prepared clips, so no statistics to fit")

    out = config.out / "rigs" / rig
    out.mkdir(parents=True, exist_ok=True)
    Normalizer.fit(arrays, spec, list(config.normalize)).save(out / "stats.npz")


def build_index(config: BuildConfig) -> CorpusIndex:
    """One flat row per clip, joining rig-level manifest fields.

    `tags` lives once, in the manifest, and is joined here -- rather than copied
    into every ClipRecord, which wrote `[quadruped, mammal]` twenty-two times
    for BrownBear.
    """
    index = CorpusIndex()
    for rig in config.corpus.rigs(config.root):
        manifest = config.manifest(rig)
        for path in sorted((config.out / "clips" / rig).glob("*.npz")):
            with np.load(path) as data:
                n_frames = int(data["rotations"].shape[0])
                fps = float(data["fps"])
            index.add(
                ClipRecord(
                    clip_id=clip_id(rig, path.stem),
                    skeleton=rig,
                    action=path.stem,
                    split="train",
                    n_frames=n_frames,
                    fps=fps,
                    path=str(Path("clips") / rig / path.name),
                    tags=tuple(manifest.tags),
                )
            )
    index.save(config.out / "index.jsonl")
    return index


def build_all(
    config: BuildConfig,
    rigs: list[str] | None = None,
    stats_only: bool = False,
) -> BuildResult:
    """Run the passes over every rig, collecting per-rig failures.

    Each rig is wrapped individually and deliberately: plan A1's Tukan crash
    killed the whole 73-rig loop mid-run because one stage raised outside a
    per-rig guard, leaving a partial corpus behind.
    """
    result = BuildResult()
    for rig in rigs or config.corpus.rigs(config.root):
        try:
            if not stats_only:
                result.clips += build_clips(config, rig)
                build_skeleton(config, rig)
            build_stats(config, rig)
            result.rigs += 1
        except Exception as error:  # noqa: BLE001 - collect, don't abort the corpus
            result.warnings.append(f"{rig}: {type(error).__name__}: {error}")

    if not stats_only:
        build_index(config)
    return result
```

Two notes on the code above:

- `build_stats` re-derives the reduction per clip rather than reusing
  `build_skeleton`'s. That is deliberate for now — a clip and its rig's rest
  pose can disagree, and re-deriving keeps the pass independently runnable
  under `--stats-only`. If the golden-parity test in Task 7 shows a
  disagreement, revisit it and report.
- `names` is stored as a plain `np.array` of strings, which numpy types as
  `<U`, not `object`. That keeps `skeleton.npz` loadable without
  `allow_pickle`, unlike `prepare.npz`. Do not "simplify" it to
  `dtype=object`.

- [ ] **Step 4: Run the tests**

Run: `docker compose run --rm test pytest tests/build/test_pipeline.py -v`
Expected: all 8 PASS. If `test_the_skeleton_pass_takes_offsets_from_the_rest_clip` fails for Crab, you are reading the wrong clip — check you called `rest_action(manifest)` rather than looking for `tpose`.

- [ ] **Step 5: `ruff` and commit**

```bash
docker compose run --rm test ruff check .
git add src/poseydon/build/pipeline.py tests/build/test_pipeline.py
git commit -m "feat(build): the four build passes

Clip, skeleton, stats and index. Rig-level offsets come from the clip the
manifest names via rest_action(), never from a filename match -- Crab's rest
clip is walk.bvh and prepared clips of one rig do not share an OFFSET block.

Each rig's work is wrapped individually: A1's Tukan crash killed all 73 rigs
mid-run because one stage was unguarded."
```

---

### Task 6: The runner and the config group

Spec: parity §4; 2026-09-08 spec §2 (the `configs/data/truebones.yaml` correction).

**Files:**
- Create: `scripts/build_features.py`, `configs/dataset/truebones.yaml`
- Modify: `configs/data/truebones.yaml`, `configs/model/anytop.yaml`, `configs/model/modiffae.yaml`
- Test: `tests/build/test_build_features_cli.py`

- [ ] **Step 1: Write the config group**

Create `configs/dataset/truebones.yaml`:

```yaml
# Stage 2. Each pluggable behaviour is a `_target_`; each value is a plain key.
root: data/truebones
schema: [ric_pos, rot6d, local_vel, foot_contact]

corpus: {_target_: poseydon.build.corpus.PerRigDirectory, pattern: "*.bvh"}
names:  {_target_: poseydon.build.names.Humanize, lowercase: true, expand_sides: true}

# T5 stays behind the poseydon[text] extra: the test image deliberately excludes
# transformers, so the suite must never require it and a corpus must build
# without it.
text: null

reduce_tolerance: 1.0e-8

# Parity spec §5. `scale: block` with `center: true` on every block is the
# reference-exact combination the parity test uses.
normalize:
  - {name: ric_pos,      center: true,  scale: joint_block}
  - {name: rot6d,        center: false, scale: joint_block}
  - {name: local_vel,    center: true,  scale: joint_block}
  - {name: foot_contact, center: false, scale: none}
```

- [ ] **Step 2: Fix the stale data config**

In `configs/data/truebones.yaml`, `manifests:` points at `data/truebones/skeletons`, a directory the Phase 1 migration deleted — it is why `poseydon train` fails at config load. Change it to `data/truebones/rigs` and add a comment saying a rig is a directory holding `manifest.yaml`.

- [ ] **Step 3: Close the model config hole**

Add `name_embedding_dim: 768` to `configs/model/anytop.yaml` and `configs/model/modiffae.yaml`, with a comment: without it `AnyTop.name_projection` stays `None` and joint-name embeddings are accepted and silently ignored (`models/anytop.py:139`). The embeddings are not built in this plan; the config hole is closed now rather than becoming a silent no-op later.

- [ ] **Step 4: Write the runner**

Create `scripts/build_features.py` — a thin Hydra runner mirroring `scripts/process_dataset_truebones.py`'s argparse style, exposing `--rigs`, `--stats-only`, `--relabel`, `--out-root`, and `--config-name`. It instantiates `BuildConfig` from the composed config via `hydra.utils.instantiate` and calls `build_all`. Print the per-rig progress and the collected warnings exactly as stage 1 does, and exit non-zero if zero clips were written.

- [ ] **Step 5: Write the CLI test**

Create `tests/build/test_build_features_cli.py` asserting: the config group composes; `schema` and `normalize` round-trip into a `BuildConfig` with the right types (`BlockPolicy` instances, not dicts); `--stats-only` runs only the stats pass (assert no clip `.npz` is written when the output directory starts empty); and an unknown `scale` mode in the config raises before any file is written.

- [ ] **Step 6: Run, `ruff`, commit**

```bash
docker compose run --rm test pytest tests/build/ -v
docker compose run --rm test ruff check .
git add scripts/build_features.py configs/ tests/build/test_build_features_cli.py
git commit -m "feat(build): the stage-2 runner and its config group

Also corrects configs/data/truebones.yaml, whose manifests: key still pointed at
the skeletons/ directory the Phase 1 migration deleted -- the reason poseydon
train fails at config load -- and adds name_embedding_dim: 768 to the model
configs, without which joint-name embeddings are accepted and silently ignored."
```

---

### Task 7: Golden feature parity against the reference

Spec: 2026-09-08 §8 test 8 / parity §11 test 4. Features extracted by PoseYdon must reproduce the reference implementation's stored `.npy` arrays block by block, foot contact bit-exact, over the joints the two representations share, matched by NAME.

**Files:**
- Create: `tests/features/test_reference_parity.py`

- [ ] **Step 1: Locate the reference arrays**

```bash
ls external/neural_motion_blending/assets/truebones/*.npy
```

Five rigs have both a `.bvh` and a `.npy`: BrownBear, Crab, Flamingo, Goat, Scorpion (plus Coyote and Skunk). Inspect one array's shape and dtype before writing assertions; record what you find in your report.

- [ ] **Step 2: Write the test**

Match joints by NAME, not index — the reference's joint order need not equal ours, and an index-matched comparison that happens to pass is worse than no test. Skip cleanly when `external/` is absent. Compare each block separately with a tolerance you MEASURE, and assert `foot_contact` bit-exact (it is a flag; a tolerance on it would hide a real disagreement).

If a block does not match, do NOT loosen the tolerance — report the measured discrepancy, which block, and which rigs. The parity spec records that `rot6d` diverges under the `EnforceRigid` rigid-bone simplification (mean ~0.3-0.5, max ~1.8-2.0), so a `rot6d` disagreement of that magnitude is EXPECTED and should be recorded as an `xfail` carrying the measured number, exactly as `test_bvh_fbx_agree_on_rest_geometry` does. `ric_pos`, `local_vel` and `foot_contact` are expected to agree.

- [ ] **Step 3: Run, record, commit**

Report the measured per-block agreement for all five rigs in your task report. Commit with the numbers in the message.

---

### Task 8: Decide the FBX/mesh divergence

Spec: 2026-09-08 §2, the amendment added at `834ebf7`. `PromoteRoot` lives in the BVH `PrepareChain`; the FBX script only rotates and scales a Blender scene. For the 14 promoted rigs the prepared BVH has the locator chain removed and the prepared FBX still has it. `test_bvh_and_fbx_agree_on_the_skeleton_structure` compares exactly that, and passes today only because the five rigs with prepared FBX artefacts include none of the 14.

**This task decides which route the spec's two options take, on evidence.** Do the cheap measurement first, then choose:

- [ ] **Step 1: Measure the cost of each route**

Run the FBX pass for ONE promoted rig with a `mesh.npz` requirement — Camel is the natural choice (it is in `SAMPLE_RIGS` and is a two-step promotion):

```bash
docker compose run --rm fbx blender -b --python scripts/process_dataset_truebones_fbx.py -- --rigs Camel
```

Report: does the FBX path produce a skeleton whose joint count matches the promoted BVH, or the un-promoted one? By how many joints do they differ?

- [ ] **Step 2: Choose and record**

The cheaper, honest route — **declare the FBX corpus un-promoted** — means `build_skeleton` refuses to fold a `mesh.npz` whose joint count disagrees with `skeleton.npz`, with an error naming both counts and pointing at this decision. The more complete route — **teach the FBX path the same promotion** — needs a Blender-side equivalent driven by the same `PROMOTE_ROOT` table.

Recommend the refusal unless Step 1 shows the Blender-side promotion is genuinely small. Record the decision and its reasoning in the task report AND as a short amendment to the spec's §2, replacing the "A2 picks a route on evidence" sentence with what you picked and why.

- [ ] **Step 3: Implement the chosen route, with a test**

Whichever route: `test_bvh_and_fbx_agree_on_the_skeleton_structure` must cover a PROMOTED rig, so the agreement stops being luck. If the route is refusal, the test asserts the refusal fires with a clear message; if it is promotion, the test asserts the two skeletons now agree.

- [ ] **Step 4: Run, `ruff`, commit**

---

### Task 9: Build the corpus and pin the artefacts

- [ ] **Step 1: Build everything**

```bash
docker compose run --rm test python scripts/build_features.py 2>&1 | tee /tmp/build.log | tail -40
```

Expected: 73 rigs, ~1145 clip `.npz`, 73 `skeleton.npz`, 73 `stats.npz`, one `index.jsonl`. Report the real counts and the full warning list.

- [ ] **Step 2: Verify `--stats-only` is cheap and correct**

Re-run with `--stats-only`, confirm only `stats.npz` files change (compare mtimes), and that it completes in seconds rather than minutes. This is the property parity §2 claims; measure it rather than assuming it.

- [ ] **Step 3: Pin the yield**

Extend `tests/build/test_corpus_yield.py` with the stage-2 totals: 73 `skeleton.npz`, 73 `stats.npz`, one `.npz` per prepared `.bvh`, and an `index.jsonl` whose row count equals the clip count. Skip cleanly if the corpus is absent. Assert each `stats.npz`'s recorded schema equals the config's, so a stale artefact from a different `features:` cannot sit unnoticed beside a fresh one.

- [ ] **Step 4: Full suite, `ruff`, commit**

- [ ] **Step 5: Write the outcome**

Append an `# Outcome` section to this plan: measured counts, the warning list, the Task 8 decision, the measured parity numbers from Task 7, and what A3 needs to know. A1's outcome is what made this plan possible; write the one that makes A3 possible.

---

## Self-Review

**Spec coverage.** Parity §5 normalization → Task 1. §7 names → Task 2. §12 index move → Task 3. §2 labels + §4 per-clip pass → Tasks 4, 5. §4 four passes → Task 5. §4 config group → Task 6. §11 test 4 golden parity → Task 7. 2026-09-08 §2's `mesh.npz` staleness and FBX divergence → Task 8. §2's `configs/data/truebones.yaml` correction and `name_embedding_dim` → Task 6. Corpus build → Task 9.

Deliberately NOT covered, and belonging to A3: the `MotionDataset` read path, `Annotations`, the `joint_names`/`crop_start` conditioners, the temporal attention band, the recipe (`core/recipe.py`), the footskate `raw_block` fix, the balanced sampler, `StepLR`, checkpointing, wandb, and the aarch64 image. Parity §8's recipe is written at TRAIN time, so it lands with the training loop rather than here.

**Type consistency.** `BlockPolicy(name, center, scale)` is constructed in Task 1's tests, Task 5's `POLICY`, and Task 6's config — same three fields throughout. `BuildConfig` gains `out` in Task 5 (the Interfaces block lists it) and is instantiated by Task 6's runner. `PerRigDirectory.rigs/clips` are defined in Task 4 and called in Task 5. `rest_action(manifest)` is imported from `scripts.process_dataset_truebones` in Task 5's implementation and its test.

**Known risks.** Task 7 may find a `rot6d` disagreement — that is expected under the `EnforceRigid` simplification and must be recorded with its measured value, not tolerated by a loose bound. Task 8 is a genuine decision, not a mechanical step, and its Step 1 measurement must happen before the choice. Task 5 is the largest task in the plan; if it proves too big to review as one unit, split it at the pass boundary (clip+skeleton, then stats+index) rather than letting it sprawl.

---

# Outcome

Task 9 ran `scripts/build_features.py` over the whole 73-rig corpus for the first time and pinned the result. Everything below is measured, not assumed — commands and their real output are in `.superpowers/sdd/2026-09-09-build-stage-and-artefacts/task-9-report.md`.

**Fix round 1.** The first pass of this task measured 71 `skeleton.npz`/`stats.npz`, not 73: `_check_mesh_matches_skeleton` (pipeline.py) raised inside `build_skeleton`, and `build_all`'s per-rig try/except turned that into a silently smaller corpus for Camel and Goat. The coordinator's review found the defect this exposed in Task 8's guard: `mesh.npz` is never folded into `skeleton.npz` (Task 5 left mesh folding out entirely), so a mesh/skeleton joint-count disagreement has no bearing on the training artefacts and must not withhold them — the design spec's "must not trust positional alignment" is about not folding a bad mesh, not about refusing the skeleton pass. Fixed: `_check_mesh_matches_skeleton` now returns a warning message (or `None`) instead of raising; `build_skeleton` always writes `skeleton.npz` and returns that warning; `build_all` appends it to `result.warnings` through the same channel it already used for hard failures. The refusal is preserved as a comment at the guard, naming the condition for whoever eventually implements mesh folding. `tests/build/test_pipeline.py`'s Camel/Goat tests were updated to assert both halves: the warning fires with both joint counts, AND `skeleton.npz`/`stats.npz` are written. The design spec's §2 sentence was corrected from "refuses to fold" to "warns and does not fold". All counts and warnings below are from the corrected build.

**Measured counts (one clean full build, from an emptied `clips/`/`rigs/*/skeleton.npz,stats.npz`/`index.jsonl`):**

- 73 rig directories under `data/truebones/clips/`, 1145 clip `.npz` (one per prepared `.bvh`, no drops).
- **73 `skeleton.npz` and 73 `stats.npz`** — one pair per rig, including Camel and Goat.
- `index.jsonl`: 1145 rows, equal to the clip count.

**Full warning list (2, both from `_check_mesh_matches_skeleton`, now non-fatal):**

```
Camel: mesh.npz has 51 joints but the rest skeleton has 49 real joints -- the FBX path does not run PromoteRoot, so this rig's mesh.npz is indexed against an un-promoted skeleton and would need PromoteRoot's mapping to fold correctly (design spec 2026-09-08 Stage 2, 'the FBX path does not promote'). skeleton.npz and stats.npz are written from the BVH rest pose regardless -- mesh.npz is not folded into them.
Goat: mesh.npz has 33 joints but the rest skeleton has 32 real joints -- the FBX path does not run PromoteRoot, so this rig's mesh.npz is indexed against an un-promoted skeleton and would need PromoteRoot's mapping to fold correctly (design spec 2026-09-08 Stage 2, 'the FBX path does not promote'). skeleton.npz and stats.npz are written from the BVH rest pose regardless -- mesh.npz is not folded into them.
```

Goat is the rig the plan text called out; Camel is the same mechanism (it is one of the 14 `PROMOTE_ROOT` rigs and also carries a stage-1 `mesh.npz`) and was already known and unit-tested at Task 8. Both are the design's documented trade-off, not a regression, and are recorded in `tests/build/test_corpus_yield.py::KNOWN_MESH_MISMATCHES` — a warning to be aware of, not a gap in the corpus.

**Timings (`--stats-only` parity claim, parity §2):** full stage-2 build 18.0s; `--stats-only` from the same clean state 10.6s. `--stats-only` is real (skips `build_clips`+`build_skeleton` entirely) and mtime-verified correct — clip `.npz`, `skeleton.npz` and `index.jsonl` were byte-for-byte/mtime-untouched, only `stats.npz` changed. The parity claim being measured is "changing `features:` costs seconds, not a corpus rebuild" — and "a corpus rebuild" means re-running STAGE 1 (Blender/FBX processing over 73 rigs), which is a multi-minute operation, not the 18s stage-2 rebuild measured here. Against that correct baseline, 10.6s for `--stats-only` vs. minutes for a stage-1 rebuild is exactly the claim, and it holds without qualification.

**Task 7 (parity test) — one line:** the reference's shipped `.bvh` clips reproduce PoseYdon's features to ~1e-6 on all 7 reference rigs because `build_reduction` is provably a no-op on them (`reduction.is_identity`, enforced by assertion, not just claimed in prose); it does not exercise the `rot6d` divergence `EnforceRigid`/reduction can otherwise cause.

**Task 8 (mesh/skeleton guard) — one line, corrected here:** `build_skeleton` warns and does not fold a `mesh.npz` whose joint count disagrees with the rest skeleton, because the FBX path never runs `PromoteRoot` so a promoted rig's mesh is indexed against the wrong joint order — but it does not withhold `skeleton.npz`/`stats.npz`, since neither reads `mesh.npz`; this is a genuine, permanent divergence between the BVH and FBX corpora (Goat's is a real raw-FBX `Null` root the BVH lacks, not a PoseYdon miscount) that matters only once mesh folding is implemented, which A3 must still refuse to do for these rigs.

**Pinned by `tests/build/test_corpus_yield.py`:** `test_stage_2_artefact_counts_match_the_build` (the counts above, skips cleanly if `data/truebones/clips` is absent) and `test_stats_npz_records_the_configured_schema` (every rig's `stats.npz` block layout equals `configs/dataset/truebones.yaml`'s current `schema:`, so a stale artefact from an old `features:` value cannot sit unnoticed beside a fresh one).

**What A3 needs to know:**

- The corpus A3 reads has a full 73/73/73/1145 set of artefacts — Camel and Goat are complete rigs with `skeleton.npz` and `stats.npz` like every other rig, just flagged in the build's warning list because their (unused-by-training) `mesh.npz` disagrees with the skeleton. No rig needs to be skipped or excluded to train.
- If A3 ever implements mesh folding (skinning, retargeting onto a mesh), it MUST re-check the condition `_check_mesh_matches_skeleton` measures (or reuse the function) and refuse to fold Camel's or Goat's `mesh.npz` — the comment at the guard names this explicitly.
- `--stats-only` is safe to run after any `features:` change and touches only `stats.npz` (verified above); against the real alternative (a stage-1 rebuild, minutes) it is the "seconds, not a corpus rebuild" the design promises.
- `data/truebones/{clips,rigs,index.jsonl}` (everything stage 2 writes) is gitignored — nothing from this run is committed; a fresh checkout has none of it, which is exactly what `test_corpus_yield.py`'s skip-if-absent guards are for. Running `scripts/build_features.py` once, followed by `--stats-only` as needed, is the full recipe A3's environment (including the aarch64 GPU image) needs to reproduce it.
- Nothing here is unresolved or hidden: both warnings are explained, all four counts are exact and match (73/73/73/1145), and the `--stats-only` timing holds the design's claim against the correct baseline.

## Final whole-branch review — fix wave

Applied after all 9 tasks were individually reviewed and complete; see
`.superpowers/sdd/2026-09-09-build-stage-and-artefacts/final-fix-report.md` for the
full list, commands and measured output. Two items belong here because A3 needs
them and nothing else records them:

- **`skeleton.npz` requires `allow_pickle=True`, permanently.** Not a bug to fix:
  `prepare/enforce_rigid/source_channels` is genuinely ragged (a tuple of channel
  names per joint, differing in length per joint) and has no fixed-width
  encoding. `prepare/rest_relative/names` was ALSO object-dtype, but only as a
  round-trip artefact of `RigTransform.load(allow_pickle=True)` — it is a plain
  list of equal-length strings, so it is now re-cast to `<U` before being written
  into `skeleton.npz`, and `build_skeleton`'s docstring says so. Whatever A3
  writes to load `skeleton.npz` must pass `allow_pickle=True` — there is no way
  to avoid it while `source_channels` lives in this file.
- **`tests/build/test_corpus_yield.py`'s 73/73/73/1145 pin does not run in CI.**
  Every guard in that file skips when `data/truebones/clips` is absent, which is
  always true on a fresh checkout (the corpus is gitignored, §2's "What A3 needs
  to know" above). A3 must not treat that file's green result as a live guard on
  a CI machine that has not first run `scripts/process_dataset_truebones.py` and
  `scripts/build_features.py` — it is a local/dev regression check only, unless
  and until CI is given the corpus.

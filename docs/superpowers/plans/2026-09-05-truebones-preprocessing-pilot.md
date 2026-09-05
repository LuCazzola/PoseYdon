# Truebones Raw Preprocessing (Pilot) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the raw Truebones BVH dump into training-ready data for the
seven skeletons that already have manifests, by cleaning the raw 3ds Max
Biped export quirks into a shape PoseYdon's existing ingest pipeline already
accepts.

**Architecture:** A new, dataset-agnostic `poseydon.datasets.raw_bvh` module
does two structural fixups (merge a redundant zero-offset root, freeze
non-root translation to a static offset) and exposes `load_raw_biped_bvh`.
A pilot script cleans each raw clip, writes it back out as an ordinary BVH
via the existing `save_bvh`, and hands the result to the *unmodified*
`poseydon.ingest.pipeline.ingest_corpus`. `poseydon.io.bvh` gets one small
internal refactor (no public contract change) so the new module can reuse
its channel-parsing logic instead of duplicating it.

**Tech Stack:** Python, NumPy, pytest. All commands run via
`docker compose run --rm test <command>` (see repo `docker-compose.yml`) --
never on the host directly, since the host has no `numpy`/`scipy` installed.

**Spec:**
`docs/superpowers/specs/2026-09-05-truebones-preprocessing-design.md`

## Global Constraints

- `poseydon.io.bvh.load_bvh`'s public behaviour is unchanged: it still
  rejects non-root position channels with `BvhParseError`. Only an internal
  helper is extracted from it.
- `poseydon.ingest.pipeline.ingest_corpus`, `ingest_clip`, and
  `poseydon.ingest.align` are not modified at all.
- The seven pilot species' existing manifests
  (`data/truebones/skeletons/{BrownBear,Coyote,Crab,Flamingo,Goat,Scorpion,Skunk}.yaml`)
  are not modified.
- Every non-root joint's animated translation is discarded (frozen to its
  declared rest offset) -- this is `freeze_non_root_translation`'s entire
  job, kept as one small, separately named function so a future relaxation
  of PoseYdon's fixed-bone-length assumption has one clear call site to
  change (spec §5).
- The redundant-root merge applies **only** when the root has exactly one
  child whose offset is within `1e-6` of zero -- confirmed empirically
  (spec §3 addendum): BrownBear, Coyote, Flamingo, Goat, Skunk merge; Crab
  (single child, non-zero offset) and Scorpion (two children) do not.
- FBX parsing and manifest auto-generation are out of scope for this plan
  (spec §2).
- All commands run inside the `test` Docker service. Use `--no-cache` on any
  `ruff` invocation (the container can't write `.ruff_cache` into the bind
  mount).

---

### Task 1: Split `io.bvh`'s channel conversion into a reusable, permissive helper

**Files:**
- Modify: `src/poseydon/io/bvh.py:125-169` (`_channels_to_local`)
- Test: `tests/io/test_bvh_read.py`

**Interfaces:**
- Produces: `_channels_to_arrays(values: np.ndarray, channels: list[list[str]], n_joints: int) -> tuple[np.ndarray, np.ndarray]` -- returns `(rotations (F, J, 4), positions (F, J, 3))` for **every** joint, zero-filled where a joint has no position channels declared. No contract enforcement.
- Consumes (later tasks): `poseydon.datasets.raw_bvh` calls this directly to get per-joint positions that `_channels_to_local` deliberately discards.

This task does not change what `load_bvh` accepts or rejects. It only moves
the "turn channel columns into rotation/position arrays" logic into a
function that doesn't also enforce "only the root may translate", so that
enforcement can be layered on top by `_channels_to_local` while
`raw_bvh.py` (Task 2+) skips it.

- [ ] **Step 1: Write the failing test**

Add to `tests/io/test_bvh_read.py`:

```python
from poseydon.io.bvh import _channels_to_arrays


def test_channels_to_arrays_returns_positions_for_every_joint():
    # Two joints: joint 0 (root) has position+rotation channels, joint 1
    # has position channels too (which _channels_to_local would reject, but
    # _channels_to_arrays has no opinion about that).
    channels = [
        ["Xposition", "Yposition", "Zposition", "Zrotation", "Xrotation", "Yrotation"],
        ["Xposition", "Yposition", "Zposition", "Zrotation", "Xrotation", "Yrotation"],
    ]
    # One frame, 12 values: joint0's 6 channels then joint1's 6 channels.
    values = np.array([[1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 4.0, 5.0, 6.0, 0.0, 0.0, 0.0]])

    rotations, positions = _channels_to_arrays(values, channels, n_joints=2)

    assert rotations.shape == (1, 2, 4)
    assert positions.shape == (1, 2, 3)
    np.testing.assert_allclose(positions[0, 0], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(positions[0, 1], [4.0, 5.0, 6.0])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `docker compose run --rm test pytest tests/io/test_bvh_read.py::test_channels_to_arrays_returns_positions_for_every_joint -v`
Expected: FAIL with `ImportError: cannot import name '_channels_to_arrays'`

- [ ] **Step 3: Extract the helper and rewrite `_channels_to_local` on top of it**

In `src/poseydon/io/bvh.py`, replace the body of `_channels_to_local`
(lines 125-169) with:

```python
def _channels_to_arrays(values, channels, n_joints):
    """Per-joint rotations and positions, straight from the MOTION columns.

    Every joint gets a (F, 3) position slot -- zero-filled if it declared no
    position channels. No contract enforcement: callers decide what a
    non-root position column means. Shared by ``_channels_to_local`` (which
    rejects non-root translation) and ``poseydon.datasets.raw_bvh`` (which
    doesn't).
    """
    n_frames = values.shape[0]
    rotations = np.broadcast_to(QUAT_IDENTITY, (n_frames, n_joints, 4)).copy()
    positions = np.zeros((n_frames, n_joints, 3), dtype=np.float64)

    by_order: dict[str, list[tuple[int, list[int]]]] = {}
    column = 0
    for joint, spec in enumerate(channels):
        if not spec:
            continue
        columns = {name: column + offset for offset, name in enumerate(spec)}
        column += len(spec)

        for axis_index, name in enumerate(_POSITION_CHANNELS):
            if name in columns:
                positions[:, joint, axis_index] = values[:, columns[name]]

        rotation_names = [n for n in spec if n in CHANNEL_AXIS]
        if rotation_names:
            order = "".join(CHANNEL_AXIS[n] for n in rotation_names)
            by_order.setdefault(order, []).append(
                (joint, [columns[n] for n in rotation_names])
            )

    for order, entries in by_order.items():
        joints = [joint for joint, _ in entries]
        picks = np.array([cols for _, cols in entries])
        angles = values[:, picks].transpose(1, 0, 2)
        rotations[:, joints] = euler_to_quat(
            angles.reshape(-1, 3), order
        ).reshape(len(joints), n_frames, 4).transpose(1, 0, 2)

    return rotations, positions


def _channels_to_local(values, channels, n_joints):
    """Per-joint rotations plus a root trajectory. Only the root may translate.

    Joints are grouped by rotation order and converted in one call per order
    rather than one per joint. A 63-joint skeleton otherwise pays 63 separate
    SciPy round trips, which dominates parsing.
    """
    for joint, spec in enumerate(channels):
        if joint != 0 and any(name in spec for name in _POSITION_CHANNELS):
            raise BvhParseError(
                f"joint {joint} has position channels; only the root may translate"
            )
    rotations, positions = _channels_to_arrays(values, channels, n_joints)
    return rotations, positions[:, 0]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `docker compose run --rm test pytest tests/io/ -v`
Expected: PASS (the new test, plus every existing `test_bvh_read.py` /
`test_bvh_write.py` test, since `_channels_to_local`'s observable behaviour
is unchanged).

- [ ] **Step 5: Commit**

```bash
git add src/poseydon/io/bvh.py tests/io/test_bvh_read.py
git commit -m "refactor(io): split channel-to-array conversion from root-only enforcement

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 2: `merge_redundant_root`

**Files:**
- Create: `src/poseydon/datasets/__init__.py` (empty)
- Create: `src/poseydon/datasets/raw_bvh.py`
- Test: `tests/datasets/test_raw_bvh.py`
- Test: `tests/datasets/__init__.py` (empty)

**Interfaces:**
- Produces: `merge_redundant_root(names: tuple[str, ...], parents: np.ndarray, offsets: np.ndarray, rotations: np.ndarray, positions: np.ndarray) -> tuple[tuple[str, ...], np.ndarray, np.ndarray, np.ndarray, np.ndarray]` -- returns `(names, parents, offsets, rotations, positions)`, each one joint shorter than the input. Raises `ValueError` if the root does not have exactly one child, or that child's offset is not within `1e-6` of zero.

```bash
mkdir -p src/poseydon/datasets tests/datasets
touch src/poseydon/datasets/__init__.py tests/datasets/__init__.py
```

- [ ] **Step 1: Write the failing tests**

Create `tests/datasets/test_raw_bvh.py`:

```python
import numpy as np
import pytest

from poseydon.core.rotations import quat_mul
from poseydon.datasets.raw_bvh import merge_redundant_root


def _chain(offset1, rot0, rot1, pos0, pos1):
    """A 3-joint chain: root -> joint1 (candidate for merging) -> joint2."""
    names = ("Hips", "Bip01_Pelvis", "Bip01_Spine")
    parents = np.array([-1, 0, 1], dtype=np.int32)
    offsets = np.array([[0.0, 0.0, 0.0], offset1, [1.0, 0.0, 0.0]])
    rotations = np.array([[rot0, rot1, [0.0, 0.0, 0.0, 1.0]]])  # (1 frame, 3 joints, 4)
    positions = np.array([[pos0, pos1, [0.0, 0.0, 0.0]]])       # (1 frame, 3 joints, 3)
    return names, parents, offsets, rotations, positions


def test_merges_zero_offset_single_child_of_root():
    root_rot = [0.0, 0.0, 0.0, 1.0]
    child_rot = [0.0, 0.7071067811865476, 0.0, 0.7071067811865476]  # 90 deg about Y
    names, parents, offsets, rotations, positions = _chain(
        offset1=[0.0, 0.0, 0.0],
        rot0=root_rot,
        rot1=child_rot,
        pos0=[5.0, 0.0, 0.0],
        pos1=[0.1, 0.0, 0.0],
    )

    new_names, new_parents, new_offsets, new_rotations, new_positions = merge_redundant_root(
        names, parents, offsets, rotations, positions
    )

    assert new_names == ("Bip01_Pelvis", "Bip01_Spine")
    np.testing.assert_array_equal(new_parents, [-1, 0])
    # New root's offset is the OLD root's offset (the true world-space rest position).
    np.testing.assert_allclose(new_offsets[0], [0.0, 0.0, 0.0])
    np.testing.assert_allclose(new_offsets[1], [1.0, 0.0, 0.0])
    # New root's rotation composes root-then-child, same order forward_kinematics uses.
    np.testing.assert_allclose(new_rotations[0, 0], quat_mul(np.array(root_rot), np.array(child_rot)))
    np.testing.assert_allclose(new_rotations[0, 1], [0.0, 0.0, 0.0, 1.0])
    # New root's translation is the two old translations summed.
    np.testing.assert_allclose(new_positions[0, 0], [5.1, 0.0, 0.0])


def test_raises_when_root_has_two_children():
    names = ("Hips", "A", "B")
    parents = np.array([-1, 0, 0], dtype=np.int32)
    offsets = np.zeros((3, 3))
    rotations = np.broadcast_to([0.0, 0.0, 0.0, 1.0], (1, 3, 4)).copy()
    positions = np.zeros((1, 3, 3))

    with pytest.raises(ValueError, match="exactly one child"):
        merge_redundant_root(names, parents, offsets, rotations, positions)


def test_raises_when_single_child_offset_is_not_near_zero():
    names, parents, offsets, rotations, positions = _chain(
        offset1=[3.0, 0.0, 0.0],  # not ~zero -- this is a real bone, not a redundant wrapper
        rot0=[0.0, 0.0, 0.0, 1.0],
        rot1=[0.0, 0.0, 0.0, 1.0],
        pos0=[0.0, 0.0, 0.0],
        pos1=[0.0, 0.0, 0.0],
    )

    with pytest.raises(ValueError, match="not.*zero"):
        merge_redundant_root(names, parents, offsets, rotations, positions)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `docker compose run --rm test pytest tests/datasets/test_raw_bvh.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'poseydon.datasets.raw_bvh'`

- [ ] **Step 3: Implement `merge_redundant_root`**

Create `src/poseydon/datasets/raw_bvh.py`:

```python
"""Raw BVH hygiene, shared across dataset preprocessing scripts.

3ds Max Biped rigs export every joint with 6 channels (position + rotation),
not just the root, and wrap the true root in a redundant zero-offset child
joint. Neither quirk is Truebones-specific -- any future dataset sourced
from the same export lineage (Mixamo, other Biped-rigged BVH dumps) hits it
too, which is why this lives here rather than under a Truebones-named
module. See docs/superpowers/specs/2026-09-05-truebones-preprocessing-design.md.
"""

from __future__ import annotations

import numpy as np

from poseydon.core.rotations import quat_mul

_ZERO_OFFSET_ATOL = 1e-6


def merge_redundant_root(
    names: tuple[str, ...],
    parents: np.ndarray,
    offsets: np.ndarray,
    rotations: np.ndarray,
    positions: np.ndarray,
) -> tuple[tuple[str, ...], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Merge a zero-offset single child of the root into the root.

    Ports the one branch of the reference's (external/neural_motion_blending
    vendored BVH.py) redundant-root handling that applies to Truebones data:
    the new root's offset is the old root's, its rotation is the old root's
    composed with the old child's (root applied first, matching
    poseydon.core.kinematics.forward_kinematics's own chaining order), and
    its translation is the two old translations summed. The old root is
    dropped entirely, including its name.

    Raises ``ValueError`` if the shape doesn't match -- callers should treat
    that as "nothing to merge", not call this function.
    """
    if parents.size < 2 or np.count_nonzero(parents == 0) != 1:
        raise ValueError(
            "root does not have exactly one child; nothing to merge"
        )
    if not np.allclose(offsets[1], 0.0, atol=_ZERO_OFFSET_ATOL):
        raise ValueError(
            f"joint 1's offset {offsets[1].tolist()} is not within "
            f"{_ZERO_OFFSET_ATOL} of zero; nothing to merge"
        )

    new_offsets = offsets.copy()
    new_offsets[1] = offsets[0]

    new_rotations = rotations.copy()
    new_rotations[:, 1] = quat_mul(rotations[:, 0], rotations[:, 1])

    new_positions = positions.copy()
    new_positions[:, 1] = positions[:, 0] + positions[:, 1]

    new_parents = parents[1:] - 1
    new_names = names[1:]

    return new_names, new_parents, new_offsets[1:], new_rotations[:, 1:], new_positions[:, 1:]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `docker compose run --rm test pytest tests/datasets/test_raw_bvh.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/poseydon/datasets/ tests/datasets/
git commit -m "feat(datasets): add merge_redundant_root for raw Biped BVH hygiene

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 3: `freeze_non_root_translation`

**Files:**
- Modify: `src/poseydon/datasets/raw_bvh.py`
- Test: `tests/datasets/test_raw_bvh.py`

**Interfaces:**
- Consumes: nothing new (plain arrays).
- Produces: `freeze_non_root_translation(positions: np.ndarray) -> np.ndarray` -- returns `root_pos (F, 3)`, i.e. `positions[:, 0]`. A single, separately-named function (not inlined into `load_raw_biped_bvh`) so a future version of `Anim` that can carry per-joint translation has one call site to change (spec §5) instead of a restructure.

- [ ] **Step 1: Write the failing test**

Add to `tests/datasets/test_raw_bvh.py`:

```python
from poseydon.datasets.raw_bvh import freeze_non_root_translation


def test_freeze_non_root_translation_keeps_only_the_root_column():
    positions = np.array(
        [
            [[1.0, 2.0, 3.0], [10.0, 20.0, 30.0], [100.0, 200.0, 300.0]],
            [[4.0, 5.0, 6.0], [40.0, 50.0, 60.0], [400.0, 500.0, 600.0]],
        ]
    )  # (F=2, J=3, 3)

    root_pos = freeze_non_root_translation(positions)

    np.testing.assert_array_equal(root_pos, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    assert root_pos.shape == (2, 3)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `docker compose run --rm test pytest tests/datasets/test_raw_bvh.py::test_freeze_non_root_translation_keeps_only_the_root_column -v`
Expected: FAIL with `ImportError: cannot import name 'freeze_non_root_translation'`

- [ ] **Step 3: Implement it**

Add to `src/poseydon/datasets/raw_bvh.py`:

```python
def freeze_non_root_translation(positions: np.ndarray) -> np.ndarray:
    """Discard every non-root joint's animated translation.

    PoseYdon's ``Anim`` has no field to carry per-joint translation (bone
    lengths are fixed; only the root moves), so this drops it rather than
    averaging or sampling it. Returns the root's own trajectory unchanged.
    """
    return positions[:, 0].copy()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `docker compose run --rm test pytest tests/datasets/test_raw_bvh.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/poseydon/datasets/raw_bvh.py tests/datasets/test_raw_bvh.py
git commit -m "feat(datasets): add freeze_non_root_translation

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 4: `load_raw_biped_bvh`

**Files:**
- Modify: `src/poseydon/datasets/raw_bvh.py`
- Test: `tests/datasets/test_raw_bvh.py`

**Interfaces:**
- Consumes: `poseydon.io.bvh._parse_hierarchy(text) -> (names, parents, offsets, channels)`, `poseydon.io.bvh._parse_motion(text, n_channels) -> (values, frame_time)`, `poseydon.io.bvh._channels_to_arrays` (Task 1), `merge_redundant_root` (Task 2), `freeze_non_root_translation` (Task 3).
- Produces: `load_raw_biped_bvh(path: str | Path) -> Anim`. This is the module's public entry point; the pilot script (Task 5) calls only this.

- [ ] **Step 1: Write the failing tests**

Add to `tests/datasets/test_raw_bvh.py`. These run against the real raw
dump, so they need the repo's `data/truebones/Truebone_Z-OO/` present (it
is -- see conversation history; it's gitignored but exists on disk):

```python
from pathlib import Path

from poseydon.core.anim import Anim
from poseydon.datasets.raw_bvh import load_raw_biped_bvh

RAW_ROOT = Path(__file__).resolve().parents[2] / "data" / "truebones" / "Truebone_Z-OO"

# (relative path, expected root name, expected joint count). Joint counts and
# merge outcomes were verified by direct inspection of these exact files --
# see the design spec's §3 addendum.
PILOT_RAW_CLIPS = [
    ("BrownBear/__RiseSwat.bvh", "Bip01_Pelvis", 48),
    ("Coyote/__Attack3.bvh", "Bip01_Pelvis", 49),
    ("Crab/__Attack3.bvh", "Hips", 64),
    ("Flamingo/Flamingo_OneLEgBEnt.bvh", "Bip01_Pelvis", 52),
    ("Goat/__HeadButt.bvh", "Bip01_Pelvis", 39),
    ("Scorpion/__SlowForward.bvh", "Hips", 78),
    ("Skunk/__Spray.bvh", "Bip01_Pelvis", 46),
]


def _skip_if_raw_dump_missing():
    import pytest

    if not RAW_ROOT.is_dir():
        pytest.skip(f"raw Truebones dump not found at {RAW_ROOT}")


def test_loads_every_pilot_species_raw_clip():
    _skip_if_raw_dump_missing()
    for relative, expected_root, expected_joints in PILOT_RAW_CLIPS:
        anim = load_raw_biped_bvh(RAW_ROOT / relative)
        assert isinstance(anim, Anim), relative
        assert anim.names[0] == expected_root, relative
        assert anim.n_joints == expected_joints, relative
        assert anim.parents[0] == -1, relative


def test_does_not_merge_when_root_has_two_children():
    _skip_if_raw_dump_missing()
    # Scorpion's root (Hips) has two children (Bip01_Neck1, Bip01_Spine), so
    # nothing merges: root name and joint count are unchanged from the raw
    # hierarchy (78 joints total, including End Sites).
    anim = load_raw_biped_bvh(RAW_ROOT / "Scorpion/__SlowForward.bvh")
    assert anim.names[0] == "Hips"
    assert anim.n_joints == 78
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `docker compose run --rm test pytest tests/datasets/test_raw_bvh.py -v`
Expected: FAIL with `ImportError: cannot import name 'load_raw_biped_bvh'`

- [ ] **Step 3: Implement it**

Add to `src/poseydon/datasets/raw_bvh.py`:

```python
from pathlib import Path

from poseydon.core.anim import Anim
from poseydon.io.bvh import _channels_to_arrays, _parse_hierarchy, _parse_motion


def _should_merge(parents: np.ndarray, offsets: np.ndarray) -> bool:
    return (
        parents.size >= 2
        and np.count_nonzero(parents == 0) == 1
        and np.allclose(offsets[1], 0.0, atol=_ZERO_OFFSET_ATOL)
    )


def load_raw_biped_bvh(path: str | Path) -> Anim:
    """Read a raw multi-channel Biped BVH export into a valid ``Anim``.

    Merges a redundant zero-offset root when the raw hierarchy has one
    (``_should_merge``), then freezes every remaining non-root joint's
    translation to its declared offset. Reuses ``poseydon.io.bvh``'s
    hierarchy/motion grammar parsing -- the "only the root may translate"
    restriction in ``load_bvh`` is the only thing this skips.
    """
    text = Path(path).read_text()
    head, marker, motion = text.partition("MOTION")
    if not marker:
        raise ValueError(f"{path}: no MOTION block found")

    names, parents, offsets, channels = _parse_hierarchy(head)
    n_channels = sum(len(spec) for spec in channels)
    values, frame_time = _parse_motion(motion, n_channels)
    rotations, positions = _channels_to_arrays(values, channels, len(names))

    if _should_merge(parents, offsets):
        names, parents, offsets, rotations, positions = merge_redundant_root(
            names, parents, offsets, rotations, positions
        )

    root_pos = freeze_non_root_translation(positions)

    return Anim(
        rotations=rotations,
        root_pos=root_pos,
        offsets=offsets,
        parents=parents,
        names=names,
        fps=1.0 / frame_time,
    )
```

`_parse_hierarchy` and `_parse_motion` raise `poseydon.io.bvh.BvhParseError`
already, so no error handling of your own is needed for a malformed file --
it propagates as-is.

- [ ] **Step 4: Run tests to verify they pass**

Run: `docker compose run --rm test pytest tests/datasets/test_raw_bvh.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/poseydon/datasets/raw_bvh.py tests/datasets/test_raw_bvh.py
git commit -m "feat(datasets): add load_raw_biped_bvh, the module's entry point

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 5: Pilot ingest script

**Files:**
- Create: `scripts/__init__.py` (empty)
- Create: `scripts/create_truebones_dataset.py`
- Test: `tests/scripts/__init__.py` (empty)
- Test: `tests/scripts/test_create_truebones_dataset.py`

**Interfaces:**
- Consumes: `poseydon.datasets.raw_bvh.load_raw_biped_bvh` (Task 4),
  `poseydon.io.bvh.save_bvh` (existing, unmodified),
  `poseydon.ingest.pipeline.ingest_corpus` (existing, unmodified),
  `poseydon.core.skeleton.SkeletonManifest` (existing).
- Produces: `PILOT_SKELETONS: tuple[str, ...]`,
  `clean_species_clips(species: str, raw_root: Path, scratch_root: Path) -> list[Path]`,
  `main() -> None`.

- [ ] **Step 1: Write the failing test**

Create `tests/scripts/test_create_truebones_dataset.py`. This copies a
*single* raw clip for two representative species (Goat, which merges, and
Crab, which doesn't) into a temp raw root, so the test runs fast without
touching the full dump:

```python
import shutil
from pathlib import Path

import pytest

from poseydon.core.anim import Anim
from poseydon.io.bvh import load_bvh
from poseydon.ingest.pipeline import ingest_corpus
from scripts.create_truebones_dataset import clean_species_clips

REAL_RAW_ROOT = Path(__file__).resolve().parents[2] / "data" / "truebones" / "Truebone_Z-OO"
MANIFEST_DIR = Path(__file__).resolve().parents[2] / "data" / "truebones" / "skeletons"

SAMPLE_CLIPS = {
    "Goat": "__HeadButt.bvh",
    "Crab": "__Attack3.bvh",
}


def _skip_if_raw_dump_missing():
    if not REAL_RAW_ROOT.is_dir():
        pytest.skip(f"raw Truebones dump not found at {REAL_RAW_ROOT}")


@pytest.fixture
def small_raw_root(tmp_path):
    _skip_if_raw_dump_missing()
    raw_root = tmp_path / "raw"
    for species, filename in SAMPLE_CLIPS.items():
        dest_dir = raw_root / species
        dest_dir.mkdir(parents=True)
        shutil.copy(REAL_RAW_ROOT / species / filename, dest_dir / filename)
    return raw_root


def test_clean_species_clips_produces_loadable_bvh(tmp_path, small_raw_root):
    scratch_root = tmp_path / "scratch"
    written = clean_species_clips("Goat", small_raw_root, scratch_root)

    assert len(written) == 1
    assert written[0] == scratch_root / "Goat" / "__HeadButt.bvh"
    anim = load_bvh(written[0])  # must round-trip through the STRICT loader
    assert isinstance(anim, Anim)
    assert anim.names[0] == "Bip01_Pelvis"


def test_cleaned_clips_ingest_through_the_unmodified_pipeline(tmp_path, small_raw_root):
    scratch_root = tmp_path / "scratch"
    out_dir = tmp_path / "out"
    all_clean = []
    for species in SAMPLE_CLIPS:
        all_clean.extend(clean_species_clips(species, small_raw_root, scratch_root))

    result = ingest_corpus(all_clean, MANIFEST_DIR, out_dir, split="train")

    assert not result.skipped, result.skipped
    assert {record.skeleton for record in result.index.records} == {"Goat", "Crab"}
    for written_path in result.written:
        Anim.load(written_path)  # the aligned npz must itself be loadable
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `docker compose run --rm test pytest tests/scripts/test_create_truebones_dataset.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.create_truebones_dataset'`

- [ ] **Step 3: Implement the script**

```bash
mkdir -p scripts tests/scripts
touch scripts/__init__.py tests/scripts/__init__.py
```

Create `scripts/create_truebones_dataset.py`:

```python
"""Pilot: raw Truebones BVH -> training-ready corpus, for the seven species
that already have hand-authored manifests.

Cleans each raw clip (poseydon.datasets.raw_bvh.load_raw_biped_bvh), writes
it back out as an ordinary BVH, and hands the result to the unmodified
ingest pipeline. See docs/superpowers/specs/2026-09-05-truebones-preprocessing-design.md.

Run inside the test container:
    docker compose run --rm test python scripts/create_truebones_dataset.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from poseydon.datasets.raw_bvh import load_raw_biped_bvh
from poseydon.ingest.pipeline import ingest_corpus
from poseydon.io.bvh import save_bvh

PILOT_SKELETONS = (
    "BrownBear",
    "Coyote",
    "Crab",
    "Flamingo",
    "Goat",
    "Scorpion",
    "Skunk",
)

RAW_ROOT = Path("data/truebones/Truebone_Z-OO")
MANIFEST_DIR = Path("data/truebones/skeletons")
OUT_DIR = Path("data/truebones")


def clean_species_clips(species: str, raw_root: Path, scratch_root: Path) -> list[Path]:
    """Clean every raw ``.bvh`` clip for one species into ``scratch_root/<species>/``.

    Output filenames match the raw filenames exactly: ``ingest_clip`` derives
    a clip's action name from the filename stem, so renaming here would
    rename every resulting clip id.
    """
    species_dir = raw_root / species
    out_dir = scratch_root / species
    out_dir.mkdir(parents=True, exist_ok=True)

    written = []
    for raw_path in sorted(species_dir.glob("*.bvh")):
        anim = load_raw_biped_bvh(raw_path)
        dest = out_dir / raw_path.name
        save_bvh(anim, dest)
        written.append(dest)
    return written


def main() -> None:
    with tempfile.TemporaryDirectory() as scratch:
        scratch_root = Path(scratch)
        all_clean: list[Path] = []
        for species in PILOT_SKELETONS:
            all_clean.extend(clean_species_clips(species, RAW_ROOT, scratch_root))

        result = ingest_corpus(all_clean, MANIFEST_DIR, OUT_DIR, split="train")

    result.index.save(OUT_DIR / "index.jsonl")
    print(f"ingested {len(result.index)} clips -> {OUT_DIR / 'index.jsonl'}")
    for path, reason in result.skipped:
        print(f"  skipped {path.name}: {reason}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `docker compose run --rm test pytest tests/scripts/test_create_truebones_dataset.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/ tests/scripts/
git commit -m "feat(scripts): add Truebones raw preprocessing pilot script

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 6: Validation report + full pilot run

**Files:**
- Create: `scripts/validate_truebones_cleanup.py`
- Test: `tests/scripts/test_validate_truebones_cleanup.py`

**Interfaces:**
- Consumes: `poseydon.datasets.raw_bvh.load_raw_biped_bvh` (Task 4),
  `poseydon.ingest.pipeline.ingest_clip`, `skeleton_alignment_params`
  (existing, unmodified), `poseydon.io.bvh.load_bvh`,
  `poseydon.features.extract_features`, `poseydon.core.skeleton.resolve`.
- Produces: `compare_clip(species: str, raw_relpath: str, fixture_stem: str) -> dict[str, dict[str, float]]` -- per feature block, `{"max_abs_error": ..., "mean_abs_error": ...}`, computed over joints present in **both** the raw-cleaned and fixture skeletons (the fixture omits End Sites -- see spec §3 addendum). `main() -> None` prints the table for all seven pilot species.

This is diagnostic, not a regression gate (spec §4): the two paths are not
expected to be bit-identical, since freezing non-root translation discards
whatever motion a joint's raw translation channel carried. The test asserts
the report *runs* and returns sane structure; it does not assert error
thresholds.

- [ ] **Step 1: Write the failing test**

Create `tests/scripts/test_validate_truebones_cleanup.py`:

```python
from pathlib import Path

import pytest

from scripts.validate_truebones_cleanup import compare_clip

RAW_ROOT = Path(__file__).resolve().parents[2] / "data" / "truebones" / "Truebone_Z-OO"
FIXTURE_ROOT = (
    Path(__file__).resolve().parents[2]
    / "external"
    / "neural_motion_blending"
    / "assets"
    / "truebones"
)


def _skip_if_data_missing():
    if not RAW_ROOT.is_dir():
        pytest.skip(f"raw Truebones dump not found at {RAW_ROOT}")
    if not FIXTURE_ROOT.is_dir():
        pytest.skip(f"fixture BVHs not found at {FIXTURE_ROOT}")


def test_compare_clip_returns_a_per_block_error_table():
    _skip_if_data_missing()
    report = compare_clip("Goat", "Goat/__HeadButt.bvh", "Goat___HeadButt_395")

    assert set(report) >= {"ric_pos", "rot6d"}
    for block_errors in report.values():
        assert block_errors["max_abs_error"] >= block_errors["mean_abs_error"] >= 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `docker compose run --rm test pytest tests/scripts/test_validate_truebones_cleanup.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.validate_truebones_cleanup'`

- [ ] **Step 3: Implement the report**

Create `scripts/validate_truebones_cleanup.py`:

```python
"""Diagnostic: compare features from (raw -> our cleanup) against features
from (curated fixture) for the same clip, per pilot species.

Not a pass/fail gate -- see docs/superpowers/specs/2026-09-05-truebones-preprocessing-design.md
§4. Freezing non-root translation (poseydon.datasets.raw_bvh) discards real
per-joint motion the fixture path never had to discard, so some divergence
is expected. This prints a per-block error table for a human to judge.

Run inside the test container:
    docker compose run --rm test python scripts/validate_truebones_cleanup.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from poseydon.core.skeleton import SkeletonManifest, resolve
from poseydon.datasets.raw_bvh import load_raw_biped_bvh
from poseydon.features import extract_features
from poseydon.io.bvh import load_bvh

RAW_ROOT = Path("data/truebones/Truebone_Z-OO")
FIXTURE_ROOT = Path("external/neural_motion_blending/assets/truebones")
MANIFEST_DIR = Path("data/truebones/skeletons")

# (species, raw clip relative to RAW_ROOT, matching fixture stem) -- the same
# clip under both a raw and a curated name, one pair per pilot species.
PILOT_PAIRS = [
    ("BrownBear", "BrownBear/__RiseSwat.bvh", "BrownBear___RiseSwat_132"),
    ("Coyote", "Coyote/__Attack3.bvh", "Coyote___Attack3_224"),
    ("Crab", "Crab/__Attack3.bvh", "Crab___Attack3_234"),
    ("Flamingo", "Flamingo/Flamingo_OneLEgBEnt.bvh", "Flamingo_Flamingo_OneLEgBEnt_353"),
    ("Goat", "Goat/__HeadButt.bvh", "Goat___HeadButt_395"),
    ("Scorpion", "Scorpion/__SlowForward.bvh", "Scorpion___SlowForward_839"),
    ("Skunk", "Skunk/__Spray.bvh", "Skunk___Spray_891"),
]


def compare_clip(species: str, raw_relpath: str, fixture_stem: str) -> dict[str, dict[str, float]]:
    """Per-block max/mean absolute error between the two paths' features.

    Compared over the intersection of joint names -- the fixture omits End
    Sites that our own cleanup keeps (spec §3 addendum) -- and truncated to
    the shorter of the two frame counts.
    """
    manifest = SkeletonManifest.load(MANIFEST_DIR / f"{species}.yaml")

    feature_names = ("ric_pos", "rot6d", "local_vel", "foot_contact")

    raw_anim = load_raw_biped_bvh(RAW_ROOT / raw_relpath)
    raw_resolved = resolve(manifest, raw_anim.names)
    raw_features, raw_spec = extract_features(raw_anim, raw_resolved, feature_names)

    fixture_anim = load_bvh(FIXTURE_ROOT / f"{fixture_stem}.bvh")
    fixture_resolved = resolve(manifest, fixture_anim.names)
    fixture_features, fixture_spec = extract_features(fixture_anim, fixture_resolved, feature_names)

    shared_names = [name for name in fixture_anim.names if name in raw_anim.names]
    raw_joint_index = {name: i for i, name in enumerate(raw_anim.names)}
    fixture_joint_index = {name: i for i, name in enumerate(fixture_anim.names)}
    raw_idx = [raw_joint_index[name] for name in shared_names]
    fixture_idx = [fixture_joint_index[name] for name in shared_names]
    n_frames = min(raw_features.shape[0], fixture_features.shape[0])

    # Both extractions were asked for the same `feature_names`, so the two
    # specs name and order blocks identically -- one loop, one spec, used to
    # slice both feature arrays.
    report: dict[str, dict[str, float]] = {}
    for block_name in fixture_spec.names:
        block_slice = fixture_spec.slice(block_name)
        raw_block = raw_features[:n_frames, raw_idx, block_slice]
        fixture_block = fixture_features[:n_frames, fixture_idx, block_slice]
        error = np.abs(raw_block - fixture_block)
        report[block_name] = {
            "max_abs_error": float(error.max()),
            "mean_abs_error": float(error.mean()),
        }
    return report


def main() -> None:
    for species, raw_relpath, fixture_stem in PILOT_PAIRS:
        print(f"{species}:")
        report = compare_clip(species, raw_relpath, fixture_stem)
        for block_name, errors in report.items():
            print(
                f"  {block_name:<14} max={errors['max_abs_error']:.4f}  "
                f"mean={errors['mean_abs_error']:.4f}"
            )


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `docker compose run --rm test pytest tests/scripts/test_validate_truebones_cleanup.py -v`
Expected: PASS

- [ ] **Step 5: Run the full pilot end to end and read the report**

```bash
docker compose run --rm test python scripts/create_truebones_dataset.py
docker compose run --rm test python scripts/validate_truebones_cleanup.py
```

Confirm `data/truebones/index.jsonl` and `data/truebones/aligned/*.npz` now
exist with all seven pilot skeletons represented, and read through the
validation report -- large `rot6d`/`ric_pos` errors concentrated in a
species known for a highly-animated non-root joint (e.g. a long tail or
neck) would be expected given `freeze_non_root_translation`; errors spread
evenly and small suggest the cleanup is working as intended. This is a
judgment call for a human, not an assertion.

- [ ] **Step 6: Commit**

```bash
git add scripts/validate_truebones_cleanup.py tests/scripts/test_validate_truebones_cleanup.py
git commit -m "feat(scripts): add raw-vs-fixture validation report for the pilot

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

## Self-Review Notes

- **Spec coverage:** §1 success criteria 1 (output location) -- Task 5.
  Criterion 2 (dataset-agnostic module) -- Tasks 2-4. Criterion 3
  (`io.bvh` contract unchanged) -- Task 1's regression run of the full
  existing `tests/io/` suite. Criterion 4 (validation report) -- Task 6.
  §2 deferred items (FBX, manifest auto-gen, bone-length relaxation) are
  not touched by any task, as intended.
- **Type consistency:** `merge_redundant_root`'s five-tuple return order
  (`names, parents, offsets, rotations, positions`) matches its parameter
  order, and `load_raw_biped_bvh` (Task 4) destructures it the same way.
  `clean_species_clips`'s return type (`list[Path]`) matches how Task 5's
  test and `main()` both consume it (concatenated into `all_clean`, passed
  straight to `ingest_corpus`).

# PoseYdon Data Augmentation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a composable, Hydra-configurable augmentation pipeline to PoseYdon's training data path, shipping the reference's two structural augmentations (drop a non-foot end-effector joint, duplicate a joint) as exact-under-forward-kinematics mutations of `Anim`.

**Architecture:** A new `poseydon.augment` package defines an `Augmentation` base class (two overridable hooks: `apply_structural` on `(Anim, ResolvedSkeleton)`, `apply_features` on the extracted array), a `JointEdit` value object that transports per-skeleton normalization statistics through a structural edit, and an `AugmentPipeline` that runs a config-ordered list of augmentations, each gated by its own probability. `MotionDataset.__getitem__` applies the pipeline to a **local, per-window copy** of the clip's `Anim`/`ResolvedSkeleton` before feature extraction — the shared per-clip cache is never mutated — so the existing `Topology`, `TPose`, and `NormalizationStats` conditioners need no code changes at all.

**Tech Stack:** Python, NumPy, Hydra/OmegaConf (`instantiate`), pytest, the existing Truebones BVH fixtures (`tests/conftest.py`).

**Spec:** `docs/superpowers/specs/2026-09-05-poseydon-augmentation-design.md`

## Global Constraints

- `augmentations: []` is the default in `configs/train.yaml` — augmentation is opt-in, matching `conditioners: []`.
- An empty `AugmentPipeline` must be a byte-identical no-op versus current `MotionDataset` behavior.
- `Topology`, `TPose`, and `NormalizationStats` in `src/poseydon/conditioners/` receive **zero** code changes.
- Structural edits (`drop_joints`, `duplicate_joint`) never mutate a cached `Anim`/`ResolvedSkeleton`/`Normalizer`/rest-frame in place — they return new objects; `MotionDataset`'s per-clip caches (`self._anims`, `self._manifests`, `self._normalizers`, `self._rest_frames`) stay untouched by augmentation.
- Joint duplication must reproduce the exact pre-edit world position of the duplicated joint and everything below it in the hierarchy (see spec §4) — not an approximation.
- Neither shipped augmentation may select a joint referenced in `resolved.facing_indices`, nor (for drop) a joint in `resolved.foot_indices`.

---

## File Structure

```
src/poseydon/augment/
├── __init__.py       # registers topology augmentations, re-exports the public API
├── base.py           # Augmentation, AugmentPipeline, AUGMENTATIONS registry
├── joint_edit.py      # JointEdit: identity / compose / transport / transport_row
└── topology.py         # leaves, child_count, drop_joints, duplicate_joint,
                        # DropEndEffector, DuplicateJoint

tests/augment/
├── __init__.py
├── test_joint_edit.py
├── test_base.py
└── test_topology.py

src/poseydon/data/dataset.py     # MODIFY: __init__ + __getitem__ wire in AugmentPipeline
src/poseydon/training/build.py   # MODIFY: build_dataset passes augmentations
src/poseydon/cli.py              # MODIFY: `poseydon list` enumerates augmentations
configs/train.yaml                # MODIFY: add `augmentations: []`
tests/data/test_pipeline_end_to_end.py  # MODIFY: add augmentation integration tests
```

---

### Task 1: `JointEdit`

**Files:**
- Create: `src/poseydon/augment/__init__.py` (empty for now, filled in Task 5)
- Create: `src/poseydon/augment/joint_edit.py`
- Test: `tests/augment/__init__.py` (empty)
- Test: `tests/augment/test_joint_edit.py`

**Interfaces:**
- Consumes: `poseydon.core.spec.FeatureSpec`, `poseydon.data.normalize.Normalizer` (both already exist).
- Produces: `JointEdit(source_of: tuple[int, ...])` with `JointEdit.identity(n_joints)`, `edit.compose(prior)`, `edit.transport(normalizer)`, `edit.transport_row(row)`. Every later task imports `JointEdit` from `poseydon.augment.joint_edit`.

- [ ] **Step 1: Write the failing tests**

Create `tests/augment/__init__.py` (empty file) and `src/poseydon/augment/__init__.py` (empty file, so `poseydon.augment` and `tests.augment` are importable packages).

`tests/augment/test_joint_edit.py`:

```python
import numpy as np

from poseydon.augment.joint_edit import JointEdit
from poseydon.core.spec import FeatureSpec
from poseydon.data.normalize import Normalizer


def test_identity_maps_every_joint_to_itself():
    edit = JointEdit.identity(4)
    assert edit.source_of == (0, 1, 2, 3)


def test_compose_reindexes_through_two_edits():
    # A 4-joint skeleton drops joint 1, yielding a 3-joint skeleton indexed
    # (0, 2, 3) into the original. That skeleton then drops its own joint 0
    # (original joint 0), yielding a 2-joint skeleton indexed (1, 2) into the
    # 3-joint intermediate.
    drop_middle = JointEdit(source_of=(0, 2, 3))
    then_drop_first = JointEdit(source_of=(1, 2))
    combined = then_drop_first.compose(drop_middle)
    assert combined.source_of == (2, 3)


def test_transport_gathers_normalizer_rows_by_source():
    spec = FeatureSpec((("x", 2),))
    normalizer = Normalizer(
        mean=np.arange(8, dtype=np.float64).reshape(4, 2),
        std=np.ones((4, 2)),
        spec=spec,
    )
    edit = JointEdit(source_of=(0, 2, 3))

    transported = edit.transport(normalizer)

    np.testing.assert_array_equal(transported.mean, normalizer.mean[[0, 2, 3]])
    np.testing.assert_array_equal(transported.std, normalizer.std[[0, 2, 3]])
    assert transported.spec is spec


def test_transport_row_gathers_a_single_array_by_source():
    edit = JointEdit(source_of=(2, 0))
    row = np.array([[1.0], [2.0], [3.0]])

    result = edit.transport_row(row)

    np.testing.assert_array_equal(result, np.array([[3.0], [1.0]]))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `docker compose run --rm test pytest tests/augment/test_joint_edit.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'poseydon.augment.joint_edit'`

- [ ] **Step 3: Implement `JointEdit`**

`src/poseydon/augment/joint_edit.py`:

```python
"""Transport of per-joint feature-space arrays through a structural edit.

A structural augmentation changes which joints exist and in what order. Two
things live in feature space, indexed by the ORIGINAL joint layout, and must
follow the same edit: per-skeleton normalization statistics
(:class:`~poseydon.data.normalize.Normalizer`) and the T-pose rest frame. A
``JointEdit`` records, for each joint in the NEW layout, which joint in the
layout it was derived from, so both can be re-derived by a gather rather than
recomputed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from poseydon.data.normalize import Normalizer


@dataclass(frozen=True)
class JointEdit:
    """``source_of[new_index]`` is the pre-edit joint index ``new_index`` came from."""

    source_of: tuple[int, ...]

    @classmethod
    def identity(cls, n_joints: int) -> JointEdit:
        return cls(source_of=tuple(range(n_joints)))

    def compose(self, prior: JointEdit) -> JointEdit:
        """``self`` is the edit applied AFTER ``prior``; the result maps straight
        through to whatever ``prior`` itself was relative to."""
        return JointEdit(source_of=tuple(prior.source_of[i] for i in self.source_of))

    def transport(self, normalizer: Normalizer) -> Normalizer:
        index = np.asarray(self.source_of, dtype=np.int64)
        return Normalizer(
            mean=normalizer.mean[index],
            std=normalizer.std[index],
            spec=normalizer.spec,
        )

    def transport_row(self, row: np.ndarray) -> np.ndarray:
        index = np.asarray(self.source_of, dtype=np.int64)
        return row[index]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `docker compose run --rm test pytest tests/augment/test_joint_edit.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add src/poseydon/augment/__init__.py src/poseydon/augment/joint_edit.py tests/augment/__init__.py tests/augment/test_joint_edit.py
git commit -m "$(cat <<'EOF'
feat(augment): add JointEdit for transporting per-joint stats through structural edits

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_012AQAdPqZMa4Zcve31dXMpp
EOF
)"
```

---

### Task 2: `Augmentation` base class, `AUGMENTATIONS` registry, `AugmentPipeline`

**Files:**
- Create: `src/poseydon/augment/base.py`
- Test: `tests/augment/test_base.py`

**Interfaces:**
- Consumes: `JointEdit` (Task 1). Duck-types its `anim`/`resolved` arguments — `AugmentPipeline` never constructs an `Anim` itself, only threads whatever `Augmentation.apply_structural` hands back.
- Produces: `Augmentation` (fields: `p: float = 1.0`; methods `apply_structural(anim, resolved, rng) -> tuple[anim, resolved, JointEdit]`, `apply_features(features, spec, resolved, rng) -> np.ndarray`, both defaulting to no-ops), `AUGMENTATIONS: Registry[Augmentation]`, `AugmentPipeline(augmentations: Sequence[Augmentation])` with `.apply_structural(anim, resolved, rng)` and `.apply_features(features, spec, resolved, rng)`. Task 3/4 subclass `Augmentation` and register on `AUGMENTATIONS`; Task 6 constructs one `AugmentPipeline` per `MotionDataset`.

- [ ] **Step 1: Write the failing tests**

`tests/augment/test_base.py`:

```python
from dataclasses import dataclass

import numpy as np

from poseydon.augment.base import AUGMENTATIONS, Augmentation, AugmentPipeline
from poseydon.augment.joint_edit import JointEdit


class _FakeAnim:
    """Stand-in for poseydon.core.anim.Anim: only `.n_joints` is needed here."""

    def __init__(self, n_joints: int) -> None:
        self.n_joints = n_joints


@dataclass
class _DropLast(Augmentation):
    """Test double: always drops the current last joint."""

    def apply_structural(self, anim, resolved, rng):
        new_anim = _FakeAnim(anim.n_joints - 1)
        edit = JointEdit(source_of=tuple(range(anim.n_joints - 1)))
        return new_anim, resolved, edit


@dataclass
class _AddOne(Augmentation):
    """Test double: always adds 1 to every feature value."""

    def apply_features(self, features, spec, resolved, rng):
        return features + 1


def test_default_hooks_are_no_ops():
    aug = Augmentation()
    anim = _FakeAnim(5)
    rng = np.random.default_rng(0)

    _, _, edit = aug.apply_structural(anim, None, rng)
    assert edit.source_of == (0, 1, 2, 3, 4)

    features = np.zeros((2, 3))
    assert aug.apply_features(features, None, None, rng) is features


def test_pipeline_skips_when_the_probability_trial_fails():
    pipeline = AugmentPipeline([_DropLast(p=0.0)])
    anim = _FakeAnim(5)

    _, _, edit = pipeline.apply_structural(anim, None, np.random.default_rng(0))

    assert edit.source_of == (0, 1, 2, 3, 4)


def test_pipeline_applies_when_the_probability_trial_passes():
    pipeline = AugmentPipeline([_DropLast(p=1.0)])
    anim = _FakeAnim(5)

    _, _, edit = pipeline.apply_structural(anim, None, np.random.default_rng(0))

    assert edit.source_of == (0, 1, 2, 3)


def test_pipeline_composes_structural_edits_across_entries_in_order():
    pipeline = AugmentPipeline([_DropLast(p=1.0), _DropLast(p=1.0)])
    anim = _FakeAnim(5)

    _, _, edit = pipeline.apply_structural(anim, None, np.random.default_rng(0))

    assert edit.source_of == (0, 1, 2)


def test_pipeline_applies_feature_hooks_in_order():
    pipeline = AugmentPipeline([_AddOne(p=1.0), _AddOne(p=1.0)])
    features = np.zeros((2, 2))

    out = pipeline.apply_features(features, None, None, np.random.default_rng(0))

    np.testing.assert_array_equal(out, np.full((2, 2), 2.0))


def test_empty_pipeline_is_a_no_op():
    pipeline = AugmentPipeline([])
    anim = _FakeAnim(5)
    rng = np.random.default_rng(0)

    _, _, edit = pipeline.apply_structural(anim, None, rng)
    assert edit.source_of == (0, 1, 2, 3, 4)

    features = np.zeros((2, 2))
    out = pipeline.apply_features(features, None, None, rng)
    np.testing.assert_array_equal(out, features)


def test_augmentations_registry_exists_and_starts_empty_before_topology_import():
    registry = AUGMENTATIONS
    assert registry.kind == "augmentation"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `docker compose run --rm test pytest tests/augment/test_base.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'poseydon.augment.base'`

- [ ] **Step 3: Implement `base.py`**

`src/poseydon/augment/base.py`:

```python
"""Augmentation contract and pipeline.

An augmentation is a small class with two independent, individually optional
hooks: ``apply_structural`` edits the skeleton itself (joint count, order,
rotations) BEFORE feature extraction, and ``apply_features`` edits the
already-extracted, already-normalized ``(frames, joints, dim)`` array. Most
augmentations need only one; the base class no-ops the other so a subclass
overrides exactly what it changes.

``AugmentPipeline`` composes a config-ordered list, each entry gated by its
own ``p`` -- an empty list, or every entry's trial failing, is a true no-op:
the identity ``JointEdit`` and the input array unchanged.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from poseydon.augment.joint_edit import JointEdit
from poseydon.core.registry import Registry


@dataclass
class Augmentation:
    p: float = 1.0

    def apply_structural(
        self, anim: Any, resolved: Any, rng: np.random.Generator
    ) -> tuple[Any, Any, JointEdit]:
        return anim, resolved, JointEdit.identity(anim.n_joints)

    def apply_features(
        self,
        features: np.ndarray,
        spec: Any,
        resolved: Any,
        rng: np.random.Generator,
    ) -> np.ndarray:
        return features


AUGMENTATIONS: Registry[Augmentation] = Registry("augmentation")


class AugmentPipeline:
    """Ordered, independently-gated composition of augmentations."""

    def __init__(self, augmentations: Sequence[Augmentation] = ()) -> None:
        self.augmentations = tuple(augmentations)

    def apply_structural(
        self, anim: Any, resolved: Any, rng: np.random.Generator
    ) -> tuple[Any, Any, JointEdit]:
        edit = JointEdit.identity(anim.n_joints)
        for augmentation in self.augmentations:
            if rng.random() < augmentation.p:
                anim, resolved, step_edit = augmentation.apply_structural(anim, resolved, rng)
                edit = step_edit.compose(edit)
        return anim, resolved, edit

    def apply_features(
        self,
        features: np.ndarray,
        spec: Any,
        resolved: Any,
        rng: np.random.Generator,
    ) -> np.ndarray:
        for augmentation in self.augmentations:
            if rng.random() < augmentation.p:
                features = augmentation.apply_features(features, spec, resolved, rng)
        return features
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `docker compose run --rm test pytest tests/augment/test_base.py -v`
Expected: PASS (7 tests)

- [ ] **Step 5: Commit**

```bash
git add src/poseydon/augment/base.py tests/augment/test_base.py
git commit -m "$(cat <<'EOF'
feat(augment): add Augmentation contract and AugmentPipeline

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_012AQAdPqZMa4Zcve31dXMpp
EOF
)"
```

---

### Task 3: `drop_joints` and `DropEndEffector`

**Files:**
- Create: `src/poseydon/augment/topology.py` (this task adds `leaves`, `drop_joints`, `DropEndEffector`; Task 4 adds the rest to the same file)
- Test: `tests/augment/test_topology.py` (this task adds the drop-related tests; Task 4 appends duplicate-related tests to the same file)

**Interfaces:**
- Consumes: `poseydon.core.anim.Anim`, `poseydon.core.skeleton.ResolvedSkeleton`, `poseydon.core.kinematics.check_topological_order`, `Augmentation`/`AUGMENTATIONS` (Task 2), `JointEdit` (Task 1), `tests.ingest.manifest_helper.resolved_for` and the `bvh_fixture` pytest fixture (both already exist).
- Produces: `leaves(parents: np.ndarray) -> list[int]`, `drop_joints(anim: Anim, resolved: ResolvedSkeleton, joints: set[int]) -> tuple[Anim, ResolvedSkeleton, JointEdit]`, `DropEndEffector(Augmentation)` with field `rate: tuple[float, ...] = (0.1, 0.2, 0.3)`, registered as `"drop_end_effector"`. Task 4 adds `child_count`, `duplicate_joint`, `DuplicateJoint` to the same module; Task 5 imports both classes.

- [ ] **Step 1: Write the failing tests**

`tests/augment/test_topology.py`:

```python
import numpy as np
import pytest

from poseydon.augment.topology import DropEndEffector, drop_joints, leaves
from poseydon.core.kinematics import check_topological_order
from poseydon.core.skeleton import ResolvedSkeleton
from poseydon.io.bvh import load_bvh
from tests.ingest.manifest_helper import resolved_for


def _excluded(resolved) -> set[int]:
    facing = {i for pair in resolved.facing_indices for i in pair}
    return facing | set(resolved.foot_indices)


def _droppable_leaf(anim, resolved) -> int:
    excluded = _excluded(resolved)
    candidates = [j for j in leaves(anim.parents) if j not in excluded and j != 0]
    if not candidates:
        pytest.skip("this fixture has no droppable end-effector")
    return candidates[0]


def test_leaves_are_joints_with_no_children(bvh_fixture):
    anim = load_bvh(bvh_fixture)
    leaf_set = set(leaves(anim.parents))
    has_child = {int(p) for p in anim.parents if p >= 0}
    assert leaf_set == set(range(anim.n_joints)) - has_child


def test_drop_joints_shrinks_every_structural_array_by_one(bvh_fixture):
    anim = load_bvh(bvh_fixture)
    resolved = resolved_for(bvh_fixture, anim)
    j = _droppable_leaf(anim, resolved)

    new_anim, new_resolved, edit = drop_joints(anim, resolved, {j})

    assert new_anim.n_joints == anim.n_joints - 1
    check_topological_order(new_anim.parents)
    assert len(edit.source_of) == new_anim.n_joints
    assert j not in edit.source_of
    assert len(new_resolved.foot_indices) == len(resolved.foot_indices)


def test_drop_joints_leaves_surviving_positions_bit_identical(bvh_fixture):
    anim = load_bvh(bvh_fixture)
    resolved = resolved_for(bvh_fixture, anim)
    j = _droppable_leaf(anim, resolved)

    original_positions = anim.global_positions()
    new_anim, _, edit = drop_joints(anim, resolved, {j})
    new_positions = new_anim.global_positions()

    for new_index, old_index in enumerate(edit.source_of):
        np.testing.assert_array_equal(
            new_positions[:, new_index], original_positions[:, old_index]
        )


def test_drop_end_effector_never_selects_a_facing_or_foot_joint(bvh_fixture):
    anim = load_bvh(bvh_fixture)
    resolved = resolved_for(bvh_fixture, anim)
    excluded = _excluded(resolved)
    augmentation = DropEndEffector(p=1.0)

    for seed in range(50):
        _, _, edit = augmentation.apply_structural(anim, resolved, np.random.default_rng(seed))
        dropped = set(range(anim.n_joints)) - set(edit.source_of)
        assert dropped.isdisjoint(excluded)


def test_drop_end_effector_is_a_no_op_when_every_leaf_is_excluded():
    # A 2-joint chain: root (0) -> foot (1). The only leaf is also the only
    # declared foot, so there is nothing legal to drop.
    from poseydon.core.anim import Anim
    from poseydon.core.rotations import QUAT_IDENTITY

    n_joints, n_frames = 2, 3
    anim = Anim(
        rotations=np.tile(QUAT_IDENTITY, (n_frames, n_joints, 1)),
        root_pos=np.zeros((n_frames, 3)),
        offsets=np.array([[0.0, 0.0, 0.0], [0.0, -1.0, 0.0]]),
        parents=np.array([-1, 0], dtype=np.int32),
        names=("root", "foot"),
        fps=30.0,
    )
    resolved = ResolvedSkeleton(manifest=None, facing_indices=(), foot_indices=(1,))
    augmentation = DropEndEffector(p=1.0)

    new_anim, new_resolved, edit = augmentation.apply_structural(
        anim, resolved, np.random.default_rng(0)
    )

    assert new_anim is anim
    assert new_resolved is resolved
    assert edit.source_of == (0, 1)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `docker compose run --rm test pytest tests/augment/test_topology.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'poseydon.augment.topology'`

- [ ] **Step 3: Implement `leaves`, `drop_joints`, `DropEndEffector`**

`src/poseydon/augment/topology.py`:

```python
"""Structural augmentations: joint drop and joint duplication.

Ports the reference's per-sample topology augmentation (see design spec
section 3) as exact mutations of :class:`~poseydon.core.anim.Anim`, so every
feature block -- present or future -- is recomputed from a genuinely valid
skeleton rather than patched after extraction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from poseydon.augment.base import AUGMENTATIONS, Augmentation
from poseydon.augment.joint_edit import JointEdit
from poseydon.core.anim import Anim
from poseydon.core.rotations import QUAT_IDENTITY
from poseydon.core.skeleton import ResolvedSkeleton


def leaves(parents: np.ndarray) -> list[int]:
    """Joint indices that are nobody's parent."""
    parents = np.asarray(parents)
    has_children = np.zeros(len(parents), dtype=bool)
    valid = parents >= 0
    has_children[parents[valid]] = True
    return [j for j in range(len(parents)) if not has_children[j]]


def child_count(parents: np.ndarray) -> np.ndarray:
    """Number of children per joint, ``(J,)``."""
    parents = np.asarray(parents)
    counts = np.zeros(len(parents), dtype=np.int64)
    valid = parents >= 0
    np.add.at(counts, parents[valid], 1)
    return counts


def _excluded_joints(resolved: ResolvedSkeleton) -> set[int]:
    facing = {i for pair in resolved.facing_indices for i in pair}
    return facing | set(resolved.foot_indices)


def _reindex_resolved(
    resolved: ResolvedSkeleton, old_to_new: dict[int, int]
) -> ResolvedSkeleton:
    facing = tuple((old_to_new[r], old_to_new[l]) for r, l in resolved.facing_indices)
    feet = tuple(old_to_new[f] for f in resolved.foot_indices)
    return ResolvedSkeleton(manifest=resolved.manifest, facing_indices=facing, foot_indices=feet)


def drop_joints(
    anim: Anim, resolved: ResolvedSkeleton, joints: set[int]
) -> tuple[Anim, ResolvedSkeleton, JointEdit]:
    """Remove ``joints`` (must all be leaves) from ``anim``.

    Every surviving joint's forward-kinematics position is unchanged: a leaf
    has no children, so nothing downstream referenced it.
    """
    keep = [j for j in range(anim.n_joints) if j not in joints]
    old_to_new = {old: new for new, old in enumerate(keep)}

    new_parents = np.array(
        [-1 if anim.parents[j] == -1 else old_to_new[int(anim.parents[j])] for j in keep],
        dtype=np.int32,
    )
    new_anim = Anim(
        rotations=anim.rotations[:, keep, :].copy(),
        root_pos=anim.root_pos.copy(),
        offsets=anim.offsets[keep].copy(),
        parents=new_parents,
        names=tuple(anim.names[j] for j in keep),
        fps=anim.fps,
    )
    new_resolved = _reindex_resolved(resolved, old_to_new)
    return new_anim, new_resolved, JointEdit(source_of=tuple(keep))


@AUGMENTATIONS.register("drop_end_effector")
@dataclass
class DropEndEffector(Augmentation):
    """Randomly drop a fraction of non-foot, non-facing end-effector joints.

    Ports the reference's ``remove_joints_augmentation``. A rate is drawn
    uniformly from ``rate``; the number of joints dropped is
    ``floor(len(candidates) * rate)``, so a small rate on a small candidate
    set can legally drop zero joints (identity edit).
    """

    rate: tuple[float, ...] = field(default_factory=lambda: (0.1, 0.2, 0.3))

    def apply_structural(self, anim: Anim, resolved: ResolvedSkeleton, rng: np.random.Generator):
        excluded = _excluded_joints(resolved)
        candidates = sorted(j for j in leaves(anim.parents) if j not in excluded and j != 0)
        if not candidates:
            return anim, resolved, JointEdit.identity(anim.n_joints)

        rate = float(rng.choice(self.rate))
        n_remove = math.floor(len(candidates) * rate)
        if n_remove == 0:
            return anim, resolved, JointEdit.identity(anim.n_joints)

        drop = set(rng.choice(candidates, size=n_remove, replace=False).tolist())
        return drop_joints(anim, resolved, drop)
```

Note: `QUAT_IDENTITY` is imported here already because Task 4 (same file) needs it — leaving the import unused between Task 3 and Task 4 would fail lint; add `duplicate_joint` in the same commit as this import, i.e. do Task 4 immediately after this step before running lint. If you run lint between Task 3 and Task 4, temporarily drop the unused `QUAT_IDENTITY`/`math` imports Task 4 needs and re-add them in Task 4's step 3.

- [ ] **Step 4: Run tests to verify they pass**

Run: `docker compose run --rm test pytest tests/augment/test_topology.py -v`
Expected: PASS (5 tests). If `ruff` complains about the unused `QUAT_IDENTITY` import (Task 4 needs it, not this task), remove that one import line for now — Task 4 adds it back.

- [ ] **Step 5: Commit**

```bash
git add src/poseydon/augment/topology.py tests/augment/test_topology.py
git commit -m "$(cat <<'EOF'
feat(augment): add DropEndEffector structural augmentation

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_012AQAdPqZMa4Zcve31dXMpp
EOF
)"
```

---

### Task 4: `duplicate_joint` and `DuplicateJoint`

**Files:**
- Modify: `src/poseydon/augment/topology.py` (append to the file from Task 3)
- Modify: `tests/augment/test_topology.py` (append tests)

**Interfaces:**
- Consumes: everything from Task 3 in the same module (`_excluded_joints`, `child_count`), plus `QUAT_IDENTITY` from `poseydon.core.rotations`.
- Produces: `duplicate_joint(anim: Anim, resolved: ResolvedSkeleton, joint: int) -> tuple[Anim, ResolvedSkeleton, JointEdit]`, `DuplicateJoint(Augmentation)`, registered as `"duplicate_joint"`. Task 5 imports both.

- [ ] **Step 1: Write the failing tests**

Append to `tests/augment/test_topology.py`:

```python
from poseydon.augment.topology import DuplicateJoint, child_count, duplicate_joint


def _duplicatable_joint(anim, resolved) -> int:
    excluded = _excluded(resolved)
    counts = child_count(anim.parents)
    candidates = [
        j
        for j in range(1, anim.n_joints)
        if counts[j] == 1 and anim.parents[j] != 0 and j not in excluded
    ]
    if not candidates:
        pytest.skip("this fixture has no duplicatable joint")
    return candidates[0]


def test_duplicate_joint_adds_exactly_one_joint(bvh_fixture):
    anim = load_bvh(bvh_fixture)
    resolved = resolved_for(bvh_fixture, anim)
    j = _duplicatable_joint(anim, resolved)

    new_anim, new_resolved, edit = duplicate_joint(anim, resolved, j)

    assert new_anim.n_joints == anim.n_joints + 1
    check_topological_order(new_anim.parents)
    assert len(edit.source_of) == new_anim.n_joints
    assert new_anim.parents[j] == anim.parents[j]  # midpoint keeps j's old parent
    assert new_anim.parents[j + 1] == j            # j itself now hangs off the midpoint


def test_duplicate_joint_preserves_downstream_positions(bvh_fixture):
    anim = load_bvh(bvh_fixture)
    resolved = resolved_for(bvh_fixture, anim)
    j = _duplicatable_joint(anim, resolved)

    original_positions = anim.global_positions()
    new_anim, _, edit = duplicate_joint(anim, resolved, j)
    new_positions = new_anim.global_positions()

    for new_index, old_index in enumerate(edit.source_of):
        if new_index == j:
            continue  # the freshly inserted midpoint has no "before" to compare
        np.testing.assert_allclose(
            new_positions[:, new_index], original_positions[:, old_index], atol=1e-9
        )


def test_duplicate_joint_new_joint_sits_at_the_bone_midpoint(bvh_fixture):
    anim = load_bvh(bvh_fixture)
    resolved = resolved_for(bvh_fixture, anim)
    j = _duplicatable_joint(anim, resolved)
    parent = int(anim.parents[j])

    original_positions = anim.global_positions()
    new_anim, _, _ = duplicate_joint(anim, resolved, j)
    new_positions = new_anim.global_positions()

    midpoint = (original_positions[:, parent] + original_positions[:, j]) / 2
    np.testing.assert_allclose(new_positions[:, j], midpoint, atol=1e-9)


def test_duplicate_joint_never_selects_a_facing_or_foot_joint(bvh_fixture):
    anim = load_bvh(bvh_fixture)
    resolved = resolved_for(bvh_fixture, anim)
    excluded = _excluded(resolved)
    augmentation = DuplicateJoint(p=1.0)

    for seed in range(50):
        new_anim, _, edit = augmentation.apply_structural(
            anim, resolved, np.random.default_rng(seed)
        )
        if new_anim.n_joints == anim.n_joints:
            continue  # no candidate this draw
        duplicated_original = {
            old for new_index, old in enumerate(edit.source_of) if new_index != new_anim.n_joints
        }
        assert duplicated_original.isdisjoint(excluded)


def test_duplicate_joint_is_a_no_op_when_nothing_qualifies():
    # A 2-joint chain: root (0) -> child (1). Child's parent IS the root, so
    # it fails the "parent is not the root" rule -- no legal candidate.
    from poseydon.core.anim import Anim
    from poseydon.core.rotations import QUAT_IDENTITY

    n_joints, n_frames = 2, 3
    anim = Anim(
        rotations=np.tile(QUAT_IDENTITY, (n_frames, n_joints, 1)),
        root_pos=np.zeros((n_frames, 3)),
        offsets=np.array([[0.0, 0.0, 0.0], [0.0, -1.0, 0.0]]),
        parents=np.array([-1, 0], dtype=np.int32),
        names=("root", "child"),
        fps=30.0,
    )
    resolved = ResolvedSkeleton(manifest=None, facing_indices=(), foot_indices=())
    augmentation = DuplicateJoint(p=1.0)

    new_anim, new_resolved, edit = augmentation.apply_structural(
        anim, resolved, np.random.default_rng(0)
    )

    assert new_anim is anim
    assert new_resolved is resolved
    assert edit.source_of == (0, 1)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `docker compose run --rm test pytest tests/augment/test_topology.py -v`
Expected: FAIL with `ImportError: cannot import name 'DuplicateJoint'`

- [ ] **Step 3: Implement `duplicate_joint` and `DuplicateJoint`**

Append to `src/poseydon/augment/topology.py` (and restore the `QUAT_IDENTITY` import at the top if Task 3 removed it for lint):

```python
def duplicate_joint(
    anim: Anim, resolved: ResolvedSkeleton, joint: int
) -> tuple[Anim, ResolvedSkeleton, JointEdit]:
    """Insert a midpoint joint between ``joint`` and its parent.

    The new joint takes slot ``joint``; every original joint at or after
    ``joint`` shifts up by one. The new joint gets an identity rotation and
    half of ``joint``'s offset; ``joint`` itself (now at ``joint + 1``) keeps
    its own rotation and the other half-offset -- so every world position,
    ``joint``'s and everything below it, is unchanged (design spec section 4).
    """
    n = anim.n_joints
    n_frames = anim.n_frames
    old_parent = int(anim.parents[joint])

    def shift(old_index: int) -> int:
        return old_index if old_index < joint else old_index + 1

    new_parents = np.empty(n + 1, dtype=np.int32)
    new_offsets = np.empty((n + 1, 3), dtype=anim.offsets.dtype)
    new_rotations = np.empty((n_frames, n + 1, 4), dtype=anim.rotations.dtype)
    new_names: list[str] = []
    source_of: list[int] = []

    for old_index in range(n):
        new_index = shift(old_index)
        new_offsets[new_index] = anim.offsets[old_index]
        new_rotations[:, new_index] = anim.rotations[:, old_index]
        parent = anim.parents[old_index]
        new_parents[new_index] = -1 if parent == -1 else shift(int(parent))
        new_names.append(anim.names[old_index])
        source_of.append(old_index)

    new_names.insert(joint, f"{anim.names[joint]}__mid")
    source_of.insert(joint, joint)
    new_offsets[joint] = anim.offsets[joint] / 2
    new_offsets[joint + 1] = anim.offsets[joint] / 2
    new_rotations[:, joint] = QUAT_IDENTITY
    new_parents[joint] = -1 if old_parent == -1 else shift(old_parent)
    new_parents[joint + 1] = joint

    new_anim = Anim(
        rotations=new_rotations,
        root_pos=anim.root_pos.copy(),
        offsets=new_offsets,
        parents=new_parents,
        names=tuple(new_names),
        fps=anim.fps,
    )
    old_to_new = {old: shift(old) for old in range(n)}
    new_resolved = _reindex_resolved(resolved, old_to_new)
    return new_anim, new_resolved, JointEdit(source_of=tuple(source_of))


@AUGMENTATIONS.register("duplicate_joint")
@dataclass
class DuplicateJoint(Augmentation):
    """Randomly duplicate one single-child, non-root-adjacent joint.

    Ports the reference's ``add_joint_augmentation``, but the split is exact
    under forward kinematics rather than an interpolation of features (see
    design spec section 4).
    """

    def apply_structural(self, anim: Anim, resolved: ResolvedSkeleton, rng: np.random.Generator):
        excluded = _excluded_joints(resolved)
        counts = child_count(anim.parents)
        candidates = [
            j
            for j in range(1, anim.n_joints)
            if counts[j] == 1 and anim.parents[j] != 0 and j not in excluded
        ]
        if not candidates:
            return anim, resolved, JointEdit.identity(anim.n_joints)

        joint = int(rng.choice(candidates))
        return duplicate_joint(anim, resolved, joint)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `docker compose run --rm test pytest tests/augment/test_topology.py -v`
Expected: PASS (11 tests total in the file)

- [ ] **Step 5: Commit**

```bash
git add src/poseydon/augment/topology.py tests/augment/test_topology.py
git commit -m "$(cat <<'EOF'
feat(augment): add DuplicateJoint structural augmentation

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_012AQAdPqZMa4Zcve31dXMpp
EOF
)"
```

---

### Task 5: Public package API and `poseydon list`

**Files:**
- Modify: `src/poseydon/augment/__init__.py`
- Modify: `src/poseydon/cli.py:170-196` (the `_list` function)
- Test: `tests/augment/test_base.py` (extend the registry test)

**Interfaces:**
- Consumes: everything from Tasks 1-4.
- Produces: `poseydon.augment.{AUGMENTATIONS, Augmentation, AugmentPipeline, JointEdit, DropEndEffector, DuplicateJoint}` as the stable import path config `_target_` entries use (e.g. `poseydon.augment.DropEndEffector`).

- [ ] **Step 1: Write the failing test**

Replace the placeholder registry test in `tests/augment/test_base.py`:

```python
def test_augmentations_registry_exists_and_starts_empty_before_topology_import():
    registry = AUGMENTATIONS
    assert registry.kind == "augmentation"
```

with:

```python
def test_topology_augmentations_are_registered_on_import():
    import poseydon.augment  # noqa: F401  (side-effecting import: registers)

    assert AUGMENTATIONS.names() == ["drop_end_effector", "duplicate_joint"]


def test_augmentations_are_importable_from_the_package_root():
    from poseydon.augment import DropEndEffector, DuplicateJoint

    assert DropEndEffector().p == 1.0
    assert DuplicateJoint().p == 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `docker compose run --rm test pytest tests/augment/test_base.py -v`
Expected: FAIL — `poseydon.augment` has no `DropEndEffector`/`DuplicateJoint` attribute yet (the `__init__.py` is still empty).

- [ ] **Step 3: Implement the package `__init__.py` and update `cli.py`**

`src/poseydon/augment/__init__.py`:

```python
"""Data augmentation.

An augmentation is a small class, registered by name so ``poseydon list``
can enumerate it, and instantiated by full ``_target_`` path from config
because (unlike features or conditioners) entries carry parameters of their
own -- a probability, and sometimes more. See the design spec for why this
mirrors ``conditioners:`` in philosophy but not in config shape.
"""

from poseydon.augment import topology as _topology  # noqa: F401  (registers)
from poseydon.augment.base import AUGMENTATIONS, Augmentation, AugmentPipeline
from poseydon.augment.joint_edit import JointEdit
from poseydon.augment.topology import DropEndEffector, DuplicateJoint

__all__ = [
    "AUGMENTATIONS",
    "Augmentation",
    "AugmentPipeline",
    "JointEdit",
    "DropEndEffector",
    "DuplicateJoint",
]
```

In `src/poseydon/cli.py`, inside `_list` (around line 170-190), add the import and registry entry:

```python
def _list(args: argparse.Namespace) -> int:
    from poseydon.augment import AUGMENTATIONS
    from poseydon.conditioners import CONDITIONERS
    from poseydon.data.window import WINDOWS
    from poseydon.features import FEATURES
    from poseydon.losses import LOSSES
    from poseydon.models import MODELS
    from poseydon.ops import OPERATIONS
    from poseydon.process import PROCESSES
    from poseydon.sampling import CONTROLS, SAMPLERS

    registries = {
        "features": FEATURES,
        "conditioners": CONDITIONERS,
        "augmentations": AUGMENTATIONS,
        "losses": LOSSES,
        "models": MODELS,
        "processes": PROCESSES,
        "samplers": SAMPLERS,
        "controls": CONTROLS,
        "operations": OPERATIONS,
        "windows": WINDOWS,
    }
```

(Only the two `import` lines and the one dict entry change; the rest of `_list` is unchanged.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `docker compose run --rm test pytest tests/augment/ -v`
Expected: PASS (all tests across `test_joint_edit.py`, `test_base.py`, `test_topology.py`)

Also smoke-test the CLI change:

Run: `docker compose run --rm test python -m poseydon.cli list --kind augmentations`
Expected:
```
augmentations:
  drop_end_effector
  duplicate_joint
```

- [ ] **Step 5: Commit**

```bash
git add src/poseydon/augment/__init__.py src/poseydon/cli.py tests/augment/test_base.py
git commit -m "$(cat <<'EOF'
feat(augment): expose public API and wire into `poseydon list`

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_012AQAdPqZMa4Zcve31dXMpp
EOF
)"
```

---

### Task 6: Wire `AugmentPipeline` into `MotionDataset`

**Files:**
- Modify: `src/poseydon/data/dataset.py`
- Test: `tests/data/test_pipeline_end_to_end.py` (append; this file already has the `corpus`/`make_dataset` fixtures this task's tests need)

**Interfaces:**
- Consumes: `poseydon.augment.base.{Augmentation, AugmentPipeline}` (Task 2), `poseydon.augment.joint_edit.JointEdit` (Task 1, used only inside `AugmentPipeline`, not directly by `dataset.py`).
- Produces: `MotionDataset.__init__` gains `augmentations: Sequence[Augmentation] = ()`; `MotionDataset.augment_pipeline: AugmentPipeline`. Task 7 passes `augmentations=` from config.

- [ ] **Step 1: Write the failing tests**

Append to `tests/data/test_pipeline_end_to_end.py`:

```python
from poseydon.augment.topology import DropEndEffector


def test_empty_augmentation_list_matches_unaugmented_output(corpus):
    plain = make_dataset(corpus, window=FullClip())
    explicit_empty = make_dataset(corpus, window=FullClip(), augmentations=())

    np.testing.assert_array_equal(plain[0].features, explicit_empty[0].features)


def test_structural_augmentation_keeps_conditioning_consistent_with_features(corpus):
    dataset = make_dataset(
        corpus,
        window=FullClip(),
        augmentations=(DropEndEffector(p=1.0),),
        conditioners=("topology", "tpose", "norm_stats"),
    )
    batch = collate([dataset[i] for i in range(len(dataset))])

    for i in range(len(batch)):
        joints = int(batch.masks.n_joints[i])
        parents = batch.cond["topology"]["parents"][i]
        mean = batch.cond["norm_stats"]["mean"][i]
        tpose = batch.cond["tpose"][i]

        assert torch.all(parents[joints:] == -1)
        assert torch.all(mean[joints:] == 0)
        assert torch.all(tpose[joints:] == 0)
        # the surviving joint count must match across the feature tensor and
        # every declared conditioner -- this is the property the whole design
        # exists to guarantee.
        assert batch.x[i, joints:].abs().sum() == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `docker compose run --rm test pytest tests/data/test_pipeline_end_to_end.py -k augmentation -v`
Expected: FAIL with `TypeError: MotionDataset.__init__() got an unexpected keyword argument 'augmentations'`

- [ ] **Step 3: Wire `AugmentPipeline` into `MotionDataset`**

In `src/poseydon/data/dataset.py`, add the import:

```python
from poseydon.augment.base import Augmentation, AugmentPipeline
```

Change `__init__`'s signature and body (the `conditioners` line is the anchor):

```python
    def __init__(
        self,
        index: CorpusIndex,
        root: str | Path,
        manifest_dir: str | Path,
        features: Sequence[str] = DEFAULT_FEATURES,
        window: Window | None = None,
        conditioners: Sequence[str] = (),
        augmentations: Sequence[Augmentation] = (),
        split: str | None = None,
        seed: int = 0,
    ) -> None:
        self.root = Path(root)
        self.manifest_dir = Path(manifest_dir)
        self.features = tuple(features)
        self.window = window or RandomCrop()
        self.conditioners: list[Conditioner] = [
            CONDITIONERS.get(name)() for name in conditioners
        ]
        self.augment_pipeline = AugmentPipeline(augmentations)
        self.records = index.query(split=split) if split else list(index.records)
```

(Everything below `self.records = ...` in `__init__` is unchanged.)

Replace `__getitem__` entirely:

```python
    def __getitem__(self, index: int) -> Item:
        clip_index, window_index = self._plan[index]
        record = self.records[clip_index]

        anim = self._anim(record)
        resolved = self._resolved(record)
        anim, resolved, edit = self.augment_pipeline.apply_structural(anim, resolved, self._rng)

        raw, spec = extract_features(anim, resolved, self.features)
        base_normalizer = self._normalizer(record.skeleton, spec)
        normalizer = edit.transport(base_normalizer)
        features = normalizer.normalize(raw)
        features = self.augment_pipeline.apply_features(features, spec, resolved, self._rng)

        start, length = self.window.bounds(features.shape[0], window_index, self._rng)
        window = features[start : start + length]

        base_rest_frame = self._rest_frame(record, spec, base_normalizer)
        rest_frame = edit.transport_row(base_rest_frame)

        view = ClipView(
            record=record,
            anim=anim,
            resolved=resolved,
            normalizer=normalizer,
            rest_frame=rest_frame,
        )
        return Item(
            features=window,
            spec=spec,
            start=start,
            source_length=features.shape[0],
            cond={c.name: c.extract(view) for c in self.conditioners},
        )
```

Note what did NOT change: `_anim`, `_resolved`, `_normalizer`, `_rest_frame`, and their caches (`self._anims`, `self._manifests`, `self._normalizers`, `self._rest_frames`) are untouched — they still always compute against the base, unaugmented skeleton, which is exactly what the design requires (Global Constraint: augmentation never mutates the shared per-clip cache).

- [ ] **Step 4: Run tests to verify they pass**

Run: `docker compose run --rm test pytest tests/data/test_pipeline_end_to_end.py -v`
Expected: PASS (all tests in the file, old and new)

Then run the full suite to confirm nothing else regressed:

Run: `docker compose run --rm test pytest`
Expected: PASS, same total minus any newly-added tests, no failures

- [ ] **Step 5: Commit**

```bash
git add src/poseydon/data/dataset.py tests/data/test_pipeline_end_to_end.py
git commit -m "$(cat <<'EOF'
feat(data): wire AugmentPipeline into MotionDataset

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_012AQAdPqZMa4Zcve31dXMpp
EOF
)"
```

---

### Task 7: Hydra config wiring

**Files:**
- Modify: `src/poseydon/training/build.py`
- Modify: `configs/train.yaml`
- Test: `tests/config/test_config.py` (append)

**Interfaces:**
- Consumes: `MotionDataset(augmentations=...)` (Task 6), `poseydon.augment` public API (Task 5), Hydra's `instantiate`.
- Produces: `poseydon train augmentations='[{_target_:poseydon.augment.DropEndEffector,p:0.2}]'` works end to end from the CLI.

`tests/config/test_config.py` already defines the exact helper this task needs:

```python
def load(overrides=()):
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        return compose(config_name="train", overrides=list(overrides))
```

and `test_default_config_composes` / `test_features_are_swappable_from_the_command_line` are the two tests this task's new tests directly mirror. Use `load(...)`, not a new helper.

- [ ] **Step 1: Write the failing tests**

Append to `tests/config/test_config.py`:

```python
def test_augmentations_default_to_empty():
    assert list(load().augmentations) == []


def test_augmentations_override_from_the_command_line():
    config = load(["augmentations=[{_target_:poseydon.augment.DropEndEffector,p:0.2}]"])
    assert config.augmentations[0]["_target_"] == "poseydon.augment.DropEndEffector"
    assert config.augmentations[0]["p"] == 0.2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `docker compose run --rm test pytest tests/config/ -k augmentations -v`
Expected: FAIL — `configs/train.yaml` has no `augmentations` key yet, so `config.augmentations` raises an OmegaConf attribute error.

- [ ] **Step 3: Add the config key and wire `build_dataset`**

In `configs/train.yaml`, add alongside the existing `conditioners:` line:

```yaml
# Only declared conditioning is loaded, collated and moved to device.
conditioners: [topology, tpose, norm_stats]

# Structural or feature-space edits applied per-sample at read time, each
# gated by its own probability and applied in the declared order. Empty by
# default -- augmentation is opt-in, like conditioners.
augmentations: []
```

In `src/poseydon/training/build.py`, update `build_dataset`:

```python
def build_dataset(config: DictConfig, split: str | None = None) -> MotionDataset:
    index = CorpusIndex.load(Path(config.data.index))
    return MotionDataset(
        index=index,
        root=config.data.root,
        manifest_dir=config.data.manifests,
        features=list(config.features),
        window=instantiate(config.data.window),
        conditioners=list(config.conditioners),
        augmentations=[instantiate(entry) for entry in config.get("augmentations", [])],
        split=split,
        seed=config.seed,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `docker compose run --rm test pytest tests/config/ -v`
Expected: PASS (all config tests, old and new)

- [ ] **Step 5: Run the full suite**

Run: `docker compose run --rm test pytest`
Expected: PASS, no regressions

- [ ] **Step 6: Commit**

```bash
git add configs/train.yaml src/poseydon/training/build.py tests/config/
git commit -m "$(cat <<'EOF'
feat(config): wire augmentations into train.yaml and build_dataset

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_012AQAdPqZMa4Zcve31dXMpp
EOF
)"
```

---

## Self-Review Notes

- **Spec coverage:** §5 contract → Tasks 1-2; §6 `DropEndEffector`/`DuplicateJoint` → Tasks 3-4; §7 config surface → Task 7 (simplified from the spec's illustrative `AugmentEntry` wrapper to a `p` field directly on `Augmentation`, since the config example puts `p` flush inside the same `_target_` block as the augmentation's own parameters — noted explicitly in Task 2 rather than left implicit); §8 normalization/rest-frame transport → Task 6; §9 testing list → covered across Tasks 1-4 and 6 (contract no-op, exactness, exclusion, empty-candidate fallback, end-to-end consistency); success criterion 4 (byte-identical empty pipeline) → directly tested in Task 6.
- **Type consistency check:** `Augmentation.apply_structural` returns `tuple[anim, resolved, JointEdit]` consistently from the base class (Task 2) through `DropEndEffector`/`DuplicateJoint` (Tasks 3-4) through `AugmentPipeline` (Task 2) through `MotionDataset.__getitem__` (Task 6) — no signature drift. `JointEdit.source_of` is the one shared vocabulary term used identically in every task.
- **Deferred, per spec §2:** no feature-space augmentation ships; `apply_features` exists on the contract (Task 2) and is exercised only by a test double (Task 2's `_AddOne`), never by a real augmentation. This is intentional, not a gap.

---

**Plan complete and saved to `docs/superpowers/plans/2026-09-05-poseydon-augmentation.md`. Two execution options:**

**1. Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

**Which approach?**

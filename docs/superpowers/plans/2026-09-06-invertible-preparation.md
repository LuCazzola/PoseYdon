# Invertible Preparation Implementation Plan (Phase 1 of 4)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the raw-corpus-to-prepared-motion transform invertible, so a clip can travel `source → prepare → reduce → expand → unprepare → source` and come back on the rig it started on.

**Architecture:** A `PrepareStage` contract with `fit`/`apply`/`invert`, six concrete stages composed into a `PrepareChain`, and a separate `JointReduction` that removes degenerate joints at feature time and puts them back. Stage parameters are fitted once per rig (from the rest pose) or once per clip (the facing rotation) and persisted to `rigs/<Rig>/prepare.npz`, because the transform is otherwise unrecoverable. The Truebones scripts become callers of the chain rather than owners of the geometry.

**Tech Stack:** Python 3.12, numpy, pytest, Hydra/OmegaConf (later phases), Docker (`docker compose run --rm test`).

**Spec:** `docs/superpowers/specs/2026-09-06-data-pipeline-and-training-parity-design.md`

## Global Constraints

- Everything runs in the test container: `docker compose run --rm test pytest …`. There is no host Python.
- `HML_MEAN_BONE_LENGTH = 0.20921428571428569` — the scale target, already in `core/skeleton.py`.
- Stage order is contractual: `RestRelative`, `FaceAxis`, `EnforceRigid`, `CentreXZ`, `ScaleToMeanBoneLength`, `PutOnGround`. It matches the reference's `process_anim` (rotate, centre, scale, ground). Reordering changes the result.
- Quaternions are scalar-last `(x, y, z, w)`; `QUAT_IDENTITY = [0, 0, 0, 1]`. Compose with `quat_mul(a, b)` meaning "apply `b`, then `a`".
- Forward kinematics convention: `global_rot[j] = global_rot[parent] · local_rot[j]`, `pos[j] = pos[parent] + apply(global_rot[parent], translations[j])`.
- `Animation` is frozen; every transform returns a new instance. `RigidBodyAnimation.from_root_motion(rotations=, root_pos=, offsets=, parents=, names=, fps=)` is the rigid constructor.
- Joint names are unique within a rig and are the canonical identity. Never key on index across a structural edit.
- Tests that need the Truebones corpus skip when it is absent; tests that need Blender skip when the FBX artefacts are absent. The suite must pass on a clean checkout with no data.
- Line length 100 (`ruff`, configured in `pyproject.toml`).
- Commit after every task.

---

## File Structure

| file | responsibility |
|---|---|
| `src/poseydon/build/__init__.py` | package marker, re-exports `PrepareChain` and the stages |
| `src/poseydon/build/prepare.py` | `PrepareStage` contract, six stages, `PrepareChain`, `RigTransform` persistence |
| `src/poseydon/features/reduce.py` | `JointReduction`, `build_reduction`, `apply_reduction`, `invert_reduction` |
| `src/poseydon/core/skeleton.py` | modified: manifests move to `rigs/<Rig>/manifest.yaml`; `strip_joint_prefix` removed |
| `src/poseydon/core/animation.py` | modified: `rest_geometry(anim)` extracted so `RestRelative.fit` need not read a file |
| `src/poseydon/io/bvh.py` | modified: `from_animation(anim, channels=None)` so a source channel layout can be restored |
| `scripts/process_dataset_truebones.py` | rewritten around `PrepareChain`; writes the new layout and `prepare.npz` |
| `scripts/process_dataset_truebones_fbx.py` | new layout; writes `mesh.npz` |
| `tests/conftest.py` | corpus fixtures and skip markers |
| `tests/build/test_prepare.py` | stage-level fit/apply/invert |
| `tests/features/test_reduce.py` | reduction exactness |
| `tests/build/test_roundtrip.py` | the end-to-end round-trip on real data |
| `tests/build/test_bvh_fbx_agreement.py` | cross-source skeleton agreement |

`src/poseydon/preproc/rest_pose.py` keeps `make_anim_rest_relative` and `establish_rest_pose` until Task 3 folds them into `build/prepare.py`; the package is deleted there.

---

### Task 1: Test scaffolding

The suite directory is empty and `pyproject.toml` still points `testpaths` at it. Nothing else can be tested until fixtures exist.

**Files:**
- Create: `tests/__init__.py`, `tests/conftest.py`, `tests/build/__init__.py`, `tests/features/__init__.py`
- Create: `tests/test_scaffold.py`

**Interfaces:**
- Produces: fixtures `corpus_root` (Path to `data/truebones`, skips if absent), `rig_names` (a fixed sample of rigs), `raw_clip(rig)` factory returning a `(clip_path, tpose_path)` pair.

- [ ] **Step 1: Write the failing test**

`tests/test_scaffold.py`:

```python
from poseydon.core.skeleton import HML_MEAN_BONE_LENGTH


def test_package_imports_and_constant_is_the_smpl_mean():
    assert HML_MEAN_BONE_LENGTH == 0.20921428571428569
```

`tests/conftest.py`:

```python
"""Shared fixtures.

Real data is primary, but the suite must pass without it: a clean checkout has
no Truebones corpus (it is a paid collection) and no Blender, so every fixture
that needs either skips rather than fails.
"""

from __future__ import annotations

from pathlib import Path

import pytest

CORPUS = Path("data/truebones")

# One biped, one quadruped, one milliped, plus the rig whose reduction differs
# from the reference's (Scorpion keeps a zero-offset joint with siblings).
SAMPLE_RIGS = ("Flamingo", "BrownBear", "Crab", "Scorpion")


@pytest.fixture(scope="session")
def corpus_root() -> Path:
    if not (CORPUS / "source").is_dir() and not (CORPUS / "Truebone_Z-OO").is_dir():
        pytest.skip("Truebones corpus not present")
    return CORPUS


@pytest.fixture(scope="session")
def source_root(corpus_root: Path) -> Path:
    """The raw corpus, under whichever name it currently has."""
    for name in ("source", "Truebone_Z-OO"):
        candidate = corpus_root / name
        if candidate.is_dir():
            return candidate
    pytest.skip("no raw corpus directory")


@pytest.fixture
def raw_clips(source_root: Path):
    """`raw_clips(rig)` -> sorted list of that rig's raw .bvh paths."""

    def _get(rig: str) -> list[Path]:
        paths = sorted((source_root / rig).glob("*.bvh"))
        if not paths:
            pytest.skip(f"no raw clips for {rig}")
        return paths

    return _get
```

`tests/__init__.py`, `tests/build/__init__.py`, `tests/features/__init__.py` are empty files.

- [ ] **Step 2: Run test to verify it fails**

Run: `docker compose run --rm test pytest tests/test_scaffold.py -v`
Expected: FAIL — `tests/` has no files yet, so collection reports "no tests ran" before the files are written; after writing, it should PASS. If it fails with `ModuleNotFoundError: poseydon`, check `PYTHONPATH=/app/src` is set by `docker-compose.yml`.

- [ ] **Step 3: Create the files**

Create the four `__init__.py` files (empty), `tests/conftest.py` and `tests/test_scaffold.py` with the content above.

- [ ] **Step 4: Run tests to verify they pass**

Run: `docker compose run --rm test pytest tests/ -v`
Expected: PASS, 1 test.

- [ ] **Step 5: Commit**

```bash
git add tests/
git commit -m "test: restore suite scaffolding with corpus fixtures that skip when data is absent"
```

---

### Task 2: The `PrepareStage` contract and `PrepareChain`

**Files:**
- Create: `src/poseydon/build/__init__.py`, `src/poseydon/build/prepare.py`
- Test: `tests/build/test_prepare.py`

**Interfaces:**
- Produces:
  - `PrepareStage` — abstract, class vars `name: str`, `scope: str` (`"rig"` or `"clip"`); methods `fit(anim, resolved) -> dict[str, np.ndarray]` (default `{}`), `apply(anim, params) -> Animation`, `invert(anim, params) -> Animation`.
  - `PrepareChain(stages: tuple[PrepareStage, ...])` with `fit_rig(rest, resolved) -> dict[str, dict]`, `apply(anim, resolved, rig_params) -> tuple[Animation, dict[str, dict]]`, `invert(anim, params) -> Animation`.
- Consumes: nothing.

- [ ] **Step 1: Write the failing test**

`tests/build/test_prepare.py`:

```python
"""Stage contract and chain composition."""

from __future__ import annotations

import numpy as np
import pytest

from poseydon.build.prepare import PrepareChain, PrepareStage
from poseydon.core.animation import Animation


def _anim(n_frames: int = 3, n_joints: int = 2) -> Animation:
    rotations = np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (n_frames, n_joints, 1))
    offsets = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    translations = np.broadcast_to(offsets, (n_frames, n_joints, 3)).copy()
    return Animation(
        rotations=rotations,
        translations=translations,
        offsets=offsets,
        parents=np.array([-1, 0], dtype=np.int32),
        names=("root", "child"),
        fps=30.0,
    )


class Shift(PrepareStage):
    """Adds a fitted constant to the root trajectory. Rig-scoped."""

    name = "shift"
    scope = "rig"

    def fit(self, anim, resolved):
        return {"amount": anim.translations[0, 0].copy()}

    def apply(self, anim, params):
        translations = anim.translations.copy()
        translations[:, 0] = translations[:, 0] - params["amount"]
        return Animation(
            rotations=anim.rotations, translations=translations, offsets=anim.offsets,
            parents=anim.parents, names=anim.names, fps=anim.fps,
        )

    def invert(self, anim, params):
        translations = anim.translations.copy()
        translations[:, 0] = translations[:, 0] + params["amount"]
        return Animation(
            rotations=anim.rotations, translations=translations, offsets=anim.offsets,
            parents=anim.parents, names=anim.names, fps=anim.fps,
        )


class Double(PrepareStage):
    """Scales every offset. Clip-scoped, so it refits per animation."""

    name = "double"
    scope = "clip"

    def fit(self, anim, resolved):
        return {"factor": np.float64(2.0)}

    def apply(self, anim, params):
        return Animation(
            rotations=anim.rotations, translations=anim.translations * params["factor"],
            offsets=anim.offsets * params["factor"], parents=anim.parents,
            names=anim.names, fps=anim.fps,
        )

    def invert(self, anim, params):
        return Animation(
            rotations=anim.rotations, translations=anim.translations / params["factor"],
            offsets=anim.offsets / params["factor"], parents=anim.parents,
            names=anim.names, fps=anim.fps,
        )


def test_fit_rig_keeps_only_rig_scoped_parameters():
    chain = PrepareChain((Shift(), Double()))
    rig = chain.fit_rig(_anim(), resolved=None)
    assert set(rig) == {"shift"}


def test_apply_then_invert_is_the_identity():
    chain = PrepareChain((Shift(), Double()))
    source = _anim()
    rig = chain.fit_rig(source, resolved=None)

    prepared, params = chain.apply(source, resolved=None, rig_params=rig)
    restored = chain.invert(prepared, params)

    np.testing.assert_allclose(restored.translations, source.translations, atol=1e-12)
    np.testing.assert_allclose(restored.offsets, source.offsets, atol=1e-12)


def test_invert_runs_stages_in_reverse_order():
    """Shift-then-double is not double-then-shift; a chain that inverted in
    forward order would still round-trip each stage but not the composition."""
    chain = PrepareChain((Shift(), Double()))
    source = _anim()
    rig = chain.fit_rig(source, resolved=None)
    prepared, params = chain.apply(source, resolved=None, rig_params=rig)

    # The root moved to the origin BEFORE doubling, so doubling cannot move it.
    np.testing.assert_allclose(prepared.translations[:, 0], 0.0, atol=1e-12)


def test_a_clip_scoped_stage_refits_on_each_animation():
    chain = PrepareChain((Double(),))
    rig = chain.fit_rig(_anim(), resolved=None)
    assert rig == {}
    _prepared, params = chain.apply(_anim(), resolved=None, rig_params=rig)
    assert "double" in params


def test_apply_rejects_a_missing_rig_parameter():
    chain = PrepareChain((Shift(),))
    with pytest.raises(KeyError, match="shift"):
        chain.apply(_anim(), resolved=None, rig_params={})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `docker compose run --rm test pytest tests/build/test_prepare.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'poseydon.build'`

- [ ] **Step 3: Write minimal implementation**

`src/poseydon/build/__init__.py`:

```python
"""Turning a raw corpus into a prepared one, invertibly."""

from poseydon.build.prepare import PrepareChain, PrepareStage

__all__ = ["PrepareChain", "PrepareStage"]
```

`src/poseydon/build/prepare.py`:

```python
"""Invertible preparation stages.

A prepared corpus is only useful to an application if the transform that
produced it can be undone: a user hands over a rig, the model generates on the
canonical one, and the result has to come back on theirs. So every stage here
declares an ``invert`` beside its ``apply``, and the parameters it fitted are
persisted rather than recomputed -- the facing rotation in particular is derived
from a clip's own frame 0 and is unrecoverable once the file already faces +Z.

Stages are fitted at one of two scopes. A ``"rig"`` stage fits once, against the
rest pose, and every clip of that character reuses the result; fitting per clip
would ground a flying creature onto the floor and destroy the height difference
between a crouch and a stand. A ``"clip"`` stage fits per animation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np

from poseydon.core.animation import Animation

RIG = "rig"
CLIP = "clip"


class PrepareStage(ABC):
    """One invertible step of the source-to-prepared transform."""

    #: Key under which this stage's fitted parameters are stored.
    name: ClassVar[str]
    #: ``RIG`` to fit once against the rest pose, ``CLIP`` to fit per animation.
    scope: ClassVar[str] = RIG

    def fit(self, anim: Animation, resolved: Any) -> dict[str, np.ndarray]:
        """Parameters derived from ``anim`` as it stands at this point in the chain."""
        return {}

    @abstractmethod
    def apply(self, anim: Animation, params: dict[str, np.ndarray]) -> Animation: ...

    @abstractmethod
    def invert(self, anim: Animation, params: dict[str, np.ndarray]) -> Animation: ...


@dataclass(frozen=True)
class PrepareChain:
    """An ordered composition of stages, invertible as a whole."""

    stages: tuple[PrepareStage, ...]

    def fit_rig(self, rest: Animation, resolved: Any) -> dict[str, dict]:
        """Fit every stage against the rest pose; keep the rig-scoped results.

        Each stage fits against the animation as the PRECEDING stages left it,
        which is why this applies as it goes: the scale factor is measured on an
        already-rest-relative skeleton, and the ground height on an already
        scaled one.
        """
        params: dict[str, dict] = {}
        current = rest
        for stage in self.stages:
            params[stage.name] = stage.fit(current, resolved)
            current = stage.apply(current, params[stage.name])
        return {stage.name: params[stage.name] for stage in self.stages if stage.scope == RIG}

    def apply(
        self, anim: Animation, resolved: Any, rig_params: dict[str, dict]
    ) -> tuple[Animation, dict[str, dict]]:
        """Prepare one animation, returning it with the full parameter set."""
        params: dict[str, dict] = dict(rig_params)
        current = anim
        for stage in self.stages:
            if stage.scope == CLIP:
                params[stage.name] = stage.fit(current, resolved)
            elif stage.name not in params:
                raise KeyError(
                    f"no fitted parameters for rig-scoped stage `{stage.name}`; "
                    "call fit_rig against this rig's rest pose first"
                )
            current = stage.apply(current, params[stage.name])
        return current, params

    def invert(self, anim: Animation, params: dict[str, dict]) -> Animation:
        """Undo the whole chain, stages in reverse order."""
        current = anim
        for stage in reversed(self.stages):
            current = stage.invert(current, params[stage.name])
        return current
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `docker compose run --rm test pytest tests/build/test_prepare.py -v`
Expected: PASS, 5 tests.

- [ ] **Step 5: Commit**

```bash
git add src/poseydon/build tests/build
git commit -m "feat(build): invertible PrepareStage contract and PrepareChain"
```

---

### Task 3: `RestRelative`

Bind removal, refactored from `preproc/rest_pose.py::make_anim_rest_relative` into fit/apply/invert. The existing function derives its constants from a rest `Animation` on every call; splitting it means the constants are fitted once, stored, and reusable in both directions.

**Files:**
- Modify: `src/poseydon/core/animation.py` — add `rest_geometry`
- Modify: `src/poseydon/build/prepare.py` — add `RestRelative`
- Modify: `src/poseydon/build/__init__.py` — export it
- Delete: `src/poseydon/preproc/` (both files)
- Test: `tests/build/test_prepare.py`

**Interfaces:**
- Consumes: `PrepareStage` from Task 2.
- Produces: `RestRelative()` with `name = "rest_relative"`, `scope = RIG`. Fitted params: `local_bind (J, 4)`, `rest_offsets (J, 3)`, `source_offsets (J, 3)`. Also `rest_geometry(anim: Animation) -> RigidBodyAnimation` in `core/animation.py`.

- [ ] **Step 1: Write the failing test**

Append to `tests/build/test_prepare.py`:

```python
from poseydon.build.prepare import RestRelative
from poseydon.core.rotations import QUAT_IDENTITY, euler_to_quat


def _bent(n_frames: int = 4) -> Animation:
    """A three-joint chain whose rest pose is NOT the identity pose."""
    rest = euler_to_quat(np.array([0.0, 0.0, 20.0]), "ZYX")
    rotations = np.tile(QUAT_IDENTITY, (n_frames, 3, 1))
    rotations[:, 1] = rest
    rotations[:, 2] = euler_to_quat(np.array([0.0, 0.0, 35.0]), "ZYX")
    offsets = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 2.0, 0.0]])
    translations = np.broadcast_to(offsets, (n_frames, 3, 3)).copy()
    translations[:, 0] = np.arange(n_frames)[:, None] * np.array([1.0, 0.0, 0.0])
    return Animation(
        rotations=rotations, translations=translations, offsets=offsets,
        parents=np.array([-1, 0, 1], dtype=np.int32),
        names=("root", "mid", "tip"), fps=30.0,
    )


def test_rest_relative_makes_the_rest_pose_read_as_identity():
    stage = RestRelative()
    rest = _bent(n_frames=1)
    params = stage.fit(rest, resolved=None)
    prepared = stage.apply(rest, params)

    dots = np.abs(np.sum(prepared.rotations * QUAT_IDENTITY, axis=-1))
    np.testing.assert_allclose(dots, 1.0, atol=1e-9)


def test_rest_relative_preserves_world_positions():
    stage = RestRelative()
    source = _bent()
    params = stage.fit(_bent(n_frames=1), resolved=None)
    prepared = stage.apply(source, params)

    np.testing.assert_allclose(
        prepared.global_positions(), source.global_positions(), atol=1e-9
    )


def test_rest_relative_inverts_exactly():
    stage = RestRelative()
    source = _bent()
    params = stage.fit(_bent(n_frames=1), resolved=None)
    restored = stage.invert(stage.apply(source, params), params)

    np.testing.assert_allclose(restored.rotations, source.rotations, atol=1e-9)
    np.testing.assert_allclose(restored.translations, source.translations, atol=1e-9)
    np.testing.assert_allclose(restored.offsets, source.offsets, atol=1e-9)


def test_rest_relative_refuses_a_rig_it_was_not_fitted_to():
    stage = RestRelative()
    params = stage.fit(_bent(n_frames=1), resolved=None)
    other = _anim()
    with pytest.raises(ValueError, match="fitted"):
        stage.apply(other, params)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `docker compose run --rm test pytest tests/build/test_prepare.py -k rest_relative -v`
Expected: FAIL with `ImportError: cannot import name 'RestRelative'`

- [ ] **Step 3: Write the implementation**

Add to `src/poseydon/core/animation.py`, after the `Animation` class:

```python
def rest_geometry(anim: Animation) -> RigidBodyAnimation:
    """Recover true bone geometry from a rest-pose frame's position channels.

    Raw Biped exports carry the real skeleton in their per-joint POSITION
    channels while the OFFSET block describes something else, so bone offsets
    have to be read back out of frame 0's world-space layout and expressed in
    each parent's own frame. Keeps the file's declared rotations, which is what
    :class:`~poseydon.build.prepare.RestRelative` needs as the bind.
    """
    first = anim.slice(0, 1)
    global_pos, global_rot = first.global_transforms()
    parent_of = first.parents[1:]

    offsets = np.zeros_like(first.offsets)
    offsets[1:] = quat_apply(
        quat_inverse(global_rot[0, parent_of]),
        global_pos[0, 1:] - global_pos[0, parent_of],
    )
    return RigidBodyAnimation.from_root_motion(
        rotations=first.rotations,
        root_pos=global_pos[:, 0],
        offsets=offsets,
        parents=first.parents,
        names=first.names,
        fps=first.fps,
    )
```

Add `from poseydon.core.rotations import quat_apply, quat_inverse` to that module's imports.

Add to `src/poseydon/build/prepare.py`:

```python
from poseydon.core.animation import RigidBodyAnimation, rest_geometry
from poseydon.core.rotations import quat_apply, quat_inverse, quat_mul


def _global_binds(local_bind: np.ndarray, parents: np.ndarray) -> np.ndarray:
    """Accumulate local rest rotations down the hierarchy. Parents precede children."""
    binds = np.empty_like(local_bind)
    binds[0] = local_bind[0]
    for joint in range(1, len(parents)):
        binds[joint] = quat_mul(binds[parents[joint]], local_bind[joint])
    return binds


class RestRelative(PrepareStage):
    """Re-express rotations so zero on every joint reproduces the rest pose.

    Raw Biped rigs bake an arbitrary per-joint bind rotation into every clip --
    an exporter axis-convention artefact, not motion. Removing it is what makes a
    T-pose read as identity, which every downstream consumer assumes.

    Offsets become plain WORLD-SPACE differences of the rest pose's own global
    positions, not offsets rotated into a joint's local frame: identity rotation
    applied to a world-space offset reproduces that offset unchanged, which is
    exactly what makes forward kinematics self-consistent at the rest frame.
    Pairing identity rotations with parent-local offsets instead is a
    coordinate-frame mismatch invisible at the rest frame and wrong everywhere
    else.
    """

    name = "rest_relative"
    scope = RIG

    def fit(self, anim: Animation, resolved: Any) -> dict[str, np.ndarray]:
        rest = rest_geometry(anim)
        rest_global = rest.global_positions()
        parent_of = rest.parents[1:]

        rest_offsets = anim.offsets.copy()
        rest_offsets[1:] = rest_global[0, 1:] - rest_global[0, parent_of]
        return {
            "local_bind": rest.rotations[0].copy(),
            "rest_offsets": rest_offsets,
            "source_offsets": anim.offsets.copy(),
            "names": np.array(anim.names, dtype=object),
        }

    @staticmethod
    def _check(anim: Animation, params: dict[str, np.ndarray]) -> None:
        fitted = tuple(str(n) for n in params["names"])
        if tuple(anim.names) != fitted:
            raise ValueError(
                "this animation's joints differ from the rig `rest_relative` was "
                f"fitted to ({len(anim.names)} vs {len(fitted)} joints); a clip "
                "rigged differently from its own rest pose cannot be prepared "
                "against it"
            )

    def apply(self, anim: Animation, params: dict[str, np.ndarray]) -> Animation:
        self._check(anim, params)
        local_bind = params["local_bind"]
        binds = _global_binds(local_bind, anim.parents)

        rotations = anim.rotations.copy()
        rotations[:, 0] = quat_mul(anim.rotations[:, 0], quat_inverse(local_bind[0]))

        translations = anim.translations.copy()
        for joint in range(1, anim.n_joints):
            parent_bind = binds[anim.parents[joint]]
            # Sandwiched between its own local bind (undone) and its parent's
            # global bind (removed, then reapplied).
            rotations[:, joint] = quat_mul(
                quat_mul(
                    quat_mul(parent_bind, anim.rotations[:, joint]),
                    quat_inverse(local_bind[joint]),
                ),
                quat_inverse(parent_bind),
            )
            # A translation is expressed in the PARENT's frame, and that frame
            # has just turned, so it turns with it.
            translations[:, joint] = quat_apply(parent_bind, anim.translations[:, joint])

        return Animation(
            rotations=rotations,
            translations=translations,
            offsets=params["rest_offsets"].copy(),
            parents=anim.parents,
            names=anim.names,
            fps=anim.fps,
        )

    def invert(self, anim: Animation, params: dict[str, np.ndarray]) -> Animation:
        self._check(anim, params)
        local_bind = params["local_bind"]
        binds = _global_binds(local_bind, anim.parents)

        rotations = anim.rotations.copy()
        rotations[:, 0] = quat_mul(anim.rotations[:, 0], local_bind[0])

        translations = anim.translations.copy()
        for joint in range(1, anim.n_joints):
            parent_bind = binds[anim.parents[joint]]
            rotations[:, joint] = quat_mul(
                quat_mul(
                    quat_mul(quat_inverse(parent_bind), anim.rotations[:, joint]),
                    parent_bind,
                ),
                local_bind[joint],
            )
            translations[:, joint] = quat_apply(
                quat_inverse(parent_bind), anim.translations[:, joint]
            )

        return Animation(
            rotations=rotations,
            translations=translations,
            offsets=params["source_offsets"].copy(),
            parents=anim.parents,
            names=anim.names,
            fps=anim.fps,
        )
```

Update `src/poseydon/build/__init__.py` to export `RestRelative`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `docker compose run --rm test pytest tests/build/test_prepare.py -v`
Expected: PASS, 9 tests.

- [ ] **Step 5: Delete the superseded module and fix its callers**

```bash
git rm -r src/poseydon/preproc
```

Then in `scripts/process_dataset_truebones.py` replace
`from poseydon.preproc.rest_pose import establish_rest_pose, make_anim_rest_relative`
with `from poseydon.build.prepare import RestRelative` — the script is rewritten in Task 9, so for now it only needs to import cleanly. Confirm with:

Run: `docker compose run --rm test python -c "import scripts.process_dataset_truebones"`
Expected: no output.

- [ ] **Step 6: Commit**

```bash
git add -A src/poseydon tests scripts
git commit -m "feat(build): RestRelative stage, fitted once and invertible

Splits preproc/rest_pose.py::make_anim_rest_relative into fit/apply/invert
so the bind rotation is a stored constant rather than something rederived
on every call, and can therefore be put back."
```

---

### Task 4: `FaceAxis`

**Files:**
- Modify: `src/poseydon/ingest/align.py` — extract `rotate_rig(anim, rotation)` from `rotate_to_face_axis`
- Modify: `src/poseydon/build/prepare.py` — add `FaceAxis`
- Test: `tests/build/test_prepare.py`

**Interfaces:**
- Consumes: `PrepareStage`; `facing_quats`, `quat_inverse`.
- Produces: `FaceAxis(axis="+Z")` with `name = "face_axis"`, `scope = CLIP`. Fitted params: `rotation (4,)`. Also `rotate_rig(anim: Animation, rotation: np.ndarray) -> Animation` in `ingest/align.py`.

- [ ] **Step 1: Write the failing test**

Append to `tests/build/test_prepare.py`:

```python
from dataclasses import dataclass

from poseydon.build.prepare import FaceAxis


@dataclass(frozen=True)
class _FakeManifest:
    extra_yaw_deg: float = 0.0


@dataclass(frozen=True)
class _FakeResolved:
    facing_indices: tuple[tuple[int, int], ...]
    manifest: _FakeManifest = _FakeManifest()


def _wide() -> Animation:
    """Root with a left and a right joint, facing +X rather than +Z."""
    offsets = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, -1.0]])
    translations = np.broadcast_to(offsets, (2, 3, 3)).copy()
    return Animation(
        rotations=np.tile(QUAT_IDENTITY, (2, 3, 1)),
        translations=translations, offsets=offsets,
        parents=np.array([-1, 0, 0], dtype=np.int32),
        names=("root", "right", "left"), fps=30.0,
    )


def test_face_axis_turns_frame_zero_onto_plus_z():
    stage = FaceAxis(axis="+Z")
    resolved = _FakeResolved(facing_indices=((1, 2),))
    source = _wide()

    params = stage.fit(source, resolved)
    prepared = stage.apply(source, params)

    positions = prepared.global_positions()
    across = positions[0, 1] - positions[0, 2]
    forward = np.cross(np.array([0.0, 1.0, 0.0]), across / np.linalg.norm(across))
    np.testing.assert_allclose(forward, [0.0, 0.0, 1.0], atol=1e-9)


def test_face_axis_inverts_exactly():
    stage = FaceAxis(axis="+Z")
    resolved = _FakeResolved(facing_indices=((1, 2),))
    source = _wide()

    params = stage.fit(source, resolved)
    restored = stage.invert(stage.apply(source, params), params)

    np.testing.assert_allclose(restored.rotations, source.rotations, atol=1e-9)
    np.testing.assert_allclose(restored.offsets, source.offsets, atol=1e-9)
    np.testing.assert_allclose(
        restored.global_positions(), source.global_positions(), atol=1e-9
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `docker compose run --rm test pytest tests/build/test_prepare.py -k face_axis -v`
Expected: FAIL with `ImportError: cannot import name 'FaceAxis'`

- [ ] **Step 3: Extract `rotate_rig`**

In `src/poseydon/ingest/align.py`, replace the body of `rotate_to_face_axis` after the rotation is computed with a call to a new public function, and add that function above it:

```python
def rotate_rig(anim: Animation, rotation: np.ndarray) -> Animation:
    """Turn a whole rig -- motion AND rest geometry -- by ``rotation``.

    ``offsets' = R . offsets``, ``rotations' = R . rotations . R^-1``,
    ``translations' = R . translations``. The conjugation is what distinguishes
    this from composing onto the root alone: every joint's local frame has
    itself turned. Both forms give identical world motion, but only this one
    leaves a BVH whose OFFSET block describes the new orientation, so a DCC
    tool draws the rest skeleton and the animation facing the same way.

    It also preserves a rest-relative representation: wherever a local rotation
    is identity, ``R . identity . R^-1`` is identity still.
    """
    inverse = quat_inverse(rotation)
    return type(anim)(
        rotations=quat_mul(quat_mul(rotation, anim.rotations), inverse),
        translations=quat_apply(rotation, anim.translations),
        offsets=quat_apply(rotation, anim.offsets),
        parents=anim.parents,
        names=anim.names,
        fps=anim.fps,
    )
```

and change the tail of `rotate_to_face_axis` to:

```python
    rotation = facing_quats(
        anim.global_positions()[:1], facing_indices, extra_yaw_deg, target
    )[0]
    return rotate_rig(anim, rotation)
```

Also export the axis parsing so `FaceAxis` can reuse it — extract the four lines that turn `axis` into `target` into:

```python
def axis_vector(axis: str) -> np.ndarray:
    """``"+Z"``, ``"Z"``, ``"-x"`` -> a unit vector. A bare name means positive."""
    spec = str(axis).strip().upper()
    name = spec[1:] if spec[:1] in "+-" else spec
    if name not in _AXES:
        raise ValueError(
            f"target axis must be X, Y or Z with an optional sign, got `{axis}`"
        )
    return (-1.0 if spec.startswith("-") else 1.0) * _AXES[name]
```

and have `rotate_to_face_axis` call it.

- [ ] **Step 4: Write `FaceAxis`**

Add to `src/poseydon/build/prepare.py`:

```python
from poseydon.ingest.align import axis_vector, facing_quats, rotate_rig


@dataclass(frozen=True)
class FaceAxis(PrepareStage):
    """Turn each clip so its frame-0 facing direction points along ``axis``.

    Clip-scoped, and that is the whole reason this module exists: the rotation
    comes from THIS clip's frame 0, so once the prepared file faces +Z the
    original orientation cannot be recovered from it. It has to be recorded.
    """

    axis: str = "+Z"

    name: ClassVar[str] = "face_axis"
    scope: ClassVar[str] = CLIP

    def fit(self, anim: Animation, resolved: Any) -> dict[str, np.ndarray]:
        rotation = facing_quats(
            anim.global_positions()[:1],
            resolved.facing_indices,
            resolved.manifest.extra_yaw_deg,
            axis_vector(self.axis),
        )[0]
        return {"rotation": rotation}

    def apply(self, anim: Animation, params: dict[str, np.ndarray]) -> Animation:
        return rotate_rig(anim, params["rotation"])

    def invert(self, anim: Animation, params: dict[str, np.ndarray]) -> Animation:
        return rotate_rig(anim, quat_inverse(params["rotation"]))
```

Add `from dataclasses import dataclass` to the imports if not already present.

- [ ] **Step 5: Run tests to verify they pass**

Run: `docker compose run --rm test pytest tests/build/test_prepare.py -v`
Expected: PASS, 11 tests.

- [ ] **Step 6: Commit**

```bash
git add src/poseydon tests
git commit -m "feat(build): FaceAxis stage; extract rotate_rig and axis_vector from align"
```

---

### Task 5: `EnforceRigid`, `CentreXZ`, `ScaleToMeanBoneLength`, `PutOnGround`

The four remaining stages. They are grouped because each is a handful of lines over the same root trajectory, and they are only meaningful in sequence.

**Files:**
- Modify: `src/poseydon/build/prepare.py`
- Modify: `src/poseydon/io/bvh.py` — `from_animation(animation, channels=None)`
- Test: `tests/build/test_prepare.py`

**Interfaces:**
- Produces:
  - `EnforceRigid(joint_translation="drop")`, `name = "enforce_rigid"`, `scope = RIG`. Fitted params: `source_channels` (object array of per-joint channel tuples) — recorded for the writer, not used by `apply`.
  - `CentreXZ()`, `name = "centre_xz"`, `scope = RIG`. Params: `root_xz (3,)`.
  - `ScaleToMeanBoneLength(target=HML_MEAN_BONE_LENGTH)`, `name = "scale"`, `scope = RIG`. Params: `factor` (0-d).
  - `PutOnGround()`, `name = "ground"`, `scope = RIG`. Params: `height` (0-d).
  - `BVH.from_animation(animation, channels=None)`.

- [ ] **Step 1: Write the failing test**

Append to `tests/build/test_prepare.py`:

```python
from poseydon.build.prepare import CentreXZ, EnforceRigid, PutOnGround, ScaleToMeanBoneLength
from poseydon.core.skeleton import HML_MEAN_BONE_LENGTH


def _rigid(n_frames: int = 3) -> Animation:
    offsets = np.array([[0.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 6.0, 0.0]])
    translations = np.broadcast_to(offsets, (n_frames, 3, 3)).copy()
    translations[:, 0] = np.array([7.0, 3.0, -2.0])
    return Animation(
        rotations=np.tile(QUAT_IDENTITY, (n_frames, 3, 1)),
        translations=translations, offsets=offsets,
        parents=np.array([-1, 0, 1], dtype=np.int32),
        names=("root", "mid", "tip"), fps=30.0,
    )


def test_centre_xz_moves_frame_zero_to_the_origin_in_xz_only():
    stage = CentreXZ()
    source = _rigid()
    params = stage.fit(source, resolved=None)
    prepared = stage.apply(source, params)

    np.testing.assert_allclose(prepared.translations[0, 0], [0.0, 3.0, 0.0], atol=1e-12)


def test_scale_makes_the_mean_bone_length_the_target():
    stage = ScaleToMeanBoneLength()
    source = _rigid()
    prepared = stage.apply(source, stage.fit(source, resolved=None))

    lengths = np.linalg.norm(prepared.offsets[1:], axis=-1)
    assert lengths.mean() == pytest.approx(HML_MEAN_BONE_LENGTH)


def test_ground_puts_the_lowest_joint_at_zero():
    stage = PutOnGround()
    source = _rigid()
    prepared = stage.apply(source, stage.fit(source, resolved=None))

    assert prepared.global_positions()[..., 1].min() == pytest.approx(0.0)


@pytest.mark.parametrize(
    "stage", [CentreXZ(), ScaleToMeanBoneLength(), PutOnGround()],
    ids=["centre", "scale", "ground"],
)
def test_geometry_stages_invert_exactly(stage):
    source = _rigid()
    params = stage.fit(source, resolved=None)
    restored = stage.invert(stage.apply(source, params), params)

    np.testing.assert_allclose(restored.translations, source.translations, atol=1e-12)
    np.testing.assert_allclose(restored.offsets, source.offsets, atol=1e-12)


def test_enforce_rigid_records_the_source_channel_layout():
    """The values are lost; the channel DECLARATION is not, so a source rig
    that gave six channels per joint gets six channels back."""
    from poseydon.io.bvh import BVH

    source = _rigid()
    moving = source.translations.copy()
    moving[:, 1] += np.array([0.0, 0.1, 0.0])
    source = Animation(
        rotations=source.rotations, translations=moving, offsets=source.offsets,
        parents=source.parents, names=source.names, fps=source.fps,
    )

    stage = EnforceRigid()
    params = stage.fit(source, resolved=None)
    rigid = stage.apply(source, params)
    restored = stage.invert(rigid, params)

    assert rigid.is_rigid()
    written = BVH.from_animation(restored, channels=params["source_channels"])
    assert "Xposition" in written.channels[1]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `docker compose run --rm test pytest tests/build/test_prepare.py -k "centre or scale or ground or rigid" -v`
Expected: FAIL with `ImportError: cannot import name 'CentreXZ'`

- [ ] **Step 3: Add the `channels` override to the BVH writer**

In `src/poseydon/io/bvh.py`, change `from_animation`'s signature and body head:

```python
    @classmethod
    def from_animation(
        cls, animation: Animation, channels: tuple[tuple[str, ...], ...] | None = None
    ) -> BVH:
        """Prepare an animation for writing.

        By default joints are given position channels only where they actually
        translate, so a rigid animation writes the ordinary ``6 channels on the
        root, 3 elsewhere`` layout. Pass ``channels`` to declare a specific
        layout instead: undoing ``EnforceRigid`` has to restore the source
        file's declaration even though the values in those channels are now the
        constant rest offsets, so the returned rig is structurally the one the
        user supplied.
        """
        if channels is None:
            moves = ~np.all(
                np.isclose(animation.translations, animation.offsets[np.newaxis], atol=1e-9),
                axis=(0, 2),
            )
            children = _children_of(animation.parents)
            built = []
            for joint in range(animation.n_joints):
                if not children[joint]:
                    built.append(())
                elif joint == 0 or moves[joint]:
                    built.append((*_POSITION_CHANNELS, "Zrotation", "Xrotation", "Yrotation"))
                else:
                    built.append(("Zrotation", "Xrotation", "Yrotation"))
            channels = tuple(built)
        elif len(channels) != animation.n_joints:
            raise ValueError(
                f"channels declares {len(channels)} joints but the animation has "
                f"{animation.n_joints}"
            )
```

and end with `channels=tuple(tuple(spec) for spec in channels),` in the constructor call.

- [ ] **Step 4: Write the four stages**

Add to `src/poseydon/build/prepare.py`:

```python
from poseydon.core.skeleton import HML_MEAN_BONE_LENGTH


def _with_root(anim: RigidBodyAnimation, root_pos: np.ndarray) -> RigidBodyAnimation:
    return RigidBodyAnimation.from_root_motion(
        rotations=anim.rotations,
        root_pos=root_pos,
        offsets=anim.offsets,
        parents=anim.parents,
        names=anim.names,
        fps=anim.fps,
    )


@dataclass(frozen=True)
class EnforceRigid(PrepareStage):
    """Impose the rigid-bone assumption, recording the source channel layout.

    Features, the IK solver and the topology augmentations all assume constant
    bone lengths, so this is where a corpus that animates per-joint translation
    stops doing so. The motion discarded is real -- roughly 10% of skeleton size
    on raw Truebones -- which is why the stage is explicit rather than something
    the parser does silently.

    ``invert`` is the identity on the animation, because a rigid animation
    already carries its offsets in every translation slot. What has to be put
    back is the FILE's channel declaration, and that is the writer's job:
    ``BVH.from_animation(anim, channels=params["source_channels"])``.
    """

    joint_translation: str = "drop"

    name: ClassVar[str] = "enforce_rigid"
    scope: ClassVar[str] = RIG

    def fit(self, anim: Animation, resolved: Any) -> dict[str, np.ndarray]:
        moves = ~np.all(
            np.isclose(anim.translations, anim.offsets[np.newaxis], atol=1e-9), axis=(0, 2)
        )
        channels = tuple(
            ("Xposition", "Yposition", "Zposition", "Zrotation", "Xrotation", "Yrotation")
            if joint == 0 or moves[joint]
            else ("Zrotation", "Xrotation", "Yrotation")
            for joint in range(anim.n_joints)
        )
        return {"source_channels": np.array(channels, dtype=object)}

    def apply(self, anim: Animation, params: dict[str, np.ndarray]) -> Animation:
        return anim.as_rigid_body(joint_translation=self.joint_translation)

    def invert(self, anim: Animation, params: dict[str, np.ndarray]) -> Animation:
        return anim


@dataclass(frozen=True)
class CentreXZ(PrepareStage):
    """Translate so the root sits at the XZ origin on the rest pose's first frame."""

    name: ClassVar[str] = "centre_xz"
    scope: ClassVar[str] = RIG

    def fit(self, anim: Animation, resolved: Any) -> dict[str, np.ndarray]:
        return {"root_xz": anim.translations[0, 0] * np.array([1.0, 0.0, 1.0])}

    def apply(self, anim, params):
        return _with_root(anim, anim.root_pos - params["root_xz"])

    def invert(self, anim, params):
        return _with_root(anim, anim.root_pos + params["root_xz"])


@dataclass(frozen=True)
class ScaleToMeanBoneLength(PrepareStage):
    """Uniformly scale so the rig's MEAN bone length equals ``target``.

    ``target`` defaults to the mean of the 21 SMPL bone lengths, which is what
    the reference scales every character to so a scorpion and an elephant occupy
    comparable numeric ranges.
    """

    target: float = HML_MEAN_BONE_LENGTH

    name: ClassVar[str] = "scale"
    scope: ClassVar[str] = RIG

    def fit(self, anim: Animation, resolved: Any) -> dict[str, np.ndarray]:
        mean_length = float(np.linalg.norm(anim.offsets[1:], axis=-1).mean())
        if mean_length < 1e-12:
            raise ValueError("skeleton has zero mean bone length; cannot scale")
        return {"factor": np.float64(self.target / mean_length)}

    def apply(self, anim, params):
        factor = float(params["factor"])
        return RigidBodyAnimation.from_root_motion(
            rotations=anim.rotations, root_pos=anim.root_pos * factor,
            offsets=anim.offsets * factor, parents=anim.parents,
            names=anim.names, fps=anim.fps,
        )

    def invert(self, anim, params):
        factor = float(params["factor"])
        return RigidBodyAnimation.from_root_motion(
            rotations=anim.rotations, root_pos=anim.root_pos / factor,
            offsets=anim.offsets / factor, parents=anim.parents,
            names=anim.names, fps=anim.fps,
        )


@dataclass(frozen=True)
class PutOnGround(PrepareStage):
    """Translate in Y so the rest pose's lowest joint sits at ``y = 0``.

    Rig-scoped deliberately: fitting per clip would ground a flying creature and
    flatten the height difference between a crouch and a stand.
    """

    name: ClassVar[str] = "ground"
    scope: ClassVar[str] = RIG

    def fit(self, anim: Animation, resolved: Any) -> dict[str, np.ndarray]:
        return {"height": np.float64(anim.global_positions()[..., 1].min())}

    def apply(self, anim, params):
        shift = np.array([0.0, float(params["height"]), 0.0])
        return _with_root(anim, anim.root_pos - shift)

    def invert(self, anim, params):
        shift = np.array([0.0, float(params["height"]), 0.0])
        return _with_root(anim, anim.root_pos + shift)
```

Export all four from `src/poseydon/build/__init__.py`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `docker compose run --rm test pytest tests/build/test_prepare.py -v`
Expected: PASS, 18 tests.

- [ ] **Step 6: Commit**

```bash
git add src/poseydon tests
git commit -m "feat(build): rigid, centre, scale and ground stages

EnforceRigid records the source channel layout so its inverse restores the
file's declaration -- the values become the rest offsets, but the rig is
structurally the one the user supplied."
```

---

### Task 6: Joint reduction

Removes joints that carry nothing learnable, and puts them back. Two cases, and conflating them loses real motion: a zero-offset **leaf** can be deleted outright, but a zero-offset **internal** joint swings its subtree, so it can only be folded into its parent — and that fold is exact only when it is the parent's sole child, because otherwise it turns the siblings too.

**Files:**
- Create: `src/poseydon/features/reduce.py`
- Test: `tests/features/test_reduce.py`

**Interfaces:**
- Produces:
  - `RemovalOp(name: str, parent: str, index: int, kind: str, children: tuple[str, ...])`
  - `JointReduction(ops: tuple[RemovalOp, ...], source_of: tuple[int, ...], source_names: tuple[str, ...])`
  - `build_reduction(anim: RigidBodyAnimation, tolerance: float = 1e-8) -> JointReduction`
  - `apply_reduction(anim: RigidBodyAnimation, reduction: JointReduction) -> RigidBodyAnimation`
  - `invert_reduction(anim: RigidBodyAnimation, reduction: JointReduction) -> RigidBodyAnimation`
  - `DropDegenerateJoints(tolerance: float = 1e-8)` — a config-facing wrapper with `.build(anim)`.

- [ ] **Step 1: Write the failing test**

`tests/features/test_reduce.py`:

```python
"""Joint reduction: what comes out, and that nothing moves."""

from __future__ import annotations

import numpy as np
import pytest

from poseydon.core.animation import RigidBodyAnimation
from poseydon.core.rotations import QUAT_IDENTITY, euler_to_quat
from poseydon.features.reduce import (
    apply_reduction,
    build_reduction,
    invert_reduction,
)


def _rig(offsets, parents, names, n_frames=4, spin=()):
    """A rigid animation; joints in `spin` get a non-identity rotation."""
    offsets = np.asarray(offsets, dtype=np.float64)
    rotations = np.tile(QUAT_IDENTITY, (n_frames, len(names), 1))
    for joint in spin:
        rotations[:, joint] = euler_to_quat(np.array([0.0, 0.0, 25.0]), "ZYX")
    return RigidBodyAnimation.from_root_motion(
        rotations=rotations,
        root_pos=np.zeros((n_frames, 3)),
        offsets=offsets,
        parents=np.asarray(parents, dtype=np.int32),
        names=tuple(names),
        fps=30.0,
    )


def _leafy():
    """root -> mid -> tip, plus a zero-offset End Site under tip."""
    return _rig(
        offsets=[[0, 0, 0], [0, 1, 0], [0, 1, 0], [0, 0, 0]],
        parents=[-1, 0, 1, 2],
        names=("root", "mid", "tip", "end"),
        spin=(1, 2),
    )


def _only_child():
    """root -> dummy(zero offset, sole child) -> a, b."""
    return _rig(
        offsets=[[0, 0, 0], [0, 0, 0], [0, 1, 0], [1, 0, 0]],
        parents=[-1, 0, 1, 1],
        names=("root", "dummy", "a", "b"),
        spin=(1, 2),
    )


def _with_sibling():
    """root -> {dummy(zero offset), other}; dummy is NOT an only child."""
    return _rig(
        offsets=[[0, 0, 0], [0, 0, 0], [0, 1, 0], [1, 0, 0]],
        parents=[-1, 0, 1, 0],
        names=("root", "dummy", "a", "other"),
        spin=(1, 3),
    )


def test_a_zero_offset_leaf_is_dropped():
    reduction = build_reduction(_leafy())
    reduced = apply_reduction(_leafy(), reduction)
    assert reduced.names == ("root", "mid", "tip")


def test_a_zero_offset_only_child_is_collapsed_into_its_parent():
    reduction = build_reduction(_only_child())
    reduced = apply_reduction(_only_child(), reduction)
    assert reduced.names == ("root", "a", "b")
    assert list(reduced.parents) == [-1, 0, 0]


def test_a_zero_offset_joint_with_siblings_is_kept():
    """Folding it into the parent would turn `other` too, so it stays."""
    reduction = build_reduction(_with_sibling())
    reduced = apply_reduction(_with_sibling(), reduction)
    assert "dummy" in reduced.names


@pytest.mark.parametrize(
    "factory", [_leafy, _only_child, _with_sibling],
    ids=["leaf", "only-child", "with-sibling"],
)
def test_reduction_does_not_move_any_surviving_joint(factory):
    source = factory()
    reduction = build_reduction(source)
    reduced = apply_reduction(source, reduction)

    source_positions = source.global_positions()
    reduced_positions = reduced.global_positions()
    for new, old in enumerate(reduction.source_of):
        np.testing.assert_allclose(
            reduced_positions[:, new], source_positions[:, old], atol=1e-9,
            err_msg=f"joint {source.names[old]} moved",
        )


@pytest.mark.parametrize(
    "factory", [_leafy, _only_child, _with_sibling],
    ids=["leaf", "only-child", "with-sibling"],
)
def test_expansion_restores_structure_and_world_positions(factory):
    source = factory()
    reduction = build_reduction(source)
    restored = invert_reduction(apply_reduction(source, reduction), reduction)

    assert restored.names == source.names
    assert list(restored.parents) == list(source.parents)
    np.testing.assert_allclose(restored.offsets, source.offsets, atol=1e-9)
    np.testing.assert_allclose(
        restored.global_positions(), source.global_positions(), atol=1e-9
    )


def test_reduction_repeats_until_no_zero_offset_leaf_remains():
    """Dropping a zero-offset leaf can leave its parent a zero-offset leaf."""
    anim = _rig(
        offsets=[[0, 0, 0], [0, 1, 0], [0, 0, 0], [0, 0, 0]],
        parents=[-1, 0, 1, 2],
        names=("root", "mid", "a", "b"),
        spin=(1,),
    )
    reduced = apply_reduction(anim, build_reduction(anim))
    assert reduced.names == ("root", "mid")


def test_reduction_never_removes_the_root():
    anim = _rig(
        offsets=[[0, 0, 0], [0, 1, 0]], parents=[-1, 0], names=("root", "tip")
    )
    assert apply_reduction(anim, build_reduction(anim)).names == ("root", "tip")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `docker compose run --rm test pytest tests/features/test_reduce.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'poseydon.features.reduce'`

- [ ] **Step 3: Write the implementation**

`src/poseydon/features/reduce.py`:

```python
"""Removing joints that carry nothing learnable, and putting them back.

A zero-length bone is invisible: its child sits exactly on its parent forever,
so the rotation orienting that bone moves nothing. Truebones rigs carry ten or
more per character -- every End Site, plus the odd dummy -- and they inflate the
feature tensor by roughly a fifth with columns that duplicate a neighbour.

Reduction is applied at FEATURE EXTRACTION time and never baked into the corpus.
Skin weights are indexed by the original joint order, so a generated clip has to
come back on the rig the user supplied; the corpus therefore keeps every joint
and this module is the lens the model looks through.

Two cases, and they are not interchangeable:

* a zero-offset LEAF can be deleted outright -- nothing hangs off it and the
  rotation reaching it is unobservable, so nothing is lost;
* a zero-offset INTERNAL joint swings its subtree, so it can only be folded into
  its parent -- and that fold turns the parent, hence every OTHER child of the
  parent too. It is exact only when the joint is its parent's sole child.
  Otherwise the joint is kept, because displacing its siblings to save one token
  is not a trade worth making.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from poseydon.augment.joint_edit import JointEdit
from poseydon.core.animation import RigidBodyAnimation
from poseydon.core.rotations import QUAT_IDENTITY, quat_mul

DROP = "drop"
COLLAPSE = "collapse"


@dataclass(frozen=True)
class RemovalOp:
    """One removal, described by NAME so it survives reindexing."""

    name: str
    parent: str
    index: int
    kind: str
    children: tuple[str, ...] = ()


@dataclass(frozen=True)
class JointReduction:
    """How to get from a full rig to the reduced one, and back."""

    ops: tuple[RemovalOp, ...]
    source_of: tuple[int, ...]
    source_names: tuple[str, ...]

    @property
    def is_identity(self) -> bool:
        return not self.ops

    def joint_edit(self) -> JointEdit:
        """The same map as a :class:`JointEdit`, for transporting statistics."""
        return JointEdit(source_of=self.source_of)


def _children_of(parents: np.ndarray, joint: int) -> list[int]:
    return [j for j in range(len(parents)) if parents[j] == joint]


def _next_removal(anim: RigidBodyAnimation, tolerance: float) -> RemovalOp | None:
    """The first removable joint, or None when the rig is fully reduced."""
    lengths = np.linalg.norm(anim.offsets, axis=-1)
    for joint in range(1, anim.n_joints):
        if lengths[joint] >= tolerance:
            continue
        parent = int(anim.parents[joint])
        children = _children_of(anim.parents, joint)
        if not children:
            return RemovalOp(anim.names[joint], anim.names[parent], joint, DROP)
        if len(_children_of(anim.parents, parent)) == 1:
            return RemovalOp(
                anim.names[joint],
                anim.names[parent],
                joint,
                COLLAPSE,
                tuple(anim.names[c] for c in children),
            )
    return None


def _remove(anim: RigidBodyAnimation, op: RemovalOp) -> RigidBodyAnimation:
    """Apply one removal, preserving every surviving joint's world transform."""
    victim = anim.names.index(op.name)
    parent = int(anim.parents[victim])

    rotations = anim.rotations.copy()
    if op.kind == COLLAPSE:
        # global_rot'[parent] becomes global_rot[victim], which is what the
        # reparented children need. Safe only because `parent` has no other
        # child to be turned by it -- checked when the op was chosen.
        rotations[:, parent] = quat_mul(rotations[:, parent], rotations[:, victim])

    keep = [j for j in range(anim.n_joints) if j != victim]
    old_to_new = {old: new for new, old in enumerate(keep)}

    parents = np.empty(len(keep), dtype=np.int32)
    for new, old in enumerate(keep):
        ancestor = int(anim.parents[old])
        if ancestor < 0:
            parents[new] = -1
        elif ancestor == victim:
            # The victim's children inherit its parent.
            parents[new] = old_to_new[parent]
        else:
            parents[new] = old_to_new[ancestor]

    return RigidBodyAnimation.from_root_motion(
        rotations=rotations[:, keep],
        root_pos=anim.root_pos,
        offsets=anim.offsets[keep],
        parents=parents,
        names=tuple(anim.names[j] for j in keep),
        fps=anim.fps,
    )


def build_reduction(
    anim: RigidBodyAnimation, tolerance: float = 1e-8
) -> JointReduction:
    """Work out which joints to remove, repeating until none is left.

    Repetition matters: dropping a zero-offset leaf can leave its parent a
    zero-offset leaf in turn.
    """
    ops: list[RemovalOp] = []
    current = anim
    while (op := _next_removal(current, tolerance)) is not None:
        ops.append(op)
        current = _remove(current, op)

    index_of = {name: i for i, name in enumerate(anim.names)}
    return JointReduction(
        ops=tuple(ops),
        source_of=tuple(index_of[name] for name in current.names),
        source_names=tuple(anim.names),
    )


def apply_reduction(
    anim: RigidBodyAnimation, reduction: JointReduction
) -> RigidBodyAnimation:
    """Remove the joints the reduction names, in the order it recorded."""
    if tuple(anim.names) != reduction.source_names:
        raise ValueError(
            "this animation's joints differ from the rig the reduction was built "
            f"for ({len(anim.names)} vs {len(reduction.source_names)} joints)"
        )
    current = anim
    for op in reduction.ops:
        current = _remove(current, op)
    return current


def invert_reduction(
    anim: RigidBodyAnimation, reduction: JointReduction
) -> RigidBodyAnimation:
    """Put every removed joint back, in reverse order.

    A collapsed joint returns with identity rotation and the whole composed
    product left on its parent. The two are coincident, so every joint lands
    exactly where it was; which of the pair stores the rotation is unobservable
    in world space, and for generated motion there was never an original split.
    """
    current = anim
    for op in reversed(reduction.ops):
        current = _reinsert(current, op)
    return current


def _reinsert(anim: RigidBodyAnimation, op: RemovalOp) -> RigidBodyAnimation:
    parent = anim.names.index(op.parent)
    n_new = anim.n_joints + 1

    order = list(range(anim.n_joints))
    order.insert(op.index, -1)  # -1 marks the joint being restored

    names = tuple(op.name if old < 0 else anim.names[old] for old in order)
    new_of = {old: new for new, old in enumerate(order) if old >= 0}
    restored = op.index

    rotations = np.empty((anim.n_frames, n_new, 4))
    offsets = np.zeros((n_new, 3))
    parents = np.empty(n_new, dtype=np.int32)

    moved = set(op.children)
    for new, old in enumerate(order):
        if old < 0:
            rotations[:, new] = QUAT_IDENTITY
            offsets[new] = 0.0
            parents[new] = new_of[parent]
            continue
        rotations[:, new] = anim.rotations[:, old]
        offsets[new] = anim.offsets[old]
        ancestor = int(anim.parents[old])
        if anim.names[old] in moved:
            parents[new] = restored
        else:
            parents[new] = -1 if ancestor < 0 else new_of[ancestor]

    return RigidBodyAnimation.from_root_motion(
        rotations=rotations,
        root_pos=anim.root_pos,
        offsets=offsets,
        parents=parents,
        names=names,
        fps=anim.fps,
    )


@dataclass(frozen=True)
class DropDegenerateJoints:
    """Config-facing wrapper, named by ``_target_`` in a dataset config."""

    tolerance: float = 1e-8

    def build(self, anim: RigidBodyAnimation) -> JointReduction:
        return build_reduction(anim, self.tolerance)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `docker compose run --rm test pytest tests/features/test_reduce.py -v`
Expected: PASS, 11 tests.

- [ ] **Step 5: Check it against the real corpus**

Run:

```bash
docker compose run --rm test python -c "
import glob
from poseydon.io.bvh import BVH
from poseydon.features.reduce import apply_reduction, build_reduction
for rig in ['BrownBear','Flamingo','Goat','Crab','Scorpion']:
    p = sorted(glob.glob(f'data/truebones/bvh/{rig}/*.bvh'))[0]
    a = BVH.read(p).to_animation().as_rigid_body(joint_translation='drop')
    r = build_reduction(a)
    print(rig, a.n_joints, '->', apply_reduction(a, r).n_joints)
"
```

Expected: `BrownBear 49 -> 38`, `Flamingo 53 -> 40`, `Goat 40 -> 31`, `Crab 64 -> 54`, `Scorpion 78 -> 64`. Scorpion is one higher than the reference's 63 by design — `Bip01_Neck1` has zero offset but its parent `Hips` has two children, so it is kept.

- [ ] **Step 6: Commit**

```bash
git add src/poseydon/features/reduce.py tests/features
git commit -m "feat(features): joint reduction and expansion

Drops zero-offset leaves and collapses zero-offset only-children into their
parent; keeps a zero-offset joint that has siblings, because folding it would
turn them too. Reduction is a lens at feature time, never baked into the
corpus, so a generated clip returns on the rig the user supplied."
```

---

### Task 7: Persisting the fitted transform

`prepare.npz` is what makes the whole thing invertible tomorrow rather than only inside one process.

**Files:**
- Modify: `src/poseydon/build/prepare.py` — add `RigTransform`
- Test: `tests/build/test_prepare.py`

**Interfaces:**
- Produces: `RigTransform(rig_params: dict[str, dict], clip_params: dict[str, dict[str, dict]], reduction: JointReduction | None)` with `save(path)` / `load(path)` classmethods.

- [ ] **Step 1: Write the failing test**

Append to `tests/build/test_prepare.py`:

```python
from poseydon.build.prepare import RigTransform


def test_rig_transform_round_trips_through_a_file(tmp_path):
    chain = PrepareChain((Shift(), Double()))
    source = _anim()
    rig = chain.fit_rig(source, resolved=None)
    _prepared, params = chain.apply(source, resolved=None, rig_params=rig)

    transform = RigTransform(rig_params=rig, clip_params={"walk": params})
    path = tmp_path / "prepare.npz"
    transform.save(path)
    loaded = RigTransform.load(path)

    np.testing.assert_allclose(
        loaded.rig_params["shift"]["amount"], rig["shift"]["amount"]
    )
    np.testing.assert_allclose(
        loaded.clip_params["walk"]["double"]["factor"], 2.0
    )


def test_rig_transform_reports_an_unknown_clip_clearly(tmp_path):
    transform = RigTransform(rig_params={}, clip_params={})
    transform.save(tmp_path / "prepare.npz")
    with pytest.raises(KeyError, match="jump"):
        RigTransform.load(tmp_path / "prepare.npz").params_for("jump")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `docker compose run --rm test pytest tests/build/test_prepare.py -k rig_transform -v`
Expected: FAIL with `ImportError: cannot import name 'RigTransform'`

- [ ] **Step 3: Write the implementation**

Add to `src/poseydon/build/prepare.py`:

```python
import json


@dataclass(frozen=True)
class RigTransform:
    """Everything the chain fitted for one rig, plus its per-clip parameters.

    Persisted because the transform is otherwise one-way. The facing rotation in
    particular is derived from a clip's own frame 0, so a prepared file already
    facing +Z no longer knows which way it started.

    Stored as a flat npz with ``/``-joined keys -- ``rig/scale/factor``,
    ``clip/walk/face_axis/rotation`` -- so the file stays inspectable with
    ``np.load`` and needs no pickle.
    """

    rig_params: dict[str, dict]
    clip_params: dict[str, dict[str, dict]]

    def params_for(self, clip: str) -> dict[str, dict]:
        """The full parameter set for one clip: rig-scoped plus its own."""
        if clip not in self.clip_params:
            known = ", ".join(sorted(self.clip_params)[:5]) or "(none)"
            raise KeyError(
                f"no recorded parameters for clip `{clip}`; known clips: {known}"
            )
        return {**self.rig_params, **self.clip_params[clip]}

    def save(self, path) -> None:
        arrays: dict[str, np.ndarray] = {}
        for stage, params in self.rig_params.items():
            for key, value in params.items():
                arrays[f"rig/{stage}/{key}"] = np.asarray(value)
        for clip, stages in self.clip_params.items():
            for stage, params in stages.items():
                for key, value in params.items():
                    arrays[f"clip/{clip}/{stage}/{key}"] = np.asarray(value)
        arrays["__clips__"] = np.array(json.dumps(sorted(self.clip_params)))
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(Path(path), **arrays)

    @classmethod
    def load(cls, path) -> RigTransform:
        rig: dict[str, dict] = {}
        clips: dict[str, dict[str, dict]] = {}
        with np.load(Path(path), allow_pickle=True) as data:
            for name in json.loads(str(data["__clips__"])):
                clips[name] = {}
            for key in data.files:
                if key == "__clips__":
                    continue
                head, *rest = key.split("/")
                if head == "rig":
                    stage, field = rest
                    rig.setdefault(stage, {})[field] = data[key]
                else:
                    clip, stage, field = rest
                    clips.setdefault(clip, {}).setdefault(stage, {})[field] = data[key]
        return cls(rig_params=rig, clip_params=clips)
```

Add `from pathlib import Path` to the module imports.

- [ ] **Step 4: Run tests to verify they pass**

Run: `docker compose run --rm test pytest tests/build/test_prepare.py -v`
Expected: PASS, 20 tests.

- [ ] **Step 5: Commit**

```bash
git add src/poseydon/build/prepare.py tests/build
git commit -m "feat(build): persist fitted transform parameters to prepare.npz"
```

---

### Task 8: The round-trip test on real data

The single test the application story rests on. It is written before the scripts change, against the raw corpus, so it validates the chain rather than the pipeline that will use it.

**Files:**
- Create: `tests/build/test_roundtrip.py`

**Interfaces:**
- Consumes: `PrepareChain`, all six stages, `build_reduction`/`apply_reduction`/`invert_reduction`.

- [ ] **Step 1: Write the test**

`tests/build/test_roundtrip.py`:

```python
"""source -> prepare -> reduce -> expand -> unprepare -> source.

If either inverse is wrong this fails, which is the point: an application that
cannot return a user's rig in the representation they supplied is a benchmark,
not a product.
"""

from __future__ import annotations

import numpy as np
import pytest

from poseydon.build.prepare import (
    CentreXZ,
    EnforceRigid,
    FaceAxis,
    PrepareChain,
    PutOnGround,
    RestRelative,
    ScaleToMeanBoneLength,
)
from poseydon.core.skeleton import SkeletonManifest, resolve
from poseydon.features.reduce import apply_reduction, build_reduction, invert_reduction
from poseydon.io.bvh import BVH

from tests.conftest import CORPUS, SAMPLE_RIGS

CHAIN = PrepareChain(
    (
        RestRelative(),
        FaceAxis(axis="+Z"),
        EnforceRigid(joint_translation="drop"),
        CentreXZ(),
        ScaleToMeanBoneLength(),
        PutOnGround(),
    )
)


def _manifest(rig: str) -> SkeletonManifest:
    for candidate in (
        CORPUS / "rigs" / rig / "manifest.yaml",
        CORPUS / "skeletons" / f"{rig}.yaml",
    ):
        if candidate.is_file():
            return SkeletonManifest.load(candidate)
    pytest.skip(f"no manifest for {rig}")


def _rest_path(clips):
    for path in clips:
        if "tpos" in path.name.lower():
            return path
    for path in clips:
        if path.name.lower().lstrip("_").startswith("idle"):
            return path
    pytest.skip("no rest-pose file for this rig")


@pytest.mark.parametrize("rig", SAMPLE_RIGS)
def test_round_trip_returns_the_source_rig(rig, raw_clips):
    clips = raw_clips(rig)
    manifest = _manifest(rig)
    rest = BVH.read(_rest_path(clips)).to_animation()
    resolved = resolve(manifest, rest.names)
    rig_params = CHAIN.fit_rig(rest, resolved)

    source_bvh = BVH.read(next(p for p in clips if p != _rest_path(clips)))
    source = source_bvh.to_animation()
    if tuple(source.names) != tuple(rest.names):
        pytest.skip(f"{rig}: this clip is rigged differently from its own rest pose")

    prepared, params = CHAIN.apply(source, resolve(manifest, source.names), rig_params)

    reduction = build_reduction(prepared)
    reduced = apply_reduction(prepared, reduction)
    expanded = invert_reduction(reduced, reduction)
    restored = CHAIN.invert(expanded, params)

    assert restored.names == source.names
    assert list(restored.parents) == list(source.parents)

    # EnforceRigid discards animated per-joint translation, so compare against
    # the source made rigid -- the structure round-trips, the values do not.
    reference = source.as_rigid_body(joint_translation="drop")
    scale = float(np.linalg.norm(reference.offsets[1:], axis=-1).mean())
    error = np.abs(restored.global_positions() - reference.global_positions()).max()
    assert error < 1e-4 * scale, f"{rig}: worst joint off by {error / scale:.2e} bone lengths"


@pytest.mark.parametrize("rig", SAMPLE_RIGS)
def test_prepared_clips_face_plus_z_and_stand_on_the_ground(rig, raw_clips):
    clips = raw_clips(rig)
    manifest = _manifest(rig)
    rest = BVH.read(_rest_path(clips)).to_animation()
    rig_params = CHAIN.fit_rig(rest, resolve(manifest, rest.names))

    prepared, _params = CHAIN.apply(rest, resolve(manifest, rest.names), rig_params)

    lengths = np.linalg.norm(prepared.offsets[1:], axis=-1)
    assert lengths.mean() == pytest.approx(0.20921428571428569, rel=1e-9)
    assert prepared.global_positions()[..., 1].min() == pytest.approx(0.0, abs=1e-9)
```

- [ ] **Step 2: Run the test**

Run: `docker compose run --rm test pytest tests/build/test_roundtrip.py -v`
Expected: PASS, 8 tests (4 rigs × 2), or SKIP if the corpus is absent.

If the round-trip fails, the two likely causes in order: `RestRelative.invert`'s conjugation order (check `Bp⁻¹ · new · Bp · Lj`, not `Bp · new · Bp⁻¹ · Lj`), and `CHAIN.invert` running stages in forward rather than reverse order.

- [ ] **Step 3: Commit**

```bash
git add tests/build/test_roundtrip.py
git commit -m "test(build): round-trip a real clip from source and back"
```

---

### Task 9: Manifests move to `rigs/<Rig>/manifest.yaml`

**Files:**
- Modify: `src/poseydon/core/skeleton.py` — drop `strip_joint_prefix` and `strip_prefix`
- Modify: `src/poseydon/ingest/pipeline.py` — `available_skeletons` lists directories
- Move: `data/truebones/skeletons/<Rig>.yaml` → `data/truebones/rigs/<Rig>/manifest.yaml`, `_base.yaml` → `rigs/_base.yaml`
- Test: `tests/build/test_layout.py`

**Interfaces:**
- Produces: `available_rigs(root: Path) -> list[str]` in `ingest/pipeline.py`, replacing `available_skeletons`.

- [ ] **Step 1: Write the failing test**

`tests/build/test_layout.py`:

```python
"""The entity-first layout: a rig is a directory."""

from __future__ import annotations

import pytest

from poseydon.core.skeleton import SkeletonManifest
from poseydon.ingest.pipeline import available_rigs

from tests.conftest import CORPUS


def test_available_rigs_lists_directories_not_yaml_stems(tmp_path):
    (tmp_path / "rigs" / "Goat").mkdir(parents=True)
    (tmp_path / "rigs" / "Crab").mkdir(parents=True)
    (tmp_path / "rigs" / "_base.yaml").write_text("fps: 30\n")
    assert available_rigs(tmp_path / "rigs") == ["Goat", "Crab"]


def test_manifests_live_beside_their_rig():
    if not (CORPUS / "rigs").is_dir():
        pytest.skip("corpus not migrated")
    manifest = SkeletonManifest.load(CORPUS / "rigs" / "Goat" / "manifest.yaml")
    assert manifest.name == "Goat"
    assert "quadruped" in manifest.tags


def test_strip_joint_prefix_is_gone():
    """It was declared, documented and read by nothing; joint-name humanizing
    computes the prefix instead of requiring it to be declared."""
    assert not hasattr(SkeletonManifest, "strip_joint_prefix")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `docker compose run --rm test pytest tests/build/test_layout.py -v`
Expected: FAIL with `ImportError: cannot import name 'available_rigs'`

- [ ] **Step 3: Migrate the data**

```bash
mkdir -p data/truebones/rigs
git mv data/truebones/skeletons/_base.yaml data/truebones/rigs/_base.yaml
for f in data/truebones/skeletons/*.yaml; do
  rig=$(basename "$f" .yaml)
  mkdir -p "data/truebones/rigs/$rig"
  git mv "$f" "data/truebones/rigs/$rig/manifest.yaml"
done
rmdir data/truebones/skeletons
sed -i 's|^base: _base.yaml$|base: ../_base.yaml|' data/truebones/rigs/*/manifest.yaml
```

- [ ] **Step 4: Update the code**

In `src/poseydon/core/skeleton.py`: remove `"strip_joint_prefix"` from `_KNOWN_KEYS`, remove the `strip_joint_prefix` field from `SkeletonManifest`, remove the corresponding lines in `_build`, and delete the `strip_prefix` function.

In `src/poseydon/ingest/pipeline.py`, replace `available_skeletons` with:

```python
def available_rigs(rig_root: str | Path) -> list[str]:
    """Rig names under ``rig_root``, longest first.

    A rig is a DIRECTORY containing a ``manifest.yaml``, which is why the old
    ``_``-prefix convention for shared fragments is no longer needed: a bare
    ``_base.yaml`` is not a directory. Longest first matters -- given both
    `Goat` and `GoatKid`, a clip named `GoatKid_walk` must match `GoatKid`.
    """
    names = [
        path.name
        for path in Path(rig_root).iterdir()
        if path.is_dir() and (path / "manifest.yaml").is_file()
    ]
    return sorted(names, key=len, reverse=True)
```

and update `infer_skeleton` and `ingest_corpus` to call `available_rigs` and to load `rig_root / skeleton / "manifest.yaml"`.

- [ ] **Step 5: Delete the empty `datasets` package**

`src/poseydon/datasets/` contains nothing but an empty `__init__.py`; its only module was removed in an earlier commit.

```bash
git rm -r src/poseydon/datasets
docker compose run --rm test python -c "import poseydon.data.dataset; print('ok')"
```

Expected: `ok`.

- [ ] **Step 6: Run tests to verify they pass**

Run: `docker compose run --rm test pytest tests/ -v`
Expected: PASS. `test_available_rigs_lists_directories_not_yaml_stems` asserts `["Goat", "Crab"]` — both are four letters, so ties keep insertion order; if it fails on ordering, sort by `(-len(name), name)` and update the expectation to `["Crab", "Goat"]`.

- [ ] **Step 7: Commit**

```bash
git add -A src/poseydon data/truebones tests
git commit -m "refactor(layout): a rig is a directory; drop dead strip_joint_prefix

Manifests move to rigs/<Rig>/manifest.yaml so every artefact about a
character lives in one place. strip_joint_prefix was declared, parsed,
documented and read by nothing."
```

---

### Task 10: Stage 1 BVH script over the chain

**Files:**
- Modify: `scripts/process_dataset_truebones.py`

**Interfaces:**
- Consumes: `PrepareChain`, all six stages, `RigTransform`, `available_rigs`.
- Produces: `data/truebones/clips/<Rig>/<action>.bvh` and `data/truebones/rigs/<Rig>/prepare.npz`.

- [ ] **Step 1: Rewrite `process_species`**

Replace the body of `process_species` in `scripts/process_dataset_truebones.py`:

```python
CHAIN = PrepareChain(
    (
        RestRelative(),
        FaceAxis(axis=TARGET_AXIS),
        EnforceRigid(joint_translation="drop"),
        CentreXZ(),
        ScaleToMeanBoneLength(),
        PutOnGround(),
    )
)


def process_species(
    manifest: SkeletonManifest, raw_root: Path, out_root: Path
) -> tuple[int, list[str]]:
    """Prepare every raw clip for one rig. Returns (n_written, warnings)."""
    warnings: list[str] = []
    species_dir = raw_root / manifest.name
    if not species_dir.is_dir():
        return 0, [f"{manifest.name}: no raw directory at {species_dir}, skipped"]

    clip_paths = sorted(species_dir.glob("*.bvh"))
    if not clip_paths:
        return 0, [f"{manifest.name}: no .bvh files found in {species_dir}, skipped"]

    rest_path = find_tpose(clip_paths)
    if rest_path is None:
        rest_path = clip_paths[0]
        warnings.append(
            f"{manifest.name}: no T-pose file found; using {rest_path.name} as an "
            "approximate rest pose"
        )

    rest = BVH.read(rest_path).to_animation()
    rig_params = CHAIN.fit_rig(rest, resolve(manifest, rest.names))

    clips_dir = out_root / "clips" / manifest.name
    clips_dir.mkdir(parents=True, exist_ok=True)

    clip_params: dict[str, dict] = {}
    n_written = 0
    for clip_path in clip_paths:
        # One bad clip must not cost a whole species. Some Truebones species
        # ship clips rigged differently from their own T-pose (Elephant's
        # __Take_001 has 43 joints against the T-pose's 51); those are reported
        # and skipped rather than aborting the other twenty.
        try:
            source = BVH.read(clip_path).to_animation()
            if manifest.fps is not None and not math.isclose(
                manifest.fps, source.fps, rel_tol=1e-3
            ):
                raise ValueError(
                    f"manifest requires {manifest.fps} fps but this clip is "
                    f"{source.fps:.4f}"
                )
            resolved = resolve(manifest, source.names)
            prepared, params = CHAIN.apply(source, resolved, rig_params)

            action = strip_skeleton_prefix(action_slug(clip_path.stem), manifest.name)
            BVH.from_animation(prepared).write(clips_dir / f"{action}.bvh")
            clip_params[action] = {
                stage.name: params[stage.name] for stage in CHAIN.stages
                if stage.scope == CLIP
            }
        except Exception as error:  # noqa: BLE001 - collect, don't abort the corpus
            warnings.append(
                f"{manifest.name}/{clip_path.name}: {type(error).__name__}: {error}"
            )
            continue
        n_written += 1

    RigTransform(rig_params=rig_params, clip_params=clip_params).save(
        out_root / "rigs" / manifest.name / "prepare.npz"
    )
    warnings.extend(_sanity_check(manifest, clips_dir, rest_path))
    return n_written, warnings
```

Update the module imports to:

```python
from poseydon.build.prepare import (
    CLIP,
    CentreXZ,
    EnforceRigid,
    FaceAxis,
    PrepareChain,
    PutOnGround,
    RestRelative,
    RigTransform,
    ScaleToMeanBoneLength,
)
from poseydon.ingest.index import action_slug, strip_skeleton_prefix
from poseydon.ingest.pipeline import available_rigs
```

and delete `load_raw_clip`, `resolve_rest_anim` and `rest_skeleton_positions`, which the chain now subsumes.

`_sanity_check` loses its `is_fallback` parameter — the fallback warning is now raised at the call site above, so the check no longer needs to know. Change its signature and its early return:

```python
def _sanity_check(
    manifest: SkeletonManifest, clips_dir: Path, rest_source: Path
) -> list[str]:
    warnings: list[str] = []
    written = sorted(clips_dir.glob("*.bvh"))
    if not written:
        return warnings
    anim = BVH.read(written[0]).to_animation()
    ...
```

keeping the facing and identity assertions in its body unchanged, and dropping the `if is_fallback: return warnings` guard.

Change `main()`'s defaults to `--raw-root data/truebones/source`, `--out-root data/truebones`, and enumerate rigs with `available_rigs(out_root / "rigs")`.

- [ ] **Step 2: Move the raw corpus**

```bash
git mv data/truebones/Truebone_Z-OO data/truebones/source 2>/dev/null || \
  mv data/truebones/Truebone_Z-OO data/truebones/source
```

- [ ] **Step 3: Run it on three rigs**

Run:

```bash
docker compose run --rm test python scripts/process_dataset_truebones.py \
    --rigs Goat Crab Flamingo
```

Expected: `wrote N clips -> data/truebones/clips/<Rig>` for each, no warnings other than any pre-existing rest-pose fallbacks, and `data/truebones/rigs/Goat/prepare.npz` on disk.

- [ ] **Step 4: Verify the prepared output against the chain**

Run:

```bash
docker compose run --rm test python -c "
import numpy as np
from poseydon.build.prepare import RigTransform
from poseydon.io.bvh import BVH
t = RigTransform.load('data/truebones/rigs/Goat/prepare.npz')
print('clips recorded:', len(t.clip_params))
a = BVH.read(sorted(__import__('glob').glob('data/truebones/clips/Goat/*.bvh'))[0]).to_animation()
print('mean bone length', np.linalg.norm(a.offsets[1:], axis=-1).mean())
print('min y', a.global_positions()[...,1].min())
"
```

Expected: a nonzero clip count, mean bone length `0.2092…`, and `min y` at or just above 0 (a clip may lift off the ground the rest pose was fitted to, but must not start below it by more than a bone length).

- [ ] **Step 5: Commit**

```bash
git add -A scripts data/truebones
git commit -m "feat(preproc): stage 1 over PrepareChain, writing the new layout

Adds centring, scaling and grounding -- previously in ingest/align.py and
never run -- and records the fitted transform to rigs/<Rig>/prepare.npz so
the corpus can be turned back into what it came from."
```

---

### Task 11: FBX script and the cross-source agreement test

**Files:**
- Modify: `scripts/process_dataset_truebones_fbx.py`
- Create: `tests/build/test_bvh_fbx_agreement.py`

**Interfaces:**
- Produces: `data/truebones/clips/<Rig>/<action>.fbx`, `data/truebones/rigs/<Rig>/mesh.npz`.

- [ ] **Step 1: Write the failing test**

`tests/build/test_bvh_fbx_agreement.py`:

```python
"""The two prepared corpora must describe the same skeleton.

Verified once by hand in a commit message; a property this load-bearing needs a
test. Skips without the FBX artefacts, because the test image has no Blender.
"""

from __future__ import annotations

import numpy as np
import pytest

from poseydon.io.bvh import BVH

from tests.conftest import CORPUS, SAMPLE_RIGS


def _pair(rig: str):
    clips = CORPUS / "clips" / rig
    bvhs = sorted(clips.glob("*.bvh"))
    if not bvhs:
        pytest.skip(f"{rig}: no prepared BVH")
    mesh = CORPUS / "rigs" / rig / "mesh.npz"
    if not mesh.is_file():
        pytest.skip(f"{rig}: no mesh.npz -- run the fbx container")
    return bvhs[0], mesh


@pytest.mark.parametrize("rig", SAMPLE_RIGS)
def test_bvh_and_fbx_agree_on_the_skeleton(rig):
    bvh_path, mesh_path = _pair(rig)
    anim = BVH.read(bvh_path).to_animation()
    with np.load(mesh_path, allow_pickle=True) as mesh:
        names = tuple(str(n) for n in mesh["joint_names"])
        parents = mesh["joint_parents"]
        offsets = mesh["joint_offsets"]

    assert anim.names == names
    assert list(anim.parents) == list(parents)

    scale = float(np.linalg.norm(anim.offsets[1:], axis=-1).mean())
    error = np.abs(anim.offsets - offsets).max()
    assert error < 1e-3 * scale, f"{rig}: rest offsets differ by {error / scale:.2e}"
```

- [ ] **Step 2: Run test to verify it skips**

Run: `docker compose run --rm test pytest tests/build/test_bvh_fbx_agreement.py -v`
Expected: 4 SKIPPED with "no mesh.npz".

- [ ] **Step 3: Update the FBX script**

In `scripts/process_dataset_truebones_fbx.py`:

- change `DEFAULT_RAW_ROOT` to `Path("data/truebones/source")` and replace `DEFAULT_FBX_DIR` / `DEFAULT_MESH_DIR` with a single `DEFAULT_OUT_ROOT = Path("data/truebones")`;
- write each prepared clip to `out_root / "clips" / species / f"{action}.fbx"`, deriving `action` with `strip_skeleton_prefix(action_slug(path.stem), species)` exactly as the BVH script does, so the two corpora share basenames;
- replace `scene.write_bind_pose_obj(...)` with a call that writes `out_root / "rigs" / species / "mesh.npz"` containing `vertices`, `faces`, `skin_weights`, `skin_joints`, `joint_names`, `joint_parents` and `joint_offsets`;
- read manifests from `out_root / "rigs" / species / "manifest.yaml"`.

The mesh writer, added to `src/poseydon/io/fbx.py`:

```python
def write_mesh_npz(self, path) -> None:
    """Mesh, skinning and rest skeleton in one file.

    One load gives a consumer everything needed to drive the skin: vertices and
    faces, the per-vertex weight matrix and the joint ordering it is indexed by,
    and the rest skeleton those weights were bound against. Splitting them
    across an OBJ and something else guarantees they drift apart.
    """
    vertices, faces, weights, joints = self.bind_pose_arrays()
    np.savez_compressed(
        Path(path),
        vertices=vertices,
        faces=faces,
        skin_weights=weights,
        skin_joints=np.array(joints, dtype=object),
        joint_names=np.array(self.joint_names, dtype=object),
        joint_parents=np.asarray(self.joint_parents, dtype=np.int32),
        joint_offsets=np.asarray(self.joint_offsets, dtype=np.float64),
    )
```

`bind_pose_arrays` does not exist yet. Extract it from `write_bind_pose_obj` in `src/poseydon/io/fbx.py` — everything that gathers geometry, up to but not including the point where the OBJ text is formatted — and have it return

```python
(vertices, faces, weights, joints)
#  (V, 3) float64      bind-pose vertex positions, +Z faced
#  (F, 3) int32        triangle indices
#  (V, J) float64      per-vertex skin weight per joint, rows summing to 1
#  tuple[str, ...]     the J joint names those columns are indexed by, in
#                      the SAME order as self.joint_names
```

Then `write_bind_pose_obj` calls it and formats, `write_mesh_npz` calls it and saves, and the OBJ stays available behind a `--write-obj` flag. If the FBX reader does not currently expose `joint_offsets`, derive it as the bind-pose parent-relative delta: `offsets[j] = bind_pos[j] - bind_pos[parents[j]]`, with `offsets[0] = bind_pos[0]`.

- [ ] **Step 4: Run the FBX pipeline on three rigs**

Run:

```bash
docker compose run --rm fbx blender --background \
    --python scripts/process_dataset_truebones_fbx.py -- --rigs Goat Crab Flamingo
```

Expected: `data/truebones/rigs/Goat/mesh.npz` exists and `clips/Goat/*.fbx` share basenames with `clips/Goat/*.bvh`.

- [ ] **Step 5: Run the agreement test**

Run: `docker compose run --rm test pytest tests/build/test_bvh_fbx_agreement.py -v`
Expected: PASS for Goat, Crab and Flamingo; SKIP for Scorpion until it is processed.

If offsets disagree by more than the tolerance, the likely cause is that the FBX path applies the facing rotation to the armature object rather than to the bones, leaving rest offsets in the source orientation.

- [ ] **Step 6: Run the whole suite and commit**

Run: `docker compose run --rm test pytest tests/ -v`
Expected: PASS, no failures.

```bash
git add -A scripts src/poseydon/io/fbx.py tests
git commit -m "feat(preproc): FBX stage 1 to the new layout, mesh.npz, agreement test

Mesh, skinning weights and the rest skeleton land in one file so they cannot
drift apart, and the BVH/FBX skeleton agreement becomes a test rather than a
claim in a commit message."
```

---

## Phase exit criteria

- `docker compose run --rm test pytest tests/ -v` passes with no failures.
- `tests/build/test_roundtrip.py` passes on all four sample rigs.
- `data/truebones/rigs/<Rig>/prepare.npz` exists for every processed rig and `RigTransform.load` returns parameters for every clip written beside it.
- `data/truebones/clips/<Rig>/` holds prepared BVH whose mean bone length is `0.20921428571428569`.
- Phase 2 (`build_features.py`, the artefacts, names, the normalization policy) can begin: it consumes `clips/<Rig>/*.bvh`, `rigs/<Rig>/prepare.npz` and `rigs/<Rig>/mesh.npz`, all of which this phase produces.

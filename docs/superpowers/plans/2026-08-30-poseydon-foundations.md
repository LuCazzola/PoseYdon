# PoseYdon Foundations Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build PoseYdon's rotation maths, forward kinematics, canonical `Anim` container, and a fast BVH reader/writer, verified against seven real Truebones files.

**Architecture:** Pure-numpy foundation layer with no torch, no config framework and no
ML dependencies. `core/` holds maths and containers; `io/` holds file formats. Rotation
maths delegates to SciPy wherever SciPy already has it, and only implements what it
lacks (6D rotation representation, quaternion-between-vectors). The BVH parser reads the
MOTION block in one vectorized pass rather than line by line.

**Tech Stack:** Python >=3.11, numpy, scipy, pytest, ruff, uv — all executed inside a Docker container defined by this plan. The host needs only Docker.

**Spec:** `docs/superpowers/specs/2026-08-30-poseydon-design.md` (sections 4, 5, 12
"Dependencies", and 14)

## Global Constraints

- Python `>=3.11`. Dependencies for this plan are exactly `numpy>=1.26` and `scipy>=1.11`. Do NOT add torch, lightning, hydra or roma — later plans introduce those.
- **Quaternion layout is scalar-last `(x, y, z, w)`**, matching SciPy and roma. Never scalar-first. This is a project-wide invariant; every function that returns a quaternion returns this layout.
- **All arrays are `float64`** in this layer. Integer arrays (`parents`) are `int32`.
- **End Sites are joints.** A BVH with `n` `ROOT`/`JOINT` entries and `m` `End Site` entries has `n + m` joints. End Sites have an offset and a name but zero channels and identity local rotation. Any code that assumes joints equal channel-groups is wrong.
- **`external/neural_motion_blending/` is read-only.** Read it for reference; never edit, move or reformat anything under it.
- Rotation matrices are row-vector-agnostic but stored so that `quat_to_matrix(q) @ v` rotates column vector `v`. The 6D representation uses the **first two rows** of the matrix, matching PyTorch3D, because later parity work compares against the reference's PyTorch3D-derived code.
- **All Python runs inside Docker.** The host has no Python packaging tooling (no `uv`, no `pip`, no `ensurepip`, no numpy/scipy) and nothing may be installed on it. Every verification command is `docker compose run --rm test ...`. Never run `pytest`, `ruff` or `uv` directly on the host — it will fail.
- Every task ends with a green `pytest` run and a commit.

---

### Task 1: Project scaffolding and test fixtures

**Files:**
- Create: `pyproject.toml`
- Create: `src/poseydon/__init__.py`
- Create: `src/poseydon/core/__init__.py`
- Create: `src/poseydon/io/__init__.py`
- Create: `tests/conftest.py`
- Create: `Dockerfile`
- Create: `docker-compose.yml`
- Create: `.dockerignore`
- Test: `tests/test_fixtures.py`

**Interfaces:**
- Consumes: nothing (first task)
- Produces: pytest fixtures `truebones_dir` (session-scoped `Path`) and `bvh_fixture`
  (function-scoped `Path`, parameterized over all seven stems); module constant
  `tests.conftest.FIXTURE_STEMS: list[str]`; helper `tests.conftest.fixture_dir() -> Path | None`

- [ ] **Step 1: Create the package layout and `pyproject.toml`**

```toml
[project]
name = "poseydon"
version = "0.1.0"
description = "Modular deep learning on motion"
requires-python = ">=3.11"
dependencies = [
    "numpy>=1.26",
    "scipy>=1.11",
]

[project.optional-dependencies]
dev = ["pytest>=8", "ruff>=0.6"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/poseydon"]

[tool.pytest.ini_options]
testpaths = ["tests"]
# Makes `import poseydon` work from the src layout without installing the
# project, which keeps the Docker image's dependency layer independent of source.
pythonpath = ["src"]

[tool.ruff]
line-length = 100
src = ["src", "tests"]
```

Create four empty files: `src/poseydon/__init__.py`, `src/poseydon/core/__init__.py`,
`src/poseydon/io/__init__.py`, and `tests/__init__.py` is NOT needed (pytest uses rootdir).

- [ ] **Step 2: Write the Docker environment**

The host has no Python tooling and must stay that way. These three files are the
only way anything in this plan gets executed.

`Dockerfile`:

```dockerfile
# Test and development environment for PoseYdon.
# The host needs only Docker: no Python, pip or uv installation required.
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app

# UV_PROJECT_ENVIRONMENT puts the virtualenv OUTSIDE /app so the bind mount in
# docker-compose.yml cannot shadow it. UV_LINK_MODE=copy avoids hardlink warnings
# when the uv cache and the venv live on different layers.
ENV UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH"

# Only pyproject.toml is copied, so editing source never invalidates this layer.
# --no-install-project means the package itself is not built here; pytest's
# `pythonpath = ["src"]` setting makes it importable from the bind mount instead.
COPY pyproject.toml ./
RUN uv sync --extra dev --no-install-project

CMD ["pytest", "-v"]
```

`docker-compose.yml`:

```yaml
services:
  test:
    build: .
    working_dir: /app
    volumes:
      # Source, tests and the read-only reference checkout are bind-mounted, so
      # edits take effect with no rebuild. Only pyproject.toml changes require one.
      - .:/app
    command: pytest -v
```

`.dockerignore` — keeps the build context small; `external/` alone is tens of MB
with its own `.git`, and it is bind-mounted at runtime anyway:

```
.git
.worktrees
external
__pycache__
*.pyc
.pytest_cache
.ruff_cache
```

- [ ] **Step 3: Build the image and verify the toolchain**

Run: `docker compose build test`
Expected: build succeeds.

Run: `docker compose run --rm test python -c "import numpy, scipy; print(numpy.__version__, scipy.__version__)"`
Expected: two version numbers printed, numpy >= 1.26 and scipy >= 1.11.

- [ ] **Step 4: Write `tests/conftest.py`**

The seven fixtures may live in either of two places. `tests/data/truebones/` is preferred
(committed to this repo); `external/neural_motion_blending/assets/truebones/` is the
read-only reference checkout. If neither has a complete set, tests skip with a message
naming both paths. Do not fail — a fresh clone without the reference must still run the
rest of the suite.

```python
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

_CANDIDATE_DIRS = (
    REPO_ROOT / "tests" / "data" / "truebones",
    REPO_ROOT / "external" / "neural_motion_blending" / "assets" / "truebones",
)

FIXTURE_STEMS = [
    "BrownBear___RiseSwat_132",
    "Coyote___Attack3_224",
    "Crab___Attack3_234",
    "Flamingo_Flamingo_OneLEgBEnt_353",
    "Goat___HeadButt_395",
    "Scorpion___SlowForward_839",
    "Skunk___Spray_891",
]

# Ground truth measured from the files themselves. Joint counts INCLUDE End Sites.
FIXTURE_FACTS = {
    "BrownBear___RiseSwat_132":          {"joints": 38, "frames": 160, "channels": 87},
    "Coyote___Attack3_224":              {"joints": 40, "frames": 91,  "channels": 96},
    "Crab___Attack3_234":                {"joints": 54, "frames": 90,  "channels": 135},
    "Flamingo_Flamingo_OneLEgBEnt_353":  {"joints": 40, "frames": 201, "channels": 87},
    "Goat___HeadButt_395":               {"joints": 31, "frames": 79,  "channels": 72},
    "Scorpion___SlowForward_839":        {"joints": 63, "frames": 80,  "channels": 153},
    "Skunk___Spray_891":                 {"joints": 35, "frames": 121, "channels": 75},
}


def fixture_dir() -> Path | None:
    """First directory holding a complete set of the seven fixture BVHs, else None."""
    for directory in _CANDIDATE_DIRS:
        if directory.is_dir() and all(
            (directory / f"{stem}.bvh").is_file() for stem in FIXTURE_STEMS
        ):
            return directory
    return None


@pytest.fixture(scope="session")
def truebones_dir() -> Path:
    directory = fixture_dir()
    if directory is None:
        pytest.skip(
            "Truebones BVH fixtures not found. Expected a complete set in either "
            f"{_CANDIDATE_DIRS[0]} or {_CANDIDATE_DIRS[1]}."
        )
    return directory


@pytest.fixture(params=FIXTURE_STEMS)
def bvh_fixture(request, truebones_dir) -> Path:
    """Parameterized: every test using this runs once per fixture file."""
    return truebones_dir / f"{request.param}.bvh"


@pytest.fixture(scope="session")
def fixture_facts() -> dict[str, dict[str, int]]:
    """Ground-truth counts, exposed as a fixture.

    Tests in subdirectories must NOT do ``from conftest import ...`` -- whether
    ``tests/`` lands on ``sys.path`` depends on pytest's import mode. Fixtures
    are resolved up the directory tree and always work.
    """
    return FIXTURE_FACTS


@pytest.fixture(scope="session")
def fixture_stems() -> list[str]:
    return list(FIXTURE_STEMS)
```

- [ ] **Step 5: Write the failing test**

`tests/test_fixtures.py`:

```python
def test_every_stem_has_recorded_facts(fixture_facts, fixture_stems):
    assert sorted(fixture_facts) == sorted(fixture_stems)


def test_fixture_file_exists(bvh_fixture):
    assert bvh_fixture.is_file()
    assert bvh_fixture.read_text().startswith("HIERARCHY")
```

- [ ] **Step 6: Run the tests**

Run: `docker compose run --rm test pytest tests/test_fixtures.py -v`
Expected: `test_every_stem_has_recorded_facts` PASSES; the seven parameterized
`test_fixture_file_exists` cases PASS if the reference checkout is present, or all SKIP
with the two-path message if not. Both outcomes are correct.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml Dockerfile docker-compose.yml .dockerignore src tests
git commit -m "chore: scaffold poseydon package, Docker test env and fixtures"
```

---

### Task 2: Rotation representations

**Files:**
- Create: `src/poseydon/core/rotations.py`
- Test: `tests/core/test_rotations.py`

**Interfaces:**
- Consumes: nothing from earlier tasks
- Produces:
  - `euler_to_quat(angles_deg: np.ndarray, order: str) -> np.ndarray` — `(..., 3) -> (..., 4)`
  - `quat_to_euler(q: np.ndarray, order: str) -> np.ndarray` — `(..., 4) -> (..., 3)` degrees
  - `quat_to_matrix(q: np.ndarray) -> np.ndarray` — `(..., 4) -> (..., 3, 3)`
  - `matrix_to_quat(m: np.ndarray) -> np.ndarray` — `(..., 3, 3) -> (..., 4)`
  - `matrix_to_rot6d(m: np.ndarray) -> np.ndarray` — `(..., 3, 3) -> (..., 6)`
  - `rot6d_to_matrix(d6: np.ndarray) -> np.ndarray` — `(..., 6) -> (..., 3, 3)`
  - `quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray`
  - `quat_apply(q: np.ndarray, v: np.ndarray) -> np.ndarray`
  - `quat_between(a: np.ndarray, b: np.ndarray) -> np.ndarray`
  - `QUAT_IDENTITY: np.ndarray` — `array([0.0, 0.0, 0.0, 1.0])`

- [ ] **Step 1: Write the failing tests**

`tests/core/test_rotations.py`. Create `tests/core/` (no `__init__.py` needed).

```python
import numpy as np

from poseydon.core.rotations import (
    QUAT_IDENTITY,
    euler_to_quat,
    matrix_to_quat,
    matrix_to_rot6d,
    quat_apply,
    quat_between,
    quat_mul,
    quat_to_matrix,
    rot6d_to_matrix,
)


def random_quats(n, seed=0):
    rng = np.random.default_rng(seed)
    q = rng.normal(size=(n, 4))
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def test_quat_layout_is_scalar_last():
    # Identity has w == 1 in the LAST slot.
    assert QUAT_IDENTITY.tolist() == [0.0, 0.0, 0.0, 1.0]


def test_euler_zyx_90_about_z_maps_x_axis_to_y_axis():
    # BVH channel order "Zrotation Yrotation Xrotation" -> order string "ZYX",
    # angles given in that same order.
    q = euler_to_quat(np.array([90.0, 0.0, 0.0]), "ZYX")
    got = quat_apply(q, np.array([1.0, 0.0, 0.0]))
    np.testing.assert_allclose(got, [0.0, 1.0, 0.0], atol=1e-12)


def test_quat_matrix_round_trip():
    q = random_quats(64)
    back = matrix_to_quat(quat_to_matrix(q))
    # q and -q are the same rotation; compare via matrices instead.
    np.testing.assert_allclose(quat_to_matrix(back), quat_to_matrix(q), atol=1e-12)


def test_rot6d_round_trip():
    m = quat_to_matrix(random_quats(64, seed=1))
    np.testing.assert_allclose(rot6d_to_matrix(matrix_to_rot6d(m)), m, atol=1e-12)


def test_rot6d_uses_first_two_rows():
    m = quat_to_matrix(random_quats(4, seed=2))
    np.testing.assert_allclose(matrix_to_rot6d(m), m[..., :2, :].reshape(-1, 6), atol=0)


def test_rot6d_reorthonormalizes_a_perturbed_input():
    m = quat_to_matrix(random_quats(8, seed=3))
    d6 = matrix_to_rot6d(m) + 0.01
    out = rot6d_to_matrix(d6)
    eye = np.einsum("...ij,...kj->...ik", out, out)
    np.testing.assert_allclose(eye, np.broadcast_to(np.eye(3), eye.shape), atol=1e-12)


def test_quat_mul_matches_matrix_product():
    a, b = random_quats(32, seed=4), random_quats(32, seed=5)
    np.testing.assert_allclose(
        quat_to_matrix(quat_mul(a, b)),
        quat_to_matrix(a) @ quat_to_matrix(b),
        atol=1e-12,
    )


def test_quat_between_rotates_a_onto_b():
    rng = np.random.default_rng(6)
    a = rng.normal(size=(32, 3))
    b = rng.normal(size=(32, 3))
    a /= np.linalg.norm(a, axis=-1, keepdims=True)
    b /= np.linalg.norm(b, axis=-1, keepdims=True)
    np.testing.assert_allclose(quat_apply(quat_between(a, b), a), b, atol=1e-10)


def test_quat_between_handles_antiparallel_vectors():
    a = np.array([[1.0, 0.0, 0.0]])
    b = -a
    np.testing.assert_allclose(quat_apply(quat_between(a, b), a), b, atol=1e-10)


def test_quat_between_handles_identical_vectors():
    a = np.array([[0.0, 1.0, 0.0]])
    np.testing.assert_allclose(quat_between(a, a), [QUAT_IDENTITY], atol=1e-12)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `docker compose run --rm test pytest tests/core/test_rotations.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'poseydon.core.rotations'`

- [ ] **Step 3: Write the implementation**

`src/poseydon/core/rotations.py`:

```python
"""Rotation representations.

Project-wide invariant: quaternions are scalar-last ``(x, y, z, w)``, matching
SciPy and roma. The 6D representation is the first two ROWS of the rotation
matrix, matching PyTorch3D.

SciPy already implements Euler/quaternion/matrix conversions correctly and
quickly; this module wraps those and adds only what SciPy lacks.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

QUAT_IDENTITY = np.array([0.0, 0.0, 0.0, 1.0])


def _as_rotation(q: np.ndarray) -> Rotation:
    return Rotation.from_quat(np.asarray(q, dtype=np.float64).reshape(-1, 4))


def euler_to_quat(angles_deg: np.ndarray, order: str) -> np.ndarray:
    """Intrinsic Euler angles (degrees) to quaternion.

    ``order`` is uppercase, e.g. ``"ZYX"``, and ``angles_deg[..., i]`` is the
    angle for ``order[i]``. Uppercase is SciPy's spelling for intrinsic
    rotations, which is what BVH means by its channel ordering.
    """
    angles = np.asarray(angles_deg, dtype=np.float64)
    flat = Rotation.from_euler(order.upper(), angles.reshape(-1, 3), degrees=True)
    return flat.as_quat().reshape(*angles.shape[:-1], 4)


def quat_to_euler(q: np.ndarray, order: str) -> np.ndarray:
    """Quaternion to intrinsic Euler angles in degrees, in ``order``."""
    q = np.asarray(q, dtype=np.float64)
    angles = _as_rotation(q).as_euler(order.upper(), degrees=True)
    return angles.reshape(*q.shape[:-1], 3)


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    return _as_rotation(q).as_matrix().reshape(*q.shape[:-1], 3, 3)


def matrix_to_quat(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=np.float64)
    flat = Rotation.from_matrix(m.reshape(-1, 3, 3))
    return flat.as_quat().reshape(*m.shape[:-2], 4)


def matrix_to_rot6d(m: np.ndarray) -> np.ndarray:
    """First two rows of the rotation matrix, flattened."""
    m = np.asarray(m, dtype=np.float64)
    return m[..., :2, :].reshape(*m.shape[:-2], 6).copy()


def rot6d_to_matrix(d6: np.ndarray) -> np.ndarray:
    """Gram-Schmidt the two 3-vectors back into an orthonormal matrix."""
    d6 = np.asarray(d6, dtype=np.float64)
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)
    b2 = a2 - (b1 * a2).sum(axis=-1, keepdims=True) * b1
    b2 = b2 / np.linalg.norm(b2, axis=-1, keepdims=True)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-2)


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Composition: applying the result equals applying ``b`` then ``a``."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    ax, ay, az, aw = np.moveaxis(a, -1, 0)
    bx, by, bz, bw = np.moveaxis(b, -1, 0)
    return np.stack(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        axis=-1,
    )


def quat_apply(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector(s) ``v`` by quaternion(s) ``q``, broadcasting."""
    q = np.asarray(q, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    q, v = np.broadcast_arrays(q, np.concatenate([v, np.zeros_like(v[..., :1])], axis=-1))
    xyz, w = q[..., :3], q[..., 3:]
    t = 2.0 * np.cross(xyz, v[..., :3])
    return v[..., :3] + w * t + np.cross(xyz, t)


def quat_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Shortest-arc rotation taking unit vector ``a`` to unit vector ``b``."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b = b / np.linalg.norm(b, axis=-1, keepdims=True)

    axis = np.cross(a, b)
    w = 1.0 + (a * b).sum(axis=-1, keepdims=True)
    q = np.concatenate([axis, w], axis=-1)

    # Antiparallel: w collapses to 0 and the axis is degenerate. Any perpendicular
    # axis is a valid 180-degree rotation; pick one deterministically.
    antiparallel = (w[..., 0] < 1e-12)
    if np.any(antiparallel):
        fallback = np.cross(a, np.array([1.0, 0.0, 0.0]))
        degenerate = np.linalg.norm(fallback, axis=-1) < 1e-8
        alt = np.cross(a, np.array([0.0, 1.0, 0.0]))
        fallback = np.where(degenerate[..., None], alt, fallback)
        q = np.where(
            antiparallel[..., None],
            np.concatenate([fallback, np.zeros_like(w)], axis=-1),
            q,
        )

    return q / np.linalg.norm(q, axis=-1, keepdims=True)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `docker compose run --rm test pytest tests/core/test_rotations.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/poseydon/core/rotations.py tests/core/test_rotations.py
git commit -m "feat(core): rotation conversions with scalar-last quaternions"
```

---

### Task 3: Forward kinematics

**Files:**
- Create: `src/poseydon/core/kinematics.py`
- Test: `tests/core/test_kinematics.py`

**Interfaces:**
- Consumes: `quat_apply`, `quat_mul` from `poseydon.core.rotations` (Task 2)
- Produces:
  - `forward_kinematics(rotations, root_pos, offsets, parents) -> tuple[np.ndarray, np.ndarray]`
    returning `(positions (F, J, 3), global_rotations (F, J, 4))`
  - `check_topological_order(parents: np.ndarray) -> None` — raises `ValueError`

- [ ] **Step 1: Write the failing tests**

`tests/core/test_kinematics.py`:

```python
import numpy as np
import pytest

from poseydon.core.kinematics import check_topological_order, forward_kinematics
from poseydon.core.rotations import QUAT_IDENTITY, euler_to_quat


def test_two_joint_chain_child_follows_root_rotation():
    # Child sits 1 unit up the Y axis from the root. Rotating the root 90 degrees
    # about Z must swing the child onto the negative X axis.
    parents = np.array([-1, 0], dtype=np.int32)
    offsets = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    root_pos = np.zeros((1, 3))
    rotations = np.stack(
        [euler_to_quat(np.array([90.0, 0.0, 0.0]), "ZYX"), QUAT_IDENTITY]
    )[None]

    positions, _ = forward_kinematics(rotations, root_pos, offsets, parents)

    np.testing.assert_allclose(positions[0, 0], [0.0, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(positions[0, 1], [-1.0, 0.0, 0.0], atol=1e-12)


def test_identity_rotations_reproduce_offset_sums():
    parents = np.array([-1, 0, 1], dtype=np.int32)
    offsets = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    root_pos = np.array([[5.0, 0.0, 0.0]])
    rotations = np.broadcast_to(QUAT_IDENTITY, (1, 3, 4)).copy()

    positions, _ = forward_kinematics(rotations, root_pos, offsets, parents)

    np.testing.assert_allclose(positions[0, 2], [6.0, 2.0, 0.0], atol=1e-12)


def test_root_translation_moves_whole_skeleton():
    parents = np.array([-1, 0], dtype=np.int32)
    offsets = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    rotations = np.broadcast_to(QUAT_IDENTITY, (2, 2, 4)).copy()
    root_pos = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 7.0]])

    positions, _ = forward_kinematics(rotations, root_pos, offsets, parents)

    np.testing.assert_allclose(positions[1, 1], [3.0, 1.0, 7.0], atol=1e-12)


def test_rejects_non_topological_parents():
    with pytest.raises(ValueError, match="topological"):
        check_topological_order(np.array([-1, 2, 0], dtype=np.int32))


def test_rejects_missing_root():
    with pytest.raises(ValueError, match="root"):
        check_topological_order(np.array([1, 0], dtype=np.int32))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `docker compose run --rm test pytest tests/core/test_kinematics.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'poseydon.core.kinematics'`

- [ ] **Step 3: Write the implementation**

`src/poseydon/core/kinematics.py`:

```python
"""Forward kinematics over a joint hierarchy."""

from __future__ import annotations

import numpy as np

from poseydon.core.rotations import quat_apply, quat_mul


def check_topological_order(parents: np.ndarray) -> None:
    """Every joint must appear after its parent, and joint 0 must be the root.

    BVH files are written depth-first, so this always holds for parsed files.
    Asserting it lets forward kinematics run as a single forward sweep.
    """
    parents = np.asarray(parents)
    if parents.ndim != 1 or parents.size == 0:
        raise ValueError(f"parents must be a non-empty 1-D array, got shape {parents.shape}")
    if parents[0] != -1:
        raise ValueError(f"joint 0 must be the root with parent -1, got {parents[0]}")
    if np.any(parents[1:] < 0):
        raise ValueError("only joint 0 may have parent -1; found another negative parent")
    bad = np.nonzero(parents[1:] >= np.arange(1, parents.size))[0]
    if bad.size:
        joint = int(bad[0]) + 1
        raise ValueError(
            f"parents must be in topological order: joint {joint} has parent "
            f"{parents[joint]}, which does not precede it"
        )


def forward_kinematics(
    rotations: np.ndarray,
    root_pos: np.ndarray,
    offsets: np.ndarray,
    parents: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Local rotations to global joint positions and rotations.

    Args:
        rotations: ``(F, J, 4)`` local rotations, scalar-last quaternions.
        root_pos:  ``(F, 3)`` global translation of joint 0.
        offsets:   ``(J, 3)`` rest-pose offset of each joint from its parent.
        parents:   ``(J,)`` parent index per joint, ``-1`` for the root.

    Returns:
        ``(positions (F, J, 3), global_rotations (F, J, 4))``.
    """
    rotations = np.asarray(rotations, dtype=np.float64)
    root_pos = np.asarray(root_pos, dtype=np.float64)
    offsets = np.asarray(offsets, dtype=np.float64)
    parents = np.asarray(parents, dtype=np.int32)

    check_topological_order(parents)
    n_frames, n_joints = rotations.shape[:2]
    if offsets.shape != (n_joints, 3):
        raise ValueError(f"offsets must be ({n_joints}, 3), got {offsets.shape}")
    if root_pos.shape != (n_frames, 3):
        raise ValueError(f"root_pos must be ({n_frames}, 3), got {root_pos.shape}")

    positions = np.empty((n_frames, n_joints, 3), dtype=np.float64)
    global_rot = np.empty((n_frames, n_joints, 4), dtype=np.float64)

    positions[:, 0] = root_pos
    global_rot[:, 0] = rotations[:, 0]

    for joint in range(1, n_joints):
        parent = parents[joint]
        positions[:, joint] = positions[:, parent] + quat_apply(
            global_rot[:, parent], offsets[joint]
        )
        global_rot[:, joint] = quat_mul(global_rot[:, parent], rotations[:, joint])

    return positions, global_rot
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `docker compose run --rm test pytest tests/core/test_kinematics.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/poseydon/core/kinematics.py tests/core/test_kinematics.py
git commit -m "feat(core): forward kinematics over joint hierarchies"
```

---

### Task 4: The `Anim` container

**Files:**
- Create: `src/poseydon/core/anim.py`
- Test: `tests/core/test_anim.py`

**Interfaces:**
- Consumes: `forward_kinematics`, `check_topological_order` (Task 3)
- Produces:
  - `Anim` frozen dataclass with fields `rotations (F, J, 4)`, `root_pos (F, 3)`,
    `offsets (J, 3)`, `parents (J,) int32`, `names tuple[str, ...]`, `fps float`
  - properties `n_frames: int`, `n_joints: int`
  - `Anim.global_positions() -> np.ndarray` — `(F, J, 3)`
  - `Anim.save(path: Path) -> None` and `Anim.load(path: Path) -> Anim` (classmethod), `.npz`
  - `Anim.slice(start: int, stop: int) -> Anim`

- [ ] **Step 1: Write the failing tests**

`tests/core/test_anim.py`:

```python
import numpy as np
import pytest

from poseydon.core.anim import Anim
from poseydon.core.rotations import QUAT_IDENTITY


def make_anim(n_frames=4, n_joints=3) -> Anim:
    rng = np.random.default_rng(0)
    q = rng.normal(size=(n_frames, n_joints, 4))
    q /= np.linalg.norm(q, axis=-1, keepdims=True)
    return Anim(
        rotations=q,
        root_pos=rng.normal(size=(n_frames, 3)),
        offsets=rng.normal(size=(n_joints, 3)),
        parents=np.array([-1] + list(range(n_joints - 1)), dtype=np.int32),
        names=tuple(f"joint_{i}" for i in range(n_joints)),
        fps=24.0,
    )


def test_shape_properties():
    anim = make_anim(n_frames=7, n_joints=5)
    assert anim.n_frames == 7
    assert anim.n_joints == 5


def test_rejects_name_count_mismatch():
    with pytest.raises(ValueError, match="names"):
        Anim(
            rotations=np.broadcast_to(QUAT_IDENTITY, (2, 3, 4)).copy(),
            root_pos=np.zeros((2, 3)),
            offsets=np.zeros((3, 3)),
            parents=np.array([-1, 0, 1], dtype=np.int32),
            names=("only", "two"),
            fps=24.0,
        )


def test_rejects_frame_count_mismatch():
    with pytest.raises(ValueError, match="root_pos"):
        Anim(
            rotations=np.broadcast_to(QUAT_IDENTITY, (2, 3, 4)).copy(),
            root_pos=np.zeros((5, 3)),
            offsets=np.zeros((3, 3)),
            parents=np.array([-1, 0, 1], dtype=np.int32),
            names=("a", "b", "c"),
            fps=24.0,
        )


def test_npz_round_trip(tmp_path):
    anim = make_anim()
    path = tmp_path / "clip.npz"
    anim.save(path)
    back = Anim.load(path)

    np.testing.assert_allclose(back.rotations, anim.rotations, atol=0)
    np.testing.assert_allclose(back.root_pos, anim.root_pos, atol=0)
    np.testing.assert_allclose(back.offsets, anim.offsets, atol=0)
    np.testing.assert_array_equal(back.parents, anim.parents)
    assert back.names == anim.names
    assert back.fps == anim.fps


def test_global_positions_places_root_at_root_pos():
    anim = make_anim()
    np.testing.assert_allclose(anim.global_positions()[:, 0], anim.root_pos, atol=1e-12)


def test_slice_selects_frames():
    anim = make_anim(n_frames=10)
    cut = anim.slice(2, 5)
    assert cut.n_frames == 3
    assert cut.n_joints == anim.n_joints
    np.testing.assert_allclose(cut.root_pos, anim.root_pos[2:5], atol=0)
    assert cut.names == anim.names
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `docker compose run --rm test pytest tests/core/test_anim.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'poseydon.core.anim'`

- [ ] **Step 3: Write the implementation**

`src/poseydon/core/anim.py`:

```python
"""The canonical animation container.

One ``Anim`` is one animation, full length. Ingest never chunks; windowing is a
load-time sampling concern (see the design spec, section 6.6).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from poseydon.core.kinematics import check_topological_order, forward_kinematics


@dataclass(frozen=True)
class Anim:
    """Local joint rotations plus a root trajectory over a fixed skeleton.

    Attributes:
        rotations: ``(F, J, 4)`` local rotations, scalar-last quaternions.
        root_pos:  ``(F, 3)`` global translation of joint 0.
        offsets:   ``(J, 3)`` rest-pose offset of each joint from its parent.
        parents:   ``(J,)`` int32 parent index, ``-1`` for the root.
        names:     ``J`` joint names, including End Sites.
        fps:       frames per second of this animation as stored.
    """

    rotations: np.ndarray
    root_pos: np.ndarray
    offsets: np.ndarray
    parents: np.ndarray
    names: tuple[str, ...]
    fps: float

    def __post_init__(self) -> None:
        if self.rotations.ndim != 3 or self.rotations.shape[-1] != 4:
            raise ValueError(f"rotations must be (F, J, 4), got {self.rotations.shape}")
        n_frames, n_joints = self.rotations.shape[:2]
        if self.root_pos.shape != (n_frames, 3):
            raise ValueError(
                f"root_pos must be ({n_frames}, 3) to match rotations, "
                f"got {self.root_pos.shape}"
            )
        if self.offsets.shape != (n_joints, 3):
            raise ValueError(f"offsets must be ({n_joints}, 3), got {self.offsets.shape}")
        if self.parents.shape != (n_joints,):
            raise ValueError(f"parents must be ({n_joints},), got {self.parents.shape}")
        if len(self.names) != n_joints:
            raise ValueError(
                f"names must have {n_joints} entries to match rotations, got {len(self.names)}"
            )
        if len(set(self.names)) != n_joints:
            raise ValueError("joint names must be unique")
        check_topological_order(self.parents)

    @property
    def n_frames(self) -> int:
        return int(self.rotations.shape[0])

    @property
    def n_joints(self) -> int:
        return int(self.rotations.shape[1])

    def global_positions(self) -> np.ndarray:
        """``(F, J, 3)`` global joint positions."""
        positions, _ = forward_kinematics(
            self.rotations, self.root_pos, self.offsets, self.parents
        )
        return positions

    def slice(self, start: int, stop: int) -> "Anim":
        """A new ``Anim`` over frames ``[start, stop)``, same skeleton."""
        return Anim(
            rotations=self.rotations[start:stop].copy(),
            root_pos=self.root_pos[start:stop].copy(),
            offsets=self.offsets,
            parents=self.parents,
            names=self.names,
            fps=self.fps,
        )

    def save(self, path: str | Path) -> None:
        np.savez_compressed(
            Path(path),
            rotations=self.rotations,
            root_pos=self.root_pos,
            offsets=self.offsets,
            parents=self.parents,
            names=np.array(self.names, dtype=object),
            fps=np.float64(self.fps),
        )

    @classmethod
    def load(cls, path: str | Path) -> "Anim":
        with np.load(Path(path), allow_pickle=True) as data:
            return cls(
                rotations=data["rotations"],
                root_pos=data["root_pos"],
                offsets=data["offsets"],
                parents=data["parents"].astype(np.int32),
                names=tuple(str(n) for n in data["names"]),
                fps=float(data["fps"]),
            )
```

Note on `allow_pickle=True`: it is required because joint names are stored as an
object array. These files are produced by our own ingest, never fetched from
elsewhere, so this is safe here. Do not extend that assumption to user-supplied `.npz`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `docker compose run --rm test pytest tests/core/test_anim.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/poseydon/core/anim.py tests/core/test_anim.py
git commit -m "feat(core): Anim container with npz round-trip"
```

---

### Task 5: BVH reader

**Files:**
- Create: `src/poseydon/io/bvh.py`
- Test: `tests/io/test_bvh_read.py`

**Interfaces:**
- Consumes: `Anim` (Task 4), `euler_to_quat`, `QUAT_IDENTITY` (Task 2)
- Produces:
  - `load_bvh(path: str | Path) -> Anim`
  - `BvhParseError(Exception)`
  - `CHANNEL_AXIS: dict[str, str]` mapping `"Xrotation" -> "X"` etc.

**Background the implementer needs.** A BVH file has a `HIERARCHY` block of nested
`ROOT` / `JOINT` / `End Site` entries followed by a `MOTION` block. In these files:

- `End Site` entries carry a nonstandard name comment: `End Site #name: Bip01_HeadNub`.
  **They are joints.** Joint counts including End Sites are 38, 40, 54, 40, 31, 63, 35 for
  the seven fixtures, and those match the reference's feature arrays exactly. A parser
  that drops End Sites is wrong.
- End Sites have an `OFFSET` but no `CHANNELS`; their local rotation is identity.
- Only two channel specs appear: the root's
  `CHANNELS 6 Xposition Yposition Zposition Zrotation Yrotation Xrotation`, and
  `CHANNELS 3 Zrotation Yrotation Xrotation` on every other jointed entry. Parse
  generally anyway — do not hardcode these two.
- Channels-per-line equals the total channel count, i.e. `6 + 3 * (jointed - 1)`.
- `Frame Time: 0.041667`, i.e. 24 fps. Store `fps = 1.0 / frame_time`.

- [ ] **Step 1: Write the failing tests**

`tests/io/test_bvh_read.py`. Create `tests/io/`.

```python
import numpy as np
import pytest

from poseydon.core.anim import Anim
from poseydon.io.bvh import BvhParseError, load_bvh


def test_loads_expected_shape(bvh_fixture, fixture_facts):
    facts = fixture_facts[bvh_fixture.stem]
    anim = load_bvh(bvh_fixture)

    assert isinstance(anim, Anim)
    assert anim.n_joints == facts["joints"], "End Sites must be counted as joints"
    assert anim.n_frames == facts["frames"]


def test_fps_is_24(bvh_fixture):
    assert load_bvh(bvh_fixture).fps == pytest.approx(24.0, abs=0.01)


def test_rotations_are_unit_quaternions(bvh_fixture):
    anim = load_bvh(bvh_fixture)
    norms = np.linalg.norm(anim.rotations, axis=-1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-10)


def test_topology_is_valid(bvh_fixture):
    anim = load_bvh(bvh_fixture)
    assert anim.parents[0] == -1
    assert np.all(anim.parents[1:] < np.arange(1, anim.n_joints))


def test_end_sites_have_identity_rotation(bvh_fixture):
    # End Sites are leaves with no channels, so their local rotation never changes.
    anim = load_bvh(bvh_fixture)
    is_leaf = np.ones(anim.n_joints, dtype=bool)
    is_leaf[anim.parents[1:]] = False
    leaf_rots = anim.rotations[:, is_leaf]
    expected = np.broadcast_to(np.array([0.0, 0.0, 0.0, 1.0]), leaf_rots.shape)
    np.testing.assert_allclose(leaf_rots, expected, atol=1e-12)


def test_flamingo_root_offset_and_name(truebones_dir):
    anim = load_bvh(truebones_dir / "Flamingo_Flamingo_OneLEgBEnt_353.bvh")
    assert anim.names[0] == "Bip01_Pelvis"
    np.testing.assert_allclose(anim.offsets[0], [0.0, 1.328589, 0.0], atol=1e-9)
    assert "BN_Tail_R_01" in anim.names, "named End Site must be kept"


def test_rejects_truncated_motion_block(tmp_path):
    path = tmp_path / "bad.bvh"
    path.write_text(
        "HIERARCHY\n"
        "ROOT a\n{\nOFFSET 0 0 0\n"
        "CHANNELS 6 Xposition Yposition Zposition Zrotation Yrotation Xrotation\n"
        "End Site #name: a_end\n{\nOFFSET 0 1 0\n}\n}\n"
        "MOTION\nFrames: 2\nFrame Time: 0.041667\n"
        "0 0 0 0 0 0\n"
    )
    with pytest.raises(BvhParseError, match="2 frames"):
        load_bvh(path)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `docker compose run --rm test pytest tests/io/test_bvh_read.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'poseydon.io.bvh'`

- [ ] **Step 3: Write the implementation**

`src/poseydon/io/bvh.py`:

```python
"""BVH reading.

End Sites are treated as ordinary joints with an offset, a name and no channels.
The Truebones files carry their names in a nonstandard ``#name:`` comment; where
that is missing a name is derived from the parent.

The MOTION block is parsed in a single vectorized pass rather than line by line,
which is where a naive parser spends nearly all its time.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from poseydon.core.anim import Anim
from poseydon.core.rotations import QUAT_IDENTITY, euler_to_quat

CHANNEL_AXIS = {
    "Xrotation": "X",
    "Yrotation": "Y",
    "Zrotation": "Z",
}
_POSITION_CHANNELS = ("Xposition", "Yposition", "Zposition")

_END_SITE_RE = re.compile(r"End\s+Site\s*(?:#\s*name:\s*(\S+))?", re.IGNORECASE)


class BvhParseError(Exception):
    """Raised when a BVH file cannot be parsed."""


def _parse_hierarchy(text: str):
    names: list[str] = []
    parents: list[int] = []
    offsets: list[list[float]] = []
    channels: list[list[str]] = []
    stack: list[int] = []

    def add(name: str) -> int:
        names.append(name)
        parents.append(stack[-1] if stack else -1)
        offsets.append([0.0, 0.0, 0.0])
        channels.append([])
        return len(names) - 1

    pending: int | None = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith(("ROOT ", "JOINT ")):
            pending = add(line.split(None, 1)[1].strip())
        elif line.upper().startswith("END SITE"):
            match = _END_SITE_RE.match(line)
            name = match.group(1) if match and match.group(1) else None
            if name is None:
                name = f"{names[stack[-1]]}_End" if stack else "End"
            pending = add(name)
        elif line.startswith("{"):
            if pending is None:
                raise BvhParseError("found '{' before any ROOT, JOINT or End Site")
            stack.append(pending)
            pending = None
        elif line.startswith("}"):
            if not stack:
                raise BvhParseError("unbalanced '}' in HIERARCHY")
            stack.pop()
        elif line.startswith("OFFSET"):
            if not stack:
                raise BvhParseError("OFFSET outside of any joint block")
            offsets[stack[-1]] = [float(v) for v in line.split()[1:4]]
        elif line.startswith("CHANNELS"):
            if not stack:
                raise BvhParseError("CHANNELS outside of any joint block")
            parts = line.split()
            count = int(parts[1])
            spec = parts[2 : 2 + count]
            if len(spec) != count:
                raise BvhParseError(f"CHANNELS declares {count} names, found {len(spec)}")
            channels[stack[-1]] = spec

    if stack:
        raise BvhParseError("unbalanced '{' in HIERARCHY")
    if not names:
        raise BvhParseError("no joints found in HIERARCHY")

    return (
        tuple(names),
        np.array(parents, dtype=np.int32),
        np.array(offsets, dtype=np.float64),
        channels,
    )


def _parse_motion(text: str, n_channels: int) -> tuple[np.ndarray, float]:
    lines = text.strip().splitlines()
    n_frames: int | None = None
    frame_time: float | None = None
    start = 0
    for index, raw in enumerate(lines):
        line = raw.strip()
        if line.lower().startswith("frames:"):
            n_frames = int(line.split(":", 1)[1])
        elif line.lower().startswith("frame time:"):
            frame_time = float(line.split(":", 1)[1])
            start = index + 1
            break
    if n_frames is None or frame_time is None:
        raise BvhParseError("MOTION block is missing 'Frames:' or 'Frame Time:'")

    block = " ".join(lines[start:])
    values = np.fromstring(block, sep=" ", dtype=np.float64)
    expected = n_frames * n_channels
    if values.size != expected:
        raise BvhParseError(
            f"MOTION block declares {n_frames} frames of {n_channels} channels "
            f"({expected} values) but contains {values.size}"
        )
    return values.reshape(n_frames, n_channels), frame_time


def _channels_to_local(values, channels, n_joints):
    n_frames = values.shape[0]
    rotations = np.broadcast_to(QUAT_IDENTITY, (n_frames, n_joints, 4)).copy()
    root_pos = np.zeros((n_frames, 3), dtype=np.float64)

    column = 0
    for joint, spec in enumerate(channels):
        if not spec:
            continue
        columns = {name: column + offset for offset, name in enumerate(spec)}
        column += len(spec)

        position_names = [n for n in _POSITION_CHANNELS if n in columns]
        if position_names:
            if joint != 0:
                raise BvhParseError(
                    f"joint {joint} has position channels; only the root may translate"
                )
            for axis_index, name in enumerate(_POSITION_CHANNELS):
                if name in columns:
                    root_pos[:, axis_index] = values[:, columns[name]]

        rotation_names = [n for n in spec if n in CHANNEL_AXIS]
        if rotation_names:
            order = "".join(CHANNEL_AXIS[n] for n in rotation_names)
            angles = np.stack([values[:, columns[n]] for n in rotation_names], axis=-1)
            rotations[:, joint] = euler_to_quat(angles, order)

    return rotations, root_pos


def load_bvh(path: str | Path) -> Anim:
    """Read a BVH file into an :class:`Anim`. End Sites become joints."""
    text = Path(path).read_text()
    head, marker, motion = text.partition("MOTION")
    if not marker:
        raise BvhParseError(f"{path}: no MOTION block found")

    names, parents, offsets, channels = _parse_hierarchy(head)
    n_channels = sum(len(spec) for spec in channels)
    values, frame_time = _parse_motion(motion, n_channels)
    rotations, root_pos = _channels_to_local(values, channels, len(names))

    if frame_time <= 0.0:
        raise BvhParseError(f"{path}: non-positive Frame Time {frame_time}")

    return Anim(
        rotations=rotations,
        root_pos=root_pos,
        offsets=offsets,
        parents=parents,
        names=names,
        fps=1.0 / frame_time,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `docker compose run --rm test pytest tests/io/test_bvh_read.py -v`
Expected: all PASS (or SKIP if fixtures are absent, except
`test_rejects_truncated_motion_block`, which uses `tmp_path` and must PASS regardless).

- [ ] **Step 5: Commit**

```bash
git add src/poseydon/io/bvh.py tests/io/test_bvh_read.py
git commit -m "feat(io): vectorized BVH reader treating End Sites as joints"
```

---

### Task 6: BVH writer and round-trip

**Files:**
- Modify: `src/poseydon/io/bvh.py` (append writing functions)
- Test: `tests/io/test_bvh_write.py`

**Interfaces:**
- Consumes: `Anim` (Task 4), `quat_to_euler` (Task 2), `load_bvh` (Task 5)
- Produces: `save_bvh(anim: Anim, path: str | Path, order: str = "ZYX") -> None`

- [ ] **Step 1: Write the failing tests**

`tests/io/test_bvh_write.py`:

```python
import numpy as np
import pytest

from poseydon.core.rotations import quat_to_matrix
from poseydon.io.bvh import load_bvh, save_bvh


def test_round_trip_preserves_structure(bvh_fixture, tmp_path):
    original = load_bvh(bvh_fixture)
    out = tmp_path / "round_trip.bvh"
    save_bvh(original, out)
    back = load_bvh(out)

    assert back.names == original.names
    np.testing.assert_array_equal(back.parents, original.parents)
    np.testing.assert_allclose(back.offsets, original.offsets, atol=1e-6)
    assert back.n_frames == original.n_frames
    # Frame Time is written with 6 decimals, so fps survives only approximately:
    # 1/24 -> "0.041667" -> 23.99981...
    assert back.fps == pytest.approx(original.fps, rel=1e-4)


def test_round_trip_preserves_rotations_as_matrices(bvh_fixture, tmp_path):
    # Euler angles are not unique, so compare the rotations they represent.
    original = load_bvh(bvh_fixture)
    out = tmp_path / "round_trip.bvh"
    save_bvh(original, out)
    back = load_bvh(out)

    np.testing.assert_allclose(
        quat_to_matrix(back.rotations), quat_to_matrix(original.rotations), atol=1e-6
    )


def test_round_trip_preserves_root_trajectory(bvh_fixture, tmp_path):
    original = load_bvh(bvh_fixture)
    out = tmp_path / "round_trip.bvh"
    save_bvh(original, out)
    back = load_bvh(out)

    np.testing.assert_allclose(back.root_pos, original.root_pos, atol=1e-5)


def test_written_file_keeps_end_site_names(bvh_fixture, tmp_path):
    original = load_bvh(bvh_fixture)
    out = tmp_path / "round_trip.bvh"
    save_bvh(original, out)

    text = out.read_text()
    assert "End Site #name:" in text
    assert text.startswith("HIERARCHY")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `docker compose run --rm test pytest tests/io/test_bvh_write.py -v`
Expected: FAIL — `ImportError: cannot import name 'save_bvh'`

- [ ] **Step 3: Append the implementation to `src/poseydon/io/bvh.py`**

```python
def _children_of(parents: np.ndarray) -> list[list[int]]:
    children: list[list[int]] = [[] for _ in range(parents.size)]
    for joint in range(1, parents.size):
        children[int(parents[joint])].append(joint)
    return children


def _write_joint(lines, anim, children, joint, depth, order) -> None:
    pad = "\t" * depth
    is_end_site = not children[joint]

    if is_end_site:
        lines.append(f"{pad}End Site #name: {anim.names[joint]}")
    elif joint == 0:
        lines.append(f"{pad}ROOT {anim.names[joint]}")
    else:
        lines.append(f"{pad}JOINT {anim.names[joint]}")

    lines.append(f"{pad}{{")
    ox, oy, oz = anim.offsets[joint]
    lines.append(f"{pad}\tOFFSET {ox:.6f} {oy:.6f} {oz:.6f}")

    if not is_end_site:
        channel_names = " ".join(f"{axis}rotation" for axis in order)
        if joint == 0:
            lines.append(
                f"{pad}\tCHANNELS 6 Xposition Yposition Zposition {channel_names}"
            )
        else:
            lines.append(f"{pad}\tCHANNELS 3 {channel_names}")
        for child in children[joint]:
            _write_joint(lines, anim, children, child, depth + 1, order)

    lines.append(f"{pad}}}")


def save_bvh(anim: Anim, path: str | Path, order: str = "ZYX") -> None:
    """Write an :class:`Anim` to a BVH file.

    Leaf joints are written as ``End Site`` entries carrying their name in the
    same ``#name:`` comment the reader understands, so a round trip preserves
    joint count and naming.
    """
    children = _children_of(anim.parents)

    lines: list[str] = ["HIERARCHY"]
    _write_joint(lines, anim, children, 0, 0, order)

    lines.append("MOTION")
    lines.append(f"Frames: {anim.n_frames}")
    lines.append(f"Frame Time: {1.0 / anim.fps:.6f}")

    jointed = [j for j in range(anim.n_joints) if children[j]]
    angles = quat_to_euler(anim.rotations[:, jointed], order)

    for frame in range(anim.n_frames):
        row = [f"{value:.6f}" for value in anim.root_pos[frame]]
        row.extend(f"{value:.6f}" for value in angles[frame].reshape(-1))
        lines.append(" ".join(row))

    Path(path).write_text("\n".join(lines) + "\n")
```

Add `quat_to_euler` to the existing import from `poseydon.core.rotations` at the top
of the file, so it reads:

```python
from poseydon.core.rotations import QUAT_IDENTITY, euler_to_quat, quat_to_euler
```

- [ ] **Step 4: Run the full suite**

Run: `docker compose run --rm test pytest -v`
Expected: all PASS or SKIP. No failures.

- [ ] **Step 5: Run the linter**

Run: `docker compose run --rm test ruff check src tests`
Expected: no findings. Fix any that appear.

- [ ] **Step 6: Commit**

```bash
git add src/poseydon/io/bvh.py tests/io/test_bvh_write.py
git commit -m "feat(io): BVH writer with round-trip preservation of End Site names"
```

---

## Follow-on plans

This plan delivers the foundations half of the spec's milestone 1. The next plan
covers ingest: skeleton manifests with inheritance and name resolution, alignment
(ground, facing, scale, T-pose-relative rotations), the GPU IK solver, the JSONL
corpus index, the ingest pipeline and CLI, and the IO/IK benchmark against the
`Motion` baseline.

**Carry into the ingest plan:** the fixtures are 24 fps (`Frame Time: 0.041667`) while
the reference hardcodes `FPS = 20` and never resamples. Golden-parity ingest must
therefore not resample either, or extracted features will not match the reference
`.npy` files. Make `fps` in the skeleton manifest a target that is a no-op when it
equals the source rate, and set it to 24 for the Truebones manifests.

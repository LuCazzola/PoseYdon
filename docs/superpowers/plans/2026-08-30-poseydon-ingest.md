# PoseYdon Ingest Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn a folder of BVH files plus skeleton manifests into aligned `Anim` files and a corpus index, driven by `poseydon ingest`.

**Architecture:** Skeleton knowledge moves out of Python constants into per-skeleton YAML
manifests that reference joints by name. Alignment is a pure function from
`(Anim, SkeletonManifest)` to an aligned `Anim`. Clip identity lives in a JSONL corpus
index, so nothing downstream ever parses a filename.

**Tech Stack:** Python >=3.11, numpy, scipy, PyYAML, pytest, ruff — all inside Docker.

**Spec:** `docs/superpowers/specs/2026-08-30-poseydon-design.md` (sections 6.1, 6.2, 6.3)

**Predecessor:** `docs/superpowers/plans/2026-08-30-poseydon-foundations.md` (merged).
`Anim`, `load_bvh`, `save_bvh`, rotations and FK already exist.

## Global Constraints

- **All Python runs inside Docker.** Every command is `docker compose run --rm test ...`. Never run `pytest`, `ruff` or `uv` on the host.
- Quaternions are scalar-last `(x, y, z, w)`. Arrays are `float64`, `parents` is `int32`.
- **End Sites are joints** (established in the predecessor plan).
- `external/neural_motion_blending/` is **read-only**.
- **`__` is a reserved separator** between skeleton name and the rest of a clip id. Ingest rejects a skeleton name containing it.
- **Alignment order is fixed and must not be reordered:** rotate to face +Z, move root XZ to origin, scale, put on ground. This matches the reference's `process_anim`; a different order gives different output.
- **Ingest never resamples.** The Truebones fixtures are 24 fps while the reference hardcodes `FPS = 20` and never resamples; resampling would break future golden-parity checks. A manifest `fps` value is a target that must be a no-op when it equals the source rate, and an error otherwise until resampling is implemented.
- **Deliberately out of scope:** T-pose-relative rotations. The reference's `compute_rots_from_tpos` relies on Holden's `Quaternions.__neg__`, whose meaning (conjugate vs component negation) cannot be verified without the `Motion` package. It belongs in the features plan, where the golden `.npy` files disambiguate it empirically. Aligned `Anim` files therefore store ordinary BVH-local rotations.
- Every task ends with a green `pytest` run and a commit.

---

### Task 1: Skeleton manifests

**Files:**
- Modify: `pyproject.toml` (add `pyyaml>=6`)
- Create: `src/poseydon/core/skeleton.py`
- Test: `tests/core/test_skeleton.py`

**Interfaces:**
- Consumes: nothing from this plan
- Produces:
  - `FacingPair` frozen dataclass: `right: str`, `left: str`
  - `ContactParams` frozen dataclass: `max_height: float`, `max_speed: float`
  - `SkeletonManifest` frozen dataclass with fields `name: str`, `tpose: Path | None`,
    `facing: tuple[FacingPair, ...]`, `extra_yaw_deg: float`, `foot_joints: tuple[str, ...]`,
    `fps: float | None`, `mean_bone_length: float | None`, `contact: ContactParams`,
    `tags: tuple[str, ...]`, `strip_joint_prefix: str | None`, `source: Path`
  - `SkeletonManifest.load(path: str | Path) -> SkeletonManifest` (classmethod, resolves `base:`)
  - `ManifestError(Exception)`
  - `HML_MEAN_BONE_LENGTH: float = 0.20921428571428569`

- [ ] **Step 1: Add PyYAML and rebuild the image**

In `pyproject.toml`, change the dependencies list to:

```toml
dependencies = [
    "numpy>=1.26",
    "scipy>=1.11",
    "pyyaml>=6",
]
```

Run: `docker compose build test`
Expected: build succeeds, pyyaml appears in the install list.

- [ ] **Step 2: Write the failing tests**

`tests/core/test_skeleton.py`:

```python
import pytest

from poseydon.core.skeleton import (
    HML_MEAN_BONE_LENGTH,
    ManifestError,
    SkeletonManifest,
)

MINIMAL = """
skeleton: Testy
facing:
  hips:      {right: R_Thigh, left: L_Thigh}
  shoulders: {right: R_Arm,   left: L_Arm}
scale: {mean_bone_length: 0.2}
contact: {max_height: 0.3, max_speed: 0.045}
"""


def write(tmp_path, text, name="Testy.yaml"):
    path = tmp_path / name
    path.write_text(text)
    return path


def test_loads_minimal_manifest(tmp_path):
    manifest = SkeletonManifest.load(write(tmp_path, MINIMAL))

    assert manifest.name == "Testy"
    assert [p.right for p in manifest.facing] == ["R_Thigh", "R_Arm"]
    assert [p.left for p in manifest.facing] == ["L_Thigh", "L_Arm"]
    assert manifest.extra_yaw_deg == 0.0
    assert manifest.mean_bone_length == 0.2
    assert manifest.contact.max_speed == 0.045
    assert manifest.tags == ()
    assert manifest.fps is None


def test_hml_constant_matches_reference():
    # mean of the 21 SMPL bone lengths in the reference's param_utils
    assert HML_MEAN_BONE_LENGTH == pytest.approx(0.20921428571428569, abs=1e-15)


def test_facing_order_is_hips_then_shoulders(tmp_path):
    # The forward axis sums the across-body vectors, so order does not change the
    # result -- but it must be deterministic for reproducibility.
    manifest = SkeletonManifest.load(write(tmp_path, MINIMAL))
    assert manifest.facing[0].right == "R_Thigh"
    assert manifest.facing[1].right == "R_Arm"


def test_base_inheritance_merges_and_overrides(tmp_path):
    (tmp_path / "_base.yaml").write_text(
        "facing:\n"
        "  hips:      {right: R_Thigh, left: L_Thigh}\n"
        "  shoulders: {right: R_Arm,   left: L_Arm}\n"
        "scale: {mean_bone_length: 0.2}\n"
        "contact: {max_height: 0.3, max_speed: 0.045}\n"
        "strip_joint_prefix: 'mixamorig:'\n"
        "tags: [humanoid]\n"
    )
    child = write(
        tmp_path,
        "base: _base.yaml\nskeleton: Goblin\ntags: [humanoid, goblin]\n",
        name="Goblin.yaml",
    )
    manifest = SkeletonManifest.load(child)

    assert manifest.name == "Goblin"
    assert manifest.strip_joint_prefix == "mixamorig:"
    assert manifest.facing[0].right == "R_Thigh"
    assert manifest.tags == ("humanoid", "goblin")


def test_tpose_path_resolves_relative_to_manifest(tmp_path):
    nested = tmp_path / "skeletons"
    nested.mkdir()
    path = write(nested, MINIMAL + "tpose: ../tposes/Testy.bvh\n")
    manifest = SkeletonManifest.load(path)
    assert manifest.tpose == (tmp_path / "tposes" / "Testy.bvh").resolve()


def test_rejects_double_underscore_in_skeleton_name(tmp_path):
    text = MINIMAL.replace("skeleton: Testy", "skeleton: Bad__Name")
    with pytest.raises(ManifestError, match="__"):
        SkeletonManifest.load(write(tmp_path, text))


def test_rejects_missing_required_section(tmp_path):
    text = "skeleton: Testy\ncontact: {max_height: 0.3, max_speed: 0.045}\n"
    with pytest.raises(ManifestError, match="facing"):
        SkeletonManifest.load(write(tmp_path, text))


def test_rejects_unknown_key_with_suggestion(tmp_path):
    text = MINIMAL + "foot_joint: [a]\n"
    with pytest.raises(ManifestError, match="foot_joints"):
        SkeletonManifest.load(write(tmp_path, text))


def test_rejects_facing_pair_missing_a_side(tmp_path):
    text = MINIMAL.replace("{right: R_Arm,   left: L_Arm}", "{right: R_Arm}")
    with pytest.raises(ManifestError, match="left"):
        SkeletonManifest.load(write(tmp_path, text))
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `docker compose run --rm test pytest tests/core/test_skeleton.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'poseydon.core.skeleton'`

- [ ] **Step 4: Write the implementation**

`src/poseydon/core/skeleton.py`:

```python
"""Skeleton manifests.

Everything the reference kept as Python literals in ``param_utils.py`` --
per-species face-joint indices, contact thresholds, taxonomy -- lives here as
per-skeleton YAML instead. Joints are referenced by NAME, never by index, so a
re-exported rig fails loudly rather than silently mirroring the character.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass
from pathlib import Path

import yaml

# Mean of the 21 SMPL bone lengths used by the reference as its scale target.
# The reference calls this HML_AVG_BONELEN; its docstring claims "longest
# armature" but the code takes the mean, and the code is what ran.
HML_MEAN_BONE_LENGTH = 0.20921428571428569

_KNOWN_KEYS = frozenset(
    {
        "base",
        "skeleton",
        "tpose",
        "facing",
        "foot_joints",
        "fps",
        "scale",
        "contact",
        "tags",
        "strip_joint_prefix",
    }
)


class ManifestError(Exception):
    """Raised when a skeleton manifest is malformed."""


@dataclass(frozen=True)
class FacingPair:
    """One across-body vector, as (right - left)."""

    right: str
    left: str


@dataclass(frozen=True)
class ContactParams:
    """Ground-contact thresholds, in scale-normalized units.

    ``max_speed`` is a genuine speed in units per frame. The reference stores
    0.002 and compares it against a SQUARED displacement, so its effective
    threshold is sqrt(0.002) ~= 0.045; that value is what belongs here.
    """

    max_height: float
    max_speed: float


@dataclass(frozen=True)
class SkeletonManifest:
    name: str
    facing: tuple[FacingPair, ...]
    contact: ContactParams
    source: Path
    tpose: Path | None = None
    extra_yaw_deg: float = 0.0
    foot_joints: tuple[str, ...] = ()
    fps: float | None = None
    mean_bone_length: float | None = None
    tags: tuple[str, ...] = ()
    strip_joint_prefix: str | None = None

    @classmethod
    def load(cls, path: str | Path) -> "SkeletonManifest":
        path = Path(path).resolve()
        data = _load_with_base(path, seen=[])
        return _build(data, path)


def _read_yaml(path: Path) -> dict:
    if not path.is_file():
        raise ManifestError(f"manifest not found: {path}")
    loaded = yaml.safe_load(path.read_text()) or {}
    if not isinstance(loaded, dict):
        raise ManifestError(f"{path}: manifest must be a YAML mapping")
    return loaded


def _load_with_base(path: Path, seen: list[Path]) -> dict:
    if path in seen:
        chain = " -> ".join(p.name for p in [*seen, path])
        raise ManifestError(f"circular `base:` chain: {chain}")
    data = _read_yaml(path)

    unknown = set(data) - _KNOWN_KEYS
    if unknown:
        key = sorted(unknown)[0]
        close = difflib.get_close_matches(key, sorted(_KNOWN_KEYS), n=1)
        hint = f", did you mean `{close[0]}`?" if close else ""
        raise ManifestError(f"{path}: unknown key `{key}`{hint}")

    base_name = data.pop("base", None)
    if base_name is None:
        data["__source__"] = path
        return data

    base_path = (path.parent / str(base_name)).resolve()
    merged = _load_with_base(base_path, [*seen, path])
    merged.update(data)
    merged["__source__"] = path
    return merged


def _require(data: dict, key: str, path: Path):
    if key not in data:
        raise ManifestError(f"{path}: missing required section `{key}`")
    return data[key]


def _build(data: dict, path: Path) -> SkeletonManifest:
    source = data.pop("__source__", path)

    name = str(_require(data, "skeleton", path))
    if "__" in name:
        raise ManifestError(
            f"{path}: skeleton name `{name}` contains the reserved separator `__`, "
            "which is used between skeleton and action in clip ids"
        )

    facing_raw = _require(data, "facing", path)
    if not isinstance(facing_raw, dict):
        raise ManifestError(f"{path}: `facing` must be a mapping of named pairs")

    extra_yaw = float(facing_raw.get("extra_yaw_deg", 0.0))
    pairs = []
    for pair_name, pair in facing_raw.items():
        if pair_name == "extra_yaw_deg":
            continue
        if not isinstance(pair, dict):
            raise ManifestError(f"{path}: facing.{pair_name} must be a mapping")
        for side in ("right", "left"):
            if side not in pair:
                raise ManifestError(f"{path}: facing.{pair_name} is missing `{side}`")
        pairs.append(FacingPair(right=str(pair["right"]), left=str(pair["left"])))
    if not pairs:
        raise ManifestError(f"{path}: `facing` declares no pairs")

    contact_raw = _require(data, "contact", path)
    for key in ("max_height", "max_speed"):
        if key not in contact_raw:
            raise ManifestError(f"{path}: contact is missing `{key}`")
    contact = ContactParams(
        max_height=float(contact_raw["max_height"]),
        max_speed=float(contact_raw["max_speed"]),
    )

    scale_raw = data.get("scale") or {}
    mean_bone_length = scale_raw.get("mean_bone_length")

    tpose = data.get("tpose")
    tpose_path = (source.parent / str(tpose)).resolve() if tpose else None

    return SkeletonManifest(
        name=name,
        facing=tuple(pairs),
        contact=contact,
        source=source,
        tpose=tpose_path,
        extra_yaw_deg=extra_yaw,
        foot_joints=tuple(str(j) for j in (data.get("foot_joints") or ())),
        fps=None if data.get("fps") is None else float(data["fps"]),
        mean_bone_length=None if mean_bone_length is None else float(mean_bone_length),
        tags=tuple(str(t) for t in (data.get("tags") or ())),
        strip_joint_prefix=(
            None if data.get("strip_joint_prefix") is None else str(data["strip_joint_prefix"])
        ),
    )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `docker compose run --rm test pytest tests/core/test_skeleton.py -v`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml src/poseydon/core/skeleton.py tests/core/test_skeleton.py
git commit -m "feat(core): skeleton manifests replacing param_utils constants"
```

---

### Task 2: Joint-name resolution

**Files:**
- Modify: `src/poseydon/core/skeleton.py` (append)
- Test: `tests/core/test_joint_resolution.py`

**Interfaces:**
- Consumes: `SkeletonManifest`, `ManifestError` (Task 1); `Anim` (predecessor plan)
- Produces:
  - `resolve_joint(name: str, names: Sequence[str], *, context: str = "") -> int`
  - `ResolvedSkeleton` frozen dataclass: `manifest: SkeletonManifest`,
    `facing_indices: tuple[tuple[int, int], ...]` (right, left per pair),
    `foot_indices: tuple[int, ...]`
  - `resolve(manifest: SkeletonManifest, names: Sequence[str]) -> ResolvedSkeleton`
  - `strip_prefix(names: Sequence[str], prefix: str | None) -> tuple[str, ...]`

- [ ] **Step 1: Write the failing tests**

`tests/core/test_joint_resolution.py`:

```python
import pytest

from poseydon.core.skeleton import (
    ContactParams,
    FacingPair,
    ManifestError,
    SkeletonManifest,
    resolve,
    resolve_joint,
    strip_prefix,
)

NAMES = ["Hips", "R_Thigh", "L_Thigh", "R_Arm", "L_Arm", "R_Foot", "L_Foot"]


def make_manifest(foot=("R_Foot", "L_Foot")) -> SkeletonManifest:
    return SkeletonManifest(
        name="Testy",
        facing=(FacingPair("R_Thigh", "L_Thigh"), FacingPair("R_Arm", "L_Arm")),
        contact=ContactParams(0.3, 0.045),
        source=__import__("pathlib").Path("Testy.yaml"),
        foot_joints=foot,
    )


def test_resolves_exact_name():
    assert resolve_joint("R_Thigh", NAMES) == 1


def test_unknown_name_suggests_nearest_match():
    with pytest.raises(ManifestError) as excinfo:
        resolve_joint("R_Thig", NAMES)
    message = str(excinfo.value)
    assert "R_Thig" in message
    assert "R_Thigh" in message


def test_unknown_name_lists_available_when_nothing_is_close():
    with pytest.raises(ManifestError) as excinfo:
        resolve_joint("zzzzzz", NAMES)
    assert "Hips" in str(excinfo.value)


def test_resolve_produces_index_pairs_in_order():
    resolved = resolve(make_manifest(), NAMES)
    assert resolved.facing_indices == ((1, 2), (3, 4))
    assert resolved.foot_indices == (5, 6)


def test_resolve_reports_context_on_failure():
    manifest = make_manifest(foot=("NoSuchFoot",))
    with pytest.raises(ManifestError, match="foot_joints"):
        resolve(manifest, NAMES)


def test_strip_prefix_removes_only_the_declared_prefix():
    names = ["mixamorig:Hips", "mixamorig:Spine", "Extra"]
    assert strip_prefix(names, "mixamorig:") == ("Hips", "Spine", "Extra")


def test_strip_prefix_is_identity_when_none():
    assert strip_prefix(NAMES, None) == tuple(NAMES)


def test_strip_prefix_rejects_collisions():
    with pytest.raises(ManifestError, match="collide"):
        strip_prefix(["a:Hips", "Hips"], "a:")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `docker compose run --rm test pytest tests/core/test_joint_resolution.py -v`
Expected: FAIL — `ImportError: cannot import name 'resolve'`

- [ ] **Step 3: Append the implementation to `src/poseydon/core/skeleton.py`**

Add `from collections.abc import Sequence` to the imports at the top of the file, then
append:

```python
def resolve_joint(name: str, names: Sequence[str], *, context: str = "") -> int:
    """Index of ``name`` in ``names``, or a readable error naming alternatives."""
    try:
        return list(names).index(name)
    except ValueError:
        pass

    where = f" ({context})" if context else ""
    close = difflib.get_close_matches(name, list(names), n=3)
    if close:
        hint = ", did you mean " + " or ".join(f"`{c}`" for c in close) + "?"
    else:
        preview = ", ".join(list(names)[:10])
        more = "" if len(names) <= 10 else f", ... ({len(names)} total)"
        hint = f". Available joints: {preview}{more}"
    raise ManifestError(f"joint `{name}`{where} is not in the skeleton{hint}")


@dataclass(frozen=True)
class ResolvedSkeleton:
    """A manifest bound to a concrete joint ordering."""

    manifest: SkeletonManifest
    facing_indices: tuple[tuple[int, int], ...]
    foot_indices: tuple[int, ...]


def resolve(manifest: SkeletonManifest, names: Sequence[str]) -> ResolvedSkeleton:
    """Bind every joint name in ``manifest`` to an index in ``names``."""
    facing = tuple(
        (
            resolve_joint(pair.right, names, context=f"{manifest.name} facing.right"),
            resolve_joint(pair.left, names, context=f"{manifest.name} facing.left"),
        )
        for pair in manifest.facing
    )
    feet = tuple(
        resolve_joint(joint, names, context=f"{manifest.name} foot_joints")
        for joint in manifest.foot_joints
    )
    return ResolvedSkeleton(manifest=manifest, facing_indices=facing, foot_indices=feet)


def strip_prefix(names: Sequence[str], prefix: str | None) -> tuple[str, ...]:
    """Drop a dataset-specific joint-name prefix, e.g. ``mixamorig:``."""
    if not prefix:
        return tuple(names)
    stripped = tuple(n[len(prefix) :] if n.startswith(prefix) else n for n in names)
    if len(set(stripped)) != len(stripped):
        raise ManifestError(
            f"stripping prefix `{prefix}` makes joint names collide; "
            "remove the prefix from the manifest or rename the joints"
        )
    return stripped
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `docker compose run --rm test pytest tests/core/test_joint_resolution.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/poseydon/core/skeleton.py tests/core/test_joint_resolution.py
git commit -m "feat(core): joint-name resolution with nearest-match errors"
```

---

### Task 3: Alignment

**Files:**
- Create: `src/poseydon/ingest/__init__.py`
- Create: `src/poseydon/ingest/align.py`
- Test: `tests/ingest/test_align.py`

**Interfaces:**
- Consumes: `Anim` (predecessor), `ResolvedSkeleton`/`resolve`/`HML_MEAN_BONE_LENGTH` (Tasks 1-2), rotations helpers
- Produces:
  - `facing_quat(positions, facing_indices, extra_yaw_deg=0.0) -> np.ndarray` — `(4,)`
  - `rotate_to_face_z(anim, rotation) -> Anim`
  - `move_xz_to_origin(anim) -> tuple[Anim, np.ndarray]`
  - `scale_to_mean_bone_length(anim, target) -> tuple[Anim, float]`
  - `put_on_ground(anim) -> tuple[Anim, float]`
  - `AlignmentParams` frozen dataclass: `rotation (4,)`, `root_xz (3,)`, `scale_factor float`, `ground_height float`
  - `align(anim, resolved, target_bone_length=HML_MEAN_BONE_LENGTH) -> tuple[Anim, AlignmentParams]`

**Background.** The reference's `process_anim` applies, in this exact order: rotate to
face +Z, move root XZ to origin, scale, put on ground. Reordering changes the output,
so the order is part of the contract. The facing rotation is computed from the FIRST
frame only and then applied to every frame.

Forward is derived as ``cross(world_up, across)`` where ``across`` is the normalized sum
of ``(right - left)`` over the facing pairs, and the rotation is the one taking that
forward onto ``+Z``. Note the reference ignores the root's BVH ``OFFSET`` when the root
has position channels, matching Holden's loader; our `Anim` already does the same.

- [ ] **Step 1: Write the failing tests**

`tests/ingest/test_align.py`. Create `tests/ingest/`.

```python
import numpy as np
import pytest

from poseydon.core.anim import Anim
from poseydon.core.rotations import QUAT_IDENTITY, euler_to_quat, quat_apply
from poseydon.core.skeleton import (
    ContactParams,
    FacingPair,
    SkeletonManifest,
    resolve,
)
from poseydon.ingest.align import (
    align,
    facing_quat,
    put_on_ground,
    scale_to_mean_bone_length,
)
from poseydon.io.bvh import load_bvh


def toy_anim(n_frames=3) -> Anim:
    # Root, two hips (right +X, left -X), two shoulders, one foot below origin.
    offsets = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [1.0, 2.0, 0.0],
            [-1.0, 2.0, 0.0],
            [0.0, -3.0, 0.0],
        ]
    )
    parents = np.array([-1, 0, 0, 0, 0, 0], dtype=np.int32)
    return Anim(
        rotations=np.broadcast_to(QUAT_IDENTITY, (n_frames, 6, 4)).copy(),
        root_pos=np.tile(np.array([5.0, 4.0, 7.0]), (n_frames, 1)),
        offsets=offsets,
        parents=parents,
        names=("root", "r_hip", "l_hip", "r_sdr", "l_sdr", "foot"),
        fps=24.0,
    )


def toy_resolved():
    manifest = SkeletonManifest(
        name="Toy",
        facing=(FacingPair("r_hip", "l_hip"), FacingPair("r_sdr", "l_sdr")),
        contact=ContactParams(0.3, 0.045),
        source=__import__("pathlib").Path("Toy.yaml"),
    )
    return resolve(manifest, ["root", "r_hip", "l_hip", "r_sdr", "l_sdr", "foot"])


def test_facing_quat_turns_x_facing_character_to_face_z():
    # across = right - left = +X, so forward = cross(+Y, +X) = -Z.
    # The rotation must take -Z onto +Z.
    anim = toy_anim()
    positions = anim.global_positions()
    q = facing_quat(positions, toy_resolved().facing_indices)
    np.testing.assert_allclose(
        quat_apply(q, np.array([0.0, 0.0, -1.0])), [0.0, 0.0, 1.0], atol=1e-10
    )


def test_extra_yaw_is_applied_on_top():
    anim = toy_anim()
    positions = anim.global_positions()
    plain = facing_quat(positions, toy_resolved().facing_indices)
    yawed = facing_quat(positions, toy_resolved().facing_indices, extra_yaw_deg=90.0)
    assert not np.allclose(plain, yawed)


def test_scale_sets_mean_bone_length():
    anim = toy_anim()
    scaled, factor = scale_to_mean_bone_length(anim, 0.2)
    lengths = np.linalg.norm(scaled.offsets[1:], axis=-1)
    assert lengths.mean() == pytest.approx(0.2, rel=1e-12)
    assert factor > 0


def test_put_on_ground_puts_lowest_joint_at_zero():
    anim = toy_anim()
    grounded, height = put_on_ground(anim)
    assert grounded.global_positions()[..., 1].min() == pytest.approx(0.0, abs=1e-12)
    assert height == pytest.approx(1.0, abs=1e-12)  # foot at 4 - 3 = 1


def test_align_is_idempotent_on_already_aligned_anim():
    anim = toy_anim()
    once, _ = align(anim, toy_resolved())
    twice, _ = align(once, toy_resolved())
    np.testing.assert_allclose(twice.global_positions(), once.global_positions(), atol=1e-9)


def test_align_grounds_and_centres_real_fixture(bvh_fixture, truebones_dir):
    from tests.ingest.manifest_helper import resolved_for  # noqa: PLC0415

    anim = load_bvh(bvh_fixture)
    resolved = resolved_for(bvh_fixture, anim)
    aligned, params = align(anim, resolved)

    positions = aligned.global_positions()
    assert positions[..., 1].min() == pytest.approx(0.0, abs=1e-9)
    np.testing.assert_allclose(aligned.root_pos[0, [0, 2]], [0.0, 0.0], atol=1e-9)
    lengths = np.linalg.norm(aligned.offsets[1:], axis=-1)
    assert lengths.mean() == pytest.approx(0.20921428571428569, rel=1e-9)
    assert aligned.n_frames == anim.n_frames
    assert aligned.names == anim.names
    assert params.scale_factor > 0


def test_align_preserves_bone_length_ratios(bvh_fixture):
    from tests.ingest.manifest_helper import resolved_for  # noqa: PLC0415

    anim = load_bvh(bvh_fixture)
    aligned, params = align(anim, resolved_for(bvh_fixture, anim))
    before = np.linalg.norm(anim.offsets[1:], axis=-1)
    after = np.linalg.norm(aligned.offsets[1:], axis=-1)
    np.testing.assert_allclose(after, before * params.scale_factor, atol=1e-9)


def test_rotation_is_rigid(bvh_fixture):
    from tests.ingest.manifest_helper import resolved_for  # noqa: PLC0415

    anim = load_bvh(bvh_fixture)
    aligned, params = align(anim, resolved_for(bvh_fixture, anim))
    # Pairwise joint distances are preserved up to the uniform scale factor.
    a = anim.global_positions()[0]
    b = aligned.global_positions()[0]
    da = np.linalg.norm(a[:, None] - a[None, :], axis=-1)
    db = np.linalg.norm(b[:, None] - b[None, :], axis=-1)
    np.testing.assert_allclose(db, da * params.scale_factor, atol=1e-7)
```

Also create `tests/ingest/manifest_helper.py`, which Task 5's manifests will satisfy:

```python
"""Bind a fixture BVH to its shipped skeleton manifest."""

from pathlib import Path

from poseydon.core.skeleton import SkeletonManifest, resolve

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_DIR = REPO_ROOT / "data" / "truebones" / "skeletons"

# Fixture stems encode the species differently from clip to clip, so map explicitly.
STEM_TO_SKELETON = {
    "BrownBear___RiseSwat_132": "BrownBear",
    "Coyote___Attack3_224": "Coyote",
    "Crab___Attack3_234": "Crab",
    "Flamingo_Flamingo_OneLEgBEnt_353": "Flamingo",
    "Goat___HeadButt_395": "Goat",
    "Scorpion___SlowForward_839": "Scorpion",
    "Skunk___Spray_891": "Skunk",
}


def resolved_for(bvh_path, anim):
    skeleton = STEM_TO_SKELETON[Path(bvh_path).stem]
    manifest = SkeletonManifest.load(MANIFEST_DIR / f"{skeleton}.yaml")
    return resolve(manifest, anim.names)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `docker compose run --rm test pytest tests/ingest/test_align.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'poseydon.ingest'`. The
fixture-based tests will additionally fail until Task 5 ships the manifests; that is
expected and they are re-run at the end of Task 5.

- [ ] **Step 3: Write the implementation**

Create empty `src/poseydon/ingest/__init__.py`, then `src/poseydon/ingest/align.py`:

```python
"""Canonical alignment: orientation, centring, scale, ground.

The order below is the reference's ``process_anim`` order and is part of the
contract: rotate, centre XZ, scale, ground. Reordering changes the result.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from poseydon.core.anim import Anim
from poseydon.core.rotations import (
    euler_to_quat,
    quat_apply,
    quat_between,
    quat_mul,
)
from poseydon.core.skeleton import HML_MEAN_BONE_LENGTH, ResolvedSkeleton

_WORLD_UP = np.array([0.0, 1.0, 0.0])
_TARGET_FORWARD = np.array([0.0, 0.0, 1.0])


@dataclass(frozen=True)
class AlignmentParams:
    """What alignment did, so it can be replayed or inverted."""

    rotation: np.ndarray
    root_xz: np.ndarray
    scale_factor: float
    ground_height: float


def facing_quat(
    positions: np.ndarray,
    facing_indices: tuple[tuple[int, int], ...],
    extra_yaw_deg: float = 0.0,
) -> np.ndarray:
    """Rotation taking the character's forward axis onto +Z, from frame 0.

    ``across`` is the normalized sum of ``(right - left)`` over the facing pairs;
    ``forward = cross(world_up, across)``.
    """
    first = positions[0]
    across = np.zeros(3)
    for right, left in facing_indices:
        across = across + (first[right] - first[left])

    norm = np.linalg.norm(across)
    if norm < 1e-8:
        raise ValueError(
            "facing joints are coincident in the first frame, so the forward "
            "direction is undefined; check the manifest's facing pairs"
        )
    across = across / norm

    forward = np.cross(_WORLD_UP, across)
    forward_norm = np.linalg.norm(forward)
    if forward_norm < 1e-8:
        raise ValueError(
            "the across-body axis is parallel to world up, so the forward "
            "direction is undefined; check the manifest's facing pairs"
        )
    forward = forward / forward_norm

    rotation = quat_between(forward, _TARGET_FORWARD)
    if extra_yaw_deg:
        yaw = euler_to_quat(np.array([0.0, extra_yaw_deg, 0.0]), "ZYX")
        rotation = quat_mul(yaw, rotation)
    return rotation


def rotate_to_face_z(anim: Anim, rotation: np.ndarray) -> Anim:
    """Apply a global rotation by composing it onto the root only."""
    rotations = anim.rotations.copy()
    rotations[:, 0] = quat_mul(np.broadcast_to(rotation, rotations[:, 0].shape), rotations[:, 0])
    return Anim(
        rotations=rotations,
        root_pos=quat_apply(rotation, anim.root_pos),
        offsets=anim.offsets,
        parents=anim.parents,
        names=anim.names,
        fps=anim.fps,
    )


def move_xz_to_origin(anim: Anim) -> tuple[Anim, np.ndarray]:
    """Translate so the root sits at XZ origin on the first frame."""
    root_xz = anim.root_pos[0] * np.array([1.0, 0.0, 1.0])
    return (
        Anim(
            rotations=anim.rotations,
            root_pos=anim.root_pos - root_xz,
            offsets=anim.offsets,
            parents=anim.parents,
            names=anim.names,
            fps=anim.fps,
        ),
        root_xz,
    )


def scale_to_mean_bone_length(anim: Anim, target: float) -> tuple[Anim, float]:
    """Uniformly scale so the MEAN bone length equals ``target``."""
    lengths = np.linalg.norm(anim.offsets[1:], axis=-1)
    mean_length = float(lengths.mean())
    if mean_length < 1e-12:
        raise ValueError("skeleton has zero mean bone length; cannot scale")
    factor = target / mean_length
    return (
        Anim(
            rotations=anim.rotations,
            root_pos=anim.root_pos * factor,
            offsets=anim.offsets * factor,
            parents=anim.parents,
            names=anim.names,
            fps=anim.fps,
        ),
        factor,
    )


def put_on_ground(anim: Anim) -> tuple[Anim, float]:
    """Translate in Y so the lowest joint over the whole clip sits at y = 0."""
    ground_height = float(anim.global_positions()[..., 1].min())
    shift = np.array([0.0, ground_height, 0.0])
    return (
        Anim(
            rotations=anim.rotations,
            root_pos=anim.root_pos - shift,
            offsets=anim.offsets,
            parents=anim.parents,
            names=anim.names,
            fps=anim.fps,
        ),
        ground_height,
    )


def align(
    anim: Anim,
    resolved: ResolvedSkeleton,
    target_bone_length: float = HML_MEAN_BONE_LENGTH,
) -> tuple[Anim, AlignmentParams]:
    """Rotate to face +Z, centre XZ, scale, then ground. Order is contractual."""
    rotation = facing_quat(
        anim.global_positions(),
        resolved.facing_indices,
        resolved.manifest.extra_yaw_deg,
    )
    rotated = rotate_to_face_z(anim, rotation)
    centred, root_xz = move_xz_to_origin(rotated)
    scaled, factor = scale_to_mean_bone_length(centred, target_bone_length)
    grounded, ground_height = put_on_ground(scaled)

    return grounded, AlignmentParams(
        rotation=rotation,
        root_xz=root_xz,
        scale_factor=factor,
        ground_height=ground_height,
    )
```

- [ ] **Step 4: Run the toy tests**

Run: `docker compose run --rm test pytest tests/ingest/test_align.py -v -k "not fixture"`
Expected: the toy tests PASS. Fixture-based tests still fail pending Task 5's manifests.

- [ ] **Step 5: Commit**

```bash
git add src/poseydon/ingest tests/ingest
git commit -m "feat(ingest): canonical alignment - orientation, centring, scale, ground"
```

---

### Task 4: Corpus index

**Files:**
- Create: `src/poseydon/ingest/index.py`
- Test: `tests/ingest/test_index.py`

**Interfaces:**
- Consumes: nothing from this plan
- Produces:
  - `action_slug(stem: str) -> str`
  - `clip_id(skeleton: str, action: str) -> str`
  - `ClipRecord` frozen dataclass: `clip_id`, `skeleton`, `action`, `split`, `n_frames`, `fps`, `path`, `tags`
  - `CorpusIndex` class: `.records`, `.add(record)`, `.save(path)`, `.load(path)` (classmethod),
    `.query(skeleton=None, action=None, split=None, tag=None) -> list[ClipRecord]`,
    `.skeletons() -> list[str]`, `.__len__`

- [ ] **Step 1: Write the failing tests**

`tests/ingest/test_index.py`:

```python
import pytest

from poseydon.ingest.index import ClipRecord, CorpusIndex, action_slug, clip_id


def record(clip="Goat__head_butt", skeleton="Goat", action="head_butt", split="train"):
    return ClipRecord(
        clip_id=clip,
        skeleton=skeleton,
        action=action,
        split=split,
        n_frames=79,
        fps=24.0,
        path=f"aligned/{clip}.npz",
        tags=("quadruped",),
    )


@pytest.mark.parametrize(
    ("stem", "expected"),
    [
        ("Goat___HeadButt_395", "goat_headbutt_395"),
        ("Flamingo_Flamingo_OneLEgBEnt_353", "flamingo_flamingo_onelegbent_353"),
        ("Male Sitting Pose", "male_sitting_pose"),
        ("__leading__", "leading"),
        ("a---b", "a_b"),
    ],
)
def test_action_slug(stem, expected):
    assert action_slug(stem) == expected


def test_clip_id_uses_double_underscore_separator():
    assert clip_id("Goblin_m", "walking") == "Goblin_m__walking"


def test_clip_id_rejects_reserved_separator_in_skeleton():
    with pytest.raises(ValueError, match="__"):
        clip_id("Bad__Name", "walking")


def test_round_trip_jsonl(tmp_path):
    index = CorpusIndex()
    index.add(record())
    index.add(record(clip="Coyote__attack", skeleton="Coyote", action="attack"))
    path = tmp_path / "index.jsonl"
    index.save(path)

    back = CorpusIndex.load(path)
    assert len(back) == 2
    assert back.records[0] == index.records[0]
    assert back.records[1].skeleton == "Coyote"


def test_saved_file_is_one_json_object_per_line(tmp_path):
    index = CorpusIndex()
    index.add(record())
    index.add(record(clip="B__b", skeleton="B", action="b"))
    path = tmp_path / "index.jsonl"
    index.save(path)
    lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
    assert len(lines) == 2
    assert all(ln.startswith("{") and ln.endswith("}") for ln in lines)


def test_duplicate_clip_id_is_rejected():
    index = CorpusIndex()
    index.add(record())
    with pytest.raises(ValueError, match="duplicate"):
        index.add(record())


def test_query_filters_independently():
    index = CorpusIndex()
    index.add(record())
    index.add(record(clip="Coyote__attack", skeleton="Coyote", action="attack"))
    index.add(record(clip="Goat__walk", action="walk", split="test"))

    assert len(index.query(skeleton="Goat")) == 2
    assert len(index.query(split="test")) == 1
    assert len(index.query(action="attack")) == 1
    assert len(index.query(skeleton="Goat", split="train")) == 1
    assert len(index.query(tag="quadruped")) == 3
    assert len(index.query(tag="flying")) == 0


def test_retargeting_pairs_are_a_query_not_a_file():
    # Same action across different skeletons -- what the reference materialized
    # into txt files with a dedicated build script.
    index = CorpusIndex()
    index.add(record(clip="Goat__walk", action="walk"))
    index.add(record(clip="Coyote__walk", skeleton="Coyote", action="walk"))
    index.add(record(clip="Goat__attack", action="attack"))

    walkers = index.query(action="walk")
    assert {r.skeleton for r in walkers} == {"Goat", "Coyote"}


def test_skeletons_are_sorted_and_unique():
    index = CorpusIndex()
    index.add(record(clip="Z__a", skeleton="Zebra"))
    index.add(record(clip="A__a", skeleton="Ant"))
    index.add(record(clip="A__b", skeleton="Ant", action="b"))
    assert index.skeletons() == ["Ant", "Zebra"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `docker compose run --rm test pytest tests/ingest/test_index.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'poseydon.ingest.index'`

- [ ] **Step 3: Write the implementation**

`src/poseydon/ingest/index.py`:

```python
"""The corpus index.

One row per clip. This replaces filename parsing (the reference re-implements it
in eight places), the per-character filelist tree, the hardcoded taxonomy subset
lists, and the script that materialized retargeting pairs into txt files.

JSONL: append-only, diffable, readable without a dependency, and small enough at
corpus scale that a columnar format would be premature.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

SEPARATOR = "__"
_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def action_slug(stem: str) -> str:
    """Lowercase, collapse non-alphanumeric runs to `_`, trim the ends."""
    return _NON_ALNUM.sub("_", stem.lower()).strip("_")


def clip_id(skeleton: str, action: str) -> str:
    """Deterministic clip identity: ``{skeleton}__{action}``."""
    if SEPARATOR in skeleton:
        raise ValueError(
            f"skeleton name `{skeleton}` contains the reserved separator "
            f"`{SEPARATOR}`, which divides skeleton from action in clip ids"
        )
    return f"{skeleton}{SEPARATOR}{action}"


@dataclass(frozen=True)
class ClipRecord:
    clip_id: str
    skeleton: str
    action: str
    split: str
    n_frames: int
    fps: float
    path: str
    tags: tuple[str, ...] = ()


@dataclass
class CorpusIndex:
    records: list[ClipRecord] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.records)

    def add(self, record: ClipRecord) -> None:
        if any(existing.clip_id == record.clip_id for existing in self.records):
            raise ValueError(f"duplicate clip_id `{record.clip_id}`")
        self.records.append(record)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as handle:
            for record in self.records:
                payload = asdict(record)
                payload["tags"] = list(record.tags)
                handle.write(json.dumps(payload, sort_keys=True) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> "CorpusIndex":
        index = cls()
        for line in Path(path).read_text().splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            payload["tags"] = tuple(payload.get("tags", ()))
            index.add(ClipRecord(**payload))
        return index

    def query(
        self,
        *,
        skeleton: str | None = None,
        action: str | None = None,
        split: str | None = None,
        tag: str | None = None,
    ) -> list[ClipRecord]:
        return [
            record
            for record in self.records
            if (skeleton is None or record.skeleton == skeleton)
            and (action is None or record.action == action)
            and (split is None or record.split == split)
            and (tag is None or tag in record.tags)
        ]

    def skeletons(self) -> list[str]:
        return sorted({record.skeleton for record in self.records})
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `docker compose run --rm test pytest tests/ingest/test_index.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/poseydon/ingest/index.py tests/ingest/test_index.py
git commit -m "feat(ingest): JSONL corpus index replacing filename parsing"
```

---

### Task 5: Truebones manifests, pipeline and CLI

**Files:**
- Create: `data/truebones/skeletons/{BrownBear,Coyote,Crab,Flamingo,Goat,Scorpion,Skunk}.yaml`
- Create: `src/poseydon/ingest/pipeline.py`
- Create: `src/poseydon/cli.py`
- Modify: `pyproject.toml` (add the console script)
- Test: `tests/ingest/test_pipeline.py`

**Interfaces:**
- Consumes: everything from Tasks 1-4 plus `load_bvh`, `Anim`
- Produces:
  - `IngestResult` frozen dataclass: `index: CorpusIndex`, `written: list[Path]`, `skipped: list[tuple[Path, str]]`
  - `ingest_clip(bvh_path, manifest, out_dir, split="train") -> ClipRecord`
  - `ingest_corpus(bvh_paths, manifest_dir, out_dir, split="train", skeleton_of=None) -> IngestResult`
  - `main(argv=None) -> int` in `cli.py`

**Facing joints, verified against the reference's `FACE_JOINTS` indices resolved to
names in these exact files.** Order within each manifest is hips then shoulders.

- [ ] **Step 1: Write the seven manifests**

All seven share the same contact thresholds and scale target. `max_speed` is
`sqrt(0.002) ~= 0.0447`, the reference's squared threshold expressed as a real speed.
`fps: null` means "keep the source rate" — these files are 24 fps and must not be
resampled.

`data/truebones/skeletons/BrownBear.yaml`:

```yaml
skeleton: BrownBear
facing:
  hips:      {right: Bip01_R_Thigh,    left: Bip01_L_Thigh}
  shoulders: {right: Bip01_R_UpperArm, left: Bip01_L_UpperArm}
  extra_yaw_deg: 0
fps: null
scale: {mean_bone_length: 0.20921428571428569}
contact: {max_height: 0.30, max_speed: 0.0447}
tags: [quadruped, mammal]
```

`data/truebones/skeletons/Coyote.yaml`:

```yaml
skeleton: Coyote
facing:
  hips:      {right: Bip01_R_Thigh,    left: Bip01_L_Thigh}
  shoulders: {right: Bip01_R_Clavicle, left: Bip01_L_Clavicle}
  extra_yaw_deg: 0
fps: null
scale: {mean_bone_length: 0.20921428571428569}
contact: {max_height: 0.30, max_speed: 0.0447}
tags: [quadruped, mammal]
```

`data/truebones/skeletons/Crab.yaml`:

```yaml
skeleton: Crab
facing:
  hips:      {right: BN_Leg_R_11, left: BN_Leg_L_11}
  shoulders: {right: BN_Arm_R_02, left: BN_Arm_L_02}
  extra_yaw_deg: 0
fps: null
scale: {mean_bone_length: 0.20921428571428569}
contact: {max_height: 0.30, max_speed: 0.0447}
tags: [milliped]
```

`data/truebones/skeletons/Flamingo.yaml`:

```yaml
skeleton: Flamingo
facing:
  hips:      {right: Bip01_R_Thigh,     left: Bip01_L_Thigh}
  shoulders: {right: BN_Forearm_R_02,   left: BN_Forearm_L_02}
  extra_yaw_deg: 0
fps: null
scale: {mean_bone_length: 0.20921428571428569}
contact: {max_height: 0.30, max_speed: 0.0447}
tags: [biped, bird]
```

`data/truebones/skeletons/Goat.yaml`:

```yaml
skeleton: Goat
facing:
  hips:      {right: Bip01_R_Thigh,    left: Bip01_L_Thigh}
  shoulders: {right: Bip01_R_UpperArm, left: Bip01_L_UpperArm}
  extra_yaw_deg: 0
fps: null
scale: {mean_bone_length: 0.20921428571428569}
contact: {max_height: 0.30, max_speed: 0.0447}
tags: [quadruped, mammal]
```

`data/truebones/skeletons/Scorpion.yaml`:

```yaml
skeleton: Scorpion
facing:
  hips:      {right: Bip01_R_Thigh_4,  left: Bip01_L_Thigh1_4}
  shoulders: {right: Bip01_R_Forearm,  left: Bip01_L_Forearm}
  extra_yaw_deg: 0
fps: null
scale: {mean_bone_length: 0.20921428571428569}
contact: {max_height: 0.30, max_speed: 0.0447}
tags: [milliped]
```

`data/truebones/skeletons/Skunk.yaml`:

```yaml
skeleton: Skunk
facing:
  hips:      {right: Bip01_R_Thigh,    left: Bip01_L_Thigh}
  shoulders: {right: Bip01_R_UpperArm, left: Bip01_L_UpperArm}
  extra_yaw_deg: 0
fps: null
scale: {mean_bone_length: 0.20921428571428569}
contact: {max_height: 0.30, max_speed: 0.0447}
tags: [quadruped, mammal]
```

- [ ] **Step 2: Write the failing pipeline tests**

`tests/ingest/test_pipeline.py`:

```python
import numpy as np
import pytest

from poseydon.core.anim import Anim
from poseydon.core.skeleton import ManifestError, SkeletonManifest
from poseydon.ingest.index import CorpusIndex
from poseydon.ingest.pipeline import ingest_clip, ingest_corpus

from tests.ingest.manifest_helper import MANIFEST_DIR, STEM_TO_SKELETON


def test_ingest_clip_writes_aligned_anim(bvh_fixture, tmp_path):
    skeleton = STEM_TO_SKELETON[bvh_fixture.stem]
    manifest = SkeletonManifest.load(MANIFEST_DIR / f"{skeleton}.yaml")
    record = ingest_clip(bvh_fixture, manifest, tmp_path)

    written = tmp_path / record.path
    assert written.is_file()

    anim = Anim.load(written)
    assert anim.n_frames == record.n_frames
    assert anim.global_positions()[..., 1].min() == pytest.approx(0.0, abs=1e-9)
    assert record.skeleton == skeleton
    assert record.clip_id.startswith(f"{skeleton}__")


def test_ingest_never_chunks(bvh_fixture, tmp_path):
    # One BVH in, exactly one Anim out, at full length.
    skeleton = STEM_TO_SKELETON[bvh_fixture.stem]
    manifest = SkeletonManifest.load(MANIFEST_DIR / f"{skeleton}.yaml")
    record = ingest_clip(bvh_fixture, manifest, tmp_path)

    from poseydon.io.bvh import load_bvh  # noqa: PLC0415

    assert record.n_frames == load_bvh(bvh_fixture).n_frames
    assert len(list(tmp_path.rglob("*.npz"))) == 1


def test_ingest_is_idempotent(bvh_fixture, tmp_path):
    skeleton = STEM_TO_SKELETON[bvh_fixture.stem]
    manifest = SkeletonManifest.load(MANIFEST_DIR / f"{skeleton}.yaml")

    first = ingest_clip(bvh_fixture, manifest, tmp_path)
    second = ingest_clip(bvh_fixture, manifest, tmp_path)

    assert first.clip_id == second.clip_id
    assert first.path == second.path
    assert len(list(tmp_path.rglob("*.npz"))) == 1


def test_ingest_corpus_builds_index(truebones_dir, tmp_path):
    paths = sorted(truebones_dir.glob("*.bvh"))
    result = ingest_corpus(
        paths,
        MANIFEST_DIR,
        tmp_path,
        skeleton_of=lambda p: STEM_TO_SKELETON[p.stem],
    )

    assert len(result.index) == 7
    assert result.skipped == []
    assert result.index.skeletons() == sorted(set(STEM_TO_SKELETON.values()))

    index_path = tmp_path / "index.jsonl"
    result.index.save(index_path)
    assert len(CorpusIndex.load(index_path)) == 7


def test_ingest_corpus_reports_failures_without_aborting(truebones_dir, tmp_path):
    paths = sorted(truebones_dir.glob("*.bvh"))
    broken = tmp_path / "broken.bvh"
    broken.write_text("HIERARCHY\nROOT a\n{\nOFFSET 0 0 0\n}\n")

    result = ingest_corpus(
        [*paths, broken],
        MANIFEST_DIR,
        tmp_path / "out",
        skeleton_of=lambda p: STEM_TO_SKELETON.get(p.stem, "Goat"),
    )

    assert len(result.index) == 7
    assert len(result.skipped) == 1
    assert result.skipped[0][0] == broken


def test_manifest_facing_joints_exist_in_every_fixture(bvh_fixture):
    from poseydon.core.skeleton import resolve  # noqa: PLC0415
    from poseydon.io.bvh import load_bvh  # noqa: PLC0415

    skeleton = STEM_TO_SKELETON[bvh_fixture.stem]
    manifest = SkeletonManifest.load(MANIFEST_DIR / f"{skeleton}.yaml")
    resolved = resolve(manifest, load_bvh(bvh_fixture).names)
    assert len(resolved.facing_indices) == 2


def test_unknown_joint_name_fails_with_suggestion(truebones_dir, tmp_path):
    bad = tmp_path / "Bad.yaml"
    bad.write_text(
        "skeleton: Bad\n"
        "facing:\n"
        "  hips:      {right: Bip01_R_Thig, left: Bip01_L_Thigh}\n"
        "  shoulders: {right: Bip01_R_UpperArm, left: Bip01_L_UpperArm}\n"
        "scale: {mean_bone_length: 0.2}\n"
        "contact: {max_height: 0.3, max_speed: 0.045}\n"
    )
    manifest = SkeletonManifest.load(bad)
    with pytest.raises(ManifestError, match="Bip01_R_Thigh"):
        ingest_clip(truebones_dir / "Goat___HeadButt_395.bvh", manifest, tmp_path)


def test_rejects_fps_mismatch(truebones_dir, tmp_path):
    text = (MANIFEST_DIR / "Goat.yaml").read_text().replace("fps: null", "fps: 20")
    path = tmp_path / "Goat20.yaml"
    path.write_text(text.replace("skeleton: Goat", "skeleton: Goat20"))
    manifest = SkeletonManifest.load(path)

    with pytest.raises(ValueError, match="resampl"):
        ingest_clip(truebones_dir / "Goat___HeadButt_395.bvh", manifest, tmp_path)
```

Add `tests/__init__.py`, `tests/ingest/__init__.py` (empty) so
`from tests.ingest.manifest_helper import ...` resolves, and add `"."` to
`pythonpath` in `pyproject.toml`:

```toml
pythonpath = ["src", "."]
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `docker compose run --rm test pytest tests/ingest/test_pipeline.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'poseydon.ingest.pipeline'`

- [ ] **Step 4: Write the pipeline**

`src/poseydon/ingest/pipeline.py`:

```python
"""BVH corpus to aligned Anim files plus a corpus index."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path

from poseydon.core.skeleton import SkeletonManifest, resolve
from poseydon.ingest.align import align
from poseydon.ingest.index import ClipRecord, CorpusIndex, action_slug, clip_id
from poseydon.io.bvh import load_bvh

ALIGNED_DIRNAME = "aligned"


@dataclass
class IngestResult:
    index: CorpusIndex = field(default_factory=CorpusIndex)
    written: list[Path] = field(default_factory=list)
    skipped: list[tuple[Path, str]] = field(default_factory=list)


def ingest_clip(
    bvh_path: str | Path,
    manifest: SkeletonManifest,
    out_dir: str | Path,
    split: str = "train",
) -> ClipRecord:
    """Align one BVH and write one full-length Anim. Never chunks."""
    bvh_path = Path(bvh_path)
    out_dir = Path(out_dir)

    anim = load_bvh(bvh_path)

    if manifest.fps is not None and abs(manifest.fps - anim.fps) > 1e-6:
        raise ValueError(
            f"{bvh_path.name}: manifest requests {manifest.fps} fps but the source "
            f"is {anim.fps:.4f} fps, and resampling is not implemented. Set "
            "`fps: null` to keep the source rate."
        )

    resolved = resolve(manifest, anim.names)
    aligned, _params = align(anim, resolved)

    identifier = clip_id(manifest.name, action_slug(bvh_path.stem))
    relative = f"{ALIGNED_DIRNAME}/{identifier}.npz"
    destination = out_dir / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    aligned.save(destination)

    return ClipRecord(
        clip_id=identifier,
        skeleton=manifest.name,
        action=action_slug(bvh_path.stem),
        split=split,
        n_frames=aligned.n_frames,
        fps=aligned.fps,
        path=relative,
        tags=manifest.tags,
    )


def ingest_corpus(
    bvh_paths: Iterable[str | Path],
    manifest_dir: str | Path,
    out_dir: str | Path,
    split: str = "train",
    skeleton_of: Callable[[Path], str] | None = None,
) -> IngestResult:
    """Ingest many BVHs, collecting failures instead of aborting the run."""
    manifest_dir = Path(manifest_dir)
    out_dir = Path(out_dir)
    result = IngestResult()
    cache: dict[str, SkeletonManifest] = {}

    for raw in bvh_paths:
        path = Path(raw)
        try:
            skeleton = skeleton_of(path) if skeleton_of else path.stem.split("__")[0]
            if skeleton not in cache:
                cache[skeleton] = SkeletonManifest.load(manifest_dir / f"{skeleton}.yaml")
            record = ingest_clip(path, cache[skeleton], out_dir, split=split)
        except Exception as error:  # noqa: BLE001 - one bad file must not stop a corpus
            result.skipped.append((path, f"{type(error).__name__}: {error}"))
            continue
        result.index.add(record)
        result.written.append(out_dir / record.path)

    return result
```

- [ ] **Step 5: Write the CLI**

`src/poseydon/cli.py`:

```python
"""poseydon command line."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from poseydon.ingest.pipeline import ingest_corpus


def _ingest(args: argparse.Namespace) -> int:
    paths = sorted(Path(args.bvh_dir).rglob("*.bvh"))
    if not paths:
        print(f"no .bvh files found under {args.bvh_dir}", file=sys.stderr)
        return 1

    result = ingest_corpus(paths, args.manifests, args.out, split=args.split)
    index_path = Path(args.out) / "index.jsonl"
    result.index.save(index_path)

    print(f"ingested {len(result.index)} clips -> {index_path}")
    for path, reason in result.skipped:
        print(f"  skipped {path.name}: {reason}", file=sys.stderr)
    return 0 if result.index.records else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="poseydon")
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest", help="align a BVH corpus and build its index")
    ingest.add_argument("bvh_dir", help="directory of source .bvh files")
    ingest.add_argument("--manifests", required=True, help="directory of skeleton manifests")
    ingest.add_argument("--out", required=True, help="output directory")
    ingest.add_argument("--split", default="train")
    ingest.set_defaults(func=_ingest)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
```

Add to `pyproject.toml`:

```toml
[project.scripts]
poseydon = "poseydon.cli:main"
```

- [ ] **Step 6: Run the whole suite**

Run: `docker compose run --rm test pytest -q`
Expected: all PASS, including the alignment tests deferred from Task 3.

- [ ] **Step 7: Lint**

Run: `docker compose run --rm test ruff check src tests`
Expected: no findings. Fix any that appear.

- [ ] **Step 8: Commit**

```bash
git add data src/poseydon/ingest/pipeline.py src/poseydon/cli.py pyproject.toml tests
git commit -m "feat(ingest): Truebones manifests, ingest pipeline and CLI"
```

---

## Follow-on plans

Next: **Solver** — a torch-based batched GPU `GradientIK` with registry-named objective
terms (position, bone length, joint limits, smoothness), plus the IO and IK benchmark
against the `Motion` baseline. That plan introduces torch, which is why it is separate:
it adds roughly a gigabyte to the image and would slow every test cycle in this plan.

Then: **Features** — the composable feature layer, where T-pose-relative rotations land
and where the golden `.npy` parity gate finally runs.

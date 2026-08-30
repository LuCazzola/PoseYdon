"""Skeleton manifests.

Everything the reference kept as Python literals in ``param_utils.py`` --
per-species face-joint indices, contact thresholds, taxonomy -- lives here as
per-skeleton YAML instead. Joints are referenced by NAME, never by index, so a
re-exported rig fails loudly rather than silently mirroring the character.
"""

from __future__ import annotations

import difflib
from collections.abc import Sequence
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
    def load(cls, path: str | Path) -> SkeletonManifest:
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
        key = min(unknown)
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
    stripped = tuple(n.removeprefix(prefix) for n in names)
    if len(set(stripped)) != len(stripped):
        raise ManifestError(
            f"stripping prefix `{prefix}` makes joint names collide; "
            "remove the prefix from the manifest or rename the joints"
        )
    return stripped

"""Skeleton manifests.

Everything the reference kept as Python literals in ``param_utils.py`` --
per-species face-joint indices, contact thresholds, taxonomy -- lives here as
per-skeleton YAML instead. Joints are referenced by NAME, never by index, so a
re-exported rig fails loudly rather than silently mirroring the character.
"""

from __future__ import annotations

import difflib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from poseydon.core.authority import AuthorityPolicy, SourceArtifacts

# Mean of the 21 SMPL bone lengths used by the reference as its scale target.
# The reference calls this HML_AVG_BONELEN; its docstring claims "longest
# armature" but the code takes the mean, and the code is what ran.
HML_MEAN_BONE_LENGTH = 0.20921428571428569

_KNOWN_KEYS = frozenset(
    {
        "base",
        "skeleton",
        "rest_pose",
        "up_axis",
        "facing",
        "foot_joints",
        "fps",
        "contact",
        "promote_root",
        "ground",
        "tags",
        "authority",
        "artifacts",
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
    #: Foot-contact thresholds; a dataset supplies them (`labels.contact`).
    contact: ContactParams | None
    source: Path
    #: Filename of the clip whose first frame is this rig's rest pose, relative
    #: to the rig's RAW source directory -- e.g. "__IdleLoop.bvh". A filename
    #: rather than a path because consumers resolve it in different domains:
    #: stage 1 wants `source/<Rig>/<file>`, everything downstream wants the
    #: prepared clip `clips/<Rig>/<action>.bvh`.
    rest_pose: str | None = None
    #: The source's up axis, ``"+Y"`` to ``"-Z"``. A BVH does not record it,
    #: so a rig authored Z-up has to say so; stage 1 turns it to +Y up first.
    up_axis: str = "+Y"
    extra_yaw_deg: float = 0.0
    foot_joints: tuple[str, ...] = ()
    fps: float | None = None
    #: The body joint to make the root when the file roots at a ground locator
    #: (`poseydon.build.prepare.PromoteRoot`).
    promote_root: str | None = None
    #: This rig's own ground mode, overriding the dataset's (`PutOnGround`).
    ground: str | None = None
    tags: tuple[str, ...] = ()
    authority: AuthorityPolicy = field(default_factory=AuthorityPolicy)
    artifacts: SourceArtifacts = field(default_factory=SourceArtifacts)

    @classmethod
    def load(cls, path: str | Path, defaults: Mapping | None = None) -> SkeletonManifest:
        """``defaults`` fills keys the manifest leaves out; a dataset passes its
        ``labels.contact`` as ``{"contact": ...}`` (`contact_defaults`)."""
        path = Path(path).resolve()
        data = _load_with_base(path, seen=[])
        for key, value in (defaults or {}).items():
            data.setdefault(key, value)
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

    contact_raw = data.get("contact")
    contact = None
    if contact_raw is not None:
        for key in ("max_height", "max_speed"):
            if key not in contact_raw:
                raise ManifestError(f"{path}: contact is missing `{key}`")
        contact = ContactParams(
            max_height=float(contact_raw["max_height"]),
            max_speed=float(contact_raw["max_speed"]),
        )

    rest_pose = data.get("rest_pose")

    ground = data.get("ground")
    if ground is not None and str(ground) not in ("joints", "mesh"):
        raise ManifestError(f"{path}: `ground` must be `joints` or `mesh`, got `{ground}`")

    up_raw = str(data.get("up_axis", "+Y")).strip().upper()
    up_axis = up_raw if up_raw[:1] in "+-" else f"+{up_raw}"
    if up_axis[1:] not in ("X", "Y", "Z") or len(up_axis) != 2:
        raise ManifestError(f"{path}: `up_axis` must be X, Y or Z with an optional sign, got `{up_raw}`")

    authority_raw = data.get("authority")
    artifacts_raw = data.get("artifacts")
    if authority_raw is None and artifacts_raw is None:
        authority = AuthorityPolicy()
        artifacts = SourceArtifacts(
            bvh=None if rest_pose is None else Path(str(rest_pose))
        )
    else:
        if not isinstance(authority_raw, dict):
            raise ManifestError(f"{path}: `authority` must be a mapping")
        if not isinstance(artifacts_raw, dict):
            raise ManifestError(f"{path}: `artifacts` must be a mapping")
        try:
            authority = AuthorityPolicy(**authority_raw)
        except (TypeError, ValueError) as error:
            raise ManifestError(f"{path}: invalid authority: {error}") from error
        unknown = set(artifacts_raw) - {"bvh", "fbx", "textures"}
        if unknown:
            raise ManifestError(f"{path}: unknown artifacts key `{min(unknown)}`")
        artifacts = SourceArtifacts(**{
            key: None if value is None else Path(str(value))
            for key, value in artifacts_raw.items()
        })
        if getattr(artifacts, authority.motion) is None:
            raise ManifestError(
                f"{path}: motion authority `{authority.motion}` requires an "
                f"artifacts.{authority.motion} source"
            )
        if getattr(artifacts, authority.skeleton) is None:
            raise ManifestError(
                f"{path}: skeleton authority `{authority.skeleton}` requires an "
                f"artifacts.{authority.skeleton} source"
            )
        if authority.surface == "fbx" and artifacts.fbx is None:
            raise ManifestError(
                f"{path}: surface authority `fbx` requires an artifacts.fbx source"
            )

    return SkeletonManifest(
        name=name,
        facing=tuple(pairs),
        contact=contact,
        source=source,
        rest_pose=None if rest_pose is None else str(rest_pose),
        up_axis=up_axis,
        extra_yaw_deg=extra_yaw,
        foot_joints=tuple(str(j) for j in (data.get("foot_joints") or ())),
        fps=None if data.get("fps") is None else float(data["fps"]),
        promote_root=None if data.get("promote_root") is None else str(data["promote_root"]),
        ground=None if ground is None else str(ground),
        tags=tuple(str(t) for t in (data.get("tags") or ())),
        authority=authority,
        artifacts=artifacts,
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


def contact_defaults(contact: Mapping | None) -> dict:
    """A dataset's ``labels.contact`` as `SkeletonManifest.load` defaults."""
    if contact is None:
        return {}
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(contact):
            contact = OmegaConf.to_container(contact, resolve=True)
    except ImportError:  # pragma: no cover
        pass
    return {"contact": dict(contact)}

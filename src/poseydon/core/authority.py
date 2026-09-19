"""Declared source authority and export capability for one character."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Literal, Mapping, TYPE_CHECKING

if TYPE_CHECKING:
    from poseydon.core.skeleton import SkeletonManifest


@dataclass(frozen=True)
class AuthorityPolicy:
    skeleton: Literal["bvh", "fbx"] = "bvh"
    motion: Literal["bvh", "fbx"] = "bvh"
    surface: Literal["fbx", "none"] = "none"
    auxiliary_joints: Literal["exclude", "retain", "bake"] = "exclude"

    def __post_init__(self) -> None:
        allowed = {
            "skeleton": {"bvh", "fbx"},
            "motion": {"bvh", "fbx"},
            "surface": {"fbx", "none"},
            "auxiliary_joints": {"exclude", "retain", "bake"},
        }
        for field, values in allowed.items():
            value = getattr(self, field)
            if value not in values:
                raise ValueError(
                    f"authority.{field} must be one of {sorted(values)}, got {value!r}"
                )


@dataclass(frozen=True)
class SourceArtifacts:
    bvh: Path | None = None
    fbx: Path | None = None
    textures: Path | None = None


class ArtifactUnavailableError(RuntimeError):
    def __init__(self, dataset: str, rig: str, representation: str, format: str, reason: str):
        self.dataset, self.rig = dataset, rig
        self.representation, self.format, self.reason = representation, format, reason
        super().__init__(f"{dataset}/{rig}: {representation} {format} unavailable: {reason}")


@dataclass(frozen=True)
class ArtifactAvailability:
    prepared_bvh: bool
    prepared_fbx: bool
    presentation_fbx: bool
    original_bvh: bool
    original_fbx: bool
    reasons: Mapping[str, str]

    @classmethod
    def from_manifest(cls, manifest: SkeletonManifest) -> ArtifactAvailability:
        has_surface = manifest.authority.surface == "fbx" and manifest.artifacts.fbx is not None
        reasons = {}
        if not has_surface:
            for name in ("prepared_fbx", "presentation_fbx", "original_fbx"):
                reasons[name] = "source_surface_missing"
        return cls(True, has_surface, has_surface, True, has_surface, MappingProxyType(reasons))

    def require(self, dataset: str, rig: str, representation: str, format: str) -> ArtifactUnavailableError | None:
        key = f"{representation}_{format}"
        if getattr(self, key):
            return None
        return ArtifactUnavailableError(dataset, rig, representation, format, self.reasons[key])

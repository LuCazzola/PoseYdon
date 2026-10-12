"""Declared source authority and export capability for one character."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


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


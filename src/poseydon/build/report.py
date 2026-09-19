"""Deterministic, machine-readable records of character preparation."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol, TypeAlias

import numpy as np

from poseydon.core.animation import Animation
from poseydon.core.authority import AuthorityPolicy

JsonValue: TypeAlias = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]


def _json(value: Any) -> JsonValue:
    if isinstance(value, np.ndarray):
        return _json(value.tolist())
    if isinstance(value, np.generic):
        return _json(value.item())
    if isinstance(value, Mapping):
        return {str(key): _json(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_json(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("preparation reports cannot contain non-finite numbers")
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(f"unsupported report value: {type(value).__name__}")


def topology_sha256(anim: Animation) -> str:
    payload = {"names": list(anim.names), "parents": anim.parents.tolist(), "offsets": anim.offsets.tolist()}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class OperationRecord:
    operation_id: str
    kind: str
    direction: str
    scope: str
    parameters: JsonValue
    input_topology_sha256: str
    output_topology_sha256: str


@dataclass(frozen=True)
class JointDecision:
    joint: str
    operation: str
    reason: str
    destination: str | None = None


@dataclass(frozen=True)
class SkinningDecision:
    mesh: str
    source_joint: str
    destination_joint: str | None
    affected_vertices: int
    transferred_weight: float


class AuditSink(Protocol):
    def record_transform(self, *, stage: str, direction: str, scope: str,
                         parameters: Mapping[str, Any], before: Animation,
                         after: Animation) -> str: ...


@dataclass
class PreparationReport:
    dataset: str
    rig: str
    policy: AuthorityPolicy
    operations: list[OperationRecord] = field(default_factory=list)
    joint_decisions: list[JointDecision] = field(default_factory=list)
    skinning_decisions: list[SkinningDecision] = field(default_factory=list)

    @classmethod
    def start(cls, dataset: str, rig: str, policy: AuthorityPolicy) -> "PreparationReport":
        return cls(dataset=dataset, rig=rig, policy=policy)

    def record_transform(self, *, stage: str, direction: str, scope: str,
                         parameters: Mapping[str, Any], before: Animation,
                         after: Animation) -> str:
        operation_id = f"{self.rig}/{direction}/{stage}/{len(self.operations):04d}"
        self.operations.append(OperationRecord(
            operation_id=operation_id, kind=stage, direction=direction, scope=scope,
            parameters=_json(parameters), input_topology_sha256=topology_sha256(before),
            output_topology_sha256=topology_sha256(after),
        ))
        return operation_id

    def semantic_dict(self) -> dict[str, JsonValue]:
        return _json({
            "dataset": self.dataset, "rig": self.rig, "policy": asdict(self.policy),
            "operations": [asdict(item) for item in self.operations],
            "joint_decisions": [asdict(item) for item in self.joint_decisions],
            "skinning_decisions": [asdict(item) for item in self.skinning_decisions],
        })

    @property
    def semantic_sha256(self) -> str:
        encoded = json.dumps(self.semantic_dict(), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.semantic_dict(), indent=2, sort_keys=True) + "\n")

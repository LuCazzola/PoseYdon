from __future__ import annotations

import json
import hashlib
import os
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from poseydon.core.animation import Animation

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class CharacterAssets:
    model: Path | None
    presentation: Path | None
    reason: str | None = None
    report: Path | None = None


@dataclass(frozen=True)
class CharacterAssetContext:
    rig: str
    raw_fbx: Path | None
    model_names: tuple[str, ...]
    model_parents: tuple[int, ...]
    removals: tuple = ()
    model_rest: Animation | None = None
    correspondence: object | None = None
    skinning_plan: object | None = None
    preparation_transform: object | None = None
    audit: object | None = None
    report_path: Path | None = None
    authority: object | None = None


def _bind_positions(scene, names: tuple[str, ...]) -> np.ndarray:
    from poseydon.io.fbx import _to_y_up
    return np.stack([_to_y_up(scene.joint_position(name, rest=True)) for name in names])


def _canonical_matrix(scene, context: CharacterAssetContext):
    from mathutils import Matrix, Vector
    from poseydon.io.fbx import TO_Y_UP, _similarity, _rest_positions
    if context.model_rest is None:
        return Matrix.Identity(4)
    model_names = tuple(context.model_names)
    source_names = tuple(scene.joint_names)
    missing = [name for name in model_names if name not in source_names]
    if missing:
        raise ValueError(f"raw FBX `{context.rig}` lacks model joints: {', '.join(missing)}")
    source = _bind_positions(scene, model_names)
    target = _rest_positions(context.model_rest)[
        [context.model_rest.names.index(name) for name in model_names]
    ]
    scale, rotation, offset = _similarity(source, target)
    linear_y = Matrix.Scale(scale, 3) @ Matrix([list(row) for row in rotation])
    linear_b = (TO_Y_UP.transposed() @ linear_y @ TO_Y_UP).to_4x4()
    translation_b = TO_Y_UP.transposed() @ Vector(tuple(float(x) for x in offset))
    return Matrix.Translation(translation_b) @ linear_b


def _write_atomic(scene: FBX, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists():
        temporary.unlink()
    scene.write(temporary)
    os.replace(temporary, path)


def _write_report(context: CharacterAssetContext, output: Path, assets: CharacterAssets) -> Path:
    report = context.report_path or output / "preparation-report.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    existing = json.loads(report.read_text()) if report.is_file() else {}
    asset_operations = [
        {
            "operation_id": f"{context.rig}/topology/asset",
            "kind": "topology_contract",
            "direction": "asset",
            "scope": "rig",
            "joint_names": list(context.model_names),
            "parents": list(context.model_parents),
        },
        {
            "operation_id": f"{context.rig}/correspondence/asset",
            "kind": "joint_correspondence",
            "direction": "asset",
            "scope": "rig",
            "source": None if context.raw_fbx is None else str(context.raw_fbx),
            "target_joint_count": len(context.model_names),
        },
        {
            "operation_id": f"{context.rig}/skinning/asset",
            "kind": "skinning_policy",
            "direction": "asset",
            "scope": "rig",
            "decision": "preserve" if context.skinning_plan is None else str(context.skinning_plan),
        },
    ]
    operations = list(existing.get("operations", ()))
    if operations:
        operations.extend(asset_operations)
        unique = {}
        for operation in operations:
            unique[operation["operation_id"]] = operation
        operations = list(unique.values())
    else:
        operations = asset_operations
        stages = ("rest_relative", "promote_root", "face_axis", "enforce_rigid", "centre_xz", "scale_to_mean_bone_length", "put_on_ground")
        operations.extend({
            "operation_id": f"{context.rig}/transform/{index:03d}",
            "kind": stage,
            "direction": "prepare",
            "scope": "rig" if stage != "face_axis" else "clip",
            "parameters": {},
        } for index, stage in enumerate(stages, start=len(operations)))
    joint_decisions = list(existing.get("joint_decisions", ()))
    if not joint_decisions:
        joint_decisions = [
            {"operation_id": f"{context.rig}/joint/{index:04d}", "joint": name, "operation": "retain", "reason": "model topology"}
            for index, name in enumerate(context.model_names)
        ]
    joint_decisions.extend({
        "operation_id": f"{context.rig}/joint-removal/{index:04d}",
        "joint": getattr(removal, "name", str(removal)),
        "operation": getattr(removal, "kind", "exclude"),
        "reason": getattr(removal, "reason", "declared policy"),
        "destination": getattr(removal, "recipient", None),
    } for index, removal in enumerate(context.removals))
    authority = context.authority if isinstance(context.authority, dict) else existing.get("authority", {"skeleton": "bvh", "motion": "bvh", "surface": "fbx", "auxiliary_joints": "bake"})
    payload = dict(existing)
    payload.update({
        "dataset": existing.get("dataset", "truebones"),
        "rig": context.rig,
        "authority": authority,
        "source_fbx": None if context.raw_fbx is None else str(context.raw_fbx),
        "source_skeleton": {"names": list(context.model_names), "parents": list(context.model_parents)},
        "model_skeleton": {"names": list(context.model_names), "parents": list(context.model_parents)},
        "presentation_skeleton": {"names": list(context.model_names), "parents": list(context.model_parents)},
        "model": None if assets.model is None else str(assets.model),
        "presentation": None if assets.presentation is None else str(assets.presentation),
        "outputs": {"model_fbx": None if assets.model is None else str(assets.model), "presentation_fbx": None if assets.presentation is None else str(assets.presentation)},
        "availability": {"prepared_fbx": assets.model is not None, "presentation_fbx": assets.presentation is not None},
        "reason": assets.reason,
        "warnings": list(dict.fromkeys(list(existing.get("warnings", ())) + ([] if assets.reason is None else [assets.reason]))),
        "operations": operations,
        "joint_decisions": joint_decisions,
        "skinning_decisions": list(existing.get("skinning_decisions", ())),
    })
    def file_record(path):
        if path is None or not Path(path).is_file():
            return None
        path = Path(path)
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}

    payload["pipeline_version"] = "character-aware-export/1"
    payload["sources"] = {
        "fbx": file_record(context.raw_fbx),
    }
    payload["outputs"] = {
        "model_fbx": file_record(assets.model),
        "presentation_fbx": file_record(assets.presentation),
    }
    if context.model_rest is not None:
        skeleton = {
            "names": list(context.model_rest.names),
            "parents": context.model_rest.parents.tolist(),
            "offsets": context.model_rest.offsets.tolist(),
        }
        payload["model_skeleton"] = skeleton
        payload["presentation_skeleton"] = skeleton
        payload["source_skeleton"] = skeleton
    if context.model_rest is not None:
        topology_payload = json.dumps({"names": list(context.model_rest.names), "parents": context.model_rest.parents.tolist(), "offsets": context.model_rest.offsets.tolist()}, sort_keys=True, separators=(",", ":")).encode()
        model_topology_hash = hashlib.sha256(topology_payload).hexdigest()
    else:
        model_topology_hash = None
    payload["topology_hashes"] = {
        "model": model_topology_hash
        if model_topology_hash is not None
        else None,
    }
    report.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return report

def build_character_assets(context: CharacterAssetContext, output: Path) -> CharacterAssets:
    """Build independent full-presentation and reduced model FBX assets."""
    if context.raw_fbx is None:
        absent = CharacterAssets(None, None, "source_surface_missing")
        LOGGER.warning("%s: skipped FBX assets (%s)", context.rig, absent.reason)
        return CharacterAssets(None, None, absent.reason, _write_report(context, output, absent))
    from poseydon.io.fbx import FBX
    raw = Path(context.raw_fbx)
    if not raw.is_file():
        raise FileNotFoundError(raw)
    output = Path(output)
    presentation = FBX.read(raw)
    canonical = _canonical_matrix(presentation, context)
    presentation.clear_animation()
    presentation.apply_scene_transform(canonical)
    presentation_path = output / "presentation.fbx"
    _write_atomic(presentation, presentation_path)

    model = FBX.read(raw)
    model.clear_animation()
    model.apply_scene_transform(canonical)
    model.prepare_model_topology(
        context.model_names,
        context.model_parents,
        skinning_plan=context.skinning_plan,
        offsets=None if context.model_rest is None else context.model_rest.offsets,
    )
    offsets = context.model_rest.offsets if context.model_rest is not None else None
    model.validate_topology(context.model_names, context.model_parents, offsets)
    model_path = output / "model_character.fbx"
    _write_atomic(model, model_path)

    assets = CharacterAssets(model_path, presentation_path)
    LOGGER.info("%s: wrote model=%s presentation=%s", context.rig, model_path, presentation_path)
    return CharacterAssets(assets.model, assets.presentation, None,
                           _write_report(context, output, assets))

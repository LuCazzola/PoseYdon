from __future__ import annotations

import json
import os
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from poseydon.core.animation import Animation
from poseydon.io.fbx import FBX, TO_Y_UP, _from_y_up, _quat_matrix, _similarity, _to_y_up, _rest_positions
from mathutils import Matrix, Vector

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


def _bind_positions(scene: FBX, names: tuple[str, ...]) -> np.ndarray:
    return np.stack([_to_y_up(scene.joint_position(name, rest=True)) for name in names])


def _canonical_matrix(scene: FBX, context: CharacterAssetContext) -> Matrix:
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
    operations = [
        {
            "operation_id": f"{context.rig}/topology/000",
            "kind": "topology_contract",
            "joint_names": list(context.model_names),
            "parents": list(context.model_parents),
        },
        {
            "operation_id": f"{context.rig}/correspondence/000",
            "kind": "joint_correspondence",
            "source": None if context.raw_fbx is None else str(context.raw_fbx),
            "target_joint_count": len(context.model_names),
        },
        {
            "operation_id": f"{context.rig}/skinning/000",
            "kind": "skinning_policy",
            "decision": "preserve" if context.skinning_plan is None else str(context.skinning_plan),
        },
    ]
    payload = {
        "dataset": "truebones",
        "rig": context.rig,
        "authority": {"skeleton": "bvh", "motion": "bvh", "surface": "fbx", "auxiliary_joints": "bake"},
        "source_fbx": None if context.raw_fbx is None else str(context.raw_fbx),
        "source_skeleton": {"names": list(context.model_names), "parents": list(context.model_parents)},
        "model_skeleton": {"names": list(context.model_names), "parents": list(context.model_parents)},
        "presentation_skeleton": {"names": list(context.model_names), "parents": list(context.model_parents)},
        "model": None if assets.model is None else str(assets.model),
        "presentation": None if assets.presentation is None else str(assets.presentation),
        "outputs": {"model_fbx": None if assets.model is None else str(assets.model), "presentation_fbx": None if assets.presentation is None else str(assets.presentation)},
        "availability": {"prepared_fbx": assets.model is not None, "presentation_fbx": assets.presentation is not None},
        "reason": assets.reason,
        "warnings": [] if assets.reason is None else [assets.reason],
        "operations": operations,
    }
    report.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return report


def build_character_assets(context: CharacterAssetContext, output: Path) -> CharacterAssets:
    """Build independent full-presentation and reduced model FBX assets."""
    if context.raw_fbx is None:
        absent = CharacterAssets(None, None, "source_surface_missing")
        LOGGER.warning("%s: skipped FBX assets (%s)", context.rig, absent.reason)
        return CharacterAssets(None, None, absent.reason, _write_report(context, output, absent))
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
    )
    offsets = context.model_rest.offsets if context.model_rest is not None else None
    model.validate_topology(context.model_names, context.model_parents, offsets)
    model_path = output / "model_character.fbx"
    _write_atomic(model, model_path)

    assets = CharacterAssets(model_path, presentation_path)
    LOGGER.info("%s: wrote model=%s presentation=%s", context.rig, model_path, presentation_path)
    return CharacterAssets(assets.model, assets.presentation, None,
                           _write_report(context, output, assets))

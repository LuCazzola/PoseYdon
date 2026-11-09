"""C6's second reader: import each file of a run in Blender and write its evaluated world vertices.

    blender -b --python tools/verify/blender_read.py -- RUN_DIR

Writes ``RUN_DIR/blender/<name>.npz`` for every file in ``case.json`` (the stage files, and the
vendor clip as ``1_source``): ``<mesh>`` -> ``(F, V, 3)`` at the case's frames, and ``names``. A BVH file gives
``names``, ``positions`` (F, J, 3) of the joint heads and ``frames`` instead (B4).
Imports nothing from `poseydon`. A stage file's frame ``i`` is the action's first frame plus ``i``; the vendor clip's frames are frame numbers.

Vertex order: Blender's evaluated vertices match the SDK's control points by index (same count and
order on every file probed), so C6 compares them index to index; no nearest-position matching.
"""

from __future__ import annotations

import builtins
import json
import sys
from pathlib import Path

import bpy
import numpy as np


def read(path: str, frames: list[int], absolute: bool) -> dict[str, np.ndarray]:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.fbx(filepath=path, automatic_bone_orientation=False)
    actions = [a.frame_range[0] for a in bpy.data.actions]
    start = 0 if absolute or not actions else round(min(actions))
    meshes = [o for o in bpy.data.objects if o.type == "MESH"]
    out: dict[str, list] = {o.name: [] for o in meshes}
    for frame in frames:
        bpy.context.scene.frame_set(start + int(frame))
        graph = bpy.context.evaluated_depsgraph_get()
        for o in meshes:
            mesh = o.evaluated_get(graph).to_mesh()
            world = np.array(o.matrix_world)
            points = np.array([v.co[:] for v in mesh.vertices]).reshape(-1, 3)
            out[o.name].append(points @ world[:3, :3].T + world[:3, 3])
            o.evaluated_get(graph).to_mesh_clear()
    return {name: np.array(rows) for name, rows in out.items()}


def read_bvh(path: str, frames: list[int]) -> dict[str, np.ndarray]:
    """Joint heads in world space at ``frames`` (0-based) from Blender's BVH importer, as ``positions`` (F, J, 3)."""
    bpy.ops.wm.read_factory_settings(use_empty=True)
    real = builtins.open
    builtins.open = lambda f, mode="r", *a, **k: real(f, mode.replace("U", ""), *a, **k)  # the importer opens with "rU", gone in Python 3.11
    try:
        bpy.ops.import_anim.bvh(filepath=path)
    finally:
        builtins.open = real
    arm = next(o for o in bpy.data.objects if o.type == "ARMATURE")
    start = round(min(a.frame_range[0] for a in bpy.data.actions))
    rows = []
    for frame in frames:
        bpy.context.scene.frame_set(start + int(frame))
        rows.append([np.array(arm.matrix_world @ b.head) for b in arm.pose.bones])
    return {"names": np.array([b.name for b in arm.pose.bones]), "positions": np.array(rows), "frames": np.array(frames)}


def main(run: Path) -> None:
    case = json.loads((run / "case.json").read_text())
    jobs = {**case["files"], "1_source": case["clip_fbx"]}
    if any(p.endswith(".bvh") for p in case["files"].values()):
        jobs.pop("1_source")  # a BVH has no mesh to read
    (run / "blender").mkdir(exist_ok=True)
    for name, path in jobs.items():
        frames = case["source_frames"] if name == "1_source" else [0] if name == "5_zero" else case["frames"]
        try:
            if path.endswith(".bvh"):
                np.savez(run / "blender" / f"{name}.npz", **read_bvh(path, frames))
                continue
            meshes = read(path, frames, name == "1_source")
            np.savez(run / "blender" / f"{name}.npz", names=np.array(list(meshes)), **meshes)
        except Exception as error:  # noqa: BLE001 - a failed import is C6's line, not a crash
            print(f"blender_read: {name}: {type(error).__name__}: {error}", file=sys.stderr)
            (run / "blender" / f"{name}.error").write_text(f"{type(error).__name__}: {error}")


if __name__ == "__main__":
    main(Path(sys.argv[sys.argv.index("--") + 1]))

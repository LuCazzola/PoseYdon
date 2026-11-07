"""The verifier's checks. Each takes a `Case` and returns `Check` lines."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from poseydon.io.fbx import FBX, Inventory
from tools.verify.case import Case
from tools.verify.report import Check, Evidence, Status


@dataclass
class _Read:
    inventory: Inventory
    bind: np.ndarray
    mesh_bind: dict[str, np.ndarray]


def _read(path: Path) -> _Read:
    scene = FBX.read(path)
    try:
        return _Read(scene.inventory(), scene.bind_matrices(),
                     {s.mesh: s.mesh_bind for s in scene.skin_data()})
    finally:
        scene.close()


def _close(a: np.ndarray, b: np.ndarray, tol: float) -> bool:
    return a.shape == b.shape and bool(np.allclose(a, b, rtol=tol, atol=tol, equal_nan=True))


def _diff(a, b) -> str:
    """A short sentence on how two inventory fields differ."""
    if isinstance(a, (str, float)):
        return f"{a} -> {b}"
    a, b = (list(x.items() if isinstance(x, dict) else x) for x in (a, b))
    return f"{len(a)} -> {len(b)} entries, missing {[x for x in a if x not in b][:3]}, new {[x for x in b if x not in a][:3]}"


def _differences(want: _Read, got: _Read, want_clip: Inventory, tol: float) -> list[str]:
    """What ``got`` lacks or changes against the reference ``want`` (clip timing against ``want_clip``)."""
    w, g = want.inventory, got.inventory
    out = []
    for field in ("nodes", "parents", "meshes", "unweighted", "materials", "textures", "unit_cm", "up_axis"):
        if getattr(w, field) != getattr(g, field):
            out.append(f"{field}: {_diff(getattr(w, field), getattr(g, field))}")
    for mesh in w.clusters.keys() & g.clusters.keys():
        if sorted(w.clusters[mesh]) != sorted(g.clusters[mesh]):
            out.append(f"clusters of {mesh}: {len(w.clusters[mesh])} -> {len(g.clusters[mesh])}")
    if w.clusters.keys() != g.clusters.keys():
        out.append(f"skinned meshes: {sorted(w.clusters)} -> {sorted(g.clusters)}")
    if not _close(want.bind, got.bind, tol):
        out.append("bind matrices differ")
    if want.mesh_bind.keys() == got.mesh_bind.keys() and not all(
        _close(want.mesh_bind[k], got.mesh_bind[k], tol) for k in want.mesh_bind
    ):
        out.append("mesh bind matrices differ")
    if want_clip.frames != g.frames:
        out.append(f"frame range: {want_clip.frames} -> {g.frames}")
    if want_clip.fps != g.fps and not (np.isnan(g.fps) and np.isnan(want_clip.fps)):
        out.append(f"fps: {want_clip.fps} -> {g.fps}")
    return out


def c1_inventory(case: Case, tol: float = 1e-6) -> list[Check]:
    """Nodes, hierarchy, meshes, skin clusters, binds, materials, textures, units, up axis,
    fps and frame range of each file against the rig's reference; the clip's own vendor
    file is held to the same, and a disagreement stays visible, never dropped."""
    evidence = Evidence.DERIVED if "repaired" in str(case.skin_fbx) else Evidence.INDEPENDENT
    try:
        reference, clip = _read(case.skin_fbx), _read(case.clip_fbx)
    except Exception as error:  # noqa: BLE001 - an unreadable reference is a finding, not a crash
        return [Check("C1", Status.FAIL, evidence, note=f"unreadable reference: {error}")]
    problems, unsupported = [], list(reference.inventory.unsupported)
    for name, path in case.files.items():
        try:
            got = _read(path)
            found = _differences(reference, got, clip.inventory, tol)
            unsupported += got.inventory.unsupported
        except Exception as error:  # noqa: BLE001
            found = [f"unreadable: {error}"]
        problems += [f"{name} ({path.name}): {line}" for line in found]
    checks = [Check("C1", Status.FAIL if problems else Status.PASS, evidence,
                    value=float(len(problems)), limit=0.0, where={"files": list(case.files)},
                    note="; ".join(problems))]
    if unsupported:
        checks.append(Check("C1:unsupported", Status.NOT_RUN, evidence, note="; ".join(dict.fromkeys(unsupported))))
    if case.clip_fbx != case.skin_fbx:
        split = _differences(reference, clip, clip.inventory, tol)
        if split:
            checks.append(Check("C1:vendor-disagreement", Status.ALLOWED, evidence,
                                where={"clip": case.clip_fbx.name}, note="; ".join(split)))
    return checks

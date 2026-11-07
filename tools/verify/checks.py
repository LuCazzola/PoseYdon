"""The verifier's checks. Each takes a `Case` and returns `Check` lines."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from poseydon.io.fbx import FBX, Inventory, SkinData
from tools.verify import laws
from tools.verify.case import Case, loss_record
from tools.verify.report import Check, Evidence, Status
from tools.verify.skin import deform, extent


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


def _evidence(case: Case) -> Evidence:
    """KingCobra's reference skin is a repaired copy of the vendor file."""
    return Evidence.DERIVED if "cache/repaired/" in case.skin_fbx.as_posix() else Evidence.INDEPENDENT


def _close(a: np.ndarray, b: np.ndarray, tol: float) -> bool:
    return a.shape == b.shape and bool(np.allclose(a, b, rtol=tol, atol=tol, equal_nan=True))


def _diff(a, b) -> str:
    """A short sentence on how two inventory fields differ."""
    if isinstance(a, (str, float)):
        return f"{a} -> {b}"
    a, b = (list(x.items() if isinstance(x, dict) else x) for x in (a, b))
    return f"{len(a)} -> {len(b)} entries, missing {[x for x in a if x not in b][:3]}, new {[x for x in b if x not in a][:3]}"


def _differences(want: _Read, got: _Read, want_clip: Inventory, tol: float,
                 window: tuple[int, int, int] | None = None) -> list[str]:
    """What ``got`` lacks or changes against the reference ``want`` (clip timing against ``want_clip``;
    with a recorded ``window`` over the vendor keys, the file's frame count against the window's length)."""
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
    if window is not None:
        vendor, count = want_clip.frames, g.frames[1] - g.frames[0] + 1
        expected = len(range(vendor[1] - vendor[0] + 1)[slice(*window)])
        if count != expected:
            out.append(f"frame count: {expected} (vendor keys {vendor[0]}-{vendor[1]}, window {window}) -> {count}")
    elif want_clip.frames != g.frames:
        out.append(f"frame range: {want_clip.frames} -> {g.frames}")
    if want_clip.fps != g.fps and not (np.isnan(g.fps) and np.isnan(want_clip.fps)):
        out.append(f"fps: {want_clip.fps} -> {g.fps}")
    return out


def c1_inventory(case: Case, tol: float) -> list[Check]:
    """Nodes, hierarchy, meshes, skin clusters, binds, materials, textures, units, up axis,
    fps and frame count (against the clip's recorded window) of each file against the rig's reference; the clip's own vendor
    file is held to the same, and a disagreement stays visible, never dropped."""
    evidence = _evidence(case)
    try:
        reference, clip = _read(case.skin_fbx), _read(case.clip_fbx)
    except Exception as error:  # noqa: BLE001 - an unreadable reference is a finding, not a crash
        return [Check("C1", Status.FAIL, evidence, note=f"unreadable reference: {error}")]
    problems, unsupported = [], list(reference.inventory.unsupported)
    for name, path in case.files.items():
        try:
            got = _read(path)
            found = _differences(reference, got, clip.inventory, tol, case.window)
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


def reference_skin(case: Case) -> list[SkinData]:
    """The rig's reference skin: mesh, clusters and bind of ``case.skin_fbx``."""
    scene = FBX.read(case.skin_fbx)
    try:
        return scene.skin_data()
    finally:
        scene.close()


def _worst(ref: np.ndarray, got: np.ndarray) -> tuple[float, int, int]:
    """Largest distance of ``(F, V, 3)`` arrays, with its frame and vertex; a non-finite one counts as infinite."""
    error = np.nan_to_num(np.linalg.norm(ref - got, axis=-1), nan=np.inf, posinf=np.inf)
    frame, vertex = np.unravel_index(error.argmax(), error.shape)
    return float(error[frame, vertex]), int(frame), int(vertex)


def _dominant(skin: SkinData, vertex: int) -> str:
    weights = [dict(zip(i, w)).get(vertex, 0.0) for i, w in zip(skin.indices, skin.weights)]
    return skin.links[int(np.argmax(weights))] if max(weights, default=0.0) > 0 else "(unweighted)"


def _skipped(inventory, mesh: str) -> str | None:
    """Why C2a cannot judge ``mesh`` of a file: no skin to compare, or a skinning the verifier does not model."""
    if mesh in inventory.meshes and not inventory.meshes[mesh][2]:
        return "has no skin deformer"
    reasons = [u.split(": ", 1)[1] for u in inventory.unsupported if u.startswith(f"{mesh}: ")]
    return f"uses unsupported {', '.join(reasons)}" if reasons else None


def _c2a_file(case: Case, name: str, path: Path, ref: tuple[list[SkinData], Inventory], tol: float) -> list[Check]:
    evidence, where = _evidence(case), {"rig": case.rig, "clip": case.action, "file": name}
    if not len(case.frames):
        return [Check("C2a", Status.NOT_RUN, evidence, where=where, note=f"{name}: no frames to compare")]
    try:
        scene = FBX.read(path)
        try:
            skinned, inventory, frames = {s.mesh: s for s in scene.skin_data()}, scene.inventory(), scene.frames()
            if len(case.frames) and case.frames.max() >= len(frames):
                return [Check("C2a", Status.FAIL, evidence, where=where,
                              note=f"{name} has {len(frames)} frames, the case samples frame {case.frames.max()}")]
            world, names = scene.world(frames[case.frames]), scene.names
        finally:
            scene.close()
    except Exception as error:  # noqa: BLE001 - an unreadable stage is NOT RUN, not a crash
        return [Check("C2a", Status.NOT_RUN, evidence, where=where, note=f"{name} ({path.name}) unreadable: {error}")]
    joints = {n: world[:, j] for j, n in enumerate(names)}
    notes, skipped, shown, worst = [], [], [], (-1.0, {})
    for mesh, skin in {s.mesh: s for s in ref[0]}.items():
        why = _skipped(ref[1], mesh) or _skipped(inventory, mesh)
        other = skinned.get(mesh)
        if why:
            skipped.append(f"mesh {mesh} {why}")
        elif other is None:
            notes.append(f"mesh {mesh} is missing")
        elif (missing := sorted({*skin.links, *other.links} - joints.keys())):
            notes.append(f"mesh {mesh}: no joint named {missing[:3]} in the file")
        elif skin.points.shape != other.points.shape or skin.faces.shape != other.faces.shape \
                or (skin.faces != other.faces).any():
            notes.append(f"mesh {mesh}: topology differs ({len(skin.points)} -> {len(other.points)} vertices)")
        else:
            try:
                want, got = deform(skin, joints), deform(other, joints)
            except np.linalg.LinAlgError as error:
                notes.append(f"mesh {mesh}: a bind matrix is singular ({error})")
                continue
            size, frame, vertex = _worst(want, got)
            shown.append(want)
            if size > worst[0]:
                worst = (size, {"mesh": mesh, "frame": int(case.frames[frame]), "vertex": vertex,
                                "joint": _dominant(skin, vertex)})
    for mesh, (_, _, has_skin) in ref[1].meshes.items():
        if not has_skin:
            skipped.append(f"mesh {mesh} has no skin deformer")
    out = []
    if shown or notes:
        size = max(extent(np.concatenate([m[f] for m in shown])) for f in range(len(case.frames))) if shown else 1.0
        value = worst[0] / size if shown and size > 0 else None
        if shown and value is None:
            notes.append("the reference skin has zero extent")
        failed = bool(notes) or (value is not None and (not np.isfinite(value) or value > tol))
        out.append(Check("C2a", Status.FAIL if failed else Status.PASS, evidence, value, tol, where | worst[1],
                         "; ".join(notes)))
    if skipped:
        out.append(Check("C2a", Status.NOT_RUN, evidence, where=where, note=f"{name}: " + "; ".join(skipped)))
    return out


def c2a_skin(case: Case, cfg: dict) -> list[Check]:
    """Each stage file's own skin against the reference skin, both driven by the joints read
    from that file; per mesh by name and per vertex by index, as a fraction of rig extent.
    A mesh with no skin, or a skinning the verifier does not model, is NOT RUN, named; a rig
    with no mesh has nothing to compare. Takes no allowances."""
    try:
        skins = reference_skin(case)
        scene = FBX.read(case.skin_fbx)
        try:
            inventory = scene.inventory()
        finally:
            scene.close()
    except Exception as error:  # noqa: BLE001
        return [Check("C2a", Status.NOT_RUN, _evidence(case), note=f"unreadable reference: {error}")]
    if not inventory.meshes:
        return []
    return [c for name, path in case.files.items() for c in _c2a_file(case, name, path, (skins, inventory), cfg["skin_tol"])]


@dataclass
class _Joints:
    names: tuple[str, ...]
    parents: np.ndarray
    skeleton: list[bool]
    world: np.ndarray


def _joints(path: Path, frames: np.ndarray | None, positions: np.ndarray | None = None) -> _Joints:
    """World matrices of every node of ``path`` at the ``positions`` of its frames, or at the frame
    numbers ``frames``."""
    scene = FBX.read(path)
    try:
        wanted = frames if positions is None else scene.frames()[positions]
        return _Joints(scene.names, scene.parents, [scene.is_skeleton(i) for i in range(len(scene.nodes))],
                       scene.world(wanted))
    finally:
        scene.close()


def _bind(path: Path) -> _Joints:
    """The vendor file's bind pose as one frame of world matrices (the first frame's where a node has none)."""
    scene = FBX.read(path)
    try:
        return _Joints(scene.names, scene.parents, [], scene.bind_or_first()[None])
    finally:
        scene.close()


def _gap(a: np.ndarray, b: np.ndarray, names: tuple[str, ...], compare: list[int], skins, size: float,
         frames: np.ndarray) -> tuple[float, str, dict]:
    """Largest distance between two ``(F, J, 4, 4)`` joint sets, as a fraction of rig extent ``size``,
    over the skin vertices (``skins``: reference skin, deformed by each) and the joint positions."""
    rows = {n: (a[:, j], b[:, j]) for j, n in enumerate(names)}
    worst, where, parts = -1.0, {}, []
    for skin in skins:
        error, frame, vertex = _worst(deform(skin, {n: r[0] for n, r in rows.items()}),
                                      deform(skin, {n: r[1] for n, r in rows.items()}))
        parts.append(f"mesh {skin.mesh} vertex {vertex} at {_dominant(skin, vertex)}, frame {frames[frame]}: {error / size:.3g}")
        if error / size > worst:
            worst, where = error / size, {"mesh": skin.mesh, "vertex": vertex, "joint": _dominant(skin, vertex),
                                          "frame": int(frames[frame])}
    error, frame, joint = _worst(a[:, compare][..., :3, 3], b[:, compare][..., :3, 3])
    parts.append(f"joint {names[compare[joint]]}, frame {frames[frame]}: {error / size:.3g}")
    if error / size > worst:
        worst, where = error / size, {"joint": names[compare[joint]], "frame": int(frames[frame])}
    return worst, "; ".join(parts), where


def c2b_fidelity(case: Case, cfg: dict) -> list[Check]:
    """The returned file's joints against what the pipeline is defined to produce from the user's
    clip: the source, with each lossy stage's law applied (`tools/verify/laws.py`). Untraced error
    (file vs expected) is the verdict, within ``untraced_tol`` of rig extent; traced loss (source vs
    expected) is reported, never judged. Vertices are the reference skin driven by each joint set;
    joint positions are compared too. A stage with no law makes it NOT RUN, naming the stage."""
    evidence, where = _evidence(case), {"rig": case.rig, "clip": case.action}

    def stop(status: Status, note: str) -> list[Check]:
        return [Check("C2b-untraced", status, evidence, where=where, note=note)]

    path = case.files.get("4_roundtrip")
    if path is None or not case.data_name:
        return stop(Status.NOT_RUN, "needs a built round trip (its 4_roundtrip file and the rig's records)")
    if not len(case.frames):
        return stop(Status.NOT_RUN, "no frames to compare")
    try:
        record = loss_record(case)
        file, asset = _joints(path, None, case.frames), _joints(record.asset, None, np.zeros(1, dtype=int))
        rest = asset if record.bind_of is None else _bind(record.bind_of)
        source = _joints(case.clip_fbx, case.source_frames)
        skins, inventory = reference_skin(case), _read(case.skin_fbx).inventory
    except Exception as error:  # noqa: BLE001 - an unreadable file or record is NOT RUN, not a crash
        return stop(Status.NOT_RUN, f"unreadable: {error}")
    names = file.names
    missing = sorted({n for other in (asset, rest, source) for n in names if n not in other.names})
    if missing:
        return stop(Status.NOT_RUN, f"the rest asset or the source clip has no node named {missing[:3]}")
    by = lambda other, w: w[:, [other.names.index(n) for n in names]]  # noqa: E731 - file order
    pose = laws.SourcePose(names, file.parents, by(source, source.world), by(rest, rest.world)[0],
                          asset=by(asset, asset.world)[0])
    expected = laws.expected_pose(pose, record.stages, record.params)
    if isinstance(expected, laws.MissingLaw):
        return stop(Status.NOT_RUN, f"no loss law for stage `{expected.stage}`: declare it in tools/verify/laws.py")
    usable = [s for s in skins if not _skipped(inventory, s.mesh)]
    links = {n for s in usable for n in s.links}
    compare = [j for j, n in enumerate(names) if file.skeleton[j] or n in links]
    if not compare or links - set(names):
        return stop(Status.NOT_RUN, f"no joints to compare, or skin links missing from the file: {sorted(links - set(names))[:3]}")
    try:
        cloud = [deform(s, {n: expected.world[:, j] for j, n in enumerate(names)}) for s in usable]
    except np.linalg.LinAlgError as error:
        return stop(Status.NOT_RUN, f"a bind matrix is singular ({error})")
    points = [np.concatenate([m[f] for m in cloud]) if cloud else expected.world[f, compare][:, :3, 3]
              for f in range(len(case.frames))]
    size = max(extent(p) for p in points)
    if not size > 0:
        return stop(Status.FAIL, "the expected pose has zero extent")
    out = []
    for check, other in (("C2b-untraced", file.world), ("C2b-traced", pose.world)):
        value, note, at = _gap(other, expected.world, names, compare, usable, size, case.frames)
        if check == "C2b-traced":
            out.append(Check(check, Status.PASS, evidence, value, where=where | at,
                             note=f"by laws of {', '.join(s for s in record.stages if s in laws.LAWS)}; {note}"))
        else:
            failed = not np.isfinite(value) or value > cfg["untraced_tol"]
            out.append(Check(check, Status.FAIL if failed else Status.PASS, evidence, value, cfg["untraced_tol"],
                             where | at, note))
    return out

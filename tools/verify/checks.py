"""The verifier's checks. Each takes a `Case` and returns `Check` lines."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import NamedTuple
import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from poseydon.build.prepare import similarity
from poseydon.io.bvh import BVH
from poseydon.io.fbx import FBX, Inventory, SkinData
from tools.verify import laws
from tools.verify.case import Case, LossRecord, rig_skin, sample_frames, ZeroRecord, loss_record, recorded_space, zero_record
from tools.verify.report import Check, Evidence, Status
from tools.verify.skin import deform, extent

ZERO = "5_zero"


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
                 window: tuple[int, int, int] | None = None, timed: bool = True) -> list[str]:
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
    if not timed:  # the zero file is one pose, not a clip
        return out
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
            timing = replace(clip.inventory, frames=got.inventory.frames) if name == "file" else clip.inventory  # a `file` run has no clip to count against
            found = _differences(reference, got, timing, tol, case.window, name != ZERO)
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
    sampled = np.zeros(1, dtype=int) if name == ZERO else case.frames
    if not len(sampled):
        return [Check("C2a", Status.NOT_RUN, evidence, where=where, note=f"{name}: no frames to compare")]
    try:
        scene = FBX.read(path)
        try:
            skinned, inventory, frames = {s.mesh: s for s in scene.skin_data()}, scene.inventory(), scene.frames()
            if sampled.max() >= len(frames):
                return [Check("C2a", Status.FAIL, evidence, where=where,
                              note=f"{name} has {len(frames)} frames, the case samples frame {sampled.max()}")]
            world, names = scene.world(frames[sampled]), scene.names
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
                worst = (size, {"mesh": mesh, "frame": int(sampled[frame]), "vertex": vertex,
                                "joint": _dominant(skin, vertex)})
    for mesh, (_, _, has_skin) in ref[1].meshes.items():
        if not has_skin:
            skipped.append(f"mesh {mesh} has no skin deformer")
    out = []
    if shown or notes:
        size = max(extent(np.concatenate([m[f] for m in shown])) for f in range(len(sampled))) if shown else 1.0
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


def _joints(path: Path, pick) -> _Joints:
    """World matrices of every node of ``path`` at the frame numbers ``pick(all frame numbers)``."""
    scene = FBX.read(path)
    try:
        return _Joints(scene.names, scene.parents, [scene.is_skeleton(i) for i in range(len(scene.nodes))],
                       scene.world(pick(scene.frames())))
    finally:
        scene.close()


def _bind(path: Path) -> _Joints:
    """The vendor file's bind pose as one frame of world matrices (the first frame's where a node has none)."""
    scene = FBX.read(path)
    try:
        return _Joints(scene.names, scene.parents, [], scene.bind_or_first()[None])
    finally:
        scene.close()


def _rotation(m: np.ndarray) -> np.ndarray:
    return m[..., :3, :3] / np.linalg.norm(m[..., :3, :3], axis=-2)[..., None, :]


def _gap(a: np.ndarray, b: np.ndarray, names: tuple[str, ...], compare: list[int], skins, size: float,
         frames: np.ndarray) -> dict:
    """Largest difference between two ``(F, J, 4, 4)`` joint sets, each with its frame and where: skin
    vertices (``skins``: the reference skin deformed by each set) and joint positions as a fraction of rig
    extent ``size``, joint world rotations as the largest rotation-matrix element difference."""
    rows = {n: (a[:, j], b[:, j]) for j, n in enumerate(names)}
    vertex = None
    for skin in skins:
        error, frame, v = _worst(deform(skin, {n: r[0] for n, r in rows.items()}),
                                 deform(skin, {n: r[1] for n, r in rows.items()}))
        if vertex is None or error / size > vertex["value"]:
            vertex = {"value": error / size, "mesh": skin.mesh, "vertex": v, "joint": _dominant(skin, v),
                      "frame": int(frames[frame])}
    error, frame, j = _worst(a[:, compare][..., :3, 3], b[:, compare][..., :3, 3])
    joint = {"value": error / size, "joint": names[compare[j]], "frame": int(frames[frame])}
    turn = np.nan_to_num(np.abs(_rotation(a[:, compare]) - _rotation(b[:, compare])).max(axis=(-2, -1)),
                         nan=np.inf, posinf=np.inf)
    frame, j = np.unravel_index(turn.argmax(), turn.shape)
    return {"vertex": vertex, "joint": joint,
            "rotation": {"value": float(turn[frame, j]), "joint": names[compare[j]], "frame": int(frames[frame])}}


def _summary(gap: dict) -> str:
    v, j, r = gap["vertex"], gap["joint"], gap["rotation"]
    return "; ".join(
        ([f"vertices {v['value']:.3g} (mesh {v['mesh']} vertex {v['vertex']} at {v['joint']}, frame {v['frame']})"] if v else [])
        + [f"joints {j['value']:.3g} ({j['joint']}, frame {j['frame']})",
           f"rotations {r['value']:.3g} ({r['joint']}, frame {r['frame']})"])


def _rest(case: Case, pose: laws.SourcePose, parents: np.ndarray, size: float, tol: float, evidence) -> Check:
    """C2b-rest: the rest inputs the laws read (bone lengths of the dataset's rest pose, node scales of the
    asset the file is written onto) against the vendor file's own bind. They come from the asset, which the
    pipeline wrote, so this is what keeps a wrong asset from moving file and expected together."""
    bind = _bind(case.skin_fbx)
    names, rest, asset = pose.names, pose.rest, pose.placed
    known = [j for j, n in enumerate(names) if n in bind.names]
    if not known:
        return Check("C2b-rest", Status.NOT_RUN, evidence, where={"rig": case.rig, "clip": case.action},
                     note="no joint of the file is named in the vendor bind")
    worst, at = -1.0, ""
    for j in known:
        p = int(parents[j])
        scale = np.abs(np.linalg.norm(asset[j, :3, :3], axis=-2)
                       - np.linalg.norm(bind.world[0, bind.names.index(names[j]), :3, :3], axis=-2)).max()
        if scale > worst:
            worst, at = float(scale), f"{names[j]} node scale"
        if p >= 0 and names[p] in bind.names:
            ours = np.linalg.norm(rest[j, :3, 3] - rest[p, :3, 3])
            theirs = np.linalg.norm(bind.world[0, bind.names.index(names[j]), :3, 3]
                                    - bind.world[0, bind.names.index(names[p]), :3, 3])
            if abs(ours - theirs) / size > worst:
                worst, at = float(abs(ours - theirs) / size), f"{names[j]} bone length {ours:.4g} against bind {theirs:.4g}"
    return Check("C2b-rest", Status.FAIL if not worst <= tol else Status.PASS, evidence, worst, tol,
                 {"rig": case.rig, "clip": case.action}, f"worst {at}")


@dataclass
class _Run:
    """What a stage file is judged with: the pipeline's record, the file's joints, the source clip as a
    ``SourcePose`` in the file's joint order, and the reference skins C2b and C3 can model."""

    record: LossRecord
    file: _Joints
    pose: laws.SourcePose
    skins: list
    skipped: list[str]
    compare: list[int]


def _load(case: Case, path: Path) -> _Run | str:
    """The inputs of C2b and C3 for the stage file ``path``, or the reason they cannot be had."""
    try:
        record = loss_record(case)
        file = _joints(path, lambda f: f[case.frames])
        asset = _joints(record.asset, lambda f: f[:1])
        rest = asset if record.bind_of is None else _bind(record.bind_of)
        source = _joints(case.clip_fbx, lambda f: case.source_frames)
        skins, inventory = reference_skin(case), _read(case.skin_fbx).inventory
    except Exception as error:  # noqa: BLE001 - an unreadable file or record is NOT RUN, not a crash
        return f"unreadable: {error}"
    names = file.names
    missing = sorted({n for other in (asset, rest, source) for n in names if n not in other.names})
    if missing:
        return f"the rest asset or the source clip has no node named {missing[:3]}"

    def order(other: _Joints) -> np.ndarray:
        return other.world[:, [other.names.index(n) for n in names]]

    pose = laws.SourcePose(names, file.parents, order(source), order(rest)[0], asset=order(asset)[0])
    modelled = _modelled(skins, inventory, file)
    return modelled if isinstance(modelled, str) else _Run(record, file, pose, *modelled)


def _modelled(skins: list[SkinData], inventory: Inventory, file: _Joints) -> tuple[list, list[str], list[int]] | str:
    """The reference skins the verifier models, why the others are skipped, and the joints to compare."""
    usable = [s for s in skins if not _skipped(inventory, s.mesh)]
    skipped = [f"mesh {m} {_skipped(inventory, m)}" for m in {s.mesh for s in skins} if _skipped(inventory, m)]
    links = {n for s in usable for n in s.links}
    compare = [j for j, n in enumerate(file.names) if file.skeleton[j] or n in links]
    if not compare or links - set(file.names):
        return f"no joints to compare, or skin links missing from the file: {sorted(links - set(file.names))[:3]}"
    return usable, skipped, compare


def _size(run: _Run, world: np.ndarray) -> float:
    """Largest extent over frames of the skin (else the compared joints) under ``world``."""
    cloud = [deform(s, {n: world[:, j] for j, n in enumerate(run.file.names)}) for s in run.skins]
    return max(extent(np.concatenate([m[f] for m in cloud]) if cloud else world[f, run.compare][:, :3, 3])
               for f in range(len(world)))


def _judged(gap: dict, cfg: dict) -> tuple[float, bool]:
    """The value and the verdict of a file-against-expected gap: vertices (else joints) and joint positions
    within ``untraced_tol`` of extent, rotations within ``untraced_rot_tol``."""
    value = (gap["vertex"] or gap["joint"])["value"]
    failed = (not value <= cfg["untraced_tol"] or not gap["joint"]["value"] <= cfg["untraced_tol"]
              or not gap["rotation"]["value"] <= cfg["untraced_rot_tol"])
    return value, failed


def c2b_fidelity(case: Case, cfg: dict) -> list[Check]:
    """The returned file's joints against what the pipeline is defined to produce from the user's
    clip: the source, with each lossy stage's law applied (`tools/verify/laws.py`). Untraced error
    (file vs expected) is the verdict: skin vertices and joint positions within ``untraced_tol`` of rig
    extent, joint world rotations within ``untraced_rot_tol``. Traced loss (source vs expected) is
    reported, never judged. Vertices, joints and rotations are reported apart. A stage with no law makes
    it NOT RUN, naming it (``C2b-rest``, which needs no law, still runs). The laws read bone lengths and node scales from the rig's rest asset, which the
    pipeline wrote: ``C2b-rest`` holds those to the vendor file's bind. A floor run (the source clip
    written with no preparation) is expected to equal the source but for the writer's own node scale and
    held joints."""
    evidence, where = _evidence(case), {"rig": case.rig, "clip": case.action}

    def stop(status: Status, note: str) -> list[Check]:
        return [Check("C2b-untraced", status, evidence, where=where, note=note)]

    path = case.files.get("floor" if case.floor else "4_roundtrip")
    if path is None or not case.data_name:
        return stop(Status.NOT_RUN, "needs a built round trip (its 4_roundtrip file and the rig's records)")
    if not len(case.frames):
        return stop(Status.NOT_RUN, "no frames to compare")
    run = _load(case, path)
    if isinstance(run, str):
        return stop(Status.NOT_RUN, run)
    file, pose, usable = run.file, run.pose, run.skins
    names, record = file.names, run.record
    stages, params = record.stages, record.params
    if case.floor:
        stages = [s for s in stages if s in ("promote_root", "prune_unskinned")]
        prune = params.get("prune_unskinned")
        if prune:
            roots = [(n, k) for n, k in zip(prune["removed"], prune["kind"]) if k == "root"]
            params = params | {"prune_unskinned": {"removed": [n for n, _ in roots], "kind": [k for _, k in roots]}}
    expected = laws.expected_pose(pose, stages, params)
    try:
        size = _size(run, pose.world if isinstance(expected, laws.MissingLaw) else expected.world)
    except np.linalg.LinAlgError as error:
        return stop(Status.NOT_RUN, f"a bind matrix is singular ({error})")
    if not size > 0:
        return stop(Status.FAIL, "the expected pose has zero extent")
    rest = _rest(case, pose, file.parents, size, cfg["untraced_tol"], evidence)
    if isinstance(expected, laws.MissingLaw):
        return [*stop(Status.NOT_RUN, f"no loss law for stage `{expected.stage}`: declare it in tools/verify/laws.py"), rest]
    inputs = "bone lengths from the " + ("rest asset" if record.bind_of is None else "vendor bind") \
        + " and node scales from the rest asset"
    out = []
    for check, other in (("C2b-untraced", file.world), ("C2b-traced", pose.world)):
        gap = _gap(other, expected.world, names, run.compare, usable, size, case.frames)
        value, failed = _judged(gap, cfg)
        note = "; ".join([_summary(gap), inputs, *run.skipped])
        if check == "C2b-traced":
            out.append(Check(check, Status.PASS, evidence, value, where=where | gap,
                             note=f"by laws of {', '.join(s for s in stages if s in laws.LAWS)}; {note}"))
        else:
            out.append(Check(check, Status.FAIL if failed else Status.PASS, evidence, value, cfg["untraced_tol"],
                             where | gap, note))
    return [*out, rest]


def _skins_of(path: Path) -> dict[str, SkinData]:
    scene = FBX.read(path)
    try:
        return {s.mesh: s for s in scene.skin_data()}
    finally:
        scene.close()


class _Cannot(NamedTuple):
    status: Status
    note: str


def _fitted(path: Path, file: _Joints, skins: list[SkinData], compare: list[int], expected: np.ndarray,
            frames: np.ndarray, cfg: dict):
    """The file's joints against ``expected`` (``(F, J, 4, 4)``, the reference skins' pose) after ONE
    similarity fitted from the expected skin vertices to the file's: ``(gap, value, failed, scale,
    rotation, shift, size)``, or a `_Cannot` where the file cannot be judged."""
    names = file.names
    if not compare:
        return _Cannot(Status.NOT_RUN, "no joints to compare")
    try:
        mine = _skins_of(path)
        odd = [s.mesh for s in skins if s.mesh not in mine or mine[s.mesh].points.shape != s.points.shape
               or set(mine[s.mesh].links) - set(names)]
        if odd:
            return _Cannot(Status.NOT_RUN, f"mesh {odd[:3]} is missing, differs in vertices or links a joint the file lacks (C2a)")
        mesh = [mine[s.mesh] for s in skins]
        wants = [deform(s, {n: expected[:, j] for j, n in enumerate(names)}) for s in skins]
        got = [deform(s, {n: file.world[:, j] for j, n in enumerate(names)}) for s in mesh]
    except Exception as error:  # noqa: BLE001 - an unreadable file or a singular bind is NOT RUN
        return _Cannot(Status.NOT_RUN, f"{type(error).__name__}: {error}")
    want, seen = (np.concatenate([a.reshape(-1, 3) for a in x]) for x in (wants, got))
    if not (np.isfinite(want).all() and np.isfinite(seen).all()):
        return _Cannot(Status.FAIL, "non-finite vertices")
    # One similarity for the whole clip: a per-frame fit would hide a placement error.
    scale, rotation, shift = similarity(want, seen)
    fit = np.eye(4)
    fit[:3, :3], fit[:3, 3] = scale * rotation, shift
    size = max(extent(np.concatenate([w[f] for w in wants]) @ fit[:3, :3].T + shift) for f in range(len(frames)))
    if not size > 0:
        return _Cannot(Status.FAIL, "the expected pose has zero extent")
    gap = _gap(file.world, fit @ expected, names, compare, mesh, size, frames)
    return (gap, *_judged(gap, cfg), scale, rotation, shift, size)


def _placement(space: np.ndarray, scale: float, rotation: np.ndarray, shift: np.ndarray, size: float, cfg: dict,
               where: dict, check: str = "C3-placement") -> Check:
    """The fitted similarity against the recorded ``space``: scale ratio, whole rotation in degrees, offset
    as a fraction of extent, each within its `placement_*` bound."""
    recorded = np.cbrt(abs(np.linalg.det(space[:3, :3])))
    turned = float(np.degrees(Rotation.from_matrix(rotation @ (space[:3, :3] / recorded).T).magnitude()))
    found = {"scale": abs(scale / recorded - 1), "rotation_deg": turned,
             "offset": float(np.linalg.norm(shift - space[:3, 3]) / size)}
    limit = {"scale": cfg["placement_scale_tol"], "rotation_deg": cfg["placement_rotation_deg"], "offset": cfg["placement_offset_tol"]}
    bad = [k for k in found if not found[k] <= limit[k]]
    return Check(check, Status.FAIL if bad else Status.PASS, Evidence.PIPELINE,
                 max(found[k] / limit[k] for k in found), 1.0, where | {"found": found, "limit": limit},
                 "fitted similarity against the recorded `space` (value: worst error over its limit; angle is the whole "
                 "rotation between them)" + (f"; over: {', '.join(bad)}" if bad else ""))


def _c3_stage(case: Case, cfg: dict, name: str, path: Path) -> list[Check]:
    evidence, where = _evidence(case), {"rig": case.rig, "clip": case.action, "file": name}

    def stop(status: Status, note: str) -> list[Check]:
        return [Check("C3-shape", status, evidence, where=where, note=note),
                Check("C3-placement", Status.NOT_RUN, Evidence.PIPELINE, where=where, note="C3-shape did not complete")]

    run = _load(case, path)
    if isinstance(run, str):
        return stop(Status.NOT_RUN, run)
    if not run.skins:
        return stop(Status.NOT_RUN, "no skinned mesh the verifier models: " + "; ".join(run.skipped))
    names, file, record = run.file.names, run.file, run.record
    stages = [s for s in record.stages if name == "3_model" or s not in record.reduction]
    expected = laws.expected_pose(run.pose, stages, record.params)
    if isinstance(expected, laws.MissingLaw):
        return stop(Status.NOT_RUN, f"no loss law for stage `{expected.stage}`: declare it in tools/verify/laws.py")
    fitted = _fitted(path, file, run.skins, run.compare, expected.world, case.frames, cfg)
    if isinstance(fitted, _Cannot):
        return stop(*fitted)
    gap, value, failed, scale, rotation, shift, size = fitted
    out = [Check("C3-shape", Status.FAIL if failed else Status.PASS, evidence, value, cfg["untraced_tol"],
                 where | gap, "after one fitted similarity; " + "; ".join([_summary(gap), *run.skipped]))]
    try:
        space = recorded_space(case, name)
    except Exception as error:  # noqa: BLE001
        return [*out, Check("C3-placement", Status.NOT_RUN, Evidence.PIPELINE, where=where, note=f"unreadable record: {error}")]
    return [*out, _placement(space, scale, rotation, shift, size, cfg, where)]


def c3_canonical(case: Case, cfg: dict) -> list[Check]:
    """The prepared and model-view exports (written in canonical space) against the source mesh: ONE
    similarity is fitted for the whole clip from the expected skin vertices (the source clip with the
    stage's laws, on the reference skin) to the file's. ``C3-shape`` is the residual, judged on vertices,
    joints and rotations like C2b-untraced; ``C3-placement`` compares the fit with the `space`
    `RigRecord.placement` recorded: scale ratio, rotation in degrees, offset as a fraction of extent. The
    model stage adds the reduction's laws. ``C3-placement`` checks a canonical file against the placement
    recomputed from the record; a wrong record moves 4_roundtrip off the source, which is C2b's catch."""
    wanted = ("2_prepared", "3_model")
    if not any(n in case.files for n in wanted) or not case.data_name or case.floor or not len(case.frames):
        return [Check("C3-shape", Status.NOT_RUN, _evidence(case), note="needs a built round trip's 2_prepared and 3_model files")]
    out = []
    for name in wanted:
        if name in case.files:
            out += _c3_stage(case, cfg, name, case.files[name])
        else:
            out += [Check(c, Status.NOT_RUN, _evidence(case) if c == "C3-shape" else Evidence.PIPELINE,
                          where={"rig": case.rig, "clip": case.action, "file": name}, note=f"the case has no {name} file")
                    for c in ("C3-shape", "C3-placement")]
    return out


_AXIS = {"X": 0, "Y": 1, "Z": 2}


def _angle(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip(a @ b / np.linalg.norm(a) / np.linalg.norm(b), -1, 1))))


def _zero_pose(case: Case, cfg: dict, rec: ZeroRecord, file: _Joints, skins: list, rotation: np.ndarray | None) -> Check:
    """C4-pose: the declared definition of the rest pose, read off the zero file. Up: the manifest's up axis
    in the vendor rest, carried by C4-mesh's fitted rotation, lands on +Y. Facing: the manifest's pairs, as
    `FaceRestAxis` defines it (forward = up x (right - left), turned by ``extra_yaw_deg``), land on +Z.
    Grounded: the rig's ground mode puts the lowest vertex (mesh) or joint (joints) at y = 0, as a fraction
    of extent. A part the chain does not declare, or the file cannot show, is named and left out."""
    where, world = {"rig": case.rig, "clip": case.action}, file.world[0]
    found, limit, notes = {}, {"up_deg": cfg["zero_facing_deg"], "facing_deg": cfg["zero_facing_deg"],
                               "ground": cfg["zero_ground_tol"]}, []

    def stop(note: str) -> Check:
        return Check("C4-pose", Status.NOT_RUN, Evidence.PIPELINE, where=where, note=note)

    if "up_axis" in rec.stages and rotation is not None:
        up = np.zeros(3)
        up[_AXIS[rec.up_axis[1]]] = 1.0 if rec.up_axis[0] == "+" else -1.0
        found["up_deg"] = _angle(rotation @ up, np.array([0.0, 1.0, 0.0]))
    else:
        notes.append("up not judged: " + ("needs C4-mesh's fit" if "up_axis" in rec.stages else "the chain has no up_axis"))
    if "face_rest_axis" in rec.stages:
        try:
            across = sum(world[file.names.index(r), :3, 3] - world[file.names.index(l), :3, 3] for r, l in rec.facing)
        except ValueError as error:
            return stop(f"facing joint missing from the file: {error}")
        yaw = np.radians(rec.extra_yaw_deg)
        found["facing_deg"] = _angle(np.cross([0.0, 1.0, 0.0], across), np.array([np.sin(yaw), 0.0, np.cos(yaw)]))
    else:
        notes.append("facing not judged: the chain has no face_rest_axis")
    if rec.ground:
        if rec.ground == "mesh":
            cloud = [deform(s, {n: file.world[:, j] for j, n in enumerate(file.names)})[0] for s in skins]
        else:
            cloud = [world[[j for j, n in enumerate(file.names) if n in rec.joints], :3, 3]]
        cloud = np.concatenate(cloud) if cloud else np.zeros((0, 3))
        if not len(cloud):
            return stop(f"ground mode `{rec.ground}`: " + ("no skin the verifier models" if rec.ground == "mesh" else "no prepared joint in the file"))
        found["ground"] = float(abs(cloud[:, 1].min()) / extent(cloud))
        where = where | {"ground_mode": rec.ground}
    else:
        notes.append("ground not judged: the chain has no ground stage")
    if not found:
        return stop("; ".join(notes))
    bad = [k for k in found if not found[k] <= limit[k]]
    return Check("C4-pose", Status.FAIL if bad else Status.PASS, Evidence.PIPELINE, max(found[k] / limit[k] for k in found), 1.0,
                 where | {"found": found, "limit": limit},
                 "pipeline evidence: the manifest's own definition (value: worst error over its limit)"
                 + (f"; over: {', '.join(bad)}" if bad else "") + ("; " + "; ".join(notes) if notes else ""))


def _zero_joints(case: Case, cfg: dict, rec: ZeroRecord, file: _Joints) -> Check:
    """C4-joints: the zero file's joint positions against the rig's rest-pose BVH (the manifest's `rest_pose`),
    after ONE fitted similarity, within ``untraced_tol`` of extent. The BVH is the source `rest.npz` was built
    from, read here on its own: `derived` where this repo wrote the file (`tools/tpose_author.py` names its output
    `*_authored.bvh`), `independent` where it is a vendor file the raw folder holds unchanged (a `__TPOSE.bvh`, or
    a vendor clip such as Dog's `__Idle.bvh`, whose first frame is the rest pose). Joints held by structure are left out.
    Blind spot: it compares positions only, so a rotation error that leaves every joint where it is goes unseen."""
    where = {"rig": case.rig, "clip": case.action}
    if rec.rest_pose is None:
        return Check("C4-joints", Status.NOT_RUN, None, where=where, note="the manifest declares no rest_pose BVH")
    authored = rec.rest_pose.name.endswith("_authored.bvh")  # `tools/tpose_author.py`'s fixed output name
    evidence = Evidence.DERIVED if authored else Evidence.INDEPENDENT
    try:
        pose = BVH.read(rec.rest_pose).to_animation().slice(0, 1)
    except Exception as error:  # noqa: BLE001 - a missing raw file is NOT RUN
        return Check("C4-joints", Status.NOT_RUN, evidence, where=where, note=f"rest_pose unreadable: {error}")
    common = [n for n in file.names if n in pose.names and n not in rec.fixed]
    if len(common) < 4:
        return Check("C4-joints", Status.NOT_RUN, evidence, where=where, note=f"{len(common)} joints shared with {rec.rest_pose.name}")
    want = pose.global_positions()[0, [pose.names.index(n) for n in common]]
    got = file.world[0, [file.names.index(n) for n in common], :3, 3]
    scale, rotation, shift = similarity(want, got)
    error = np.linalg.norm(want * scale @ rotation.T + shift - got, axis=-1)
    size, at = extent(got), int(error.argmax())
    if not size > 0:
        return Check("C4-joints", Status.FAIL, evidence, where=where, note="the zero file has zero extent")
    value = float(error[at] / size)
    return Check("C4-joints", Status.FAIL if not value <= cfg["untraced_tol"] else Status.PASS, evidence, value, cfg["untraced_tol"],
                 where | {"joint": common[at]},
                 f"{len(common)} joints against {rec.rest_pose.name} ({'authored by tools/tpose_author.py' if authored else 'the vendor file, unchanged'})"
                 f" after one fitted similarity (worst {common[at]})")


def c4_zero(case: Case, cfg: dict) -> list[Check]:
    """Identity rotations in PoseYdon's frame (the `5_zero` file) against the rig's reference pose.
    ``C4-mesh`` (`pipeline`): the file's skin under its joints against the same skin under `rest.npz:rest`
    (the rest pose the corpus was built from, in the vendor's space), after ONE fitted similarity, judged like
    `C3-shape`; the vendor skin enters through the fit and C2a holds the file's skin to it. A chain without
    `rest_relative` (`data=unimate`) has no rest at identity: NOT RUN. ``C4-placement`` compares that fit
    with the placement recorded for the zero, like `C3-placement`. ``C4-joints`` is the pose's independent
    leg (see `_zero_joints`). ``C4-pose`` (`pipeline`, the manifest's own definition): up +Y, facing +Z,
    grounded. ``C4-symmetry`` needs a limit format the owner has not set (`configs/verify/limits.yaml`,
    empty): NOT RUN."""
    evidence, where, path = _evidence(case), {"rig": case.rig, "clip": case.action}, case.files.get(ZERO)
    declared = case.rig in cfg.get("limits", {})
    symmetry = Check("C4-symmetry", Status.NOT_RUN, Evidence.PIPELINE, where=where,
                     note=(f"limits declared for {case.rig}, but symmetry is not implemented: it needs a limit format"
                           if declared else f"no symmetry limits declared for {case.rig} (configs/verify/limits.yaml is owner-set)"))
    if path is None or not case.data_name or case.floor:
        why = f"needs a built round trip's {ZERO} file"
        return [Check(c, Status.NOT_RUN, Evidence.PIPELINE, note=why) for c in ("C4-mesh", "C4-placement", "C4-joints", "C4-pose")] + [symmetry]

    def stop(check: str, status: Status, note: str, ev=Evidence.PIPELINE) -> Check:
        return Check(check, status, ev, where=where, note=note)

    try:
        rec = zero_record(case)
        file = _joints(path, lambda f: f[:1])
        skins, inventory = reference_skin(case), _read(case.skin_fbx).inventory
    except Exception as error:  # noqa: BLE001 - an unreadable file or record is NOT RUN
        return [stop(c, Status.NOT_RUN, f"unreadable: {error}") for c in ("C4-mesh", "C4-placement", "C4-joints", "C4-pose")] + [symmetry]
    modelled = _modelled(skins, inventory, file)
    usable = [] if isinstance(modelled, str) else modelled[0]
    rotation, placement = None, None
    rest = {}
    if "rest_relative" in rec.stages:
        with np.load(rec.rest, allow_pickle=True) as z:
            rest = {str(n): m for n, m in zip(z["names"], z["rest"])}
    if "rest_relative" not in rec.stages:
        mesh = stop("C4-mesh", Status.NOT_RUN, "the chain has no rest_relative, so identity is not the rig's rest pose")
    elif isinstance(modelled, str) or not usable:
        mesh = stop("C4-mesh", Status.NOT_RUN, modelled if isinstance(modelled, str) else "no skinned mesh the verifier models: " + "; ".join(modelled[1]))
    elif (lacking := [n for n in file.names if n not in rest]):
        mesh = stop("C4-mesh", Status.NOT_RUN, f"rest.npz has no node named {lacking[:3]}")
    else:
        links = {n for s in usable for n in s.links}
        held = [j for j in modelled[2] if file.names[j] in rec.fixed and file.names[j] not in links]
        compare = [j for j in modelled[2] if j not in held]
        fitted = _fitted(path, file, usable, compare, np.array([[rest[n] for n in file.names]]), np.zeros(1, dtype=int), cfg)
        if isinstance(fitted, _Cannot):
            mesh = stop("C4-mesh", *fitted)
        else:
            gap, value, failed, scale, rotation, shift, size = fitted
            mesh = Check("C4-mesh", Status.FAIL if failed else Status.PASS, Evidence.PIPELINE, value, cfg["untraced_tol"], where | gap,
                         "after one fitted similarity; " + "; ".join([_summary(gap), *modelled[1]])
                         + (f"; {len(held)} joints held by structure not compared (`RigRecord.fixed`)" if held else ""))
            placement = _placement(rec.space, scale, rotation, shift, size, cfg, where, "C4-placement")
    placement = placement or stop("C4-placement", Status.NOT_RUN, "C4-mesh did not complete")
    return [mesh, placement, _zero_joints(case, cfg, rec, file), _zero_pose(case, cfg, rec, file, usable, rotation), symmetry]


# C5: sanity of a file that has no reference ---------------------------------------------------

C5 = ("range", "stretch", "area", "bone-distance")
LIMITS = Path("tmp/verify/limits")
LIMITS_VERSION = 4  # bump with any change to what `_measure` computes: it invalidates every cached limit


def _frame_numbers(scene: FBX) -> np.ndarray:
    first, last = scene.key_range()  # not `frames()`: a vendor file's time span can end a key early
    return np.arange(round(first), round(last) + 1)


def _worst_ratio(a: np.ndarray, b: np.ndarray, floor: float) -> float:
    """Largest ``max(a/b, b/a)`` over entries whose ``b`` exceeds ``floor``; non-finite counts as infinite,
    and NaN (not infinite) means there was nothing to measure."""
    keep = b > floor
    if not keep.any():
        return float("nan")
    r = np.nan_to_num(a[..., keep] / b[keep], nan=np.inf, posinf=np.inf)
    return float(np.maximum(r, 1 / np.maximum(r, 1e-300)).max())


def _mesh_ratios(skin: SkinData, world: dict[str, np.ndarray]) -> tuple[float, float, float]:
    """Worst edge stretch, triangle area ratio (both ``max(x, 1/x)``) and change of the vertex-to-dominant-bone
    distance (fraction of the rest mesh's extent) of ``skin`` under ``world`` against its own bind. NaN: nothing to measure."""
    rest = deform(skin, {n: m[None] for n, m in zip(skin.links, skin.link_bind)})[0]
    moved = deform(skin, world)
    size, tri = extent(rest), skin.faces
    edges = np.unique(np.sort(np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]]), axis=1), axis=0)

    def length(p, e):
        return np.linalg.norm(p[..., e[:, 0], :] - p[..., e[:, 1], :], axis=-1)

    def area(p):
        return 0.5 * np.linalg.norm(np.cross(p[..., tri[:, 1], :] - p[..., tri[:, 0], :], p[..., tri[:, 2], :] - p[..., tri[:, 0], :]), axis=-1)

    best, dominant = np.zeros(len(rest)), np.full(len(rest), -1)
    for c, (index, weight) in enumerate(zip(skin.indices, skin.weights)):
        win = weight > best[index]
        best[index[win]], dominant[index[win]] = weight[win], c
    held = dominant >= 0
    bone = np.stack([world[n][:, :3, 3] for n in skin.links], axis=1)  # (F, C, 3)
    away = np.linalg.norm(moved[:, held] - bone[:, dominant[held]], axis=-1)
    away0 = np.linalg.norm(rest[held] - skin.link_bind[dominant[held], :3, 3], axis=-1)
    change = np.nan_to_num(np.abs(away - away0) / size, nan=np.inf, posinf=np.inf).max() if held.any() else float("nan")
    return (_worst_ratio(length(moved, edges), length(rest, edges), 1e-9 * size),
            _worst_ratio(area(moved), area(rest), 1e-8 * size**2), float(change))


@dataclass
class _Ref:
    """The rig's reference skin as C5 uses it: the meshes the verifier models, why others are left out, and the
    rig's rest extent (the rest mesh, else the bind joints). Limits and files share it, so they share one baseline."""

    skins: list[SkinData]
    why: list[str]
    size: float
    bind: dict[str, np.ndarray]  # joint -> its world matrix in the reference's bind pose
    axis: dict[str, np.ndarray | None]  # joint -> bone axis in its own frame (towards its children), None for a leaf


def _reference(path: Path) -> _Ref:
    scene = FBX.read(path)
    try:
        skins, inventory = scene.skin_data(), scene.inventory()
        bind_world, names, parents = scene.bind_or_first(), scene.names, scene.parents
        bind = bind_world[:, :3, 3]
        skeleton = [i for i in range(len(scene.names)) if scene.is_skeleton(i)] or list(range(len(scene.names)))
    finally:
        scene.close()
    usable = [s for s in skins if not _skipped(inventory, s.mesh)]
    why = [f"mesh {m} {_skipped(inventory, m)}" for m in {s.mesh for s in skins} if _skipped(inventory, m)]
    rest = [deform(s, {n: m[None] for n, m in zip(s.links, s.link_bind)})[0] for s in usable]
    size = extent(np.concatenate(rest)) if rest else extent(bind[skeleton])
    axis = {}
    for j, n in enumerate(names):
        kids = [(np.linalg.inv(bind_world[j]) @ bind_world[c])[:3, 3] for c, p in enumerate(parents) if p == j]
        kids = [k / np.linalg.norm(k) for k in kids if np.linalg.norm(k) > 0]
        mean = np.mean(kids, axis=0) if kids else None
        axis[n] = None if mean is None or not np.linalg.norm(mean) > 1e-9 else mean / np.linalg.norm(mean)
    return _Ref(usable, why, size, {n: bind_world[j] for j, n in enumerate(names)}, axis)


def _unscaled(m: np.ndarray) -> np.ndarray:
    return m / np.linalg.norm(m, axis=-2, keepdims=True)


def _joint_motion(world: np.ndarray, names, parents, fps: float, ref: _Ref, held=frozenset()) -> dict[str, list[float]]:
    """Per joint of the reference, over every key: its largest swing (degrees of bend off its rest rotation, the part
    that is not a turn about the bone axis), largest |twist| (degrees about the bone axis, from the rest rotation) and
    largest angular speed (degrees per frame at 30 fps). A joint with no bone axis (a leaf) has its whole turn as swing.
    A joint the file holds by structure (``held``, `RigRecord.fixed`) is left out, and one whose parent is held is
    measured against its nearest ancestor that is not, in the file and in the reference alike: a held joint's
    motion would otherwise land on its child. A non-finite matrix is infinite."""
    out = {}
    for j, n in enumerate(names):
        if n not in ref.bind or n in held:
            continue
        a = parents[j]
        while a >= 0 and names[a] in held:
            a = parents[a]
        if a >= 0 and names[a] not in ref.bind:
            continue
        m = world[:, j] if a < 0 else np.linalg.inv(world[:, a]) @ world[:, j]
        rest = _unscaled((ref.bind[n] if a < 0 else np.linalg.inv(ref.bind[names[a]]) @ ref.bind[n])[:3, :3])
        bad = ~np.isfinite(m).all(axis=(1, 2))
        local = _unscaled(np.where(bad[:, None, None], np.eye(4), m)[:, :3, :3])
        axis = ref.axis[n]
        delta = Rotation.from_matrix(rest.T @ local)
        if axis is None:
            swing, twist = delta.magnitude(), np.zeros(len(local))
        else:
            q = delta.as_quat() * np.where(delta.as_quat()[:, 3:] < 0, -1.0, 1.0)
            theta = 2 * np.arctan2(q[:, :3] @ axis, q[:, 3])
            swing = (delta * Rotation.from_rotvec(theta[:, None] * axis).inv()).magnitude()
            twist = np.abs(theta)
        step = (Rotation.from_matrix(local[:-1]).inv() * Rotation.from_matrix(local[1:])).magnitude() if len(local) > 1 else np.zeros(1)
        vals = [np.degrees(swing), np.degrees(twist), np.degrees(step) * fps / 30]
        out[n] = [float("inf") if bad.any() else float(v.max()) for v in vals]
    return out


def joint_limits(clip_joints: dict[str, dict[str, list[float]]], margin: float, floor: float) -> dict[str, list[float]]:
    """Per joint, the largest swing, |twist| and speed over the clips, times ``margin``, at least ``floor`` degrees
    (a joint no clip moves must not fail on float noise). A joint with a non-finite value has no limit."""
    seen: dict[str, np.ndarray] = {}
    for joints in clip_joints.values():
        for n, v in joints.items():
            seen[n] = np.maximum(seen.get(n, 0.0), v)
    return {n: np.maximum(v * margin, floor).tolist() for n, v in seen.items() if np.isfinite(v).all()}


def _measure(path: Path, n_frames: int, ref: _Ref, held=frozenset()) -> dict:
    """``range`` (every key: how far a skeleton joint strays from the skeleton's centroid, as a fraction of the
    reference's rest extent; infinite if any is not a number) and the worst ``stretch``, ``area`` and ``bone-distance``
    of the REFERENCE skin driven by the file's joints (as C2a does), over ``n_frames`` sampled frames. ``why`` says what is missing."""
    scene = FBX.read(path)
    try:
        world, names, parents, fps = scene.world(_frame_numbers(scene)), scene.names, scene.parents, scene.fps
        skeleton = [i for i in range(len(names)) if scene.is_skeleton(i)] or list(range(len(names)))
    finally:
        scene.close()
    joints = {n: world[:, j] for j, n in enumerate(names)}
    pos = world[:, skeleton][..., :3, 3]
    reach = np.nan_to_num(np.linalg.norm(pos - pos.mean(axis=1, keepdims=True), axis=-1), nan=np.inf, posinf=np.inf)
    out = {"range": float(reach.max() / max(ref.size, 1e-12)), "joints": _joint_motion(world, names, parents, fps, ref, held)}
    sampled = {n: m[sample_frames(world, n_frames)] for n, m in joints.items()}
    why, rows = list(ref.why), []
    for skin in ref.skins:
        if set(skin.links) - joints.keys():
            why.append(f"mesh {skin.mesh}: skin links missing from the file: {sorted(set(skin.links) - joints.keys())[:3]}")
            continue
        try:
            rows.append(_mesh_ratios(skin, sampled))
        except np.linalg.LinAlgError as error:
            why.append(f"mesh {skin.mesh}: a bind matrix is singular ({error})")
    if rows:
        columns = np.array(rows).T  # a mesh with nothing to measure is NaN, and drops out unless every mesh is
        out |= {k: float("nan") if np.isnan(c).all() else float(np.nanmax(c)) for k, c in zip(C5[1:], columns)}
    elif not why:
        why.append("the reference holds no skinned mesh")
    return out | {"why": why}


def _stamp(path: Path) -> list:
    stat = path.stat()
    return [path.name, stat.st_size, stat.st_mtime]


def rig_limits(data_name: str, rig: str, cfg: dict) -> dict:
    """The largest ``range``, edge stretch, triangle area ratio and vertex-to-dominant-bone distance change seen
    across the rig's vendor clips, times ``stretch_margin``: ``derived`` evidence. ``joints`` holds the same per joint for
    its swing, |twist| (degrees off the rest rotation) and angular speed, with a floor of ``joint_floor_deg``. Each clip's joints drive the rig's
    REFERENCE skin (as C2a does), so the limits and the file under test share one bind and one rest extent; ``n_frames``
    frames are sampled per clip. Cached at ``tmp/verify/limits/<data>/<rig>.json`` under a stamp of `LIMITS_VERSION`,
    ``n_frames``, the margin, the reference file and every clip's name, size and mtime: any change re-measures.
    A quantity no clip could measure is absent from ``limit``; a clip that cannot drive the skin is named in ``skipped``."""
    from poseydon.build.pipeline import compose_build  # noqa: PLC0415
    from poseydon.build.sources import fbx_index  # noqa: PLC0415
    from poseydon.export import RigRecord  # noqa: PLC0415

    data = compose_build([f"data={data_name}"]).data
    reference = rig_skin(data, rig)
    held = sorted(RigRecord.load(Path(data.paths.root), rig).fixed)
    clips = sorted((a, Path(data.paths.raw) / "source" / f) for r, a, f in fbx_index(data.source.index) if r == rig)
    stamp = {"version": LIMITS_VERSION, "n_frames": cfg["n_frames"], "margin": cfg["stretch_margin"],
             "held": held, "reference": _stamp(reference), "clips": [_stamp(p) for _, p in clips]}
    cache = LIMITS / data_name / f"{rig}.json"
    try:
        saved = json.loads(cache.read_text())
        if saved.get("stamp") == stamp:
            return _with_joints(saved, cfg)
    except (OSError, ValueError):
        pass
    ref = _reference(reference)
    seen: dict[str, float] = {}
    clip_joints: dict[str, dict] = {}
    skipped: dict[str, str] = {}
    used = 0
    for action, path in clips:
        try:
            got = _measure(path, cfg["n_frames"], ref, frozenset(held))
        except Exception as error:  # noqa: BLE001 - a clip that cannot be read gives no limit
            skipped[action] = f"unreadable: {error}"
            continue
        if not np.isfinite(got["range"]):
            skipped[action] = "has keys that are not numbers"
            continue
        if not all(np.isfinite(v).all() for v in got["joints"].values()):
            skipped[action] = "has joint keys that are not numbers"
            continue
        used += 1
        clip_joints[action] = got["joints"]
        if got["why"]:
            skipped[action] = "; ".join(got["why"])
        for k in C5:
            if k in got and np.isfinite(got[k]):
                seen[k] = max(seen.get(k, 0.0), got[k])
    out = {"stamp": stamp, "n_frames": cfg["n_frames"], "reference": str(reference), "used": used, "seen": seen,
           "limit": {k: v * cfg["stretch_margin"] for k, v in seen.items()}, "skipped": skipped,
           "held": held, "clip_joints": clip_joints}
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(out, indent=2))
    return _with_joints(out, cfg)


def _with_joints(limits: dict, cfg: dict) -> dict:
    """``limits`` with the joint limits for the current ``joint_margin`` and ``joint_floor_deg``: the cache holds each
    clip's raw per-joint maxima, so these two can change without re-measuring."""
    params = [cfg["joint_margin"], cfg["joint_floor_deg"]]
    return limits | {"joint_params": params, "joints": joint_limits(limits["clip_joints"], *params)}


def c5_sanity(file: Path, rig: str, limits: dict) -> list[Check]:
    """``C5-range``, ``-stretch``, ``-area`` and ``-bone-distance`` of one FBX against ``rig_limits`` (`derived`:
    the rig's own vendor clips, with a margin), the rig's reference skin driven by the file's joints. A quantity with
    no limit, or that the reference cannot give (no mesh, nothing to measure), is NOT RUN, named; a value that is
    not a finite number within its limit is FAIL. The global worst values only catch gross faults (a flipped limb), not a
    30 degree joint error or a twist (docs/reference/tooling.md)."""
    stamp = hashlib.sha256(json.dumps([limits.get("stamp"), limits.get("joint_params")], sort_keys=True).encode()).hexdigest()[:12]
    where = {"rig": rig, "file": file.name, "limits": stamp}

    def line(k: str, status: Status, note: str, value=None, limit=None, evidence=None) -> Check:
        return Check(f"C5-{k}", status, evidence, value, limit, where, note)

    try:
        got = _measure(file, limits.get("n_frames", 8), _reference(Path(limits["reference"])), frozenset(limits.get("held", ())))
    except Exception as error:  # noqa: BLE001 - an unreadable file is NOT RUN
        return [line(k, Status.NOT_RUN, f"unreadable: {error}") for k in (*C5, "joints")]
    out = [_c5_joints(got["joints"], limits, where)]
    for k in C5:
        limit = limits.get("limit", {}).get(k)
        if limit is None or not limit > 0:
            clips = f"{limits.get('used', 0)} usable vendor clips"
            note = f"no limit: {clips}" + (f" ({'; '.join(sorted(set(limits['skipped'].values()))[:2])})" if limits.get("skipped") else "")
            out.append(line(k, Status.NOT_RUN, note))
        elif k not in got or np.isnan(got[k]):
            out.append(line(k, Status.NOT_RUN, "nothing to measure on the reference skin" + (": " + "; ".join(got["why"]) if got["why"] else "")))
        else:
            ok = got[k] <= limit
            out.append(line(k, Status.PASS if ok else Status.FAIL,
                            f"limit: the largest of {limits['used']} vendor clips x margin" + ("" if np.isfinite(got[k]) else "; not a finite number")
                            + (f"; left out: {'; '.join(got['why'])}" if got["why"] else ""),
                            got[k], limit, Evidence.DERIVED))
    return out


def _c5_joints(found: dict, limits: dict, where: dict) -> Check:
    """``C5-joints``: each joint's swing, |twist| and angular speed over its vendor limit; the value is the worst ratio
    (FAIL above 1), the joint and component are named. Held joints are not measured (`_joint_motion`). A joint with no
    finite limit is left out and counted, as are the reference's joints missing from the file."""
    bound = limits.get("joints", {})
    ratios = {(n, c): v[i] / bound[n][i] for n, v in found.items() if n in bound for i, c in enumerate(("swing", "twist", "speed"))}
    if not ratios:
        return Check("C5-joints", Status.NOT_RUN, None, where=where, note="no joint of the file has a vendor limit")
    (joint, part), worst = max(ratios.items(), key=lambda kv: np.inf if np.isnan(kv[1]) else kv[1])
    unbounded = len(found) - len({n for n, _ in ratios})
    missing = len({n for joints in limits["clip_joints"].values() for n in joints} - found.keys())
    margin, floor = limits["joint_params"]
    return Check("C5-joints", Status.PASS if worst <= 1 else Status.FAIL, Evidence.DERIVED, float(worst), 1.0, where | {"joint": joint, "part": part},
                 f"worst of {len(ratios)} joint quantities over the largest of {limits['used']} vendor clips x {margin} (floor {floor} deg)"
                 + (f"; {unbounded} joints have no finite limit" if unbounded else "")
                 + (f"; {missing} reference joints are missing from the file" if missing else ""))


def c5_stability(file: Path, record, prepared, data, cfg: dict) -> Check:
    """Read ``file`` (the clip on the user's rig) back as the pipeline reads a vendor clip, prepare it again with the
    rig's recorded chain, and compare with ``prepared`` (the generator's clip: the model view, or the prepared rig)
    after ONE fitted similarity, as a fraction of extent against ``stability_tol``: `face_axis` re-turns each clip to
    +Z, so the two agree only up to a similarity. `pipeline` evidence: the reference itself went through the pipeline.
    Blind spots: joint positions only (a rotation that leaves positions in place is unseen), and bone lengths, which
    reading restores from the rest asset."""
    from poseydon.build.prepare import similarity  # noqa: PLC0415
    from poseydon.build.sources import SOURCES  # noqa: PLC0415
    from poseydon.core.skeleton import SkeletonManifest, resolve  # noqa: PLC0415
    from poseydon.features.reduce import apply_reduction  # noqa: PLC0415
    from scripts.build_fbx_cache import read  # noqa: PLC0415

    where = {"rig": record.rig, "file": file.name}

    def line(status: Status, note: str, value=None) -> Check:
        return Check("C5-stability", status, Evidence.PIPELINE, value, cfg["stability_tol"], where, note)

    try:
        manifest = SkeletonManifest.load(Path(record.corpus) / "rigs" / record.rig / "manifest.yaml")
        source = SOURCES.build(data.source)
        # Couples to `FbxCache` internals (`_rest_offsets` primes the rig from the real cache, `cache` is then pointed
        # at a temp npz of the file, whose empty `source` skips the staleness check): a refactor of it turns this NOT RUN.
        source._rest_offsets(data, manifest)
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / record.rig).mkdir()
            np.savez(Path(tmp) / record.rig / "stable.npz", **(read(file) | {"source": np.array("")}))
            source.cache = Path(tmp)
            clip = source.clip(data, manifest, "stable").anim
        again, _ = record.chain.apply(clip, resolve(manifest, clip.names), record.params)
        model = tuple(prepared.names) == record._model_names
        want = record.expand(prepared if model else apply_reduction(prepared, record.reduction))
    except Exception as error:  # noqa: BLE001
        return line(Status.NOT_RUN, f"could not prepare the file again: {type(error).__name__}: {error}")
    if again.n_frames != want.n_frames:
        return line(Status.FAIL, f"{again.n_frames} frames after preparing again, {want.n_frames} expected")
    common = [n for n in want.names if n in again.names]
    if len(common) < 4:
        return line(Status.NOT_RUN, f"{len(common)} joints in common")
    a = again.global_positions()[:, [again.names.index(n) for n in common]]
    b = want.global_positions()[:, [want.names.index(n) for n in common]]
    scale, rotation, shift = similarity(a.reshape(-1, 3), b.reshape(-1, 3))
    error = np.nan_to_num(np.linalg.norm(a * scale @ rotation.T + shift - b, axis=-1), nan=np.inf, posinf=np.inf)
    size = extent(b.reshape(-1, 3))
    if not size > 0:
        return line(Status.FAIL, "the expected clip has zero extent")
    value = float(error.max() / size)
    at = np.unravel_index(error.argmax(), error.shape)
    return line(Status.PASS if value <= cfg["stability_tol"] else Status.FAIL,
                f"{len(common)} joints after one fitted similarity (worst {common[at[1]]}, frame {at[0]})", value)

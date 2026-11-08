"""The verifier's checks. Each takes a `Case` and returns `Check` lines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from poseydon.build.prepare import similarity
from poseydon.io.fbx import FBX, Inventory, SkinData
from tools.verify import laws
from tools.verify.case import Case, LossRecord, ZeroRecord, loss_record, recorded_space, zero_record
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
            found = _differences(reference, got, clip.inventory, tol, case.window, name != ZERO)
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
    recorded = np.cbrt(abs(np.linalg.det(space[:3, :3])))
    turned = float(np.degrees(Rotation.from_matrix(rotation @ (space[:3, :3] / recorded).T).magnitude()))
    found = {"scale": abs(scale / recorded - 1), "rotation_deg": turned,
             "offset": float(np.linalg.norm(shift - space[:3, 3]) / size)}
    limit = {"scale": cfg["placement_scale_tol"], "rotation_deg": cfg["placement_rotation_deg"], "offset": cfg["placement_offset_tol"]}
    bad = [k for k in found if not found[k] <= limit[k]]
    out.append(Check("C3-placement", Status.FAIL if bad else Status.PASS, Evidence.PIPELINE,
                     max(found[k] / limit[k] for k in found), 1.0, where | {"found": found, "limit": limit},
                     "fitted similarity against the recorded `space` (value: worst error over its limit; angle is the whole "
                     "rotation between them)" + (f"; over: {', '.join(bad)}" if bad else "")))
    return out


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
            return Check("C4-pose", Status.NOT_RUN, Evidence.PIPELINE, where=where, note=f"facing joint missing from the file: {error}")
        yaw = np.radians(rec.extra_yaw_deg)
        found["facing_deg"] = _angle(np.cross([0.0, 1.0, 0.0], across), np.array([np.sin(yaw), 0.0, np.cos(yaw)]))
    else:
        notes.append("facing not judged: the chain has no face_rest_axis")
    if rec.ground:
        names = set(rec.joints) & set(file.names)
        if rec.ground == "mesh":
            if not skins:
                return Check("C4-pose", Status.NOT_RUN, Evidence.PIPELINE, where=where, note="ground mode `mesh`, no skin the verifier models")
            cloud = np.concatenate([deform(s, {n: file.world[:, j] for j, n in enumerate(file.names)})[0] for s in skins])
        else:
            cloud = np.array([world[j, :3, 3] for j, n in enumerate(file.names) if n in names])
        found["ground"] = float(abs(cloud[:, 1].min()) / extent(cloud))
        where = where | {"ground_mode": rec.ground}
    else:
        notes.append("ground not judged: the chain has no ground stage")
    if not found:
        return Check("C4-pose", Status.NOT_RUN, Evidence.PIPELINE, where=where, note="; ".join(notes))
    bad = [k for k in found if not found[k] <= limit[k]]
    return Check("C4-pose", Status.FAIL if bad else Status.PASS, Evidence.PIPELINE, max(found[k] / limit[k] for k in found), 1.0,
                 where | {"found": found, "limit": limit},
                 "pipeline evidence: the manifest's own definition (value: worst error over its limit)"
                 + (f"; over: {', '.join(bad)}" if bad else "") + ("; " + "; ".join(notes) if notes else ""))


def c4_zero(case: Case, cfg: dict) -> list[Check]:
    """Identity rotations in PoseYdon's frame (the `5_zero` file) against the rig's reference pose.
    ``C4-mesh`` (independent): the file's skin against the vendor reference skin driven by `rest.npz:rest`
    (the authored rest pose in the vendor's space), after ONE fitted similarity, judged like C3-shape; a
    chain without `rest_relative` (`data=unimate`) has no rest at identity, so it is NOT RUN. ``C4-pose``
    (`pipeline` evidence only, the manifest's own definition): up +Y, facing +Z, grounded. ``C4-symmetry``
    needs limits the owner sets in `configs/verify/limits.yaml`, none today: NOT RUN."""
    evidence, where, path = _evidence(case), {"rig": case.rig, "clip": case.action}, case.files.get(ZERO)
    symmetry = Check("C4-symmetry", Status.NOT_RUN, Evidence.PIPELINE, where=where,
                     note=f"no symmetry limits declared for {case.rig} (configs/verify/limits.yaml is owner-set)")
    if path is None or not case.data_name or case.floor:
        why = f"needs a built round trip's {ZERO} file"
        return [Check("C4-mesh", Status.NOT_RUN, evidence, note=why), Check("C4-pose", Status.NOT_RUN, Evidence.PIPELINE, note=why), symmetry]

    def stop(check: str, status: Status, note: str, ev=evidence) -> Check:
        return Check(check, status, ev, where=where, note=note)

    try:
        rec = zero_record(case)
        file = _joints(path, lambda f: f[:1])
        skins, inventory = reference_skin(case), _read(case.skin_fbx).inventory
    except Exception as error:  # noqa: BLE001 - an unreadable file or record is NOT RUN
        return [stop("C4-mesh", Status.NOT_RUN, f"unreadable: {error}"), stop("C4-pose", Status.NOT_RUN, f"unreadable: {error}", Evidence.PIPELINE), symmetry]
    modelled = _modelled(skins, inventory, file)
    usable = [] if isinstance(modelled, str) else modelled[0]
    rotation = None
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
            gap, value, failed, _, rotation, _, _ = fitted
            mesh = Check("C4-mesh", Status.FAIL if failed else Status.PASS, evidence, value, cfg["untraced_tol"], where | gap,
                         "after one fitted similarity; " + "; ".join([_summary(gap), *modelled[1]])
                         + (f"; {len(held)} joints held by structure not compared (`RigRecord.fixed`)" if held else ""))
    return [mesh, _zero_pose(case, cfg, rec, file, usable, rotation), symmetry]

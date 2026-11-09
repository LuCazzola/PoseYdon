"""BVH files as the verifier's input: B1 layout, B2 joints, B3 mesh (driven), B4 reader.

A BVH carries no mesh. B3 runs only with a `Case` (a rig with an FBX skin); without one the mesh checks
are not claimed and the verdict rests on the joints."""

from __future__ import annotations

import re
import tempfile
from pathlib import Path
from typing import Literal

import numpy as np

from poseydon.build.prepare import similarity
from poseydon.io.bvh import BVH
from poseydon.io.fbx import FBX
from tools.verify.case import Case
from tools.verify.report import Check, Evidence, Status
from tools.verify.skin import extent


def _layout(path: Path) -> list[str]:
    """HIERARCHY with its numbers left out (joints, nesting, channels), so a moved offset is B2's finding."""
    head = path.read_text().partition("MOTION")[0]
    return [re.sub(r"^(OFFSET).*", r"\1", line.strip()) for line in head.splitlines() if line.strip()]


def _positions(path: Path) -> tuple[BVH, np.ndarray]:
    bvh = BVH.read(path)
    return bvh, bvh.to_animation().global_transforms()[0]


def _line(id: str, ok: bool, evidence: Evidence | None, note: str, value=None, limit=None, where=None) -> Check:
    return Check(id, Status.PASS if ok else Status.FAIL, evidence, value, limit, where or {}, note)


def _b1(file: Path, source: Path, kind: str) -> Check | None:
    if kind != "bvh":
        return None  # no user BVH to match: not applicable, so a meshless BVH can PASS on its joints
    want, got = _layout(source), _layout(file)
    notes = [f"line {i}: {a!r} -> {b!r}" for i, (a, b) in enumerate(zip(want, got)) if a != b][:3]
    if len(want) != len(got):
        notes.append(f"{len(want)} -> {len(got)} header lines")
    fps = (BVH.read(source).fps, BVH.read(file).fps)
    if not np.isclose(*fps, rtol=1e-6):
        notes.append(f"fps {fps[0]:.6g} -> {fps[1]:.6g}")
    return _line("B1-layout", not notes, Evidence.INDEPENDENT, "; ".join(notes), float(len(notes)), 0.0, {"file": file.name})


def _b2(file: Path, source: Path, kind: str, tol: float) -> Check:
    got, pos = _positions(file)
    evidence = Evidence.DERIVED if "cache/repaired/" in source.as_posix() else Evidence.INDEPENDENT
    where = {"file": file.name}
    if kind == "bvh":
        want, ref = _positions(source)
        if len(ref) != len(pos):
            return _line("B2-joints", False, evidence, f"{len(ref)} -> {len(pos)} frames", where=where)
        gone = [n for n in want.names if n not in got.names]
        if gone:
            return _line("B2-joints", False, evidence, f"missing joints {gone[:3]}", where=where)
        error = np.linalg.norm(pos[:, [got.names.index(n) for n in want.names]] - ref, axis=-1)
        finite = np.isfinite(error).all()
        worst = float(error.max() / extent(ref[0])) if finite else np.nan
        at = want.names[int(error.max(axis=0).argmax())] if finite else "a non-finite position"
        return _line("B2-joints", worst <= tol, evidence, f"world positions, worst at {at}", worst, tol, where)
    scene = FBX.read(source)
    try:
        numbers = scene.frames()
        world = scene.world(numbers) if len(numbers) == len(pos) else None
        names = scene.names
    finally:
        scene.close()
    if world is None:  # a bone length is no stand-in: vendor rigs animate position channels, so it changes with the clip
        return Check("B2-joints", Status.NOT_RUN, evidence, where=where,
                     note=f"the source FBX has {len(numbers)} frames, the file {len(pos)}: it is not the file's own clip")
    common = [(got.names.index(n), names.index(n)) for n in got.names if n in names]
    if not common:
        return Check("B2-joints", Status.NOT_RUN, evidence, where=where, note="no joint of the file is named in the source")
    a, b = pos[:, [c[0] for c in common]], world[..., :3, 3][:, [c[1] for c in common]]
    s, r, t = similarity(a.reshape(-1, 3), b.reshape(-1, 3))  # one similarity for units and axes, then every frame
    error = np.linalg.norm(s * a @ r.T + t - b, axis=-1)
    worst = float(error.max() / extent(b[0])) if np.isfinite(error).all() else np.nan
    return _line("B2-joints", worst <= tol, evidence, f"world positions after one similarity (scale {s:.5g}) over {len(common)} joints", worst, tol, where)


def _b3(file: Path, case: Case, cfg: dict) -> Check:
    """The file's joints drive the rig's reference skin through `FBX.from_animation`, and C5's mesh limits
    judge that: `driven` evidence, a probe the verifier made. Joints with no rotation channels (leaves) stay at the asset's rest."""
    from poseydon.build.pipeline import compose_build  # noqa: PLC0415
    from poseydon.export import RigRecord  # noqa: PLC0415
    from tools.verify.checks import c5_sanity, rig_limits  # noqa: PLC0415

    root = Path(compose_build([f"data={case.data_name}"]).data.paths.root)
    record = RigRecord.load(root, case.rig)
    bvh = BVH.read(file)
    anim = bvh.to_animation()
    held = (*record.fixed, *(n for n, spec in zip(bvh.names, bvh.channels) if not any("rotation" in c for c in spec)))  # a BVH has no rotation for these
    with tempfile.TemporaryDirectory() as tmp:
        driven = Path(tmp) / "driven.fbx"
        FBX.from_animation(anim, record.asset, fixed=held).write(driven)
        lines = [c for c in c5_sanity(driven, case.rig, rig_limits(case.data_name, case.rig, cfg)) if c.id != "C5-joints"]
    bad = [c for c in lines if c.status is not Status.PASS]
    status = Status.FAIL if any(c.status is Status.FAIL for c in bad) else Status.NOT_RUN if bad or not lines else Status.PASS
    return Check("B3-mesh", status, Evidence.DRIVEN, where={"file": file.name}, note="; ".join(f"{c.id} {c.status.value} {c.note}" for c in bad))


def _b4(file: Path, blender: Path | None, tol: float) -> Check:
    """Blender's BVH importer against our reader: joint positions after one similarity (axes) whose scale must stay 1."""
    where = {"file": file.name}
    if blender is None or not blender.is_file():
        error = None if blender is None else blender.with_suffix(".error")
        if error is not None and error.is_file():
            return _line("B4-reader", False, Evidence.INDEPENDENT, f"Blender import failed: {error.read_text()}", where=where)
        return Check("B4-reader", Status.NOT_RUN, Evidence.INDEPENDENT, where=where, note="no Blender output: run tools/verify/blender_read.py")
    bvh, pos = _positions(file)
    with np.load(blender) as z:
        names, theirs, frames = [str(n) for n in z["names"]], z["positions"], z["frames"]
    common = [(bvh.names.index(n), i) for i, n in enumerate(names) if n in bvh.names]
    if not common or not len(frames) or frames.max() >= len(pos):
        return Check("B4-reader", Status.NOT_RUN, Evidence.INDEPENDENT, where=where, note="no joints or frames to compare")
    a, b = pos[frames][:, [c[0] for c in common]], theirs[:, [c[1] for c in common]]
    s, r, t = similarity(a.reshape(-1, 3), b.reshape(-1, 3))
    error, size = np.linalg.norm(s * a @ r.T + t - b, axis=-1), extent(a[0])
    worst = float(error.max() / size) if np.isfinite(error).all() and size > 0 else np.nan
    return _line("B4-reader", worst <= tol and abs(s - 1) <= tol, Evidence.INDEPENDENT,
                 f"{len(common)} joints, {len(frames)} frames, fitted scale {s:.6g}", worst, tol, where)


def bvh_checks(file: Path, source: Path, kind: Literal["bvh", "fbx"], case: Case | None, cfg: dict,
               blender: Path | None = None) -> list[Check]:
    """B1 layout, B2 joints, B4 reader of a BVH ``file`` against its ``source`` (a BVH, or the FBX it came from), and
    B3 mesh when ``case`` names a rig with an FBX skin. ``blender`` is `blender_read.py`'s output for the file.
    An unreadable file fails every line it would have judged."""
    try:
        got = [c for c in (_b1(file, source, kind), _b2(file, source, kind, cfg["bvh_tol"])) if c is not None]
        if case is not None:
            got.append(_b3(file, case, cfg))
        return got + [_b4(file, blender, cfg["blender_tol"])]
    except Exception as error:  # noqa: BLE001 - an unreadable file is a finding
        return [Check(i, Status.FAIL, Evidence.INDEPENDENT, note=f"unreadable: {error}") for i in ("B1-layout", "B2-joints")]

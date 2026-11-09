"""Judge the FBX files the pipeline returns against the user's vendor rig.

    build  --data D --rig R [--clip C] [--smoke] [--all-frames] [--floor] [--out DIR]   write the stage files of a round trip
    file   FILE.fbx|FILE.bvh (--reference REF | --rig R --data D [--clip C.npz]) [--out DIR]          one FBX with no source: C1, C2a (against REF, or the
                                                                              rig's reference skin) and, with --rig, C5 sanity limits
    judge  DIR                                                                run the checks, write report.json / report.txt

Runs in the `fbxsdk` container; the run directory is `<out>/<Rig>__<clip>/`, `--out` defaults to
`tmp/verify`. Exit status is non-zero unless every check is PASS or ALLOWED
(docs/parallax/pages/user-guide/verification.md).

    docker compose run --rm -T --entrypoint python -e HOME=/tmp fbxsdk scripts/verify_motion.py build --data truebones --rig Camel
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np
import yaml

from poseydon.build.pipeline import compose_build
from poseydon.build.sources import SOURCES, fbx_index
from poseydon.core.skeleton import SkeletonManifest
from poseydon.core.animation import RigidBodyAnimation
from poseydon.export import RigRecord
from poseydon.io.bvh import BVH
from poseydon.io.fbx import FBX
from scripts.build_stage_samples import build as build_stages
from tools.verify.case import Case, rig_skin, commit, load_case, sample_frames, sha256, zero_clip
from tools.verify.bvh import bvh_checks
from tools.verify.checks import ZERO, c1_inventory, c2a_skin, c6_second_reader, c2b_fidelity, c3_canonical, c4_zero, c5_sanity, c5_stability, rig_limits
from tools.verify.checks import C5
from tools.verify.report import Check, Report, Status, apply_allowances

CONFIG = Path("configs/verify")
STAGES = ("2_prepared", "3_model", "4_roundtrip")


def write_zero(record: RigRecord, path: Path) -> None:
    """The rig's zero pose (`tools/verify/case.py:zero_clip`) placed in canonical space."""
    placed, space = record.placement(zero_clip(record))
    FBX.from_animation(placed, record.asset, space, record.fixed).write(path, embed_textures=True)


def _write_case(run: Path, case: Case, extra: dict) -> None:
    named = {"clip_fbx": case.clip_fbx, "skin_fbx": case.skin_fbx, **case.files}
    (run / "case.json").write_text(json.dumps({
        "data": case.data_name, "rig": case.rig, "action": case.action,
        "files": {k: str(v) for k, v in case.files.items()},
        "clip_fbx": str(case.clip_fbx), "skin_fbx": str(case.skin_fbx),
        "frames": case.frames.tolist(), "source_frames": case.source_frames.tolist(),
        "window": None if case.window is None else [int(x) for x in case.window], "floor": case.floor,
        "sha256": {k: sha256(v) for k, v in named.items()},
        "commit": commit(), "container": os.environ.get("HOSTNAME", "unknown"), **extra,
    }, indent=2))


def build(data_name: str, rig: str, clip: str | None, out: Path, all_frames: bool = False,
          floor: bool = False) -> Path:
    """Write the stage files of one clip and ``case.json``; returns the run directory.
    ``floor`` swaps the stages for the source clip alone, written onto the rest asset with no
    preparation: what the writer and reader cost with nothing of the pipeline in between."""
    data = compose_build([f"data={data_name}"]).data
    skin = rig_skin(data, rig)
    stage = out / f".stage_{rig}"
    shutil.rmtree(stage, ignore_errors=True)  # a failed earlier build leaves one
    action = build_stages(data, rig, stage, clip)
    run = out / f"{rig}__{action}"
    shutil.rmtree(run, ignore_errors=True)
    shutil.move(stage / rig, run)
    stage.rmdir()
    record = RigRecord.load(Path(data.paths.root), rig)
    if floor:
        manifest = SkeletonManifest.load(Path(data.paths.root) / "rigs" / rig / "manifest.yaml")
        anim = SOURCES.build(data.source).clip(data, manifest, action).anim
        for stale in (*(run / f"{s}.fbx" for s in STAGES), *run.glob("*.bvh")):
            stale.unlink()
        FBX.from_animation(anim, record.asset, fixed=record.fixed).write(run / "floor.fbx", embed_textures=True)
    stages = ("floor",) if floor else STAGES
    if not floor:
        write_zero(record, run / f"{ZERO}.fbx")
    scene = FBX.read(run / f"{stages[-1]}.fbx")
    world = scene.world()
    scene.close()
    frames = np.arange(len(world)) if all_frames else sample_frames(world, load_config()["n_frames"])
    source = FBX.read(run / "1_source.fbx")
    first, last = source.key_range()  # not `frames()`: a vendor file's time span can end one key early
    numbers = np.arange(round(first), round(last) + 1)
    source.close()
    # The clip's recorded window, in the vendor file's frames; the floor file keeps every frame.
    window = None if floor else record.clip_params.get(action, {}).get("frames", {}).get("window")
    window = None if window is None else tuple(int(x) for x in window)
    numbers = numbers[slice(*window)] if window is not None else numbers
    if len(numbers) != len(world):
        raise ValueError(f"{rig} {action}: the clip's window holds {len(numbers)} frames, the file {len(world)}")
    files = stages + (() if floor else (ZERO,))
    case = Case(data_name, rig, action, {s: run / f"{s}.fbx" for s in files}, run / "1_source.fbx", skin,
                frames, numbers[frames], window, floor)
    _write_case(run, case, {})
    return run


def load_config() -> dict:
    return yaml.safe_load((CONFIG / "verify.yaml").read_text())


def judge(run: Path, accept_changed: bool = False, sheets: bool = True) -> Report:
    """Run every check on a run directory and write its report. A file that changed since
    `build` raises unless ``accept_changed``: a report must not outlive the file it describes."""
    case = load_case(run)
    case.draw_sheets = sheets
    if sheets:
        for old in (run / "sheets").glob("*.png"):  # a sheet from an earlier judge must not pass for this one
            old.unlink()
    recorded = json.loads((run / "case.json").read_text())["sha256"]
    named = {"clip_fbx": case.clip_fbx, "skin_fbx": case.skin_fbx, **case.files}
    now = {k: sha256(v) for k, v in named.items()}
    changed = sorted(k for k in now if now[k] != recorded.get(k))
    if changed and not accept_changed:
        raise ValueError(f"{run}: {', '.join(changed)} changed since build (rebuild, or judge with accept_changed=True from Python)")
    meta = json.loads((run / "case.json").read_text())
    report = Report({"data": case.data_name, "rig": case.rig, "action": case.action, "sha256": now,
                     "changed_since_build": changed, "built_by": meta["commit"], "verifier": commit(),
                     "container": os.environ.get("HOSTNAME", "unknown")})
    cfg = load_config() | {"limits": yaml.safe_load((CONFIG / "limits.yaml").read_text()) or {}}
    if case.files.get("file", Path()).suffix == ".bvh":  # a BVH has no mesh: the joints decide
        report.run["not_applicable"] = ("C1, C2a, C6, C7-sheets, B3-mesh: no mesh in this input; C5: reads FBX files, not BVH; "
                                        "C2b, C3, C4: a file has no source clip or stages")
        file = case.files["file"]
        report.checks += bvh_checks(file, case.skin_fbx, case.skin_fbx.suffix.lstrip(".") or "fbx", None, cfg,
                                    None if case.run is None else case.run / "blender" / "file.npz")
        apply_allowances(report, yaml.safe_load((CONFIG / "allowances.yaml").read_text()) or {})
        report.write(run)
        return report
    report.checks += c1_inventory(case, cfg["inventory_tol"])
    report.checks += c2a_skin(case, cfg)
    if "file" in case.files:  # a file with no source: what needs the source or the stages is not claimed
        report.run["not_applicable"] = "C2b, C3, C4: a file has no source clip or stages"
        generated = meta.get("generated")
        if generated:
            report.run["generated_sha256"] = sha256(generated)
            if report.run["generated_sha256"] != meta["generated_sha256"]:
                raise ValueError(f"{run}: {generated} changed since the case was made")
        report.checks += _c5(case, cfg, generated)
        report.checks += c6_second_reader(case, cfg)
        _sheets(report, case, run)
        apply_allowances(report, yaml.safe_load((CONFIG / "allowances.yaml").read_text()) or {})
        report.write(run)
        return report
    report.checks += c2b_fidelity(case, cfg)
    report.checks += c3_canonical(case, cfg)
    report.checks += c4_zero(case, cfg)
    report.checks += _c5(case, cfg)
    report.checks += c6_second_reader(case, cfg)
    _sheets(report, case, run)
    apply_allowances(report, yaml.safe_load((CONFIG / "allowances.yaml").read_text()) or {})
    report.write(run)
    return report


def _sheets(report: Report, case: Case, run: Path) -> None:
    """Name the sheets the checks wrote. They are for reading: only a run that should have written some
    and wrote none is `NOT RUN`; a rig with no mesh has nothing to draw."""
    report.run["sheets"] = [str(p.relative_to(run)) for p in case.sheets]
    if case.sheets or not case.draw_sheets:
        return
    if any(c.id == "C2a" for c in report.checks):
        report.checks.append(Check("C7-sheets", Status.NOT_RUN, None, note="no sheet written"))
    else:
        none = "C7-sheets: no mesh in this input"
        report.run["not_applicable"] = f"{report.run['not_applicable']}; {none}" if "not_applicable" in report.run else none


def _c5(case: Case, cfg: dict, generated: str | None = None) -> list[Check]:
    """C5 on the round trip (a built run) or the file (`file`). Stability needs the clip the file should be:
    the corpus clip of a built run, or ``generated`` (`file --clip`, the generator's .npz); without one it is NOT RUN."""
    path = case.files.get("4_roundtrip") or case.files.get("file")
    if case.floor or path is None:
        return [Check(f"C5-{k}", Status.NOT_RUN, None, note="needs a file under test") for k in (*C5, "joints", "stability")]
    if not (case.data_name and case.rig):
        return [Check(f"C5-{k}", Status.NOT_RUN, None, note="needs --rig and --data for the rig's vendor limits") for k in (*C5, "joints", "stability")]
    out = c5_sanity(path, case.rig, rig_limits(case.data_name, case.rig, cfg))
    if "file" in case.files and generated is None:
        return out + [Check("C5-stability", Status.NOT_RUN, None, note="no clip to compare with: pass --clip CLIP.npz (the generator's output)")]
    data = compose_build([f"data={case.data_name}"]).data
    root = Path(data.paths.root)
    prepared = RigidBodyAnimation.load(generated or root / "clips" / case.rig / f"{case.action}.npz").as_rigid_body(joint_translation="drop")
    return out + [c5_stability(path, RigRecord.load(root, case.rig), prepared, data, cfg)]


def _vendor_clip(file: Path, data, rig: str) -> Path | None:
    """The rig's FBX clip a vendor BVH was exported with (same action name), the source its joints compare with."""
    action = file.stem.lstrip("_")
    return next((Path(data.paths.raw) / "source" / f for r, a, f in fbx_index(data.source.index) if (r, a) == (rig, action)), None)


def file_case(file: Path, reference: Path | None, out: Path, data_name: str = "", rig: str = "", clip: Path | None = None) -> Path:
    if reference is None:
        data = compose_build([f"data={data_name}"]).data
        reference = (file.suffix == ".bvh" and _vendor_clip(file, data, rig)) or rig_skin(data, rig)
    run = out / f"{file.stem}_{hashlib.sha256(str(file.resolve()).encode()).hexdigest()[:8]}"
    run.mkdir(parents=True, exist_ok=True)
    try:
        if file.suffix == ".bvh":  # B4's frames: evenly spaced, the first and last included
            frames = np.unique(np.linspace(0, len(BVH.read(file).rotations) - 1, load_config()["n_frames"]).round().astype(int))
        else:
            scene = FBX.read(file)
            try:
                frames = sample_frames(scene.world(), load_config()["n_frames"])
            finally:
                scene.close()
    except Exception:  # noqa: BLE001 - judge reports an unreadable file
        frames = np.zeros(0, dtype=int)
    _write_case(run, Case(data_name, rig, file.stem, {"file": file}, reference, reference, frames, frames),
                {} if clip is None else {"generated": str(clip), "generated_sha256": sha256(clip)})
    return run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build")
    b.add_argument("--data", default="truebones")
    b.add_argument("--rig")
    b.add_argument("--clip")
    b.add_argument("--smoke", action="store_true", help="the rigs of configs/verify/verify.yaml")
    b.add_argument("--all-frames", action="store_true")
    b.add_argument("--floor", action="store_true", help="the source clip alone on the rest asset, to measure the float floor")
    b.add_argument("--out", type=Path, default=Path("tmp/verify"))
    f = sub.add_parser("file")
    f.add_argument("file", type=Path)
    f.add_argument("--reference", type=Path)
    f.add_argument("--rig")
    f.add_argument("--data")
    f.add_argument("--clip", type=Path, help="the generator's clip (.npz) the file should be, for C5-stability")
    f.add_argument("--out", type=Path, default=Path("tmp/verify"))
    j = sub.add_parser("judge")
    j.add_argument("dir", type=Path)
    args = parser.parse_args()

    if args.command == "build":
        smoke = load_config()["smoke"]
        jobs = ([(d, r, c) for d in ("truebones", "unimate") for r in smoke[d] for c in [None, *smoke["extra"].get(r, [])]]
                if args.smoke else [(args.data, args.rig, args.clip)])
        failed = []
        for data, rig, clip in jobs:
            try:
                print(build(data, rig, clip, args.out, args.all_frames, args.floor), flush=True)
            except Exception as error:  # noqa: BLE001 - one rig must not cost the others
                failed.append(rig)
                print(f"{rig}: FAILED {type(error).__name__}: {error}", flush=True)
        raise SystemExit(f"failed: {', '.join(failed)}" if failed else 0)
    if args.command == "file" and not (args.reference or (args.rig and args.data)):
        parser.error("file needs --reference, or --rig and --data")
    run = file_case(args.file, args.reference, args.out, args.data or "", args.rig or "", args.clip) if args.command == "file" else args.dir
    runs = [run] if (run / "case.json").is_file() else sorted(p.parent for p in run.glob("*/case.json"))
    if not runs:
        raise SystemExit(f"{run}: no case.json here or in a subdirectory")
    code = 0
    for p in sorted(run.iterdir()) if run.is_dir() and run not in runs else []:
        if p.is_dir() and not p.name.startswith(".") and not (p / "case.json").is_file():
            print(f"{p}: ERROR no case.json", flush=True)
            code = 1
    for r in runs:
        try:
            report = judge(r)
        except Exception as error:  # noqa: BLE001 - one run must not cost the others
            print(f"{r}: ERROR {type(error).__name__}: {error}", flush=True)
            code = 1
            continue
        print(f"{r}: {report.verdict.value}\n{report.text()}")
        code = code or report.exit_code
    raise SystemExit(code)


if __name__ == "__main__":
    main()

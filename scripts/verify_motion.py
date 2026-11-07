"""Judge the FBX files the pipeline returns against the user's vendor rig.

    build  --data D --rig R [--clip C] [--smoke] [--all-frames] [--floor] [--out DIR]   write the stage files of a round trip
    file   FILE --reference REF [--out DIR]                                   one FBX against a vendor reference
    judge  DIR                                                                run the checks, write report.json / report.txt

Runs in the `fbxsdk` container; the run directory is `<out>/<Rig>__<clip>/`, `--out` defaults to
`tmp/verify`. Exit status is non-zero unless every check is PASS or ALLOWED
(docs/reference/tooling.md).

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
from poseydon.build.sources import REST_CACHE, SOURCES, rig_cache
from poseydon.core.skeleton import SkeletonManifest
from poseydon.export import RigRecord
from poseydon.io.fbx import FBX
from scripts.build_stage_samples import build as build_stages
from tools.verify.case import Case, commit, load_case, sample_frames, sha256
from tools.verify.checks import c1_inventory, c2a_skin, c2b_fidelity
from tools.verify.report import Report, apply_allowances

CONFIG = Path("configs/verify")
STAGES = ("2_prepared", "3_model", "4_roundtrip")


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
    rest = rig_cache(data.paths.root, rig) / REST_CACHE
    with np.load(rest) as z:
        if "source_fbx" not in z.files:
            raise LookupError(f"{rest} records no source_fbx: rebuild it with tools/build_rest_fbx.py")
        skin = Path(str(z["source_fbx"]))
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
    case = Case(data_name, rig, action, {s: run / f"{s}.fbx" for s in stages}, run / "1_source.fbx", skin,
                frames, numbers[frames], window, floor)
    _write_case(run, case, {})
    return run


def load_config() -> dict:
    return yaml.safe_load((CONFIG / "verify.yaml").read_text())


def judge(run: Path, accept_changed: bool = False) -> Report:
    """Run every check on a run directory and write its report. A file that changed since
    `build` raises unless ``accept_changed``: a report must not outlive the file it describes."""
    case = load_case(run)
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
    cfg = load_config()
    report.checks += c1_inventory(case, cfg["inventory_tol"])
    report.checks += c2a_skin(case, cfg)
    report.checks += c2b_fidelity(case, cfg)
    apply_allowances(report, yaml.safe_load((CONFIG / "allowances.yaml").read_text()) or {})
    report.write(run)
    return report


def file_case(file: Path, reference: Path, out: Path) -> Path:
    run = out / f"{file.stem}_{hashlib.sha256(str(file.resolve()).encode()).hexdigest()[:8]}"
    run.mkdir(parents=True, exist_ok=True)
    try:
        scene = FBX.read(file)
        try:
            frames = sample_frames(scene.world(), load_config()["n_frames"])
        finally:
            scene.close()
    except Exception:  # noqa: BLE001 - judge reports an unreadable file
        frames = np.zeros(0, dtype=int)
    _write_case(run, Case("", "", file.stem, {"file": file}, reference, reference, frames, frames), {})
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
    f.add_argument("--reference", type=Path, required=True)
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
    run = file_case(args.file, args.reference, args.out) if args.command == "file" else args.dir
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

"""Judge the FBX files the pipeline returns against the user's vendor rig.

    build  --data D --rig R [--clip C] [--smoke] [--all-frames] [--out DIR]   write the stage files of a round trip
    file   FILE --reference REF [--out DIR]                                   one FBX against a vendor reference
    judge  DIR                                                                run the checks, write report.json / report.txt

Runs in the `fbxsdk` container; the run directory is `<out>/<Rig>__<clip>/`, `--out` defaults to
`tmp/verify`. Exit status is non-zero unless every check is PASS or ALLOWED
(docs/reference/tooling.md).

    docker compose run --rm -T --entrypoint python -e HOME=/tmp fbxsdk scripts/verify_motion.py build --data truebones --rig Camel
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import yaml

from poseydon.build.pipeline import compose_build
from poseydon.build.sources import REST_CACHE, rig_cache
from poseydon.io.fbx import FBX
from scripts.build_stage_samples import build as build_stages
from tools.verify.case import Case, commit, load_case, sample_frames, sha256
from tools.verify.checks import c1_inventory
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
        "sha256": {k: sha256(v) for k, v in named.items()},
        "commit": commit(), "container": os.environ.get("HOSTNAME", "unknown"), **extra,
    }, indent=2))


def build(data_name: str, rig: str, clip: str | None, out: Path, all_frames: bool = False) -> Path:
    """Write the stage files of one clip and ``case.json``; returns the run directory."""
    data = compose_build([f"data={data_name}"]).data
    rest = rig_cache(data.paths.root, rig) / REST_CACHE
    with np.load(rest) as z:
        if "source_fbx" not in z.files:
            raise LookupError(f"{rest} records no source_fbx: rebuild it with tools/build_rest_fbx.py")
        skin = Path(str(z["source_fbx"]))
    stage = out / ".stage"
    action = build_stages(data, rig, stage, clip)
    run = out / f"{rig}__{action}"
    shutil.rmtree(run, ignore_errors=True)
    shutil.move(stage / rig, run)
    stage.rmdir()
    scene = FBX.read(run / "4_roundtrip.fbx")
    world = scene.world()
    scene.close()
    frames = np.arange(len(world)) if all_frames else sample_frames(world, load_config()["n_frames"])
    source = FBX.read(run / "1_source.fbx")
    n_source = len(source.frames())
    source.close()
    # Same fraction of the clip in the source file, whose length may differ from the round trip's.
    source_frames = np.round(frames * (n_source - 1) / max(len(world) - 1, 1)).astype(int)
    case = Case(data_name, rig, action, {s: run / f"{s}.fbx" for s in STAGES}, run / "1_source.fbx", skin,
                frames, source_frames)
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
        raise ValueError(f"{run}: {', '.join(changed)} changed since build (pass accept_changed to judge anyway)")
    meta = json.loads((run / "case.json").read_text())
    report = Report({"data": case.data_name, "rig": case.rig, "action": case.action, "sha256": now,
                     "changed_since_build": changed, "built_by": meta["commit"], "verifier": commit(),
                     "container": os.environ.get("HOSTNAME", "unknown")})
    report.checks += c1_inventory(case)
    apply_allowances(report, yaml.safe_load((CONFIG / "allowances.yaml").read_text()) or {})
    report.write(run)
    return report


def file_case(file: Path, reference: Path, out: Path) -> Path:
    run = out / file.stem
    run.mkdir(parents=True, exist_ok=True)
    _write_case(run, Case("", "", file.stem, {"file": file}, reference, reference,
                          np.zeros(0, dtype=int), np.zeros(0, dtype=int)), {})
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
                print(build(data, rig, clip, args.out, args.all_frames), flush=True)
            except Exception as error:  # noqa: BLE001 - one rig must not cost the others
                failed.append(rig)
                print(f"{rig}: FAILED {type(error).__name__}: {error}", flush=True)
        raise SystemExit(f"failed: {', '.join(failed)}" if failed else 0)
    run = file_case(args.file, args.reference, args.out) if args.command == "file" else args.dir
    runs = [run] if (run / "case.json").is_file() else sorted(p.parent for p in run.glob("*/case.json"))
    code = 0
    for r in runs:
        report = judge(r)
        print(f"{r}: {report.verdict.value}\n{report.text()}")
        code = code or report.exit_code
    raise SystemExit(code)


if __name__ == "__main__":
    main()

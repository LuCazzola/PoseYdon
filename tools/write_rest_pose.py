"""Emit a `rest_pose:` line for every rig, for review before it is applied.

The rule is T-pose, then idle, then walk -- or fly, for a flying creature --
applied WITHIN the rig's modal joint set. The 17 rigs whose named T-pose is
missing or unusable are listed explicitly below, because no rule gets them
right: matching "idle" as a substring picks Lion's __DeathIdle.bvh, Jaguar's
__LieIdle.bvh and Trex's __idle_attack.bvh.

Run, review the output, then re-run with --apply.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from poseydon.io.bvh import BVH

RAW = Path("data/truebones/source")
RIGS = Path("data/truebones/rigs")

#: Rigs whose named T-pose is absent or declares a minority skeleton.
AUTHORED: dict[str, str] = {
    "Anaconda": "__Idle.bvh",
    "Ant": "__Idle.bvh",
    "Bird": "__IdleLoop.bvh",
    "Camel": "__IdleLoop.bvh",
    "Crab": "__Walk.bvh",
    "Cricket": "__Idle.bvh",
    "Deer": "__Idle.bvh",
    "Dog": "__Idle.bvh",
    "Goat": "__Idle.bvh",
    "Jaguar": "__Idle.bvh",
    "Lion": "__SlowIdle.bvh",
    "Monkey": "__Idle1.bvh",
    "Pteranodon": "__FlyLoop.bvh",
    "Rat": "__Trottle.bvh",
    "SabreToothTiger": "__Startwalk.bvh",
    "Scorpion-2": "__Idle.bvh",
    "Trex": "__walk_loop.bvh",
}


def choose(rig: str) -> tuple[str | None, str]:
    rig_dir = RAW / rig
    bvhs = sorted(rig_dir.glob("*.bvh"))
    if not bvhs:
        return None, "no raw clips"

    names = {p: BVH.read_names(p) for p in bvhs}
    modal, count = Counter(names.values()).most_common(1)[0]
    modal_paths = [p for p in bvhs if names[p] == modal]

    authored = AUTHORED.get(rig)
    if authored is not None:
        match = next((p for p in modal_paths if p.name == authored), None)
        if match is None:
            return None, f"AUTHORED {authored} absent or non-modal"
        return match.name, f"authored ({count}/{len(bvhs)} modal)"

    for key in ("tpos",):
        for p in modal_paths:
            if key in p.name.lower():
                return p.name, f"tpose ({count}/{len(bvhs)} modal)"
    return None, "no T-pose in the modal set and no authored entry"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    problems = []
    for manifest_path in sorted(RIGS.glob("*/manifest.yaml")):
        rig = manifest_path.parent.name
        filename, why = choose(rig)
        print(f"{rig:<18} {filename!s:<24} {why}")
        if filename is None:
            problems.append(rig)
            continue
        if not args.apply:
            continue
        text = manifest_path.read_text()
        if "rest_pose:" in text:
            continue
        lines = text.splitlines()
        # After `skeleton:`, so the file reads identity-first.
        at = next(i for i, line in enumerate(lines) if line.startswith("skeleton:"))
        lines.insert(at + 1, f"rest_pose: {filename}")
        manifest_path.write_text("\n".join(lines) + "\n")

    if problems:
        print(f"\nUNRESOLVED: {problems}")


if __name__ == "__main__":
    main()

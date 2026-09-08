"""Walk the top of the hierarchy until the first branch, showing cumulative offset."""
from collections import Counter
from pathlib import Path

import numpy as np

from poseydon.io.bvh import BVH

RAW = Path("data/truebones/source")
RIGS = ["Tukan", "Crow", "SabreToothTiger", "Lynx", "BrownBear"]


def rest_file(d: Path):
    bvhs = sorted(d.glob("*.bvh"))
    sets = {p: tuple(BVH.read_names(p)) for p in bvhs}
    modal, _ = Counter(sets.values()).most_common(1)[0]
    files = [p for p in bvhs if sets[p] == modal]
    for key in ("TPOS", "IDLE", "WALK", "FLY"):
        for p in files:
            if key in p.name.upper():
                return p
    return files[0]


for rig in RIGS:
    d = RAW / rig
    anim = BVH.read(str(rest_file(d))).to_animation()
    names, parents, offsets = list(anim.names), anim.parents, anim.offsets
    lengths = np.linalg.norm(offsets[1:], axis=-1)
    bone = float(lengths[lengths > 1e-8].mean())
    n_children = {j: sum(1 for p in parents if p == j) for j in range(len(names))}

    glob = anim.global_positions()[0]          # rest frame, world space
    root_y = glob[0, 1]

    print(f"\n=== {rig}   mean bone {bone:.3f}   root world Y {root_y / bone:+.3f} bone")
    j, depth, cum = 0, 0, 0.0
    while depth < 8:
        off = 0.0 if j == 0 else float(np.linalg.norm(offsets[j])) / bone
        cum += off
        y = glob[j, 1] / bone
        print(f"   {'  ' * depth}{names[j]:<24} off={off:6.3f}  cum={cum:6.3f}  "
              f"worldY={y:+7.3f}  children={n_children[j]}")
        kids = [k for k, p in enumerate(parents) if p == j]
        if len(kids) != 1:
            print(f"   {'  ' * (depth + 1)}-> branches into {len(kids)}")
            break
        j, depth = kids[0], depth + 1

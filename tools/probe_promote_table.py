"""Emit the promoted-root table: for each ground-locator rig, the first branching joint."""
from collections import Counter
from pathlib import Path

import numpy as np

from poseydon.io.bvh import BVH

RAW = Path("data/truebones/source")
LOCATORS = ["Bear", "Camel", "Crow", "Dog", "Dog-2", "Horse", "Pirrana", "Pteranodon",
            "Raptor3", "SabreToothTiger", "Scorpion-2", "Spider", "Trex", "Tukan"]


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


print(f"{'rig':<18}{'promote to':<22}{'steps':>6}{'newY frac':>11}{'dropped bones':>15}")
for rig in LOCATORS:
    d = RAW / rig
    anim = BVH.read(str(rest_file(d))).to_animation()
    names, parents, offsets = list(anim.names), anim.parents, anim.offsets
    lengths = np.linalg.norm(offsets[1:], axis=-1)
    bone = float(lengths[lengths > 1e-8].mean())
    g = anim.global_positions()[0] / bone
    lo, hi = float(g[:, 1].min()), float(g[:, 1].max())

    j, steps, dropped = 0, 0, []
    while True:
        kids = [k for k, p in enumerate(parents) if p == j]
        if len(kids) != 1:
            break
        dropped.append(float(np.linalg.norm(offsets[kids[0]])) / bone)
        j, steps = kids[0], steps + 1

    frac = (g[j, 1] - lo) / (hi - lo)
    note = "" if steps else "   <-- chain rule cannot reach; must be authored"
    drop_s = ",".join(f"{x:.2f}" for x in dropped) or "-"
    print(f"{rig:<18}{names[j]:<22}{steps:>6}{frac:>11.3f}{drop_s:>15}{note}")

# Tukan by hand: follow the real-skeleton branch, not the MESH branch.
d = RAW / "Tukan"
anim = BVH.read(str(rest_file(d))).to_animation()
names, parents, offsets = list(anim.names), anim.parents, anim.offsets
lengths = np.linalg.norm(offsets[1:], axis=-1)
bone = float(lengths[lengths > 1e-8].mean())
g = anim.global_positions()[0] / bone
lo, hi = float(g[:, 1].min()), float(g[:, 1].max())
k = names.index("locator")
print(f"\nTukan authored target `locator`: worldY frac "
      f"{(g[k,1]-lo)/(hi-lo):.3f}, children={sum(1 for p in parents if p == k)}")

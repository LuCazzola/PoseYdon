"""Which rigs have a root that is a ground locator rather than a body joint?

Compares the root's rest-frame height against the skeleton's own vertical
extent, in units of the rig's mean bone length.
"""
from collections import Counter
from pathlib import Path

import numpy as np

from poseydon.io.bvh import BVH

RAW = Path("data/truebones/source")


def rest_file(d: Path):
    bvhs = sorted(d.glob("*.bvh"))
    if not bvhs:
        return None
    sets = {p: tuple(BVH.read_names(p)) for p in bvhs}
    modal, _ = Counter(sets.values()).most_common(1)[0]
    files = [p for p in bvhs if sets[p] == modal]
    for key in ("TPOS", "IDLE", "WALK", "FLY"):
        for p in files:
            if key in p.name.upper():
                return p
    return files[0]


rows = []
for d in sorted(p for p in RAW.iterdir() if p.is_dir()):
    path = rest_file(d)
    if path is None:
        continue
    anim = BVH.read(str(path)).to_animation()
    parents, offsets = anim.parents, anim.offsets
    lengths = np.linalg.norm(offsets[1:], axis=-1)
    bone = float(lengths[lengths > 1e-8].mean())
    g = anim.global_positions()[0] / bone
    lo, hi = float(g[:, 1].min()), float(g[:, 1].max())
    root_y = float(g[0, 1])
    n_kids = sum(1 for p in parents if p == 0)

    # How far up the body does the root sit, as a fraction of total height?
    frac = (root_y - lo) / (hi - lo) if hi > lo else 0.0
    # Cumulative offset along the single-child chain to the first branch.
    j, cum = 0, 0.0
    while True:
        kids = [k for k, p in enumerate(parents) if p == j]
        if len(kids) != 1:
            break
        j = kids[0]
        cum += float(np.linalg.norm(offsets[j])) / bone
    rows.append((d.name, n_kids, root_y, lo, hi, frac, cum, list(anim.names)[0]))

print(f"{'rig':<18}{'kids':>5}{'rootY':>8}{'minY':>8}{'maxY':>8}{'height frac':>12}"
      f"{'chain off':>10}  root")
locators, reachable = [], []
for name, kids, ry, lo, hi, frac, cum, rname in sorted(rows, key=lambda r: r[5]):
    tag = ""
    if frac < 0.05:
        tag = "  <-- ground locator"
        locators.append(name)
        if cum > 1e-6:
            reachable.append(name)
    print(f"{name:<18}{kids:>5}{ry:>8.2f}{lo:>8.2f}{hi:>8.2f}{frac:>12.3f}"
          f"{cum:>10.3f}  {rname}{tag}")

print(f"\nground locators: {len(locators)}")
print(f"  fixable by promoting along a single-child chain: {len(reachable)} {reachable}")
print(f"  NOT fixable that way (root branches immediately): "
      f"{sorted(set(locators) - set(reachable))}")

"""Per rig: what sits at the top of the hierarchy, and how far is the first real joint?

Classifies each rig's root by its children and the offset to its single child,
measured in units of that rig's own mean bone length so rigs are comparable.
"""
from collections import Counter
from pathlib import Path

import numpy as np

from poseydon.io.bvh import BVH

RAW = Path("data/truebones/source")


def rest_file(d: Path) -> Path | None:
    bvhs = sorted(d.glob("*.bvh"))
    if not bvhs:
        return None
    sets = {p: tuple(BVH.read_names(p)) for p in bvhs}
    modal, _ = Counter(sets.values()).most_common(1)[0]
    modal_files = [p for p in bvhs if sets[p] == modal]
    for key in ("TPOS", "IDLE", "WALK", "FLY"):
        for p in modal_files:
            if key in p.name.upper():
                return p
    return modal_files[0]


rows = []
for d in sorted(p for p in RAW.iterdir() if p.is_dir()):
    path = rest_file(d)
    if path is None:
        continue
    anim = BVH.read(str(path)).to_animation()
    names, parents, offsets = list(anim.names), anim.parents, anim.offsets

    lengths = np.linalg.norm(offsets[1:], axis=-1)
    mean_bone = float(lengths[lengths > 1e-8].mean())

    kids = [j for j, p in enumerate(parents) if p == 0]
    if len(kids) != 1:
        rows.append((d.name, names[0], len(kids), None, None, mean_bone))
        continue
    child = kids[0]
    rows.append((
        d.name, names[0], 1, names[child],
        float(np.linalg.norm(offsets[child])) / mean_bone, mean_bone,
    ))

print(f"{'rig':<18}{'root':<20}{'kids':>5}  {'first child':<22}{'|offset|/bone':>14}")
floaters, merged, forked = [], [], []
for name, root, n_kids, child, ratio, _ in rows:
    if n_kids != 1:
        print(f"{name:<18}{root:<20}{n_kids:>5}  {'(multiple children)':<22}{'-':>14}")
        forked.append(name)
        continue
    tag = ""
    if ratio < 1e-6:
        merged.append(name)
    elif ratio > 0.05:
        tag = "  <-- offset root"
        floaters.append((name, ratio))
    print(f"{name:<18}{root:<20}{n_kids:>5}  {child:<22}{ratio:>14.4f}{tag}")

print(f"\nzero-offset single child (collapsed by reduction today): {len(merged)}")
print(f"multiple children at the root (untouched):                {len(forked)} {forked}")
print(f"\nOFFSET root -- the reported problem: {len(floaters)}")
for name, ratio in sorted(floaters, key=lambda kv: -kv[1]):
    print(f"    {name:<18} {ratio:8.4f} bone lengths")

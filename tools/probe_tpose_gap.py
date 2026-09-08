"""Which joints do the gutted rigs' T-poses omit, and what are those joints?"""
import numpy as np
from pathlib import Path
from poseydon.io.bvh import BVH

RAW = Path("data/truebones/source")

for rig, tpose in [("Crab", "__TPOSE.bvh"), ("Ant", None), ("Deer", None), ("Jaguar", None)]:
    d = RAW / rig
    if not d.is_dir():
        print(f"{rig}: no source dir"); continue
    tp = [p for p in d.glob("*.bvh") if "TPOSE" in p.name.upper() or "TPOS" in p.name.upper()]
    if not tp:
        print(f"{rig}: no T-pose file among {[p.name for p in d.glob('*.bvh')][:5]}"); continue
    rest = BVH.read(str(tp[0])).to_animation()
    others = sorted(p for p in d.glob("*.bvh") if p != tp[0])
    if not others:
        print(f"{rig}: only a T-pose"); continue
    clip = BVH.read(str(others[0])).to_animation()
    rest_names, clip_names = list(rest.names), list(clip.names)
    missing = [n for n in clip_names if n not in rest_names]
    extra = [n for n in rest_names if n not in clip_names]
    print(f"\n=== {rig}: rest={len(rest_names)}j  clip={len(clip_names)}j ({others[0].name})")
    print(f"  missing from rest ({len(missing)}): {missing}")
    print(f"  extra in rest ({len(extra)}): {extra}")
    # for each missing joint: offset length, is it a leaf, and rotation spread across the clip
    children = {n: [] for n in clip_names}
    for j, p in enumerate(clip.parents):
        if p >= 0:
            children[clip_names[p]].append(clip_names[j])
    for n in missing:
        j = clip_names.index(n)
        off = np.linalg.norm(clip.offsets[j])
        q = clip.rotations[:, j]                      # (T,4)
        ang = 2 * np.degrees(np.arccos(np.clip(np.abs(q[:, 3]), -1, 1)))
        print(f"    {n:28s} |offset|={off:9.4f}  leaf={not children[n]:5}  "
              f"rot deg: max={ang.max():7.2f} mean={ang.mean():7.2f}")

print("\n\n########## across ALL clips: is the missing joint's rotation identity everywhere?")
for rig in ("Crab", "Ant", "Deer", "Jaguar"):
    d = RAW / rig
    tp = [p for p in d.glob("*.bvh") if "TPOS" in p.name.upper()]
    if not tp:
        print(f"{rig}: no T-pose"); continue
    rest_names = set(BVH.read(str(tp[0])).to_animation().names)
    worst: dict[str, float] = {}
    n_clips = 0
    for path in sorted(d.glob("*.bvh")):
        if path == tp[0]:
            continue
        a = BVH.read(str(path)).to_animation()
        n_clips += 1
        for j, n in enumerate(a.names):
            if n in rest_names:
                continue
            ang = 2 * np.degrees(np.arccos(np.clip(np.abs(a.rotations[:, j, 3]), -1, 1)))
            worst[n] = max(worst.get(n, 0.0), float(ang.max()))
    print(f"\n=== {rig} ({n_clips} clips): worst deviation from identity, per absent joint")
    for n, v in sorted(worst.items(), key=lambda kv: -kv[1]):
        print(f"    {n:28s} max {v:8.3f} deg")

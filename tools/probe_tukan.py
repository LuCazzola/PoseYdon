import numpy as np

from poseydon.io.bvh import BVH

anim = BVH.read("data/truebones/source/Tukan/__Tpose.bvh").to_animation()
names, parents, offsets = list(anim.names), anim.parents, anim.offsets
lengths = np.linalg.norm(offsets[1:], axis=-1)
bone = float(lengths[lengths > 1e-8].mean())
g = anim.global_positions()[0] / bone
n_kids = {j: sum(1 for p in parents if p == j) for j in range(len(names))}

print(f"Tukan: {len(names)} joints, mean bone {bone:.3f}, "
      f"Y range {g[:,1].min():+.2f}..{g[:,1].max():+.2f}")
for j in range(len(names)):
    depth = 0
    p = parents[j]
    while p >= 0:
        depth += 1
        p = parents[p]
    if depth > 2:
        continue
    off = 0.0 if j == 0 else float(np.linalg.norm(offsets[j])) / bone
    print(f"  {'  ' * depth}{names[j]:<20} off={off:6.3f}  worldY={g[j,1]:+7.3f}  "
          f"children={n_kids[j]}")

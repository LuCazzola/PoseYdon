"""THROWAWAY spike probe. Delete after reading the answer.

Is the residual 3-7% BVH/mesh.npz disagreement a pure scale mismatch caused
by the two sides averaging over different bone sets?

If it is, the per-bone length ratio over shared non-degenerate bones will be
UNIFORM at some value near 1.03-1.07, and that value will equal the ratio of
the two denominators. If the spread is wide, it is not scale.

Runs in the test container -- mesh.npz is an npz and the BVH reader is pure
Python, so no Blender is needed.

    docker compose run --rm test python tools/probe_scale_residual.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from poseydon.io.bvh import BVH

CORPUS = Path("data/truebones")
TOL = 1e-8


def rest_clip(rig: str) -> Path | None:
    """The prepared clip built from the rig's rest pose, as the test picks it."""
    paths = sorted((CORPUS / "clips" / rig).glob("*.bvh"))
    for p in paths:
        if "tpos" in p.stem.lower():
            return p
    for p in paths:
        if p.stem.lower().startswith("idle"):
            return p
    return paths[0] if paths else None


def main() -> None:
    rigs = sorted(p.parent.name for p in CORPUS.glob("rigs/*/mesh.npz"))
    print(f"rigs with mesh.npz: {rigs}\n")

    for rig in rigs:
        clip = rest_clip(rig)
        if clip is None:
            print(f"{rig}: no prepared clip")
            continue
        anim = BVH.read(clip).to_animation()
        with np.load(CORPUS / "rigs" / rig / "mesh.npz", allow_pickle=True) as m:
            fnames = tuple(str(n) for n in m["joint_names"])
            fparents = m["joint_parents"]
            foffsets = m["joint_offsets"]

        fset = set(fnames)
        ratios, blens, flens = [], [], []
        for name in anim.names:
            if name not in fset:
                continue
            b, f = anim.names.index(name), fnames.index(name)
            if anim.parents[b] < 0 or fparents[f] < 0:
                continue
            bl = float(np.linalg.norm(anim.offsets[b]))
            fl = float(np.linalg.norm(foffsets[f]))
            if bl > TOL and fl > TOL:
                ratios.append(fl / bl)
                blens.append(bl)
                flens.append(fl)

        if not ratios:
            print(f"{rig}: no comparable bones")
            continue

        r = np.array(ratios)
        spread = (r.max() - r.min()) / np.median(r)
        # What each side's own mean bone length is, over the bones it counted.
        bvh_all = np.linalg.norm(anim.offsets[1:], axis=-1)
        bvh_real = bvh_all[bvh_all > TOL]
        fbx_all = np.linalg.norm(foffsets[1:], axis=-1)
        fbx_real = fbx_all[fbx_all > TOL]

        print(f"{rig}:  {len(r)} shared non-degenerate bones")
        print(f"    per-bone ratio mesh/bvh: median {np.median(r):.6f}  "
              f"min {r.min():.6f}  max {r.max():.6f}")
        print(f"    relative spread: {spread:.3e}   <-- uniform if << 1e-2")
        print(f"    mean bone length: bvh(real) {bvh_real.mean():.6f}  "
              f"n={len(bvh_real)}   mesh(real) {fbx_real.mean():.6f}  n={len(fbx_real)}")
        print(f"    predicted ratio from the two means: "
              f"{fbx_real.mean() / bvh_real.mean():.6f}")
        # If both sides had averaged over the SHARED set instead:
        print(f"    if both averaged over the shared set: bvh {np.mean(blens):.6f}  "
              f"mesh {np.mean(flens):.6f}  ratio {np.mean(flens) / np.mean(blens):.6f}")
        print()


if __name__ == "__main__":
    main()

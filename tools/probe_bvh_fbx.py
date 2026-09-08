"""THROWAWAY spike probe. Delete after reading the answer.

Spike 1: does BVH and FBX reading of the same clip give the same Animation,
at the RAW stage and at the PREPARED stage? Reports structure, per-bone
length ratio, and world joint positions.

Spike 2: for rigs whose clips are rejected against their own T-pose, is the
joint-name difference a reordering (code bug) or a genuine set difference
(dirty data)? And does the raw FBX show the same mismatch?

Run: docker compose run --rm fbx blender -b --python tools/probe_bvh_fbx.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/app/src")
sys.path.insert(0, "/app")

from poseydon.io.bvh import BVH  # noqa: E402
from poseydon.io.fbx import FBX  # noqa: E402

SOURCE = Path("data/truebones/source")
CLIPS = Path("data/truebones/clips")

PAIR_RIGS = ["Flamingo", "BrownBear", "Scorpion", "Crab"]
REJECT_RIGS = ["Ant", "Centipede", "Crab", "Deer", "Elephant", "HermitCrab", "Jaguar", "Trex"]


def norm(stem: str) -> str:
    """Collapse a filename stem so `Flamingo_BendIdle` and `Flamingo-BendIdle` match."""
    return re.sub(r"[^a-z0-9]", "", stem.lower())


def fbx_positions(fbx: FBX, names: list[str], n_frames: int) -> np.ndarray:
    start, _end = fbx.frame_range()
    out = np.zeros((n_frames, len(names), 3))
    for f in range(n_frames):
        fbx.set_frame(start + f)
        for j, name in enumerate(names):
            out[f, j] = np.asarray(fbx.joint_position(name))
    return out


def compare(bvh_path: Path, fbx_path: Path, label: str) -> None:
    print(f"\n--- {label}")
    print(f"    bvh {bvh_path.name}   fbx {fbx_path.name}")
    try:
        anim = BVH.read(bvh_path).to_animation()
        fbx = FBX.read(fbx_path)
    except Exception as error:  # noqa: BLE001
        print(f"    READ FAILED: {type(error).__name__}: {error}")
        return

    bset, fset = set(anim.names), set(fbx.joint_names)
    shared = [n for n in anim.names if n in fset]
    print(f"    joints: bvh {len(anim.names)}  fbx {len(fbx.joint_names)}  shared {len(shared)}")
    print(f"            bvh-only {len(bset - fset)}  fbx-only {len(fset - bset)}")
    if not shared:
        print("    NO SHARED JOINTS -- cannot compare")
        return

    # Parents, by name, over shared joints.
    fparents = fbx.joint_parents
    bad_parent = 0
    for name in shared:
        b, f = anim.names.index(name), fbx.joint_names.index(name)
        bp = anim.names[anim.parents[b]] if anim.parents[b] >= 0 else None
        fp = fbx.joint_names[fparents[f]] if fparents[f] >= 0 else None
        if bp != fp and bp in fset and fp in bset:
            bad_parent += 1
    print(f"    parent disagreements over shared joints: {bad_parent}")

    # Bone-length ratio per shared non-root bone: catches a uniform scale.
    boff, foff = anim.offsets, fbx.joint_offsets
    ratios = []
    for name in shared:
        b, f = anim.names.index(name), fbx.joint_names.index(name)
        if anim.parents[b] < 0 or fparents[f] < 0:
            continue
        bl, fl = np.linalg.norm(boff[b]), np.linalg.norm(foff[f])
        if bl > 1e-8 and fl > 1e-8:
            ratios.append(fl / bl)
    if ratios:
        r = np.array(ratios)
        print(f"    bone-length ratio fbx/bvh: median {np.median(r):.6f}  "
              f"min {r.min():.6f}  max {r.max():.6f}  n={len(r)}")
        k = float(np.median(r))
    else:
        k = 1.0
        print("    bone-length ratio: no comparable bones")

    # World joint positions, first few frames, after removing the median scale.
    n_frames = min(5, anim.n_frames)
    bpos = anim.global_positions()[:n_frames]
    fpos = fbx_positions(fbx, shared, n_frames) / k
    bidx = [anim.names.index(n) for n in shared]
    scale = float(np.linalg.norm(anim.offsets[1:], axis=-1).mean())

    raw = np.abs(bpos[:, bidx] - fpos).max()
    # Also report after removing a per-clip constant translation (root placement).
    delta = (bpos[:, bidx] - fpos).mean(axis=(0, 1))
    centred = np.abs((bpos[:, bidx] - fpos) - delta).max()
    print(f"    world positions over {n_frames} frames, scale-normalised:")
    print(f"        max diff {raw / scale:.4e} bone lengths")
    print(f"        after removing a constant offset: {centred / scale:.4e} bone lengths")


def spike1() -> None:
    print("=" * 78)
    print("SPIKE 1 -- BVH vs FBX, raw stage and prepared stage")
    print("=" * 78)
    for rig in PAIR_RIGS:
        raw_bvh = {norm(p.stem): p for p in sorted((SOURCE / rig).glob("*.bvh"))}
        raw_fbx = {norm(p.stem): p for p in sorted((SOURCE / rig).glob("*.fbx"))}
        both = sorted(set(raw_bvh) & set(raw_fbx))
        print(f"\n########## {rig}: {len(raw_bvh)} raw bvh, {len(raw_fbx)} raw fbx, "
              f"{len(both)} pair by normalised stem")
        if both:
            key = both[0]
            compare(raw_bvh[key], raw_fbx[key], f"{rig} RAW")

        prep_bvh = {p.stem: p for p in sorted((CLIPS / rig).glob("*.bvh"))} if (CLIPS / rig).is_dir() else {}
        prep_fbx = {p.stem: p for p in sorted((CLIPS / rig).glob("*.fbx"))} if (CLIPS / rig).is_dir() else {}
        shared_prep = sorted(set(prep_bvh) & set(prep_fbx))
        print(f"    prepared: {len(prep_bvh)} bvh, {len(prep_fbx)} fbx, {len(shared_prep)} share a stem")
        if shared_prep:
            key = shared_prep[0]
            compare(prep_bvh[key], prep_fbx[key], f"{rig} PREPARED")


def find_tpose(paths: list[Path]) -> Path | None:
    for p in paths:
        if "tpos" in p.name.lower():
            return p
    for p in paths:
        if p.name.lower().lstrip("_").startswith("idle"):
            return p
    return None


def spike2() -> None:
    print("\n" + "=" * 78)
    print("SPIKE 2 -- rejected clips: reordering (code) or set difference (data)?")
    print("=" * 78)
    for rig in REJECT_RIGS:
        paths = sorted((SOURCE / rig).glob("*.bvh"))
        if not paths:
            print(f"\n########## {rig}: no raw bvh")
            continue
        tp = find_tpose(paths) or paths[0]
        rest = BVH.read(tp).to_animation()
        rset = set(rest.names)

        reorder = superset = subset = other = equal = 0
        example = None
        for p in paths:
            names = BVH.read(p).to_animation().names
            if names == rest.names:
                equal += 1
            elif set(names) == rset:
                reorder += 1
                example = example or ("REORDER", p.name, None)
            elif set(names) > rset:
                superset += 1
                example = example or ("SUPERSET", p.name, sorted(set(names) - rset)[:5])
            elif set(names) < rset:
                subset += 1
                example = example or ("SUBSET", p.name, sorted(rset - set(names))[:5])
            else:
                other += 1
                example = example or ("DISJOINT", p.name, sorted(set(names) ^ rset)[:5])
        print(f"\n########## {rig}  T-pose={tp.name} ({rest.n_joints} joints), {len(paths)} clips")
        print(f"    identical {equal}   REORDER-ONLY {reorder}   superset {superset}   "
              f"subset {subset}   other {other}")
        if example:
            print(f"    example {example[0]}: {example[1]}  diff sample: {example[2]}")

        fbxs = sorted((SOURCE / rig).glob("*.fbx"))
        if fbxs:
            try:
                f = FBX.read(fbxs[0])
                fset = set(f.joint_names)
                print(f"    raw FBX {fbxs[0].name}: {len(f.joint_names)} joints; "
                      f"matches T-pose set: {fset == rset}; "
                      f"vs T-pose  fbx-only {len(fset - rset)}  tpose-only {len(rset - fset)}")
            except Exception as error:  # noqa: BLE001
                print(f"    raw FBX read failed: {type(error).__name__}: {error}")


if __name__ == "__main__":
    spike1()
    spike2()
    print("\nDONE")

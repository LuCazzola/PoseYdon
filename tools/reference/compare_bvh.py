"""Compare PoseYdon's BVH loader against Motion's, for speed and agreement.

Runs only in the `compat` image, which is the only place Motion is installed.
"""

from __future__ import annotations

import time
from pathlib import Path

import BVH  # Motion
import numpy as np
from Animation import positions_global

from poseydon.io.bvh import BVH as PoseydonBVH

ASSETS = Path("external/neural_motion_blending/assets/truebones")
REPEATS = 5


def timed(fn, *args):
    best = float("inf")
    for _ in range(REPEATS):
        start = time.perf_counter()
        out = fn(*args)
        best = min(best, time.perf_counter() - start)
    return out, best


def main() -> None:
    print(f"{'file':40s} {'Motion (s)':>11s} {'PoseYdon (s)':>13s} {'speedup':>8s} "
          f"{'joints':>7s} {'pos err':>10s}")
    total_motion = total_ours = 0.0

    for path in sorted(ASSETS.glob("*.bvh")):
        (anim_ref, _names, _frame_time), t_ref = timed(BVH.load, str(path))
        anim_ours, t_ours = timed(lambda p: PoseydonBVH.read(p).to_animation().as_rigid_body(), path)

        total_motion += t_ref
        total_ours += t_ours

        # Motion drops End Sites from `names` but keeps them as joints in the
        # animation, so compare on the joint arrays.
        pos_ref = positions_global(anim_ref)
        pos_ours = anim_ours.global_positions()
        if pos_ref.shape == pos_ours.shape:
            error = f"{np.abs(pos_ref - pos_ours).max():.2e}"
        else:
            error = f"shape {pos_ref.shape[1]}v{pos_ours.shape[1]}"

        print(f"{path.stem[:40]:40s} {t_ref:11.4f} {t_ours:13.4f} "
              f"{t_ref / t_ours:7.2f}x {anim_ours.n_joints:7d} {error:>10s}")

    print(f"{'TOTAL':40s} {total_motion:11.4f} {total_ours:13.4f} "
          f"{total_motion / total_ours:7.2f}x")


if __name__ == "__main__":
    main()

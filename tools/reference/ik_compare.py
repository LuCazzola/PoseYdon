"""Three views of one generated clip, and two IK solvers compared.

The 13-dim representation is redundant: it carries rotations AND positions, and
a generated clip need not keep them consistent. That gives three legitimate ways
to look at the same output:

  rotations  forward kinematics from the rot6d channels. Rigid by construction,
             needs no solver. What PoseYdon renders by default.
  positions  the ric_pos channels as-is. What the reference renders. Tracks the
             intended pose more closely but does not respect bone lengths.
  solved     positions projected back onto a rigid skeleton by IK. What the
             reference writes to BVH, since a BVH stores rotations.

Also times PoseYdon's GradientIK against Motion's animation_from_positions on
the same targets.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from InverseKinematics import animation_from_positions  # Motion

from poseydon.core.spec import FeatureSpec
from poseydon.core.torch_kinematics import forward_kinematics
from poseydon.features import features_to_anim, positions_from_features
from poseydon.io.bvh import BVH
from poseydon.io.render import render_skeleton
from poseydon.solvers import IK_TERMS, GradientIK, SolverSkeleton

SPEC = FeatureSpec((("ric_pos", 3), ("rot6d", 6), ("local_vel", 3), ("foot_contact", 1)))
ASSETS = Path("external/neural_motion_blending/assets/truebones")


def bone_error(positions, parents, offsets) -> float:
    lengths = np.linalg.norm(positions[:, 1:] - positions[:, parents[1:]], axis=-1)
    rest = np.linalg.norm(offsets[1:], axis=-1)[None]
    return float(np.abs(lengths - rest).max())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", default="truebones_attnpool")
    parser.add_argument("--root", default="artifacts/transfer",
                        help="directory transfer_compare.py wrote its .npy into")
    parser.add_argument("--sampler", default="ddpm")
    parser.add_argument("--who", default="poseydon")
    parser.add_argument("--target", default="Scorpion")
    parser.add_argument("--iterations", type=int, default=150)
    args = parser.parse_args()

    root = Path(args.root) / args.save / args.sampler
    features = np.load(root / f"{args.who}_Flamingo_to_{args.target}.npy")
    template = BVH.read(ASSETS / "Scorpion___SlowForward_839.bvh").to_animation().as_rigid_body()
    parents, offsets = template.parents, template.offsets

    from_rotations = features_to_anim(features, SPEC, template).global_positions()
    from_positions = positions_from_features(features, SPEC)

    print(f"{args.save} / {args.sampler} / {args.who}   frames={features.shape[0]} "
          f"joints={features.shape[1]}")
    print(f"mean bone length: {np.linalg.norm(offsets[1:], axis=-1).mean():.4f}")
    print()
    print(f"{'view':12s} {'bone-length error':>18s} {'vs target positions':>21s} {'time (s)':>10s}")
    print(f"{'rotations':12s} {bone_error(from_rotations, parents, offsets):18.4f} "
          f"{np.abs(from_rotations - from_positions).max():21.4f} {0.0:10.3f}")
    print(f"{'positions':12s} {bone_error(from_positions, parents, offsets):18.4f} "
          f"{0.0:21.4f} {0.0:10.3f}")

    targets = torch.tensor(from_positions, dtype=torch.float32)
    skeleton = SolverSkeleton(
        parents=torch.from_numpy(parents.astype("int64")),
        offsets=torch.from_numpy(offsets).float(),
    )

    # PoseYdon: batched gradient IK, position term plus a light smoothness term.
    start = time.perf_counter()
    solver = GradientIK(
        terms=[(1.0, IK_TERMS.get("position")()), (0.05, IK_TERMS.get("smoothness")())],
        iterations=args.iterations,
        learning_rate=0.1,
    )
    rotations = solver.solve(targets, skeleton)
    ours_positions = forward_kinematics(
        rotations, targets[:, 0], skeleton.offsets, skeleton.parents
    ).numpy()
    ours_time = time.perf_counter() - start

    # Motion: the reference's own solver, same targets and iteration count.
    start = time.perf_counter()
    ref_anim, _, _ = animation_from_positions(
        positions=from_positions, parents=list(parents), offsets=offsets,
        iterations=args.iterations,
    )
    from Animation import positions_global

    ref_positions = positions_global(ref_anim)
    ref_time = time.perf_counter() - start

    print(f"{'solved (ours)':11s} {bone_error(ours_positions, parents, offsets):18.4f} "
          f"{np.abs(ours_positions - from_positions).max():21.4f} {ours_time:10.3f}")
    print(f"{'solved (Motion)':11s} {bone_error(ref_positions, parents, offsets):18.4f} "
          f"{np.abs(ref_positions - from_positions).max():21.4f} {ref_time:10.3f}")
    print()
    print(f"mean tracking error  ours {np.abs(ours_positions - from_positions).mean():.4f}   "
          f"Motion {np.abs(ref_positions - from_positions).mean():.4f}")
    print(f"speedup: {ref_time / ours_time:.2f}x")

    out = Path("artifacts/views") / args.save
    out.mkdir(parents=True, exist_ok=True)
    fps = round(template.fps)
    for label, positions in (
        ("rotations", from_rotations),
        ("positions", from_positions),
        ("solved_poseydon", ours_positions),
        ("solved_motion", ref_positions),
    ):
        render_skeleton(out / f"{label}.mp4", parents, positions, fps=fps, title=label)
    BVH.from_animation(features_to_anim(features, SPEC, template)).write(out / "rotations.bvh")
    print(f"wrote four MP4s to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

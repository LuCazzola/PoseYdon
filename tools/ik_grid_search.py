"""Grid search over GradientIK's terms, with the foot-contact term included.

`positions_ik` currently solves with `position` + `smoothness` only. The
registry also carries `contact_pin`, which penalises a joint's velocity on the
frames the MODEL itself marked as planted -- the foot-contact channel it is
trained to predict and that the reference's exporter discards. Nothing wires it
in, so this measures what it would buy.

The trade-off to watch is not whether skate falls -- pinning a joint will always
reduce its sliding -- but what that costs in fidelity to the positions the model
predicted. A solver that holds every foot perfectly still and ignores the motion
is not an improvement.

Run:
    docker compose run --rm --entrypoint python test tools/ik_grid_search.py
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from poseydon.core.spec import FeatureSpec
from poseydon.core.torch_kinematics import forward_kinematics
from poseydon.features import positions_from_features
from poseydon.io.bvh import BVH
from poseydon.solvers import IK_TERMS, GradientIK, SolverSkeleton
from poseydon.training.metrics import foot_skate

SPEC = FeatureSpec((("ric_pos", 3), ("rot6d", 6), ("local_vel", 3), ("foot_contact", 1)))
FEATURES = "artifacts/headtohead-full/truebones_attnpool/ddpm/reference_Flamingo_to_Scorpion.npy"
TEMPLATE = "external/neural_motion_blending/assets/truebones/Scorpion___SlowForward_839.bvh"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default=FEATURES)
    parser.add_argument("--template", default=TEMPLATE)
    parser.add_argument("--iterations", type=int, default=150)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--contact", type=float, nargs="+",
                        default=[0.0, 0.1, 0.5, 2.0, 10.0])
    parser.add_argument("--smoothness", type=float, nargs="+", default=[0.0, 0.05, 0.2])
    args = parser.parse_args()

    features = np.load(args.features)
    template = BVH.read(args.template).to_animation().as_rigid_body()
    targets_np = positions_from_features(features, SPEC)
    targets = torch.tensor(targets_np, dtype=torch.float32)
    skeleton = SolverSkeleton(
        parents=torch.from_numpy(template.parents.astype("int64")),
        offsets=torch.from_numpy(template.offsets).float(),
    )

    # The model's OWN contact prediction, not one re-derived from the output.
    # Re-deriving it would make the metric agree with the solver by
    # construction: both would be reading the same positions.
    contact = features[..., SPEC.slice("foot_contact")][..., 0] > 0.5  # (F, J)
    scale = float(np.linalg.norm(template.offsets[1:], axis=-1).mean())
    rest = np.linalg.norm(template.offsets[1:], axis=-1)[None]

    planted = contact[:-1]
    print(f"frames {features.shape[0]}  joints {features.shape[1]}  "
          f"mean bone {scale:.4f}")
    print(f"predicted contacts: {100 * planted.mean():.1f}% of (frame, joint) cells, "
          f"{int(contact.any(axis=0).sum())} joints ever planted")
    print(f"raw predicted positions: skate {foot_skate(targets_np, contact, scale=scale):.5f}\n")

    header = (f"{'contact':>8s} {'smooth':>7s} {'skate':>9s} {'vs raw':>8s} "
              f"{'track':>9s} {'bone err':>9s} {'time':>7s}")
    print(header)
    print("-" * len(header))

    baseline_skate = None
    rows = []
    for smooth in args.smoothness:
        for weight in args.contact:
            terms = [(1.0, IK_TERMS.get("position")())]
            if smooth > 0:
                terms.append((smooth, IK_TERMS.get("smoothness")()))
            if weight > 0:
                terms.append((weight, IK_TERMS.get("contact_pin")(
                    torch.from_numpy(contact.astype("float32")))))

            start = time.perf_counter()
            rotations = GradientIK(
                terms=terms, iterations=args.iterations, learning_rate=args.learning_rate
            ).solve(targets, skeleton)
            elapsed = time.perf_counter() - start

            solved = forward_kinematics(
                rotations, targets[:, 0], skeleton.offsets, skeleton.parents
            ).numpy()

            skate = foot_skate(solved, contact, scale=scale)
            track = float(np.abs(solved - targets_np).mean())
            live = np.linalg.norm(solved[:, 1:] - solved[:, template.parents[1:]], axis=-1)
            bone = float(np.abs(live - rest).max())
            if baseline_skate is None:
                baseline_skate = skate
            print(f"{weight:8.2f} {smooth:7.2f} {skate:9.5f} "
                  f"{100 * (skate / baseline_skate - 1):+7.1f}% {track:9.5f} "
                  f"{bone:9.6f} {elapsed:6.1f}s")
            rows.append((weight, smooth, skate, track, bone))

    print("\nThe cost of pinning, against the no-contact solve at each smoothness:")
    for smooth in args.smoothness:
        at = [r for r in rows if r[1] == smooth]
        base = next(r for r in at if r[0] == 0.0)
        for weight, _, skate, track, _ in at:
            if weight == 0.0:
                continue
            print(f"  smooth {smooth:4.2f}  contact {weight:5.2f}: "
                  f"skate {100 * (skate / base[2] - 1):+6.1f}%   "
                  f"tracking {100 * (track / base[3] - 1):+6.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

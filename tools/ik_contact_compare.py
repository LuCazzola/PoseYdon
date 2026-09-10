"""Default vs best-from-grid solve, with the predicted contacts shown.

Two questions in one render:

* Does the contact term earn its place? Default (`position` + `smoothness`,
  what `positions_ik` ships) against the grid's best trade (`contact_pin` at
  0.5), same targets and iterations.
* Does the model's foot-contact channel mean anything? The highlighted joints
  are the ones IT marked planted, per frame. A joint lit up in mid-air is a
  wrong prediction, and no solver setting can fix that -- it would only pin the
  motion to a lie.

Run:
    docker compose run --rm --entrypoint python test tools/ik_contact_compare.py
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from poseydon.core.spec import FeatureSpec
from poseydon.core.torch_kinematics import forward_kinematics
from poseydon.features import positions_from_features
from poseydon.io.bvh import BVH
from poseydon.io.render import render_skeleton
from poseydon.solvers import IK_TERMS, GradientIK, SolverSkeleton
from poseydon.training.metrics import foot_skate

SPEC = FeatureSpec((("ric_pos", 3), ("rot6d", 6), ("local_vel", 3), ("foot_contact", 1)))
FEATURES = "artifacts/headtohead-full/truebones_attnpool/ddpm/reference_Flamingo_to_Scorpion.npy"
TEMPLATE = "external/neural_motion_blending/assets/truebones/Scorpion___SlowForward_839.bvh"
OUT = "artifacts/contact"


def solve(targets, skeleton, contact, weight, smooth, iterations, lr):
    terms = [(1.0, IK_TERMS.get("position")())]
    if smooth > 0:
        terms.append((smooth, IK_TERMS.get("smoothness")()))
    if weight > 0:
        terms.append((weight, IK_TERMS.get("contact_pin")(
            torch.from_numpy(contact.astype("float32")))))
    rotations = GradientIK(terms=terms, iterations=iterations, learning_rate=lr).solve(
        targets, skeleton
    )
    return forward_kinematics(
        rotations, targets[:, 0], skeleton.offsets, skeleton.parents
    ).numpy()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default=FEATURES)
    parser.add_argument("--template", default=TEMPLATE)
    parser.add_argument("--iterations", type=int, default=150)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--out", default=OUT)
    args = parser.parse_args()

    features = np.load(args.features)
    template = BVH.read(args.template).to_animation().as_rigid_body()
    targets_np = positions_from_features(features, SPEC)
    targets = torch.tensor(targets_np, dtype=torch.float32)
    skeleton = SolverSkeleton(
        parents=torch.from_numpy(template.parents.astype("int64")),
        offsets=torch.from_numpy(template.offsets).float(),
    )
    contact = features[..., SPEC.slice("foot_contact")][..., 0] > 0.5
    scale = float(np.linalg.norm(template.offsets[1:], axis=-1).mean())

    # Is the contact channel telling the truth? A planted joint should be ON the
    # floor. This is the model's claim measured against the model's own
    # positions, so it needs no ground truth -- an internal consistency check.
    floor = float(targets_np[..., 1].min())
    height = targets_np[..., 1] - floor
    planted_h, free_h = height[contact], height[~contact]
    print(f"predicted contacts: {100 * contact.mean():.1f}% of cells, "
          f"{int(contact.any(axis=0).sum())}/{contact.shape[1]} joints ever planted")
    print(f"height above floor, in mean-bone-lengths ({scale:.3f} each):")
    print(f"  planted joints : median {np.median(planted_h)/scale:6.2f}  "
          f"p90 {np.percentile(planted_h, 90)/scale:6.2f}")
    print(f"  free joints    : median {np.median(free_h)/scale:6.2f}  "
          f"p90 {np.percentile(free_h, 90)/scale:6.2f}")
    airborne = (planted_h / scale > 1.0).mean()
    print(f"  planted but over one bone-length up: {100 * airborne:.1f}%\n")

    for label, weight, smooth in (("default", 0.0, 0.05), ("best_grid", 0.5, 0.05)):
        solved = solve(targets, skeleton, contact, weight, smooth,
                       args.iterations, args.learning_rate)
        skate = foot_skate(solved, contact, scale=scale)
        track = float(np.abs(solved - targets_np).mean())
        out = f"{args.out}/{label}.mp4"
        render_skeleton(
            out, template.parents, solved, 30,
            f"{label}  (contact={weight}, smooth={smooth})  skate={skate:.5f}",
            highlight={"#e6194b": contact},
        )
        print(f"{label:10s} contact={weight:<4} skate {skate:.5f}  tracking {track:.5f}"
              f"  -> {out}")

    # The raw predicted positions, same highlighting, as the reference point.
    render_skeleton(
        f"{args.out}/raw_positions.mp4", template.parents, targets_np, 30,
        f"raw predicted positions  skate={foot_skate(targets_np, contact, scale=scale):.5f}",
        highlight={"#e6194b": contact},
    )
    print(f"{'raw':10s} (no solve)      -> {args.out}/raw_positions.mp4")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

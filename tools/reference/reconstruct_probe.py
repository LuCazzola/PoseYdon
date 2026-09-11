"""Reconstruction through the model: encode a clip, DDIM-decode it back.

The training objective is reconstruction, so the sharpest question to ask a
trained checkpoint is how well it reproduces a clip it is shown -- and WHICH
part of the representation it gets wrong. A single error number averages a
position in metres against a 6D rotation entry, so it cannot answer that.

Same rig in and out: the semantic latent is taken from the clip, and the decode
is conditioned on the clip's own skeleton. Anything lost is the autoencoder's
bottleneck plus the sampler, not a topology change.

DDIM at eta=0, so the result is deterministic and repeatable -- a stochastic
sampler would fold its own variance into the error and make the blocks
incomparable between runs.

Run:
    docker compose run --rm --entrypoint python compat \
        tools/reference/reconstruct_probe.py --save truebones_attnpool
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path("tools/reference").resolve()))
from run_reference_transfer import ASSETS, build_model_and_diffusion, build_y, load_skeleton

from poseydon.core.batch import Cond, Masks
from poseydon.core.spec import FeatureSpec
from poseydon.models.modiffae import JOINT_NAMES, TEMPORAL_VALID, Z_SEM, MoDiffAE
from poseydon.process import GaussianDiffusion
from poseydon.sampling import DDIM

SPEC = FeatureSpec((("ric_pos", 3), ("rot6d", 6), ("local_vel", 3), ("foot_contact", 1)))
REF_DIR = Path("data/truebones/reference")
CLIPS = {
    "BrownBear": "BrownBear___RiseSwat_132",
    "Coyote": "Coyote___Attack3_224",
    "Crab": "Crab___Attack3_234",
    "Flamingo": "Flamingo_Flamingo_OneLEgBEnt_353",
    "Goat": "Goat___HeadButt_395",
    "Scorpion": "Scorpion___SlowForward_839",
    "Skunk": "Skunk___Spray_891",
}


def rest_row(pair: torch.Tensor) -> torch.Tensor:
    out = pair.clone().bool()
    out[:, 0, :] = False
    out[:, 0, 0] = True
    return out


class _Trace:
    """Wraps the denoiser and keeps its x0 estimate at every step.

    The estimate is what converges; the STATE `z_t` is still carrying scheduled
    noise, so measuring that instead reports the schedule rather than the model.
    A Control cannot see this -- it is handed the state, not the prediction --
    so the recording goes around the model itself.
    """

    def __init__(self, model) -> None:
        self.model = model
        self.steps: list[tuple[int, torch.Tensor]] = []

    def __call__(self, z_t, t, cond, masks=None):
        prediction = self.model(z_t, t, cond, masks)
        self.steps.append((int(t[0]), prediction.out.detach().clone()))
        return prediction

    def __getattr__(self, name):
        return getattr(self.model, name)


#: Below this a block does not vary, so there is no spread to measure against.
CONSTANT = 1e-6


def _relative(got: np.ndarray, want: np.ndarray) -> tuple[float, bool]:
    """Error over the block's own spread, and whether that spread exists.

    Dividing by a near-zero standard deviation manufactures a number rather
    than measuring one: Crab's `foot_contact` is identically zero across its
    clip, and the first version of this probe reported an error of 1.17e7 for
    it -- a metric artifact that reads as a catastrophic model failure.

    A constant block is reported as its RAW mean absolute error, marked, and
    kept out of the averages, because a relative error against no variation is
    undefined rather than large.
    """
    spread = float(want.std())
    error = float(np.abs(got - want).mean())
    if spread < CONSTANT:
        return error, True
    return error / spread, False


def to_cond(y) -> Cond:
    return Cond({
        "topology": {"hops": y["graph_dist"], "relations": y["joints_relations"]},
        "tpose": y["tpos_first_frame"],
        JOINT_NAMES: y["joints_names_embs"],
        TEMPORAL_VALID: rest_row(y["mask"][:, 0, 0]),
        "crop_start": y["crop_start_ind"],
    })


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", default="truebones_attnpool")
    parser.add_argument("--frames", type=int, default=40)
    parser.add_argument("--steps", type=int, default=None, help="DDIM steps; default = all")
    parser.add_argument("--start", default="noise", choices=["noise", "zeros"],
                        help="z_T. `noise` is the real trajectory; `zeros` is "
                             "out of distribution and was the first version's "
                             "mistake")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--every", type=int, default=10,
                        help="report every Nth step of the trajectory")
    args = parser.parse_args()

    reference_model, _, raw = build_model_and_diffusion(args.save)
    reference_model.eval()
    model = MoDiffAE(
        feature_dim=13, d_model=raw["latent_dim"],
        n_layers_semantic=raw["num_layers_semantic"],
        n_layers_stochastic=raw["num_layers_stochastic"],
        n_heads=4, ff_size=1024,
        n_virtual_joints=raw["num_virtual_joints"],
        projection_head_depth=raw["projection_head_depth"],
    )
    model.load_state_dict(reference_model.state_dict(), strict=True)
    model = model.eval().to(args.device)

    process = GaussianDiffusion(
        num_steps=raw.get("diffusion_steps", 100),
        schedule=raw["noise_schedule"], parameterization="x0",
    )
    cache = dict(np.load(REF_DIR / "t5_cache.npz", allow_pickle=False))

    print(f"{args.save}  DDIM eta=0  {args.steps or process.num_steps} steps  "
          f"{args.frames} frames  z_T={args.start}  on {args.device}\n")
    header = f"{'rig':11s} {'joints':>6s} " + " ".join(f"{n:>13s}" for n in SPEC.names)
    print(header)
    print("-" * len(header))

    totals: dict[str, list[float]] = {name: [] for name in SPEC.names}
    trajectories: dict[str, list] = {}
    truths: dict[str, tuple] = {}
    for rig, stem in CLIPS.items():
        skeleton = load_skeleton(rig)
        joints = len(skeleton["parents"])
        y = build_y(skeleton, cache, joints, args.frames, raw["temporal_window"], "cpu")

        clean = np.load(ASSETS / f"{stem}.npy")[: args.frames]
        if clean.shape[0] < args.frames:
            print(f"{rig:11s} skipped: only {clean.shape[0]} frames")
            continue
        mean, std = skeleton["mean"][None], skeleton["std"][None] + 1e-6
        normalized = np.nan_to_num((clean - mean) / std)
        x = torch.tensor(normalized, dtype=torch.float32).permute(1, 2, 0)[None].to(args.device)

        masks = Masks(
            frames=torch.ones(1, args.frames, dtype=torch.bool, device=args.device),
            joints=torch.ones(1, joints, dtype=torch.bool, device=args.device),
        )
        cond = to_cond(y).to(args.device)
        traced = _Trace(model)
        with torch.no_grad():
            z_sem, _ = model.encode(x, cond, masks)
            out = DDIM(steps=args.steps).sample(
                traced, process, tuple(x.shape),
                Cond({**cond.payloads, Z_SEM: z_sem}), masks=masks,
                device=args.device,
                start=(
                    torch.randn(
                        x.shape, generator=torch.Generator().manual_seed(args.seed)
                    ).to(args.device)
                    if args.start == "noise"
                    else torch.zeros_like(x)
                ),
            )
        trajectories[rig] = traced.steps

        got = out[0].permute(2, 0, 1).cpu().numpy() * std + mean
        want = clean
        truths[rig] = (want, mean, std)
        row = f"{rig:11s} {joints:6d} "
        for name in SPEC.names:
            error, constant = _relative(SPEC.take(got, name), SPEC.take(want, name))
            if not constant:
                totals[name].append(error)
            row += f"{error:12.4f}{'*' if constant else ' '} "
        print(row)

    print("-" * len(header))
    print(f"{'mean':11s} {'':6s} " + " ".join(
        f"{np.mean(totals[n]):13.4f}" if totals[n] else f"{'--':>13s}"
        for n in SPEC.names))
    print("\nError is mean |difference| over the block's own standard deviation,")
    print("so the columns are comparable. 1.0 means the model is as wrong as")
    print("simply predicting that block's mean. A `*` marks a block that does")
    print("not vary in that clip: there is no spread to divide by, so the raw")
    print("mean absolute error is shown and it is left out of the averages.")

    _by_step(trajectories, truths, args.every)
    return 0


def _by_step(trajectories, truths, every: int = 10) -> None:
    """The same error, but for the x0 estimate at each point on the ladder.

    Averaged over rigs. Steps run high (noisy) to low (clean), which is the
    order DDIM walks them, so reading downwards follows the trajectory.
    """
    if not trajectories:
        return
    order = [step for step, _ in next(iter(trajectories.values()))]
    print(f"\n\nx0 estimate error along the trajectory ({len(order)} steps, "
          f"averaged over {len(trajectories)} rigs)\n")
    header = f"{'step':>6s} " + " ".join(f"{n:>13s}" for n in SPEC.names)
    print(header)
    print("-" * len(header))

    show = set(order[::every]) | {order[-1]}
    for position, step in enumerate(order):
        if step not in show:
            continue
        row = f"{step:6d} "
        for name in SPEC.names:
            errors = []
            for rig, steps in trajectories.items():
                want, mean, std = truths[rig]
                estimate = steps[position][1][0].permute(2, 0, 1).cpu().numpy() * std + mean
                error, constant = _relative(SPEC.take(estimate, name), SPEC.take(want, name))
                if not constant:
                    errors.append(error)
            row += f"{np.mean(errors):13.4f} " if errors else f"{'--':>13s} "
        print(row)


if __name__ == "__main__":
    raise SystemExit(main())

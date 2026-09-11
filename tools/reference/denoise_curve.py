"""Does the model actually get better as the input gets cleaner?

The training loss says yes, by a factor of 11 between the noisiest and cleanest
quartile. A sampling trajectory probe said the x0 estimate barely moves. Both
cannot describe the same model, so this measures the thing training measures:
corrupt a REAL clip to a known t and score the prediction.

If the curve is steep, the model is fine and the flat sampling result is about
the trajectory -- z_t during sampling is the model's own output, not corrupted
ground truth, so a drifted state never becomes the clean input training rewards.

If the curve is flat, the discrepancy is a bug in the read path, and this is
where it shows.

Also reports the prediction with the semantic latent WITHHELD. If the numbers
barely move without it, the decoder is ignoring z_t and leaning on the latent,
which would explain a flat trajectory on its own.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path("tools/reference").resolve()))
from reconstruct_probe import CLIPS, REF_DIR, SPEC, to_cond
from run_reference_transfer import ASSETS, build_model_and_diffusion, build_y, load_skeleton

from poseydon.core.batch import Cond, Masks
from poseydon.models.modiffae import Z_SEM, MoDiffAE
from poseydon.process import GaussianDiffusion


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", default="truebones_attnpool")
    parser.add_argument("--frames", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    reference_model, _, raw = build_model_and_diffusion(args.save)
    model = MoDiffAE(
        feature_dim=13, d_model=raw["latent_dim"],
        n_layers_semantic=raw["num_layers_semantic"],
        n_layers_stochastic=raw["num_layers_stochastic"],
        n_heads=4, ff_size=1024,
        n_virtual_joints=raw["num_virtual_joints"],
        projection_head_depth=raw["projection_head_depth"],
    )
    model.load_state_dict(reference_model.state_dict(), strict=True)
    model.eval()
    process = GaussianDiffusion(
        num_steps=raw.get("diffusion_steps", 100),
        schedule=raw["noise_schedule"], parameterization="x0",
    )
    cache = dict(np.load(REF_DIR / "t5_cache.npz", allow_pickle=False))
    generator = torch.Generator().manual_seed(args.seed)

    levels = [0, 10, 25, 50, 75, 99]
    print(f"{args.save}: x0 prediction from CORRUPTED GROUND TRUTH, as training sees it")
    print(f"normalized MSE over {len(CLIPS)} rigs, {args.frames} frames\n")
    print(f"{'t':>4s} {'noise%':>7s} {'with z_sem':>12s} {'no z_sem':>12s}   per-block (with z_sem)")
    print("-" * 92)

    for t_value in levels:
        with_latent, without_latent = [], []
        blocks: dict[str, list[float]] = {n: [] for n in SPEC.names}
        for rig, stem in CLIPS.items():
            skeleton = load_skeleton(rig)
            joints = len(skeleton["parents"])
            y = build_y(skeleton, cache, joints, args.frames, raw["temporal_window"], "cpu")
            clean = np.load(ASSETS / f"{stem}.npy")[: args.frames]
            if clean.shape[0] < args.frames:
                continue
            mean, std = skeleton["mean"][None], skeleton["std"][None] + 1e-6
            x0 = torch.tensor(
                np.nan_to_num((clean - mean) / std), dtype=torch.float32
            ).permute(1, 2, 0)[None]

            masks = Masks(
                frames=torch.ones(1, args.frames, dtype=torch.bool),
                joints=torch.ones(1, joints, dtype=torch.bool),
            )
            cond = to_cond(y)
            t = torch.full((1,), t_value, dtype=torch.long)
            noise = torch.randn(x0.shape, generator=generator)
            z_t = process.corrupt(x0, t, noise)

            with torch.no_grad():
                z_sem, _ = model.encode(x0, cond, masks)
                got = model(z_t, t, Cond({**cond.payloads, Z_SEM: z_sem}), masks).out
                # Shuffle the latent between rigs to see how much of the
                # prediction is the latent rather than z_t.
                shuffled = model(
                    z_t, t, Cond({**cond.payloads, Z_SEM: torch.roll(z_sem, 1, dims=0)}),
                    masks,
                ).out

            with_latent.append(float(((got - x0) ** 2).mean()))
            without_latent.append(float(((shuffled - x0) ** 2).mean()))
            for name in SPEC.names:
                a, b = SPEC.take(got, name, axis=2), SPEC.take(x0, name, axis=2)
                blocks[name].append(float(((a - b) ** 2).mean()))

        detail = "  ".join(f"{n}={np.mean(blocks[n]):.3f}" for n in SPEC.names)
        print(f"{t_value:4d} {100 * t_value / process.num_steps:6.0f}% "
              f"{np.mean(with_latent):12.4f} {np.mean(without_latent):12.4f}   {detail}")

    print("\nTraining logs `simple` per noise quartile; q1 is t in [0,25) and q4")
    print("is [75,100). A steep curve here means the model does use z_t.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

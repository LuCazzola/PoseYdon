"""Separate loop differences from model differences under DDPM.

Three runs on one shared noise stream:

  A  reference loop + reference model
  B  PoseYdon loop  + reference model   -- A vs B isolates the SAMPLER
  C  PoseYdon loop  + PoseYdon model    -- B vs C isolates the MODEL
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path("tools/reference").resolve()))
from run_reference_transfer import build_model_and_diffusion, load_inputs  # noqa: E402
from transfer_compare import ReplayNoise, to_cond, to_masks  # noqa: E402

from poseydon.core.batch import Cond  # noqa: E402
from poseydon.models.base import Denoiser, Prediction  # noqa: E402
from poseydon.models.modiffae import Z_SEM, MoDiffAE  # noqa: E402
from poseydon.process import GaussianDiffusion  # noqa: E402
from poseydon.sampling import DDPM  # noqa: E402


class ReferenceAdapter(Denoiser):
    """Presents the reference model through PoseYdon's Denoiser interface."""

    def __init__(self, model, y, control):
        super().__init__()
        self.model = model
        self.y = y
        self.control = control

    def forward(self, z_t, t, cond, masks=None):
        out, _ = self.model(z_t, t, y=self.y, control=self.control)
        return Prediction(out=out["out"])


def report(name, a, b):
    diff = np.abs(a - b)
    print(f"  {name:34s} max {diff.max():.3e}   mean {diff.mean():.3e}   "
          f"rel {diff.max() / np.abs(a).max():.3e}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", default="truebones_globpool")
    parser.add_argument("--frames", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    reference_model, diffusion, raw = build_model_and_diffusion(args.save)
    reference_model.eval()
    inputs = load_inputs("Flamingo", "Scorpion", args.frames, raw["temporal_window"])
    joints_pad = inputs["n_joints_pad"]
    n_target = len(inputs["target"]["parents"])

    ours = MoDiffAE(
        feature_dim=13, d_model=raw["latent_dim"],
        n_layers_semantic=raw["num_layers_semantic"],
        n_layers_stochastic=raw["num_layers_stochastic"],
        n_heads=4, ff_size=1024,
        n_virtual_joints=raw["num_virtual_joints"],
        projection_head_depth=raw["projection_head_depth"],
    )
    ours.load_state_dict(reference_model.state_dict(), strict=True)
    ours.eval()

    from model.motion_diffusion_ae import ControlConfig  # noqa: PLC0415

    control = ControlConfig(x=inputs["control_x"], y=inputs["y_source"], alpha=torch.zeros(1))
    model_kwargs = {"y": inputs["y_target"], "control": control}

    shape = (1, joints_pad, 13, args.frames)
    generator = torch.Generator().manual_seed(args.seed)
    steps = int(raw.get("diffusion_steps", 100))
    stream = [torch.randn(shape, generator=generator) for _ in range(steps + 2)]

    process = GaussianDiffusion(
        num_steps=steps, schedule=raw["noise_schedule"], parameterization="x0"
    )
    cond_target = to_cond(inputs["y_target"])
    masks_target = to_masks(inputs["y_target"], joints_pad, args.frames)
    cond_source = to_cond(inputs["y_source"])
    masks_source = to_masks(inputs["y_source"], joints_pad, args.frames)

    def crop(tensor):
        return tensor[0].permute(2, 0, 1).numpy()[:, :n_target]

    with torch.no_grad():
        with ReplayNoise(stream):
            a = crop(diffusion.p_sample_loop(
                reference_model, shape, clip_denoised=False,
                model_kwargs=model_kwargs, progress=False,
            ))

        adapter = ReferenceAdapter(reference_model, inputs["y_target"], control)
        with ReplayNoise(stream):
            b = crop(DDPM().sample(adapter, process, shape, Cond({}), masks=masks_target))

        z_sem, _ = ours.encode(inputs["control_x"][:, 0], cond_source, masks_source)
        target_cond = Cond({**cond_target.payloads, Z_SEM: z_sem})
        with ReplayNoise(stream):
            c = crop(DDPM().sample(ours, process, shape, target_cond, masks=masks_target))

    print(f"{args.save}, DDPM x {steps} steps, shared noise stream")
    report("A vs B  (sampler only)", a, b)
    report("B vs C  (model only)", b, c)
    report("A vs C  (end to end)", a, c)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

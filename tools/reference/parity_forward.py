"""Single-forward parity: PoseYdon's MoDiffAE against the reference's.

Same weights, same inputs, same eval mode. This is the honest test of the port:
sampling differences could hide behind stochastic loops, but one forward pass
cannot.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path("tools/reference").resolve()))
from run_reference_transfer import (
    build_model_and_diffusion,
    load_inputs,
)

from poseydon.core.batch import Cond, Masks
from poseydon.models.modiffae import JOINT_NAMES, TEMPORAL_VALID, Z_SEM, MoDiffAE


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--save", default="truebones_globpool")
    args = parser.parse_args()

    torch.manual_seed(0)
    reference_model, _, raw = build_model_and_diffusion(args.save)
    inputs = load_inputs("Flamingo", "Scorpion", frames=40, window=raw["temporal_window"])
    print(f"checkpoint: {args.save}  virtual_joints={raw['num_virtual_joints']}")
    reference_model.eval()

    ours = MoDiffAE(
        feature_dim=13,
        d_model=raw["latent_dim"],
        n_layers_semantic=raw["num_layers_semantic"],
        n_layers_stochastic=raw["num_layers_stochastic"],
        n_heads=4,
        ff_size=1024,
        n_virtual_joints=raw["num_virtual_joints"],
        projection_head_depth=raw["projection_head_depth"],
    )
    ours.load_state_dict(reference_model.state_dict(), strict=True)
    ours.eval()
    print("state_dict loaded into PoseYdon with strict=True")

    x_t = torch.randn(1, inputs["n_joints_pad"], 13, 40, generator=torch.Generator().manual_seed(1))
    t = torch.tensor([17])

    from model.motion_diffusion_ae import ControlConfig

    control = ControlConfig(x=inputs["control_x"], y=inputs["y_source"], alpha=torch.zeros(1))
    with torch.no_grad():
        ref_out, _ = reference_model(x_t, t, y=inputs["y_target"], control=control)

    y_t, y_s = inputs["y_target"], inputs["y_source"]
    cond_target = Cond({
        "topology": {"hops": y_t["graph_dist"], "relations": y_t["joints_relations"]},
        "tpose": y_t["tpos_first_frame"],
        JOINT_NAMES: y_t["joints_names_embs"],
        TEMPORAL_VALID: _fix_rest_row(y_t["mask"][:, 0, 0]),
        "crop_start": y_t["crop_start_ind"],
    })
    cond_source = Cond({
        "topology": {"hops": y_s["graph_dist"], "relations": y_s["joints_relations"]},
        "tpose": y_s["tpos_first_frame"],
        JOINT_NAMES: y_s["joints_names_embs"],
        TEMPORAL_VALID: _fix_rest_row(y_s["mask"][:, 0, 0]),
        "crop_start": y_s["crop_start_ind"],
    })
    masks_target = _masks(y_t, inputs["n_joints_pad"], 40)
    masks_source = _masks(y_s, inputs["n_joints_pad"], 40)

    with torch.no_grad():
        z_sem, _ = ours.encode(inputs["control_x"][:, 0], cond_source, masks_source)
        ours_out = ours(
            x_t, t, Cond({**cond_target.payloads, Z_SEM: z_sem}), masks_target
        ).out

    ref_np = ref_out["out"].numpy()
    our_np = ours_out.numpy()
    diff = np.abs(ref_np - our_np)
    scale = np.abs(ref_np).max()
    print(f"reference out : {ref_np.shape}  max|.| = {scale:.4f}")
    print(f"poseydon  out : {our_np.shape}")
    print(f"max abs diff  : {diff.max():.3e}")
    print(f"mean abs diff : {diff.mean():.3e}")
    print(f"relative      : {diff.max() / scale:.3e}")
    print("PARITY: PASS" if diff.max() / scale < 1e-4 else "PARITY: FAIL")
    return 0


def _fix_rest_row(pair: torch.Tensor) -> torch.Tensor:
    """Reproduce the reference's rest-frame mask correction."""
    out = pair.clone().bool()
    out[:, 0, :] = False
    out[:, 0, 0] = True
    return out


def _masks(y, n_joints_pad, frames) -> Masks:
    joints = torch.zeros(1, n_joints_pad, dtype=torch.bool)
    joints[0, : int(y["n_joints"][0])] = True
    return Masks(frames=torch.ones(1, frames, dtype=torch.bool), joints=joints)


if __name__ == "__main__":
    raise SystemExit(main())

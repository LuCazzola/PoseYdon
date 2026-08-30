"""Flamingo -> Scorpion transfer through BOTH implementations, from one noise.

DDIM is deterministic, so seeding the starting noise once and handing the same
tensor to each makes the two trajectories directly comparable: any divergence is
the models, not the sampler's randomness.

Writes, for each implementation, the raw features, a BVH, and an MP4.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path("tools/reference").resolve()))
from run_reference_transfer import (
    ASSETS,
    build_model_and_diffusion,
    load_inputs,
)

from poseydon.core.batch import Cond, Masks
from poseydon.core.spec import FeatureSpec
from poseydon.features import extract_features, features_to_anim
from poseydon.io.bvh import load_bvh, save_bvh
from poseydon.models.modiffae import JOINT_NAMES, TEMPORAL_VALID, Z_SEM, MoDiffAE

SPEC = FeatureSpec((("ric_pos", 3), ("rot6d", 6), ("local_vel", 3), ("foot_contact", 1)))
STEMS = {"Flamingo": "Flamingo_Flamingo_OneLEgBEnt_353", "Scorpion": "Scorpion___SlowForward_839"}


def rest_row(pair: torch.Tensor) -> torch.Tensor:
    out = pair.clone().bool()
    out[:, 0, :] = False
    out[:, 0, 0] = True
    return out


def to_cond(y, joint_names_key=JOINT_NAMES) -> Cond:
    return Cond({
        "topology": {"hops": y["graph_dist"], "relations": y["joints_relations"]},
        "tpose": y["tpos_first_frame"],
        joint_names_key: y["joints_names_embs"],
        TEMPORAL_VALID: rest_row(y["mask"][:, 0, 0]),
        "crop_start": y["crop_start_ind"],
    })


def to_masks(y, joints_pad, frames) -> Masks:
    joints = torch.zeros(1, joints_pad, dtype=torch.bool)
    joints[0, : int(y["n_joints"][0])] = True
    return Masks(frames=torch.ones(1, frames, dtype=torch.bool), joints=joints)


def render(path: Path, parents, positions, fps, title, face_joints=None) -> None:
    from tools.render import render_skeleton

    render_skeleton(path, parents, positions, fps=fps, title=title)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", default="truebones_globpool")
    parser.add_argument("--source", default="Flamingo")
    parser.add_argument("--target", default="Scorpion")
    parser.add_argument("--frames", type=int, default=40)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="artifacts/transfer")
    args = parser.parse_args()

    out_dir = Path(args.out) / args.save
    out_dir.mkdir(parents=True, exist_ok=True)

    reference_model, diffusion, raw = build_model_and_diffusion(args.save)
    reference_model.eval()
    inputs = load_inputs(args.source, args.target, args.frames, raw["temporal_window"])
    joints_pad = inputs["n_joints_pad"]
    target = inputs["target"]
    n_target = len(target["parents"])

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

    shape = (1, joints_pad, 13, args.frames)
    noise = torch.randn(shape, generator=torch.Generator().manual_seed(args.seed))

    # ---- reference, deterministic DDIM ----
    from model.motion_diffusion_ae import ControlConfig

    control = ControlConfig(x=inputs["control_x"], y=inputs["y_source"], alpha=torch.zeros(1))
    with torch.no_grad():
        ref_out = diffusion.ddim_sample_loop(
            reference_model, shape, noise=noise, clip_denoised=False,
            model_kwargs={"y": inputs["y_target"], "control": control}, progress=False,
        )
    ref_features = ref_out[0].permute(2, 0, 1).numpy()[:, :n_target]

    # ---- PoseYdon, same noise, same schedule ----
    cond_source = to_cond(inputs["y_source"])
    cond_target = to_cond(inputs["y_target"])
    masks_source = to_masks(inputs["y_source"], joints_pad, args.frames)
    masks_target = to_masks(inputs["y_target"], joints_pad, args.frames)

    from poseydon.process import GaussianDiffusion
    from poseydon.sampling import DDIM

    process = GaussianDiffusion(num_steps=100, schedule=raw["noise_schedule"], parameterization="x0")
    with torch.no_grad():
        z_sem, _ = ours.encode(inputs["control_x"][:, 0], cond_source, masks_source)
        our_out = DDIM(steps=None).sample(
            ours, process, shape,
            Cond({**cond_target.payloads, Z_SEM: z_sem}),
            masks=masks_target, start=noise,
        )
    our_features = our_out[0].permute(2, 0, 1).numpy()[:, :n_target]

    # ---- numerical comparison ----
    diff = np.abs(ref_features - our_features)
    scale = np.abs(ref_features).max()
    print(f"features       : {ref_features.shape}")
    print(f"max abs diff   : {diff.max():.3e}   (values up to {scale:.3f})")
    print(f"mean abs diff  : {diff.mean():.3e}")
    print(f"relative       : {diff.max() / scale:.3e}")
    for name in SPEC.names:
        block = SPEC.slice(name)
        print(f"  {name:14s} max diff {diff[..., block].max():.3e}")

    # ---- export ----
    template = load_bvh(ASSETS / f"{STEMS[args.target]}.bvh")
    source_anim = load_bvh(ASSETS / f"{STEMS[args.source]}.bvh")
    target_face = _face_joints(args.target, template)
    source_face = _face_joints(args.source, source_anim)
    mean, std = target["mean"][None], target["std"][None] + 1e-6

    outputs = {"reference": ref_features, "poseydon": our_features}
    for label, features in outputs.items():
        raw_feats = features * std + mean
        np.save(out_dir / f"{label}_{args.source}_to_{args.target}.npy", raw_feats)
        anim = features_to_anim(raw_feats, SPEC, template)
        bvh = out_dir / f"{label}_{args.source}_to_{args.target}.bvh"
        save_bvh(anim, bvh)
        render(
            out_dir / f"{label}_{args.source}_to_{args.target}.mp4",
            template.parents, anim.global_positions(), round(template.fps),
            f"{label}: {args.source} -> {args.target}", target_face,
        )
        print(f"wrote {bvh.name} and .mp4")

    # Source clip, for visual reference.
    src_feats, _ = extract_features(source_anim, _resolved(args.source, source_anim))
    src_anim = features_to_anim(src_feats[: args.frames], SPEC, source_anim)
    save_bvh(src_anim, out_dir / f"source_{args.source}.bvh")
    render(
        out_dir / f"source_{args.source}.mp4", source_anim.parents,
        src_anim.global_positions(), round(source_anim.fps), f"source: {args.source}",
        source_face,
    )
    print(f"wrote source_{args.source}.bvh and .mp4")
    return 0


def _face_joints(name, anim):
    """Flattened face-joint indices, in the order the reference's plotter wants."""
    resolved = _resolved(name, anim)
    return [index for pair in resolved.facing_indices for index in pair]


def _resolved(name, anim):
    from poseydon.core.skeleton import SkeletonManifest, resolve

    manifest = SkeletonManifest.load(Path("data/truebones/skeletons") / f"{name}.yaml")
    return resolve(manifest, anim.names)


if __name__ == "__main__":
    raise SystemExit(main())

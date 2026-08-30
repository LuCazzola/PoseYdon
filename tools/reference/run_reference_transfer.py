"""Run the REFERENCE MoDiffAE for a cross-skeleton transfer, as ground truth.

Encodes a clean Flamingo clip into the semantic latent and decodes it onto the
Scorpion topology. Runs only in the `compat` image, and writes raw feature
arrays that PoseYdon can be compared against.
"""

from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path

import numpy as np
import torch

REFERENCE = Path("external/neural_motion_blending").resolve()
sys.path.insert(0, str(REFERENCE))
if "spacy" not in sys.modules:
    stub = types.ModuleType("spacy")
    stub.load = stub.blank = lambda *a, **k: None  # type: ignore[attr-defined]
    sys.modules["spacy"] = stub

from data_loaders.tensors import lengths_to_mask, n_joints_to_mask
from data_loaders.truebones.data.dataset import create_temporal_mask_for_window
from model.motion_diffusion_ae import ControlConfig
from utils.model_util import create_model_and_diffusion_general_skeleton

REF_DIR = Path("data/truebones/reference")
ASSETS = REFERENCE / "assets/truebones"
STEM = {"Flamingo": "Flamingo_Flamingo_OneLEgBEnt_353", "Scorpion": "Scorpion___SlowForward_839"}


def load_skeleton(name: str):
    with np.load(REF_DIR / f"{name}.npz", allow_pickle=True) as d:
        return {k: d[k] for k in d.files}


def name_embeddings(names, cache) -> np.ndarray:
    return np.stack([cache[str(n)] for n in names])


def pad_to(array: np.ndarray, joints: int, axis: int = 0, value: float = 0.0) -> np.ndarray:
    pad = [(0, 0)] * array.ndim
    pad[axis] = (0, joints - array.shape[axis])
    return np.pad(array, pad, constant_values=value)


def build_y(skeleton, cache, n_joints_pad, n_frames, window, device):
    n_joints = len(skeleton["parents"])
    names = name_embeddings(skeleton["joints_names"], cache)

    tpos = pad_to(
        (skeleton["tpos_first_frame"] - skeleton["mean"]) / (skeleton["std"] + 1e-6),
        n_joints_pad,
    )
    tpos = np.nan_to_num(tpos)

    lengths = torch.tensor([n_frames])
    joints = torch.tensor([n_joints])
    template = create_temporal_mask_for_window(window, n_frames)[None]
    length_mask = (
        torch.arange(n_frames + 1).expand(1, n_frames + 1) < (lengths[:, None] + 1)
    ).float()
    temporal = (length_mask[:, :, None] * length_mask[:, None, :]).logical_and(template.bool())

    return {
        "mask": temporal[:, None, None].to(device),
        "lengths": lengths.to(device),
        "lengths_mask": lengths_to_mask(lengths, n_frames)[:, None, None].to(device),
        "tpos_first_frame": torch.tensor(tpos, dtype=torch.float32)[None].to(device),
        "crop_start_ind": torch.zeros(1, dtype=torch.long).to(device),
        "joints_mask": n_joints_to_mask(joints, n_joints_pad)[:, None, None].to(device),
        "n_joints": joints.to(device),
        "joints_names_embs": torch.tensor(pad_to(names, n_joints_pad), dtype=torch.float32)[None].to(device),
        "joints_relations": torch.tensor(
            pad_to(pad_to(skeleton["joint_relations"], n_joints_pad, 0), n_joints_pad, 1)
        )[None].to(device),
        "graph_dist": torch.tensor(
            pad_to(pad_to(skeleton["joints_graph_dist"], n_joints_pad, 0), n_joints_pad, 1)
        )[None].to(device),
    }


def build_model_and_diffusion(save_name: str):
    """Instantiate the reference model and load its checkpoint."""
    save_dir = REFERENCE / "save" / save_name
    with open(save_dir / "args.json") as fh:
        raw = json.load(fh)

    # These checkpoints predate some flags, so their args.json lacks keys the
    # current builder reads. Fill from the parser defaults rather than guessing.
    defaults = {
        "condition_strategy": "semantic_modulation", "semantic_feat_mode": "full",
        "second_temporal_attn": False, "kl_bottleneck": False, "lambda_kl": 0.0,
        "lambda_fs": 0.0, "lambda_geo": 0.0, "sigma_small": True, "dataset": "truebones",
        "num_virtual_joints": 0, "compress_virtual_joints": True,
        "projection_head_depth": 0, "skip_t5": False, "value_emb": False,
    }
    for key, value in defaults.items():
        raw.setdefault(key, value)

    model_args = argparse.Namespace(**raw)
    model_args.sampler = "ddpm"
    model_args.ddim_steps = -1

    model, diffusion = create_model_and_diffusion_general_skeleton(model_args)
    checkpoint = next(save_dir.glob("model*.pt"))
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state.get("model", state), strict=True)
    # MoDiffAE overrides train() without returning self, so eval() returns None
    # and cannot be chained. Call them separately.
    model.eval()
    return model, diffusion, raw


def load_inputs(source_name: str, target_name: str, frames: int, window: int = 31):
    """Build the reference-format conditioning for a source/target pair."""
    cache = dict(np.load(REF_DIR / "t5_cache.npz", allow_pickle=False))
    source = load_skeleton(source_name)
    target = load_skeleton(target_name)
    n_joints_pad = max(len(source["parents"]), len(target["parents"]))

    raw_source = np.load(ASSETS / f"{STEM[source_name]}.npy")[:frames]
    norm_source = np.nan_to_num(
        (raw_source - source["mean"][None]) / (source["std"][None] + 1e-6)
    )
    control_x = torch.tensor(
        pad_to(norm_source, n_joints_pad, axis=1), dtype=torch.float32
    ).permute(1, 2, 0)[None, None]

    return {
        "source": source,
        "target": target,
        "n_joints_pad": n_joints_pad,
        "control_x": control_x,
        "y_source": build_y(source, cache, n_joints_pad, frames, window, "cpu"),
        "y_target": build_y(target, cache, n_joints_pad, frames, window, "cpu"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", default="truebones_globpool")
    parser.add_argument("--source", default="Flamingo")
    parser.add_argument("--target", default="Scorpion")
    parser.add_argument("--frames", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="artifacts/transfer")
    args_cli = parser.parse_args()

    torch.manual_seed(args_cli.seed)
    np.random.seed(args_cli.seed)
    device = "cpu"

    model, diffusion, raw = build_model_and_diffusion(args_cli.save)
    model.to(device)

    frames = args_cli.frames
    inputs = load_inputs(args_cli.source, args_cli.target, frames, raw["temporal_window"])
    target = inputs["target"]
    n_joints_pad = inputs["n_joints_pad"]
    y_source, y_target = inputs["y_source"], inputs["y_target"]

    control = ControlConfig(x=inputs["control_x"].to(device), y=y_source, alpha=torch.zeros(1))
    shape = (1, n_joints_pad, 13, frames)

    with torch.no_grad():
        out = diffusion.p_sample_loop(
            model, shape,
            clip_denoised=False,
            model_kwargs={"y": y_target, "control": control},
            progress=True,
        )

    features = out[0].permute(2, 0, 1).cpu().numpy()[:, : len(target["parents"])]
    denormalized = features * (target["std"][None] + 1e-6) + target["mean"][None]

    out_dir = Path(args_cli.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"reference_{args_cli.source}_to_{args_cli.target}"
    np.save(out_dir / f"{stem}.npy", denormalized)
    np.save(out_dir / f"{stem}_normalized.npy", features[:, : len(target["parents"])])
    print(f"wrote {out_dir / stem}.npy  shape={denormalized.shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

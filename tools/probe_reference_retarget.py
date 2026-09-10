"""THROWAWAY probe: retarget Flamingo -> Scorpion using the REFERENCE weights.

Not a test and not part of the pipeline. It answers one question: does
PoseYdon's retarget path produce real motion when driven by a model that is
actually trained, rather than the untrained one the CLI smoke run used?

Two things make this faithful rather than merely runnable:

* **The reference's own statistics.** `model000599998.pt` was trained against
  the normalization in the reference's `cond.npy`, and ours differs from it
  substantially (median relative std difference 0.43-0.82 on these two rigs,
  rot6d means off by up to 1.86). Feeding those weights our normalization
  would test nothing. So the dataset's per-rig `Normalizer` cache is
  overridden with the reference's mean/std, reordered into our joint order.
* **Everything else is the shipping code path.** `run_retarget` is called
  unmodified, so what this exercises is what training will watch.

Run:
    docker compose run --rm train python tools/probe_reference_retarget.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from poseydon.build.index import CorpusIndex
from poseydon.data.dataset import MotionDataset
from poseydon.data.normalize import Normalizer
from poseydon.io.render import render_skeleton
from poseydon.models.modiffae import MoDiffAE
from poseydon.ops.retarget_run import run_retarget
from poseydon.process.gaussian import GaussianDiffusion
from poseydon.sampling.diffusion import DDPM

CHECKPOINT = Path("external/neural_motion_blending/save/truebones_attnpool/model000599998.pt")
REFERENCE = Path("data/truebones/reference")
FEATURES = ["ric_pos", "rot6d", "local_vel", "foot_contact"]
OUT = Path("runs/reference-retarget")

#: The one joint whose name PoseYdon and the reference disagree on. Everything
#: else matches by name; only Flamingo's root was renamed.
ALIASES = {"Hips": "Bip01_Pelvis"}


def reference_normalizer(rig: str, dataset: MotionDataset) -> Normalizer:
    """The reference's mean/std for `rig`, in OUR reduced joint order.

    Row order matters and is not incidental: `Normalizer.mean` is indexed by
    joint, so using the reference's rows as-is would pair each joint's values
    with a different joint's statistics -- silently, and with plausible-looking
    output.
    """
    ref = np.load(REFERENCE / f"{rig}.npz", allow_pickle=True)
    ref_names = [str(n) for n in ref["joints_names"]]

    with np.load(REFERENCE.parent / "rigs" / rig / "skeleton.npz", allow_pickle=True) as sk:
        full = [str(n) for n in sk["names"]]
        ours = [full[int(i)] for i in sk["reduction_source_of"]]

    rows = []
    for name in ours:
        wanted = ALIASES.get(name, name)
        if wanted not in ref_names:
            raise KeyError(f"{rig}: joint {name!r} has no counterpart in the reference")
        rows.append(ref_names.index(wanted))

    spec = dataset.spec
    return Normalizer(mean=ref["mean"][rows], std=ref["std"][rows], spec=spec)


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    OUT.mkdir(parents=True, exist_ok=True)

    dataset = MotionDataset(
        index=CorpusIndex.load("data/truebones/index.jsonl"),
        root="data/truebones",
        manifest_dir="data/truebones/rigs",
        features=FEATURES,
        conditioners=["topology", "tpose", "norm_stats"],
        split="train",
        seed=0,
    )

    # Override BEFORE anything reads them; the dataset caches per rig.
    for rig in ("Flamingo", "Scorpion"):
        dataset._normalizers[rig] = reference_normalizer(rig, dataset)
        dataset._rest_frames.pop(rig, None)
    print("using the reference's normalization for Flamingo and Scorpion")

    model = MoDiffAE(
        feature_dim=dataset.spec.dim,
        d_model=128,
        n_layers_semantic=4,
        n_layers_stochastic=4,
        n_heads=4,
        ff_size=1024,
        n_virtual_joints=5,
        temporal_window=31,
    )
    state = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    state = state.get("state_dict", state)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"loaded {len(state)} tensors; missing={len(missing)} unexpected={len(unexpected)}")
    if missing or unexpected:
        print("  missing   :", list(missing)[:6])
        print("  unexpected:", list(unexpected)[:6])
        raise SystemExit("architecture does not match the checkpoint")
    model = model.eval().to(device)

    result = run_retarget(
        model=model,
        # The reference trained with 100 cosine steps (its args.json), not the
        # 1000 a diffusion model usually assumes.
        process=GaussianDiffusion(num_steps=100, schedule="cosine", parameterization="x0"),
        sampler=DDPM(),
        dataset=dataset,
        content="Flamingo/onelegbent",
        target_rig="Scorpion",
        out_dir=OUT,
        device=device,
        generator=torch.Generator(device=device).manual_seed(0),
    )

    # The shipped mp4 renders the SOLVED skeleton. The pre-IK render is the
    # interesting one here: it shows what the model predicted before the
    # solver had a chance to make the bone lengths right.
    pre_ik = OUT / "Flamingo_onelegbent__to__Scorpion.PRE-IK.mp4"
    render_skeleton(
        pre_ik,
        result.target_anim.parents,
        result.predicted_positions,
        round(result.target_anim.fps),
        "PRE-IK: raw predicted positions, Flamingo -> Scorpion (reference weights)",
    )

    print(f"\npre-IK  render : {pre_ik}")
    print(f"post-IK bvh    : {result.bvh}")
    print(f"post-IK render : {result.mp4}")
    print(f"npz            : {result.npz}")

    predicted, solved = result.predicted_positions, result.solved_positions
    print(f"\nframes {predicted.shape[0]}, joints {predicted.shape[1]}, method {result.reconstruct}")
    residual = np.linalg.norm(predicted - solved, axis=-1)
    print(f"IK residual    : mean {residual.mean():.4f}  max {residual.max():.4f} (bone lengths)")

    offsets = result.target_anim.offsets
    parents = result.target_anim.parents
    real = [j for j in range(len(parents)) if parents[j] >= 0]
    rest = np.linalg.norm(offsets[real], axis=-1)
    live = np.linalg.norm(predicted[:, real] - predicted[:, [parents[j] for j in real]], axis=-1)
    drift = np.abs(live - rest[None])
    print(f"bone drift PRE : mean {drift.mean():.4f}  max {drift.max():.4f} (bone lengths)")
    print(f"motion extent  : {np.ptp(solved.reshape(-1, 3), axis=0)}")


if __name__ == "__main__":
    main()

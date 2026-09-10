"""THROWAWAY: push a parity retarget run to wandb, for inspection before training.

Not a test and not part of the pipeline. It runs the REAL `RetargetValidation`
callback -- the same one the 600k run fires every 25 000 steps -- against the
REFERENCE's trained weights, on the same three pairs, and uploads the result to
a wandb run named `parity-reference-retarget`.

Two things this shows that an untrained smoke run cannot:

* **What good output looks like.** The callback's own step-0 firing retargets
  with random weights; this retargets with a model that finished 600k steps, so
  the videos and the six scalars are a reference point to judge training
  against rather than noise.
* **That the whole chain works on real data.** Dataset, joint-name
  conditioning, sampler, IK, BVH writing, artifact upload -- end to end,
  through the shipping code path, not a stub.

Faithfulness comes from two overrides, both POSITIONAL:

* The reference's `cond.npy` statistics, because these weights were trained
  against them and ours differ substantially.
* The reference's own T5 joint-name embeddings, because it trained with
  `skip_t5: False` and `cond_mask_prob: 0.0` -- the embeddings are on every
  forward pass and never masked.

Both map by joint INDEX. Every one of the six rigs has an identical joint
ORDER to the reference's; the only differences are renames in place (`Hips` vs
`Bip01_Pelvis` on four rigs, `...Nub` vs `..._end_site` on Crab's ten end
sites). Mapping by name would have to special-case each; mapping by index is
exact, and `_check_order` refuses a rig whose order ever stops matching.

Run:
    docker compose run --rm --entrypoint python train tools/probe_wandb_parity.py
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from poseydon.build.index import CorpusIndex
from poseydon.conditioners.skeleton import JointNames
from poseydon.data.dataset import MotionDataset
from poseydon.data.normalize import Normalizer
from poseydon.io.render import render_skeleton
from poseydon.models.modiffae import MoDiffAE
from poseydon.process.gaussian import GaussianDiffusion
from poseydon.sampling.diffusion import DDPM
from poseydon.training.validation import RetargetValidation

CHECKPOINT = Path("external/neural_motion_blending/save/truebones_attnpool/model000599998.pt")
REFERENCE = Path("data/truebones/reference")
RIGS = Path("data/truebones/rigs")
FEATURES = ["ric_pos", "rot6d", "local_vel", "foot_contact"]
OUT = Path("runs/wandb-parity")

PAIRS = [
    {"content": {"rig": "Flamingo", "action": "onelegbent"}, "target": "Scorpion"},
    {"content": {"rig": "Coyote", "action": "attack3"}, "target": "Crab"},
    {"content": {"rig": "Goat", "action": "headbutt"}, "target": "Raptor"},
]
INVOLVED = ["Flamingo", "Scorpion", "Coyote", "Crab", "Goat", "Raptor"]

#: Below this share of names agreeing in place, the two joint orders have
#: genuinely diverged and a positional map would pair the wrong rows.
ORDER_AGREEMENT = 0.8


def our_names(rig: str) -> list[str]:
    with np.load(RIGS / rig / "skeleton.npz", allow_pickle=True) as data:
        full = [str(n) for n in data["names"]]
        return [full[int(i)] for i in data["reduction_source_of"]]


def reference_names(rig: str) -> list[str]:
    with np.load(REFERENCE / f"{rig}.npz", allow_pickle=True) as data:
        return [str(n) for n in data["joints_names"]]


def _check_order(rig: str) -> list[str]:
    """The reference's names for `rig`, once its order is confirmed to be ours.

    A positional map is only sound while the two orders agree. Measured today:
    Scorpion 63/63 identical, Flamingo/Coyote/Goat/Raptor all but the root
    rename, Crab 44/54 with the other ten being end-site renames in place.
    """
    ours, theirs = our_names(rig), reference_names(rig)
    if len(ours) != len(theirs):
        raise SystemExit(f"{rig}: {len(ours)} joints here against {len(theirs)} there")
    agree = sum(a == b for a, b in zip(ours, theirs))
    if agree < ORDER_AGREEMENT * len(ours):
        raise SystemExit(
            f"{rig}: only {agree}/{len(ours)} names agree in place, so the joint "
            f"orders have diverged and a positional map would pair the wrong rows"
        )
    return theirs


class ReferenceJointNames(JointNames):
    """The reference's joint-name embeddings, mapped by joint index."""

    def __init__(self, rigs: list[str]) -> None:
        with np.load(REFERENCE / "t5_cache.npz", allow_pickle=True) as data:
            table = {name: data[name].astype("float32") for name in data.files}
        self._dim = 768
        self._by_rig = {
            rig: np.stack([table[name] for name in _check_order(rig)]) for rig in rigs
        }

    def extract(self, item) -> torch.Tensor:
        rows = self._by_rig[item.rig]
        if rows.shape[0] != len(item.anim.names):
            raise SystemExit(
                f"{item.rig}: {len(item.anim.names)} joints at read time against "
                f"{rows.shape[0]} embeddings -- augmentation must be off for parity"
            )
        return torch.from_numpy(rows.copy())


def reference_normalizer(rig: str, spec) -> Normalizer:
    """The reference's mean/std for `rig`, in our joint order (which is theirs)."""
    _check_order(rig)
    with np.load(REFERENCE / f"{rig}.npz", allow_pickle=True) as data:
        return Normalizer(mean=data["mean"], std=data["std"], spec=spec)


def main() -> int:
    import wandb

    device = "cuda" if torch.cuda.is_available() else "cpu"
    OUT.mkdir(parents=True, exist_ok=True)

    dataset = MotionDataset(
        index=CorpusIndex.load("data/truebones/index.jsonl"),
        root="data/truebones",
        manifest_dir="data/truebones/rigs",
        features=FEATURES,
        conditioners=["topology", "tpose", "norm_stats"],
        augmentations=(),  # parity: the reference does not augment at sample time
        split="train",
        seed=0,
    )
    dataset.conditioners.append(ReferenceJointNames(INVOLVED))

    spec = dataset.spec
    for rig in INVOLVED:
        dataset._normalizers[rig] = reference_normalizer(rig, spec)
        dataset._rest_frames.pop(rig, None)
    print(f"reference normalization and joint names for: {', '.join(INVOLVED)}")

    model = MoDiffAE(
        feature_dim=spec.dim,
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
    if missing or unexpected:
        raise SystemExit(f"architecture mismatch: {list(missing)[:4]} {list(unexpected)[:4]}")
    print(f"loaded {len(state)} tensors; missing=0 unexpected=0")
    model = model.eval().to(device)

    run = wandb.init(
        project="poseydon",
        entity=os.environ.get("WANDB_ENTITY") or None,
        name="parity-reference-retarget",
        job_type="parity",
        dir="/tmp/wandb",
        notes=(
            "Reference checkpoint model000599998.pt driven through PoseYdon's own "
            "RetargetValidation callback. NOT a training run -- this is what good "
            "output looks like, for judging the 600k run against."
        ),
        config={
            "checkpoint": CHECKPOINT.name,
            "normalization": "reference cond.npy",
            "joint_names": "reference t5_cache.npz",
            "ff_size": 1024,
            "diffusion_steps": 100,
        },
    )

    callback = RetargetValidation(
        pairs=PAIRS,
        dataset=dataset,
        process=GaussianDiffusion(num_steps=100, schedule="cosine", parameterization="x0"),
        sampler=DDPM(),
        out_dir=OUT,
        every_n_steps=1,
        run_at_start=False,
        seed=0,
    )
    trainer = SimpleNamespace(
        global_step=0, logger=SimpleNamespace(experiment=run, log_metrics=_printer(run))
    )
    scalars = callback.run(trainer, SimpleNamespace(task=SimpleNamespace(model=model)))
    if not scalars:
        raise SystemExit("no pair produced metrics -- see the warnings above")

    _pre_ik_videos(run)
    print(f"\nrun: {run.url}")
    run.finish()
    return 0


def _printer(run):
    """Log to wandb and to the console, so a headless run still reports."""

    def log_metrics(metrics, step=None):
        run.log(dict(metrics), step=step)
        for key in sorted(metrics):
            print(f"  {key:62s} {metrics[key]:.4f}")

    return log_metrics


def _pre_ik_videos(run) -> None:
    """The raw predicted positions, beside the solved output.

    The shipped mp4 renders the SOLVED skeleton, so a viewer cannot tell a model
    that predicted good positions from a solver that rescued bad ones. This
    renders what the model actually claimed, per pair.
    """
    import wandb

    for npz in sorted(OUT.rglob("*.npz")):
        with np.load(npz, allow_pickle=True) as data:
            if "predicted_positions" not in data:
                continue
            positions = data["predicted_positions"]
            parents = data["parents"]
            fps = int(data["fps"]) if "fps" in data else 30
        out = npz.with_suffix(".PRE-IK.mp4")
        render_skeleton(out, parents, positions, fps, f"PRE-IK: {npz.stem}")
        run.log({f"parity/{npz.stem}/pre_ik": wandb.Video(str(out))})
        print(f"pre-IK render: {out}")


if __name__ == "__main__":
    raise SystemExit(main())

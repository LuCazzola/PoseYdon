"""poseydon command line."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from poseydon.ingest.pipeline import ingest_corpus

CONFIG_DIR = "configs"


def _ingest(args: argparse.Namespace) -> int:
    paths = sorted(Path(args.bvh_dir).rglob("*.bvh"))
    if not paths:
        print(f"no .bvh files found under {args.bvh_dir}", file=sys.stderr)
        return 1

    skeleton_of = (lambda _p: args.skeleton) if args.skeleton else None
    result = ingest_corpus(
        paths, args.manifests, args.out, split=args.split, skeleton_of=skeleton_of
    )
    index_path = Path(args.out) / "index.jsonl"
    result.index.save(index_path)

    print(f"ingested {len(result.index)} clips -> {index_path}")
    for path, reason in result.skipped:
        print(f"  skipped {path.name}: {reason}", file=sys.stderr)
    return 0 if result.index.records else 1


def _compose(config_dir: str, config_name: str, overrides: list[str]):
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(config_dir=str(Path(config_dir).resolve()), version_base=None):
        return compose(config_name=config_name, overrides=overrides)


def _train(args: argparse.Namespace) -> int:
    import lightning as L
    from omegaconf import OmegaConf

    from poseydon.training.build import (
        build_callbacks,
        build_datamodule,
        build_logger,
        build_module,
        run_name,
    )
    from poseydon.training.recipe import write_recipe

    config = _compose(args.config_dir, args.config_name, args.overrides)
    if args.print_config:
        print(OmegaConf.to_yaml(config))
        return 0

    # `--out` is a directory OF runs; this run gets its own inside it, holding
    # its recipe, its checkpoints and its logs together. Named from the config
    # rather than from the clock so that re-running the same configuration --
    # or resuming it -- lands back in the same place.
    run_dir = Path(args.out) / run_name(config)

    L.seed_everything(config.seed, workers=True)
    data = build_datamodule(config)
    module = build_module(config, feature_dim=data.train_dataset.spec.dim)

    # What this run was trained with, beside the checkpoints it writes. The
    # module carries the same mapping into every checkpoint's hyperparameters,
    # so a checkpoint moved away from its run directory stays self-describing.
    print(f"wrote {write_recipe(config, run_dir)}")

    logger = build_logger(config, run_dir)
    trainer = L.Trainer(
        max_steps=config.trainer.max_steps,
        accelerator=config.trainer.accelerator,
        devices=config.trainer.devices,
        precision=config.trainer.precision,
        log_every_n_steps=config.trainer.log_every_n_steps,
        gradient_clip_val=config.trainer.gradient_clip_val,
        default_root_dir=str(run_dir),
        # Built once and shared: `build_callbacks` needs to know whether a
        # logger exists, because LearningRateMonitor cannot run without one.
        callbacks=build_callbacks(run_dir, logger),
        # None means no `WANDB_API_KEY`; `False` is how Lightning is told to
        # log nowhere at all, rather than to fall back to its own default.
        logger=logger or False,
        enable_progress_bar=not args.quiet,
        # A corpus with no validation split should not have one invented for it.
        limit_val_batches=1.0 if data.has_validation else 0,
        num_sanity_val_steps=2 if data.has_validation else 0,
    )
    trainer.fit(module, datamodule=data, ckpt_path=args.resume)
    return 0


def _sample(args: argparse.Namespace) -> int:
    import numpy as np
    import torch
    from hydra.utils import instantiate

    from poseydon.core.batch import Cond, Masks
    from poseydon.data.collate import collate
    from poseydon.features import reconstruct
    from poseydon.io.bvh import BVH
    from poseydon.training.build import build_dataset, build_model, build_process
    from poseydon.training.recipe import check_recipe, recipe_from_checkpoint

    config = _compose(args.config_dir, args.config_name, args.overrides)
    torch.manual_seed(config.seed)

    # Before anything is built: this config recomposes from `configs/sample.yaml`
    # and knows nothing about the checkpoint, so a `features:` that disagrees
    # with the one it was trained under would otherwise sample silent garbage.
    state = None
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        recorded = recipe_from_checkpoint(state, args.checkpoint)
        if recorded is None:
            print(
                f"{args.checkpoint} carries no recipe and none sits beside it: "
                "cannot verify it against this config",
                file=sys.stderr,
            )
        else:
            try:
                check_recipe(config, recorded)
            except ValueError as mismatch:
                print(mismatch, file=sys.stderr)
                return 1
    else:
        print("no --checkpoint given: sampling from an untrained model", file=sys.stderr)

    dataset = build_dataset(config)
    skeleton = config.skeleton or dataset.records[0].skeleton
    matching = [i for i, r in enumerate(dataset.records) if r.skeleton == skeleton]
    if not matching:
        print(f"no clips for skeleton `{skeleton}`", file=sys.stderr)
        return 1

    # One real item supplies the skeleton's conditioning: topology, rest pose and
    # normalization statistics all describe the rig, not the clip.
    template = collate([dataset[matching[0]]])
    process = build_process(config)
    model = build_model(config, feature_dim=template.spec.dim).eval()

    if state is not None:
        model.load_state_dict(
            {k.removeprefix("task.model."): v
             for k, v in state.get("state_dict", state).items()
             if k.startswith("task.model.")}
        )

    n, frames = config.n_samples, config.n_frames
    joints = template.n_joints
    cond = Cond({k: _tile(v, n) for k, v in template.cond.payloads.items()})
    masks = Masks(
        frames=torch.ones(n, frames, dtype=torch.bool),
        joints=template.masks.joints[:1].repeat(n, 1),
    )

    sampler = instantiate(config.sampler)
    operation = instantiate(config.operation)
    with torch.no_grad():
        out = operation.run(
            model=model,
            process=process,
            sampler=sampler,
            shape=(n, joints, template.spec.dim, frames),
            cond=cond,
            masks=masks,
        )

    normalizer = dataset._normalizer(skeleton, template.spec)
    reference = dataset._anim(dataset.records[matching[0]])

    settings = dict(config.reconstruct)
    method = str(settings.pop("name"))
    if method != "positions_ik":
        # These knobs only mean anything to the solver.
        settings = {}

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        features = normalizer.denormalize(out[i].permute(2, 0, 1).cpu().numpy().astype("float64"))
        positions, anim = reconstruct(method, features, template.spec, reference, **settings)

        stem = out_dir / f"{skeleton}__sample_{i:03d}"
        np.save(f"{stem}.positions.npy", positions)
        if anim is None:
            print(f"wrote {stem}.positions.npy  (`{method}` yields no rotations, so no BVH)")
        else:
            BVH.from_animation(anim).write(f"{stem}.bvh")
            print(f"wrote {stem}.bvh  (reconstruct={method})")

        if args.render:
            from poseydon.io.render import render_skeleton

            render_skeleton(
                f"{stem}.mp4", reference.parents, positions,
                fps=round(reference.fps), title=f"{skeleton}  ({method})", zoom=args.zoom,
            )
            print(f"wrote {stem}.mp4")
    return 0


def _tile(value, times: int):
    """Repeat one skeleton's conditioning across a batch of samples."""
    if isinstance(value, dict):
        return {k: _tile(v, times) for k, v in value.items()}
    return value[:1].repeat(times, *([1] * (value.ndim - 1)))


def _list(args: argparse.Namespace) -> int:
    from poseydon.augment import AUGMENTATIONS
    from poseydon.conditioners import CONDITIONERS
    from poseydon.data.window import WINDOWS
    from poseydon.features import FEATURES
    from poseydon.losses import LOSSES
    from poseydon.models import MODELS
    from poseydon.ops import OPERATIONS
    from poseydon.process import PROCESSES
    from poseydon.sampling import CONTROLS, SAMPLERS

    registries = {
        "features": FEATURES,
        "conditioners": CONDITIONERS,
        "augmentations": AUGMENTATIONS,
        "losses": LOSSES,
        "models": MODELS,
        "processes": PROCESSES,
        "samplers": SAMPLERS,
        "controls": CONTROLS,
        "operations": OPERATIONS,
        "windows": WINDOWS,
    }
    for kind, registry in registries.items():
        if args.kind in (None, kind):
            print(f"{kind}:")
            for name in registry.names():
                print(f"  {name}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="poseydon")
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest", help="align a BVH corpus and build its index")
    ingest.add_argument("bvh_dir", help="directory of source .bvh files")
    ingest.add_argument("--manifests", required=True, help="directory of skeleton manifests")
    ingest.add_argument("--out", required=True, help="output directory")
    ingest.add_argument("--split", default="train")
    ingest.add_argument(
        "--skeleton",
        default=None,
        help="force every file to this skeleton, for a flat directory of one character",
    )
    ingest.set_defaults(func=_ingest)

    train = sub.add_parser("train", help="train a model from a composed config")
    train.add_argument("overrides", nargs="*", help="Hydra overrides, e.g. model=anytop")
    train.add_argument("--config-dir", default=CONFIG_DIR)
    train.add_argument("--config-name", default="train")
    train.add_argument("--out", default="runs", help="directory of runs; this run gets one inside")
    train.add_argument(
        "--resume", default=None, help="checkpoint to continue from, e.g. <run>/checkpoints/last.ckpt"
    )
    train.add_argument("--print-config", action="store_true", help="show the config and exit")
    train.add_argument("--quiet", action="store_true")
    train.set_defaults(func=_train)

    sample = sub.add_parser("sample", help="generate motion and write BVH")
    sample.add_argument("overrides", nargs="*")
    sample.add_argument("--config-dir", default=CONFIG_DIR)
    sample.add_argument("--config-name", default="sample")
    sample.add_argument("--checkpoint", default=None)
    sample.add_argument("--out", default="samples")
    sample.add_argument("--render", action="store_true", help="also write an MP4 per sample")
    sample.add_argument(
        "--zoom", type=float, default=1.0, help="render framing; >1 moves the camera closer"
    )
    sample.set_defaults(func=_sample)

    listing = sub.add_parser("list", help="show the components available by name")
    listing.add_argument("kind", nargs="?", default=None)
    listing.set_defaults(func=_list)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())

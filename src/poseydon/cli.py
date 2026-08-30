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

    from poseydon.training.build import build_datamodule, build_module

    config = _compose(args.config_dir, args.config_name, args.overrides)
    if args.print_config:
        print(OmegaConf.to_yaml(config))
        return 0

    L.seed_everything(config.seed, workers=True)
    data = build_datamodule(config)
    module = build_module(config, feature_dim=data.train_dataset.spec.dim)

    trainer = L.Trainer(
        max_steps=config.trainer.max_steps,
        accelerator=config.trainer.accelerator,
        devices=config.trainer.devices,
        precision=config.trainer.precision,
        log_every_n_steps=config.trainer.log_every_n_steps,
        gradient_clip_val=config.trainer.gradient_clip_val,
        default_root_dir=args.out,
        enable_progress_bar=not args.quiet,
        # A corpus with no validation split should not have one invented for it.
        limit_val_batches=1.0 if data.has_validation else 0,
        num_sanity_val_steps=2 if data.has_validation else 0,
    )
    trainer.fit(module, datamodule=data)
    return 0


def _list(args: argparse.Namespace) -> int:
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
    train.add_argument("--out", default="runs")
    train.add_argument("--print-config", action="store_true", help="show the config and exit")
    train.add_argument("--quiet", action="store_true")
    train.set_defaults(func=_train)

    listing = sub.add_parser("list", help="show the components available by name")
    listing.add_argument("kind", nargs="?", default=None)
    listing.set_defaults(func=_list)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())

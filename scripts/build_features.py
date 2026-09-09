"""Prepared BVH -> canonical features, per-rig statistics and the clip index.

Stage 2 (parity spec §4): runs the four build passes in
``poseydon.build.pipeline`` -- clips, skeleton, stats, index -- over every rig
with a manifest under the composed dataset config's ``root``. Reads
``configs/dataset/<--config-name>.yaml`` (default ``dataset/truebones``),
instantiates it into a ``BuildConfig`` and calls ``build_all``.

Run inside the test container:
    docker compose run --rm test python scripts/build_features.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from hydra.utils import instantiate

from poseydon.build.pipeline import BuildConfig, build_all
from poseydon.data.normalize import BlockPolicy

CONFIG_DIR = "configs"
DEFAULT_CONFIG_NAME = "dataset/truebones"


def _compose(config_dir: str, config_name: str):
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(config_dir=str(Path(config_dir).resolve()), version_base=None):
        return compose(config_name=config_name)


def build_config(
    config_dir: str,
    config_name: str,
    out_root: Path | None,
    relabel: bool,
) -> BuildConfig:
    """Compose ``configs/<config_name>.yaml`` into a ``BuildConfig``.

    The dataset group lives under ``configs/dataset/``, so Hydra packages its
    content under a top-level ``dataset:`` key -- that key is unwrapped here.

    ``_convert_="all"`` matters: without it, `hydra.utils.instantiate`
    recursively creates each `_target_`-bearing node, but a *frozen dataclass*
    result (`BlockPolicy`, `PerRigDirectory`, `Humanize`) gets folded straight
    back into a `DictConfig` by OmegaConf's structured-config boxing when it is
    assigned into the surrounding `DictConfig` tree -- so `normalize` would
    silently come back as a list of dict-like `DictConfig` nodes, not
    `BlockPolicy` instances, with no error at all. `_convert_="all"` makes
    `instantiate` return plain Python objects the whole way down.
    """
    raw = _compose(config_dir, config_name)
    dataset = raw.dataset if "dataset" in raw else raw
    built = instantiate(dataset, _convert_="all")

    normalize = tuple(built["normalize"])
    for entry in normalize:
        if not isinstance(entry, BlockPolicy):
            raise TypeError(
                f"{config_name}'s `normalize` entries must instantiate into "
                f"`BlockPolicy`, got {type(entry).__name__} ({entry!r}) -- "
                "does every entry carry a `_target_: poseydon.data.normalize.BlockPolicy`?"
            )

    return BuildConfig(
        root=Path(built["root"]),
        schema=tuple(built["schema"]),
        corpus=built["corpus"],
        names=built["names"],
        normalize=normalize,
        out=out_root,
        reduce_tolerance=float(built["reduce_tolerance"]),
        relabel=relabel,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", default=CONFIG_DIR)
    parser.add_argument("--config-name", default=DEFAULT_CONFIG_NAME)
    parser.add_argument(
        "--rigs",
        nargs="*",
        default=None,
        help="Rig names to build (default: every rig under the dataset's `root`)",
    )
    parser.add_argument(
        "--stats-only", action="store_true", help="only run the stats pass (no clips/skeleton/index)"
    )
    parser.add_argument(
        "--relabel", action="store_true", help="refresh derived label keys, keeping authored ones"
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=None,
        help="where artefacts are written (default: the dataset's `root`)",
    )
    args = parser.parse_args()

    config = build_config(args.config_dir, args.config_name, args.out_root, args.relabel)

    rig_names = args.rigs if args.rigs is not None else config.corpus.rigs(config.root)
    print(f"building {len(rig_names)} rig(s) -> {config.out}")
    for rig in rig_names:
        print(f"  {rig}")

    result = build_all(config, rigs=rig_names, stats_only=args.stats_only)

    print(f"\ntotal: {result.clips} clips written across {result.rigs} rigs")
    if result.warnings:
        print(f"\n{len(result.warnings)} warning(s):")
        for warning in result.warnings:
            print(f"  - {warning}")

    failed = False
    if result.failures:
        failed = True
        print(f"\n{len(result.failures)} rig(s) FAILED to build:", file=sys.stderr)
        for failure in result.failures:
            print(f"  - {failure}", file=sys.stderr)

    if result.clips == 0 and not args.stats_only:
        failed = True
        print("\nno clips written", file=sys.stderr)

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()

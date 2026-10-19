"""Prepared BVH -> canonical features, per-rig statistics and the clip index (stage 2).

Runs the four build passes in ``poseydon.build.pipeline`` -- clips, skeleton,
stats, index -- over every rig of the dataset `configs/build.yaml` composes
(its ``data`` group, plus the model-side ``features`` group the stats are
measured for):

    docker compose run --rm test python scripts/build_features.py [--rigs Lion ...] [data=truebones_bone]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from poseydon.build.pipeline import BuildConfig, build_all, compose_build

CONFIG_DIR = "configs"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", default=CONFIG_DIR)
    parser.add_argument("--config-name", default="build")
    parser.add_argument(
        "--rigs",
        nargs="*",
        default=None,
        help="Rig names to build (default: every rig under the dataset's `paths.root`)",
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
        help="where artefacts are written (default: the dataset's `paths.root`)",
    )
    parser.add_argument("overrides", nargs="*", help="Hydra overrides, e.g. data=truebones_bone")
    args = parser.parse_args()

    composed = compose_build(args.overrides, args.config_dir, args.config_name)
    config = BuildConfig.from_config(composed, out=args.out_root, relabel=args.relabel)

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
        # Not "rig(s)": `build_all` also files the once-per-corpus index pass
        # here, and it is not a rig.
        print(f"\n{len(result.failures)} FAILED:", file=sys.stderr)
        for failure in result.failures:
            print(f"  - {failure}", file=sys.stderr)

    if result.clips == 0 and not args.stats_only:
        failed = True
        print("\nno clips written", file=sys.stderr)

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()

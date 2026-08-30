"""poseydon command line."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from poseydon.ingest.pipeline import ingest_corpus


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

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())

"""Pilot: raw Truebones BVH -> training-ready corpus, for the seven species
that already have hand-authored manifests.

Cleans each raw clip (poseydon.datasets.raw_bvh.load_raw_biped_bvh), writes
it back out as an ordinary BVH, and hands the result to the unmodified
ingest pipeline. See docs/superpowers/specs/2026-09-05-truebones-preprocessing-design.md.

Run inside the test container:
    docker compose run --rm test python scripts/create_truebones_dataset.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from poseydon.datasets.raw_bvh import load_raw_biped_bvh
from poseydon.ingest.pipeline import ingest_corpus
from poseydon.io.bvh import save_bvh

PILOT_SKELETONS = (
    "BrownBear",
    "Coyote",
    "Crab",
    "Flamingo",
    "Goat",
    "Scorpion",
    "Skunk",
)

RAW_ROOT = Path("data/truebones/Truebone_Z-OO")
MANIFEST_DIR = Path("data/truebones/skeletons")
OUT_DIR = Path("data/truebones")


def clean_species_clips(species: str, raw_root: Path, scratch_root: Path) -> list[Path]:
    """Clean every raw ``.bvh`` clip for one species into ``scratch_root/<species>/``.

    Output filenames match the raw filenames exactly: ``ingest_clip`` derives
    a clip's action name from the filename stem, so renaming here would
    rename every resulting clip id.
    """
    species_dir = raw_root / species
    out_dir = scratch_root / species
    out_dir.mkdir(parents=True, exist_ok=True)

    written = []
    for raw_path in sorted(species_dir.glob("*.bvh")):
        anim = load_raw_biped_bvh(raw_path)
        dest = out_dir / raw_path.name
        save_bvh(anim, dest)
        written.append(dest)
    return written


def main() -> None:
    with tempfile.TemporaryDirectory() as scratch:
        scratch_root = Path(scratch)
        all_clean: list[Path] = []
        for species in PILOT_SKELETONS:
            all_clean.extend(clean_species_clips(species, RAW_ROOT, scratch_root))

        result = ingest_corpus(all_clean, MANIFEST_DIR, OUT_DIR, split="train")

    result.index.save(OUT_DIR / "index.jsonl")
    print(f"ingested {len(result.index)} clips -> {OUT_DIR / 'index.jsonl'}")
    for path, reason in result.skipped:
        print(f"  skipped {path.name}: {reason}")


if __name__ == "__main__":
    main()

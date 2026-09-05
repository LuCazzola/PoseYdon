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

from poseydon.core.anim import Anim
from poseydon.datasets.raw_bvh import establish_rest_pose, load_raw_biped_bvh, remove_bind_pose
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


def resolve_rest_anim(species: str, raw_root: Path) -> Anim:
    """The cleaned, single-frame rest pose to remove every clip's bind rotation against.

    Prefers a raw T-pose file (matched case-insensitively, e.g.
    ``__TPOSE.bvh``) when the species has one. Otherwise falls back to the
    first frame of the alphabetically-first raw clip -- the same fallback
    ``poseydon.data.dataset.MotionDataset._rest_frame`` already uses
    elsewhere in this codebase when a manifest names no T-pose.

    Uses ``establish_rest_pose`` (IK-based offset/rotation recovery) rather
    than ``load_raw_biped_bvh``, since a rest reference's OFFSETS matter --
    unlike an ordinary clip, which reuses whatever offsets this returns and
    never re-derives its own.
    """
    species_dir = raw_root / species
    tpose_candidates = sorted(species_dir.glob("*[Tt][Pp][Oo][Ss][Ee]*.bvh"))
    if tpose_candidates:
        source = tpose_candidates[0]
    else:
        source = sorted(species_dir.glob("*.bvh"))[0]

    return establish_rest_pose(source)


def clean_species_clips(species: str, raw_root: Path, scratch_root: Path) -> list[Path]:
    """Clean every raw ``.bvh`` clip for one species into ``scratch_root/<species>/``.

    Output filenames match the raw filenames exactly: ``ingest_clip`` derives
    a clip's action name from the filename stem, so renaming here would
    rename every resulting clip id.
    """
    species_dir = raw_root / species
    out_dir = scratch_root / species
    out_dir.mkdir(parents=True, exist_ok=True)

    rest_anim = resolve_rest_anim(species, raw_root)

    written = []
    for raw_path in sorted(species_dir.glob("*.bvh")):
        anim = load_raw_biped_bvh(raw_path)
        anim = remove_bind_pose(anim, rest_anim)
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

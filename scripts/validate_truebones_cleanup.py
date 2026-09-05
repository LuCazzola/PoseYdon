"""Diagnostic: compare features from (raw -> our cleanup) against features
from (curated fixture) for the same clip, per pilot species.

Not a pass/fail gate -- see docs/superpowers/specs/2026-09-05-truebones-preprocessing-design.md
§4. Freezing non-root translation (poseydon.datasets.raw_bvh) discards real
per-joint motion the fixture path never had to discard, so some divergence
is expected. This prints a per-block error table for a human to judge.

Run inside the test container:
    docker compose run --rm test python scripts/validate_truebones_cleanup.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from poseydon.core.skeleton import SkeletonManifest, resolve
from poseydon.datasets.raw_bvh import load_raw_biped_bvh
from poseydon.features import extract_features
from poseydon.ingest.align import align, compute_alignment_params
from poseydon.io.bvh import load_bvh

RAW_ROOT = Path("data/truebones/Truebone_Z-OO")
FIXTURE_ROOT = Path("external/neural_motion_blending/assets/truebones")
MANIFEST_DIR = Path("data/truebones/skeletons")

# (species, raw clip relative to RAW_ROOT, matching fixture stem) -- the same
# clip under both a raw and a curated name, one pair per pilot species.
PILOT_PAIRS = [
    ("BrownBear", "BrownBear/__RiseSwat.bvh", "BrownBear___RiseSwat_132"),
    ("Coyote", "Coyote/__Attack3.bvh", "Coyote___Attack3_224"),
    ("Crab", "Crab/__Attack3.bvh", "Crab___Attack3_234"),
    ("Flamingo", "Flamingo/Flamingo_OneLEgBEnt.bvh", "Flamingo_Flamingo_OneLEgBEnt_353"),
    ("Goat", "Goat/__HeadButt.bvh", "Goat___HeadButt_395"),
    ("Scorpion", "Scorpion/__SlowForward.bvh", "Scorpion___SlowForward_839"),
    ("Skunk", "Skunk/__Spray.bvh", "Skunk___Spray_891"),
]


def compare_clip(species: str, raw_relpath: str, fixture_stem: str) -> dict[str, dict[str, float]]:
    """Per-block max/mean absolute error between the two paths' features.

    Both anims are put through the same alignment step ``ingest_clip`` would
    apply (rotate to face +Z, scale to the manifest's target bone length,
    ground) before extracting features -- raw files are in a different unit
    scale and orientation from the curated fixtures (confirmed by direct
    inspection: fixture OFFSETs are roughly 1/65th the raw ones), and that
    difference is exactly what alignment is for, not something this report
    should be measuring. Compared over the intersection of joint names --
    the fixture omits End Sites that our own cleanup keeps (spec §3
    addendum) -- and truncated to the shorter of the two frame counts.
    """
    manifest = SkeletonManifest.load(MANIFEST_DIR / f"{species}.yaml")
    feature_names = ("ric_pos", "rot6d", "local_vel", "foot_contact")

    raw_anim = load_raw_biped_bvh(RAW_ROOT / raw_relpath)
    raw_resolved = resolve(manifest, raw_anim.names)
    raw_aligned = align(raw_anim, raw_resolved, compute_alignment_params(raw_anim, raw_resolved))
    raw_features, _ = extract_features(raw_aligned, raw_resolved, feature_names)

    fixture_anim = load_bvh(FIXTURE_ROOT / f"{fixture_stem}.bvh")
    fixture_resolved = resolve(manifest, fixture_anim.names)
    fixture_aligned = align(
        fixture_anim, fixture_resolved, compute_alignment_params(fixture_anim, fixture_resolved)
    )
    fixture_features, fixture_spec = extract_features(fixture_aligned, fixture_resolved, feature_names)

    shared_names = [name for name in fixture_anim.names if name in raw_anim.names]
    raw_joint_index = {name: i for i, name in enumerate(raw_anim.names)}
    fixture_joint_index = {name: i for i, name in enumerate(fixture_anim.names)}
    raw_idx = [raw_joint_index[name] for name in shared_names]
    fixture_idx = [fixture_joint_index[name] for name in shared_names]
    n_frames = min(raw_features.shape[0], fixture_features.shape[0])

    # Both extractions were asked for the same `feature_names`, so the two
    # specs name and order blocks identically -- one loop, one spec, used to
    # slice both feature arrays.
    report: dict[str, dict[str, float]] = {}
    for block_name in fixture_spec.names:
        block_slice = fixture_spec.slice(block_name)
        raw_block = raw_features[:n_frames, raw_idx, block_slice]
        fixture_block = fixture_features[:n_frames, fixture_idx, block_slice]
        error = np.abs(raw_block - fixture_block)
        report[block_name] = {
            "max_abs_error": float(error.max()),
            "mean_abs_error": float(error.mean()),
        }
    return report


def main() -> None:
    for species, raw_relpath, fixture_stem in PILOT_PAIRS:
        print(f"{species}:")
        report = compare_clip(species, raw_relpath, fixture_stem)
        for block_name, errors in report.items():
            print(
                f"  {block_name:<14} max={errors['max_abs_error']:.4f}  "
                f"mean={errors['mean_abs_error']:.4f}"
            )


if __name__ == "__main__":
    main()

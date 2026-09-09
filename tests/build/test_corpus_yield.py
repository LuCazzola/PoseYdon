"""The recovered rigs stay recovered, and the promoted rigs stay promoted.

b2b0151 recovered 60 clips across four rigs by choosing the rest-pose file by
modal joint set, and nothing guarded it. A regression in rest selection would
quietly cost them again -- quietly being the point: the clips are skipped with
a warning, not an error.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from scripts.process_dataset_truebones import PROMOTE_ROOT

from poseydon.data.normalize import Normalizer
from poseydon.io.bvh import BVH
from tests.conftest import CORPUS

RECOVERED = {"Ant": 17, "Crab": 10, "Deer": 20, "Jaguar": 13}

#: Rigs whose `mesh.npz` (FBX-sourced, un-promoted) disagrees in joint count
#: with the promoted BVH-sourced rest skeleton. `build_skeleton`'s guard
#: (`_check_mesh_matches_skeleton`, pipeline.py) reports this as a WARNING
#: through `build_all`'s per-rig warning list -- it does NOT withhold
#: `skeleton.npz`/`stats.npz`, since neither reads `mesh.npz` (mesh folding
#: does not exist yet; task 9 fix round 1 corrected an earlier version of the
#: guard that raised and silently cost both rigs their training artefacts).
#: Goat is the rig the plan's design spec calls out explicitly; Camel hits
#: the identical mechanism (it is also in `PROMOTE_ROOT` and also has a
#: stage-1 `mesh.npz`) and was found by running the actual build, not by
#: inspection -- see task-9-report.md. This is the design's documented
#: trade-off (2026-09-08 Stage 2, "the FBX path does not promote"), not a
#: regression, so it is recorded here rather than "fixed".
KNOWN_MESH_MISMATCHES = {"Goat", "Camel"}


@pytest.mark.parametrize("rig, expected", sorted(RECOVERED.items()))
def test_the_recovered_rigs_keep_their_clips(rig, expected):
    clips = CORPUS / "clips" / rig
    if not clips.is_dir():
        pytest.skip(f"{rig}: corpus not built -- run scripts/process_dataset_truebones.py")
    assert len(sorted(clips.glob("*.bvh"))) == expected


def test_corpus_totals_match_the_rebuild():
    """Pins the headline numbers Task 9's rebuild produced.

    Measured by counting the corpus on disk after the rebuild: 1145 prepared
    `.bvh` clips across 73 rig directories under `data/truebones/clips/`.
    Nothing else in this file pins the totals -- only the four recovered
    rigs' individual counts -- so a regression that cost clips or dropped a
    rig anywhere else in the corpus would go unnoticed.
    """
    clips_root = CORPUS / "clips"
    if not clips_root.is_dir():
        pytest.skip("corpus not built -- run scripts/process_dataset_truebones.py")

    rig_dirs = sorted(p for p in clips_root.iterdir() if p.is_dir())
    total_clips = sum(len(list(rig_dir.glob("*.bvh"))) for rig_dir in rig_dirs)

    assert len(rig_dirs) == 73, f"expected 73 rig directories, found {len(rig_dirs)}"
    assert total_clips == 1145, f"expected 1145 prepared .bvh clips, found {total_clips}"


@pytest.mark.parametrize("rig", sorted(PROMOTE_ROOT))
def test_promoted_rigs_are_rooted_on_the_body(rig):
    """Measured on the PREPARED corpus, so it checks what stage 1 wrote rather
    than what the table says."""
    clips = CORPUS / "clips" / rig
    if not clips.is_dir():
        pytest.skip(f"{rig}: corpus not built")
    written = sorted(clips.glob("*.bvh"))
    if not written:
        pytest.skip(f"{rig}: no prepared clips")

    anim = BVH.read(written[0]).to_animation()
    assert anim.names[0] == PROMOTE_ROOT[rig]

    y = anim.global_positions()[0][:, 1]
    fraction = (float(y[0]) - float(y.min())) / (float(y.max()) - float(y.min()))
    assert fraction > 0.05, (
        f"{rig}: prepared root sits at height fraction {fraction:.3f}, still a "
        "ground locator"
    )


def test_stage_2_artefact_counts_match_the_build():
    """Pins task 9's measured stage-2 yield: `python scripts/build_features.py`
    run once over the full, freshly-built corpus.

    Every number here was produced by counting the artefacts stage 2 wrote to
    disk after that run, not computed from a formula:

    - 1145 clip `.npz` (one per prepared `.bvh` -- stage 2's `build_clips`
      never drops or merges a clip).
    - 73 `skeleton.npz` and 73 `stats.npz`, ONE PER RIG: `build_skeleton`'s
      mesh/skeleton mismatch check (see `KNOWN_MESH_MISMATCHES` above) is a
      warning, not a refusal, so Goat and Camel get their artefacts like
      every other rig -- only their entry in the build's warning list marks
      them as different.
    - `index.jsonl` has exactly 1145 rows, equal to the clip count.
    """
    clips_root = CORPUS / "clips"
    rigs_root = CORPUS / "rigs"
    index_path = CORPUS / "index.jsonl"
    if not clips_root.is_dir():
        pytest.skip("corpus not built -- run scripts/build_features.py")

    bvh_count = sum(len(list(d.glob("*.bvh"))) for d in clips_root.iterdir() if d.is_dir())
    clip_npz_count = sum(len(list(d.glob("*.npz"))) for d in clips_root.iterdir() if d.is_dir())
    skeleton_count = len(list(rigs_root.glob("*/skeleton.npz")))
    stats_count = len(list(rigs_root.glob("*/stats.npz")))

    if clip_npz_count == 0:
        pytest.skip("stage 2 not built -- run scripts/build_features.py")

    with index_path.open() as handle:
        index_rows = sum(1 for _ in handle)

    assert bvh_count == 1145, f"expected 1145 prepared .bvh, found {bvh_count}"
    assert clip_npz_count == bvh_count, (
        f"expected one clip .npz per prepared .bvh: {clip_npz_count} .npz vs "
        f"{bvh_count} .bvh"
    )
    assert skeleton_count == 73, (
        f"expected 73 skeleton.npz (one per rig -- a mesh/skeleton mismatch "
        f"is a warning, not a refusal), found {skeleton_count}"
    )
    assert stats_count == skeleton_count, (
        f"stats.npz ({stats_count}) and skeleton.npz ({skeleton_count}) counts "
        "diverged -- a full (non --stats-only) build writes both together per "
        "rig, in the same per-rig try block, so they should match"
    )
    assert index_rows == clip_npz_count, (
        f"index.jsonl has {index_rows} rows but {clip_npz_count} clip .npz were "
        "written -- build_index globs the output clips directory per rig, so a "
        "rig whose CLIP pass failed would contribute zero rows and this "
        "equality would break"
    )


def test_stats_npz_records_the_configured_schema():
    """A `stats.npz` fit under a stale `features:` schema must not sit
    unnoticed beside a fresh one -- `Normalizer.save` records the block names
    it was fit against, and this checks every rig's file against what
    `configs/dataset/truebones.yaml` currently declares.
    """
    rigs_root = CORPUS / "rigs"
    if not rigs_root.is_dir():
        pytest.skip("corpus not built -- run scripts/build_features.py")

    stats_paths = sorted(rigs_root.glob("*/stats.npz"))
    if not stats_paths:
        pytest.skip("stage 2 not built -- run scripts/build_features.py")

    with initialize_config_dir(config_dir=str((Path.cwd() / "configs").resolve()), version_base=None):
        cfg = compose(config_name="dataset/truebones")
    configured_schema = tuple(cfg.dataset.schema)

    for path in stats_paths:
        normalizer = Normalizer.load(path)
        assert normalizer.spec.names == configured_schema, (
            f"{path}: stats.npz records blocks {normalizer.spec.names}, but "
            f"configs/dataset/truebones.yaml currently declares "
            f"{configured_schema} -- this rig's stats.npz is stale (fit under "
            "a different `features:` schema) and should be refreshed with "
            "`--stats-only`"
        )

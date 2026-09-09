"""The recovered rigs stay recovered, and the promoted rigs stay promoted.

b2b0151 recovered 60 clips across four rigs by choosing the rest-pose file by
modal joint set, and nothing guarded it. A regression in rest selection would
quietly cost them again -- quietly being the point: the clips are skipped with
a warning, not an error.
"""

from __future__ import annotations

import pytest
from scripts.process_dataset_truebones import PROMOTE_ROOT

from poseydon.io.bvh import BVH
from tests.conftest import CORPUS

RECOVERED = {"Ant": 17, "Crab": 10, "Deer": 20, "Jaguar": 13}


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

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

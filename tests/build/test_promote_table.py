"""§1.2's promotion table: every entry must be reachable and must fix the rig.

The gate is the ground-locator classification, never the presence of an offset.
Lynx and BrownBear have a correct root on the pelvis and a chain continuing to
Bip01_Spine at a real offset (0.79 and 0.89 bone lengths); a rule keyed on "the
chain carries an offset" would promote their root onto the spine and break two
healthy rigs. They are asserted absent from the table explicitly.
"""

from __future__ import annotations

import numpy as np
import pytest
from scripts.process_dataset_truebones import PROMOTE_ROOT, rest_source

from poseydon.core.skeleton import SkeletonManifest
from poseydon.io.bvh import BVH
from tests.conftest import CORPUS

LOCATOR_BAND = 0.05


def _rest_anim(rig: str):
    rig_dir = CORPUS / "source" / rig
    if not rig_dir.is_dir():
        pytest.skip(f"{rig}: no source directory")
    bvhs = sorted(rig_dir.glob("*.bvh"))
    manifest = SkeletonManifest.load(CORPUS / "rigs" / rig / "manifest.yaml")
    return BVH.read(rest_source(manifest, bvhs)).to_animation()


def _height_fraction(anim, joint: int) -> float:
    """Where a joint sits between the skeleton's lowest and highest point."""
    y = anim.global_positions()[0][:, 1]
    lo, hi = float(y.min()), float(y.max())
    return 0.0 if hi <= lo else (float(y[joint]) - lo) / (hi - lo)


@pytest.mark.parametrize("rig", sorted(PROMOTE_ROOT))
def test_the_promoted_joint_exists_and_leaves_the_locator_band(rig):
    anim = _rest_anim(rig)
    target = PROMOTE_ROOT[rig]
    assert target in anim.names, f"{rig}: no joint named {target}"

    assert _height_fraction(anim, 0) < LOCATOR_BAND, (
        f"{rig} is in the table but its root is not a ground locator"
    )
    promoted = _height_fraction(anim, anim.names.index(target))
    assert promoted > LOCATOR_BAND, (
        f"{rig}: promoting to {target} leaves the root at height fraction "
        f"{promoted:.3f}, still inside the locator band"
    )


@pytest.mark.parametrize("rig", ["Lynx", "BrownBear", "Tyranno", "PolarBear"])
def test_healthy_rigs_are_absent_from_the_table(rig):
    """These have a correct pelvis root; some also have a chain with a real
    offset, which is precisely why the offset is not the gate."""
    assert rig not in PROMOTE_ROOT
    anim = _rest_anim(rig)
    assert _height_fraction(anim, 0) > LOCATOR_BAND


def test_promotion_moves_no_surviving_joint_on_a_real_rig():
    """World positions, on real data -- not a joint count, which would restate
    the table."""
    from poseydon.build.prepare import PromoteRoot

    rig = "Camel"
    anim = _rest_anim(rig)
    stage = PromoteRoot(target=PROMOTE_ROOT[rig])
    params = stage.fit(anim, None)
    promoted = stage.apply(anim, params)

    before = anim.global_positions()
    after = promoted.global_positions()
    for index, name in enumerate(promoted.names):
        np.testing.assert_allclose(
            after[:, index], before[:, anim.names.index(name)], atol=1e-7,
            err_msg=f"{rig}/{name} moved",
        )

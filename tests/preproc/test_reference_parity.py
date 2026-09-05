"""Cross-checks preproc.raw_bvh/rest_pose against the real reference
implementation, including a real accuracy benchmark of GradientIK against
the reference's own CPU IK solver on identical raw T-pose data.

Compat-image only:
    docker compose run --rm compat pytest tests/preproc/test_reference_parity.py -v
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from poseydon.preproc.raw_bvh import _parse_and_merge, load_raw_biped_bvh
from poseydon.preproc.rest_pose import establish_rest_pose, recover_raw_global_pose
from tests.reference_compat import reference_quats_to_poseydon, require_reference

require_reference()

import Animation  # noqa: E402
import BVH  # noqa: E402
import InverseKinematics  # noqa: E402
from Quaternions import Quaternions  # noqa: E402

RAW_ROOT = Path(__file__).resolve().parents[2] / "data" / "truebones" / "Truebone_Z-OO"


def _skip_if_raw_dump_missing():
    if not RAW_ROOT.is_dir():
        pytest.skip(f"raw Truebones dump not found at {RAW_ROOT}")


def test_merge_redundant_root_matches_the_real_loader():
    _skip_if_raw_dump_missing()
    raw_path = RAW_ROOT / "BrownBear" / "__RiseSwat.bvh"

    anim = load_raw_biped_bvh(raw_path)
    ref_anim, ref_names, _ = BVH.load(str(raw_path))

    # The real loader drops zero-offset End Sites entirely (its own
    # documented special case); poseydon.io.bvh.load_bvh deliberately keeps
    # every End Site as an ordinary joint (its own documented convention --
    # see the design spec §3). So the reference's names are a SUBSET of
    # ours, not an exact match -- both sides applied the SAME redundant-root
    # merge, which is what this test actually checks.
    assert set(ref_names) <= set(anim.names)
    assert ref_names[0] == anim.names[0] == "Bip01_Pelvis"

    np.testing.assert_allclose(anim.offsets[0], ref_anim.offsets[0], atol=1e-6)
    # Root's own translation is never frozen -- only NON-root joints are.
    np.testing.assert_allclose(anim.root_pos, ref_anim.positions[:, 0], atol=1e-4)

    from poseydon.core.rotations import quat_to_matrix

    ref_root_matrix = quat_to_matrix(reference_quats_to_poseydon(ref_anim.rotations.qs[:, 0]))
    our_root_matrix = quat_to_matrix(anim.rotations[:, 0])
    np.testing.assert_allclose(our_root_matrix, ref_root_matrix, atol=1e-4)


def test_gradient_ik_matches_the_real_solver_on_a_raw_tpose():
    _skip_if_raw_dump_missing()
    path = RAW_ROOT / "BrownBear" / "__Tpose.bvh"

    names, parents, offsets, rotations, positions, fps = _parse_and_merge(path)
    global_pos, _ = recover_raw_global_pose(rotations[:1], positions[:1], parents)
    scale = np.linalg.norm(offsets[1:], axis=1).mean()

    # This is an underdetermined fit (raw declared bone lengths don't match
    # the true position data -- see establish_rest_pose's docstring), so it
    # has no single unique solution: two different solvers landing on
    # different final positions is not evidence of a bug. The meaningful
    # comparison is each solver's OWN residual against the real target --
    # GradientIK should fit at least as well as the reference's own solver,
    # not reproduce its exact (possibly under-converged) answer.

    # The reference's own real solver, given the EXACT same target and the
    # EXACT same fixed offsets GradientIK uses.
    ref_start = Animation.Animation(
        rotations=Quaternions.id((1, len(names))),
        positions=np.repeat(offsets[np.newaxis], 1, axis=0),
        orients=Quaternions.id(0),
        offsets=offsets,
        parents=parents,
    )
    # The reference's own real animation_from_positions pins the root's
    # position to the target before running IK ("keep root positions --
    # important for IK"). Root translation is not something IK needs to
    # search for; skipping this pin makes the reference's job harder than
    # its real usage ever asks of it.
    ref_start.positions[:, 0] = global_pos[:, 0]
    ik = InverseKinematics.BasicInverseKinematics(ref_start, global_pos, iterations=150, silent=True)
    ref_fitted = ik()
    ref_positions = Animation.positions_global(ref_fitted)
    ref_residual = float(np.median(np.linalg.norm(ref_positions - global_pos, axis=-1) / scale))

    rest = establish_rest_pose(path, iterations=1000)
    our_positions = rest.global_positions()[:1]
    our_residual = float(np.median(np.linalg.norm(our_positions - global_pos, axis=-1) / scale))

    assert our_residual <= ref_residual * 1.1, (
        f"GradientIK's own fit residual against the real target ({our_residual:.4f}, "
        f"median relative to mean bone length) is worse than the reference's own "
        f"solver's residual ({ref_residual:.4f}) at its real iteration count (150). "
        "First increase establish_rest_pose's `iterations` (try 3000) before "
        "loosening this threshold -- if GradientIK is still worse after that, it is "
        "a real finding about its fit quality for this use case, not noise to "
        "average away."
    )

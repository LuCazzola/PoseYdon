"""Do OUR extracted features reproduce the REFERENCE's, on the reference's own data?

The reference (``external/neural_motion_blending/``, read-only) ships prepared
Truebones clips at ``assets/truebones/*.bvh`` alongside the exact feature
tensor its own pipeline computed from each one, ``assets/truebones/*.npy``
(``data_loaders/truebones/truebones_utils/motion_process.py::get_motion``).
Seven rigs carry both: BrownBear, Crab, Flamingo, Goat, Scorpion, Coyote, Skunk.

**Orientation, recorded so the comparison is legible.**

- Each ``.npy`` is ``(frames-1, joints, 13)`` float64: one row of
  ``concat([ric_pos(3), rot6d(6), local_vel(3), foot_contact(1)])`` per joint,
  per frame -- ``get_motion_features`` builds exactly this concatenation, in
  exactly this order, and it matches our own block order
  (``poseydon.features.DEFAULT_FEATURES``) so no block reordering is needed.
- The paired ``.bvh`` is not raw Truebones -- it is the reference's OWN
  processed output. ``get_motion`` builds ``new_anim`` (rotated to face +Z,
  scaled to ``HML_AVG_BONELEN``, grounded, and reduced to a rotation-only
  representation via ``compute_rots_from_tpos``) and feeds that SAME object to
  both ``get_motion_features`` (the ``.npy``) and ``BVH.save`` (the ``.bvh``,
  ``motion_process.py:402-403``). So the ``.bvh`` on disk is self-consistent
  with its ``.npy`` by construction: loading it and running FK reproduces the
  positions the reference features were computed from.
- Its joint order is confirmed (via the ``compat`` service's own ``BVH.load``,
  for BrownBear) to equal a plain top-to-bottom parse of the file -- the same
  order ``poseydon.io.bvh.BVH.read`` produces. The reference never reorders by
  species; ``FACE_JOINTS`` indexes into this same order to find the facing
  joints, it does not permute anything. So the reference's per-row joint
  identity is recoverable as ``BVH.read(path).to_animation().names``.
  Because ``get_motion`` computes rot6d as ``joint_6d[:, parents[j]]`` (holding
  each joint's PARENT's rotation) with no separate reduction step of its own,
  the reference's implicit "reduced" joint set is just whatever the input BVH
  already contains -- these particular files carry no zero-offset joints (see
  the per-rig `dropped` sets logged during orientation), so our own
  `build_reduction` is a no-op on all seven and the two joint sets are equal by
  name, not merely by construction. This is the load-bearing claim behind the
  `rot6d` agreement below, so it is not just narrated here -- the test asserts
  `reduction.is_identity` per rig, right after building it, and fails loudly
  (naming the rig and the dropped joints) rather than silently changing what
  the comparison measures if a future change to `build_reduction` starts
  pruning these files.

Because the reference's own preparation already produced a rotation-only,
already-reduced animation, running OUR ``build_stats``-shaped pipeline
(``as_rigid_body(joint_translation="drop")`` -> ``build_reduction`` ->
``apply_reduction`` -> ``extract_features``) directly on that ``.bvh`` exercises
no translation-dropping and no joint removal -- there is nothing left for either
stage to do. So the `EnforceRigid`-under-reduction `rot6d` divergence the
design spec anticipated (mean ~0.3-0.5, max ~1.8-2.0, from a joint's `rot6d`
slot resolving to a DIFFERENT parent once reduction removes an intermediate
joint) does not apply to this particular comparison: measured below, `rot6d`
agrees with the reference to ~1e-6 on every one of the seven rigs, the same
order of agreement as `ric_pos` and `local_vel`. That is a genuine, positive
parity result, not a loosened tolerance -- recorded as a passing assertion,
not an `xfail`, because there is no failure to park. `foot_contact` is
compared bit-exact, per the design spec's requirement, and matches exactly on
every rig.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from poseydon.core.skeleton import ManifestError, SkeletonManifest, resolve
from poseydon.features import DEFAULT_FEATURES, extract_features
from poseydon.features.reduce import apply_reduction, build_reduction
from poseydon.io.bvh import BVH

REFERENCE_ASSETS = Path("external/neural_motion_blending/assets/truebones")
RIG_MANIFESTS = Path("data/truebones/rigs")

RIGS = ("BrownBear", "Crab", "Flamingo", "Goat", "Scorpion", "Coyote", "Skunk")

# Block layout both sides share: reference's `get_motion_features` concatenates
# [ric_pos(3), rot6d(6), local_vel(3), foot_contact(1)] in this exact order,
# matching `poseydon.features.DEFAULT_FEATURES` (see module docstring).
_BLOCK_SLICES = {"ric_pos": (0, 3), "rot6d": (3, 9), "local_vel": (9, 12), "foot_contact": (12, 13)}

# Measured across all seven rigs (see this file's docstring and the task-7
# report): worst observed max was 3.02e-6 (Crab, ric_pos), worst mean 5.85e-7
# (Goat, ric_pos). The gap between that and 1e-4 is deliberate slack for text
# BVH's 6-decimal-place rounding on other rigs/actions, not a fitted bound.
_FLOAT_ATOL = 1e-4


def _pair(rig: str) -> tuple[Path, Path]:
    bvhs = sorted(REFERENCE_ASSETS.glob(f"{rig}*.bvh"))
    npys = sorted(REFERENCE_ASSETS.glob(f"{rig}*.npy"))
    if not bvhs or not npys:
        pytest.skip(f"{rig}: no matching reference .bvh/.npy pair")
    # Every rig here currently ships exactly one `.bvh`/`.npy` pair, so `[0]`
    # is the only candidate, not a representative pick among several -- if a
    # rig ever gained a second clip this would arbitrarily choose one with no
    # claim that it stands in for the others.
    return bvhs[0], npys[0]


@pytest.mark.parametrize("rig", RIGS)
def test_features_reproduce_reference_arrays(rig):
    if not REFERENCE_ASSETS.is_dir():
        pytest.skip("external/ reference checkout not present")

    bvh_path, npy_path = _pair(rig)
    manifest_path = RIG_MANIFESTS / rig / "manifest.yaml"
    if not manifest_path.is_file():
        pytest.skip(f"{rig}: no manifest at {manifest_path}, cannot resolve facing/feet")

    reference = np.load(npy_path, allow_pickle=True)

    # Build the reduced animation the way `poseydon.build.pipeline.build_stats`
    # does: rigid-body, then joint reduction, both applied at feature-extraction
    # time rather than baked into a corpus.
    raw = BVH.read(str(bvh_path)).to_animation()
    rigid = raw.as_rigid_body(joint_translation="drop")
    reduction = build_reduction(rigid, tolerance=1e-8)
    reduced = apply_reduction(rigid, reduction)

    # Load-bearing for the docstring's explanation of the ~1e-6 `rot6d`
    # agreement below: that story only holds if the reference's shipped clip
    # gives `build_reduction` nothing to remove. If a future change to the
    # tolerance or the drop/collapse rule started pruning joints from these
    # particular files, `rot6d`'s parent-reindexing would start reading a
    # DIFFERENT parent than the reference's un-reduced array does, and this
    # comparison would quietly start measuring something else -- so assert
    # the no-op rather than just narrate it.
    assert reduction.is_identity, (
        f"{rig}: build_reduction removed {len(reduction.ops)} joint(s) "
        f"({[op.name for op in reduction.ops]}) from the reference's own "
        "clip -- this invalidates the docstring's explanation for why rot6d "
        "agrees so closely (it assumes reduction is a no-op here, so no "
        "joint's parent changes under reduction); the agreement on this rig "
        "needs a different explanation, not a wider tolerance"
    )

    manifest = SkeletonManifest.load(manifest_path)
    try:
        resolved = resolve(manifest, reduced.names)
    except ManifestError as error:
        pytest.skip(f"{rig}: manifest does not resolve against this clip's joints: {error}")

    features, spec = extract_features(reduced, resolved, DEFAULT_FEATURES)
    assert tuple(name for name, _ in spec.blocks) == DEFAULT_FEATURES

    # NAME correspondence, never index: the reference's joint order is the raw
    # BVH parse order (verified against the reference's own `BVH.load` for
    # BrownBear -- see this file's docstring), which our reduction may have
    # trimmed. An index-matched comparison would silently be wrong the moment
    # a rig's reduction removes a joint from the middle of the raw order.
    reference_names = raw.names
    ours_of = {name: i for i, name in enumerate(reduced.names)}
    reference_of = {name: i for i, name in enumerate(reference_names)}
    shared = [name for name in reduced.names if name in reference_of]
    if not shared:
        pytest.skip(f"{rig}: no joint-name correspondence with the reference clip")

    ours_index = [ours_of[name] for name in shared]
    reference_index = [reference_of[name] for name in shared]

    n_frames = min(features.shape[0], reference.shape[0])
    ours = features[:n_frames][:, ours_index, :]
    theirs = reference[:n_frames][:, reference_index, :]

    for block, (start, end) in _BLOCK_SLICES.items():
        a, b = ours[..., start:end], theirs[..., start:end]

        if block == "foot_contact":
            # A binary flag: a tolerance here would hide a real disagreement
            # about which frames are contacts. Bit-exact, per the design spec.
            mismatched = int((a != b).sum())
            assert mismatched == 0, (
                f"{rig}: foot_contact disagrees on {mismatched}/{a.size} cells "
                f"(bit-exact comparison, over {len(shared)}/{len(reduced.names)} "
                "shared joints)"
            )
            continue

        diff = np.abs(a - b)
        assert diff.max() < _FLOAT_ATOL, (
            f"{rig}: {block} disagrees with the reference, mean={diff.mean():.3e} "
            f"max={diff.max():.3e} (over {len(shared)}/{len(reduced.names)} shared "
            f"joints) -- see this file's docstring for why an EnforceRigid/"
            "reduction-driven rot6d divergence was anticipated but not needed "
            "as an xfail here, since the reference's own clip is already "
            "rigid and already at our reduced joint set"
        )

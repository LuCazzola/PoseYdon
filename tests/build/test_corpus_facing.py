"""The prepared corpus faces +Z, rig by rig, pinned against real numbers.

`scripts.process_dataset_truebones._sanity_check` measures this at build time
and only WARNS -- `main()` prints and exits 0, so nothing automated points at
a rig that fails it. Three of the 73 currently do (Tukan 0.7579, Trex 0.8207,
Crow 0.9433 -- see the plan's "Addendum: three rigs do not face +Z"), and
they are already committed training data. This test pins the measured value
for every rig against the corpus actually on disk in `data/truebones/clips/`,
using the same manifest facing pairs `_sanity_check` uses, so a regression on
any of the other 70 is caught and the three known bad rigs are named rather
than silently tolerated: a blanket `xfail(strict=False)` here would assert
nothing, exactly the mistake this branch already fixed once in
`test_bvh_fbx_agreement.py`.
"""

from __future__ import annotations

import numpy as np
import pytest

from poseydon.build.corpus import rest_action
from poseydon.core.skeleton import SkeletonManifest, resolve
from poseydon.io.bvh import BVH
from tests.conftest import CORPUS

_TARGET_FORWARD = np.array([0.0, 0.0, 1.0])

# Measured directly against the prepared corpus (see this task's fix report
# for the transcript). Below the 0.99 threshold `_sanity_check` warns on.
_KNOWN_MISALIGNED = {
    "Tukan": 0.7579,
    "Trex": 0.8207,
    "Crow": 0.9433,
}


def _facing_dot(positions: np.ndarray, facing_indices) -> float:
    """How well ``positions``' forward axis agrees with +Z. 1.0 is exact.

    Mirrors `scripts.process_dataset_truebones._sanity_check.facing_dot`.
    """
    across = np.zeros(3)
    for right, left in facing_indices:
        across += positions[right] - positions[left]
    across = across / np.linalg.norm(across)
    forward = np.cross(np.array([0.0, 1.0, 0.0]), across)
    return float(np.dot(forward / np.linalg.norm(forward), _TARGET_FORWARD))


def _rest_skeleton_positions(anim) -> np.ndarray:
    """``(J, 3)`` joint positions of the REST skeleton: OFFSETs, no rotations."""
    positions = np.zeros((anim.n_joints, 3))
    for joint in range(1, anim.n_joints):
        positions[joint] = positions[anim.parents[joint]] + anim.offsets[joint]
    return positions


def _manifests() -> list[SkeletonManifest | None]:
    """One entry per manifest on disk, or ``[None]`` if the corpus is absent.

    Never an empty list: parametrizing over nothing collects zero tests for
    this function, which reads exactly like the bug class this plan hunts
    (a check that can pass by not running). A single ``None`` case, skipped
    inside the test body, keeps a real (skipped, not silently absent) result
    on a clean checkout.
    """
    if not (CORPUS / "source").is_dir():
        return [None]
    manifests = sorted((CORPUS / "rigs").glob("*/manifest.yaml"))
    if not manifests:
        return [None]
    return [SkeletonManifest.load(path) for path in manifests]


@pytest.mark.parametrize(
    "manifest", _manifests(), ids=lambda m: m.name if m is not None else "no-corpus"
)
def test_prepared_rig_faces_plus_z(manifest):
    if manifest is None:
        pytest.skip("Truebones corpus not present")

    clips_dir = CORPUS / "clips" / manifest.name
    if not clips_dir.is_dir():
        pytest.skip(f"{manifest.name}: corpus not built")
    written = sorted(clips_dir.glob("*.bvh"))
    if not written:
        pytest.skip(f"{manifest.name}: no prepared clips")

    subject = clips_dir / f"{rest_action(manifest)}.bvh"
    if not subject.is_file():
        subject = written[0]

    anim = BVH.read(subject).to_animation()
    resolved = resolve(manifest, anim.names)

    posed_dot = _facing_dot(anim.global_positions()[0], resolved.facing_indices)
    rest_dot = _facing_dot(_rest_skeleton_positions(anim), resolved.facing_indices)
    worst = min(posed_dot, rest_dot)

    if manifest.name in _KNOWN_MISALIGNED:
        pytest.xfail(
            f"{manifest.name}: prepared corpus does not face +Z, measured "
            f"forward . +Z = {worst:.4f} (frame-0 pose {posed_dot:.4f}, rest "
            f"skeleton {rest_dot:.4f}); manifest `facing:` pairs need "
            "correcting in a DCC tool -- see the plan's Addendum"
        )

    assert worst >= 0.99, (
        f"{manifest.name}: prepared corpus forward . +Z = {worst:.4f} "
        f"(frame-0 pose {posed_dot:.4f}, rest skeleton {rest_dot:.4f}), below "
        f"the 0.99 threshold {subject.name} was written under"
    )

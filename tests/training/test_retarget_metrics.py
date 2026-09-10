"""Spec test 14, "metric sanity": the five retarget scalars.

Every assertion here compares a metric against a definition that exists OUTSIDE
`training/metrics.py` -- a geometric identity, or arithmetic done by hand on the
inputs -- rather than against what the implementation happens to compute. Phase
1's retrospective is blunt that its weakest tests were the ones that restated
the code, so:

* a rigid skeleton's forward kinematics reproduces its rest bone lengths
  EXACTLY, so an FK-generated clip has no bone-length drift and an IK solve
  fitted to it has nowhere to travel;
* a foot that does not move cannot slide, whatever flag is on it;
* velocity is invariant under translation, so a clip shifted by a constant has
  the same root trajectory as the clip it was shifted from -- and a clip that
  drifts by a constant PER FRAME differs by exactly that drift;
* a distance from a thing to itself is zero.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from poseydon.build.index import CorpusIndex
from poseydon.core.animation import RigidBodyAnimation
from poseydon.core.skeleton import (
    ContactParams,
    FacingPair,
    SkeletonManifest,
    resolve,
)
from poseydon.data.dataset import MotionDataset
from poseydon.data.window import RandomCrop
from poseydon.features import extract_features, positions_from_features
from poseydon.features.reconstruct import RECONSTRUCTORS
from poseydon.models.base import Denoiser, Prediction
from poseydon.ops.retarget_run import run_retarget
from poseydon.process.gaussian import GaussianDiffusion
from poseydon.sampling.diffusion import DDPM
from poseydon.training.metrics import (
    bone_length_drift,
    contact_flags,
    foot_skate,
    ik_residual,
    latent_round_trip,
    mean_bone_length,
    retarget_metrics,
    root_trajectory_error,
    target_skeleton,
)

FEATURES = ["ric_pos", "rot6d", "local_vel", "foot_contact"]

# A seven-joint biped: two legs (so there is a facing pair and two feet), a
# spine and a head. Small enough that a 150-iteration IK solve is milliseconds,
# real enough that `extract_features` and `positions_ik` run their true code.
NAMES = ("Hips", "Spine", "Head", "LegR", "FootR", "LegL", "FootL")
PARENTS = np.array([-1, 0, 1, 0, 3, 0, 5], dtype=np.int32)
OFFSETS = np.array(
    [
        [0.00, 0.00, 0.00],
        [0.00, 0.30, 0.00],
        [0.00, 0.25, 0.00],
        [0.15, -0.10, 0.00],
        [0.00, -0.35, 0.00],
        [-0.15, -0.10, 0.00],
        [0.00, -0.35, 0.00],
    ]
)
FEET = (4, 6)


def _manifest() -> SkeletonManifest:
    return SkeletonManifest(
        name="Stick",
        facing=(FacingPair(right="LegR", left="LegL"),),
        contact=ContactParams(max_height=0.05, max_speed=0.045),
        source=Path("tests/synthetic/Stick.yaml"),
        foot_joints=("FootR", "FootL"),
    )


def _resolved():
    return resolve(_manifest(), NAMES)


def _smooth_quats(n_frames: int, n_joints: int, amplitude: float, seed: int) -> np.ndarray:
    """``(F, J, 4)`` unit quaternions, ``(x, y, z, w)``, varying smoothly in time.

    Smooth on purpose: `positions_ik` carries a smoothness term, so a clip of
    independent per-frame noise would be one the solver is entitled to refuse
    to follow, and the test would be measuring the regularizer.
    """
    rng = np.random.default_rng(seed)
    axes = rng.normal(size=(n_joints, 3))
    axes /= np.linalg.norm(axes, axis=-1, keepdims=True)
    phase = rng.uniform(0, 2 * np.pi, size=n_joints)
    t = np.arange(n_frames)[:, None] / max(n_frames - 1, 1)
    angle = amplitude * np.sin(2 * np.pi * t + phase[None, :])  # (F, J)
    half = angle / 2
    quats = np.empty((n_frames, n_joints, 4))
    quats[..., :3] = np.sin(half)[..., None] * axes[None, :, :]
    quats[..., 3] = np.cos(half)
    return quats


def _fk_clip(n_frames: int = 24, amplitude: float = 0.5, seed: int = 0) -> RigidBodyAnimation:
    """A clip whose positions are, by construction, exactly forward kinematics
    of a rigid skeleton."""
    root = np.zeros((n_frames, 3))
    root[:, 1] = 0.7
    root[:, 0] = 0.01 * np.arange(n_frames)
    root[:, 2] = 0.02 * np.arange(n_frames)
    return RigidBodyAnimation.from_root_motion(
        rotations=_smooth_quats(n_frames, len(NAMES), amplitude, seed),
        root_pos=root,
        offsets=OFFSETS,
        parents=PARENTS,
        names=NAMES,
        fps=30.0,
    )


# --------------------------------------------------------------------------
# foot skate
# --------------------------------------------------------------------------


def test_a_static_clip_does_not_skate():
    """The spec's own sanity case. A foot that never moves cannot slide, so the
    metric must be zero even with every foot flagged planted on every frame --
    which is what a static clip's contact detector reports."""
    still = np.repeat(_fk_clip(n_frames=1).global_positions(), 20, axis=0)
    planted = np.zeros((20, len(NAMES)), dtype=bool)
    planted[:, FEET] = True
    assert foot_skate(still, planted, scale=mean_bone_length(OFFSETS)) == pytest.approx(0.0)


def test_a_planted_foot_scores_the_distance_it_actually_slid():
    """Arithmetic done outside the implementation: one foot, planted on every
    frame, moved a known 0.02 per frame in X. Twenty frames of that is a
    displacement of 0.02 per step, and nothing else moves."""
    positions = np.zeros((21, len(NAMES), 3))
    positions[:, 4, 0] = 0.02 * np.arange(21)
    planted = np.zeros((21, len(NAMES)), dtype=bool)
    planted[:, 4] = True
    assert foot_skate(positions, planted, scale=1.0) == pytest.approx(0.02)
    # "in bone-length units": the same slide against a half-length bone reads
    # twice as large.
    assert foot_skate(positions, planted, scale=0.01) == pytest.approx(2.0)


def test_skate_is_horizontal_only():
    """A foot lifted straight up is not sliding. Vertical motion under a
    contact flag is a different failure and is not what this metric names."""
    positions = np.zeros((21, len(NAMES), 3))
    positions[:, 4, 1] = 0.02 * np.arange(21)
    planted = np.zeros((21, len(NAMES)), dtype=bool)
    planted[:, 4] = True
    assert foot_skate(positions, planted, scale=1.0) == pytest.approx(0.0)


def test_an_unplanted_foot_may_move_freely():
    """The flag is what makes movement a fault; without it, a swinging foot is
    just a step."""
    positions = np.zeros((21, len(NAMES), 3))
    positions[:, 4, 0] = 0.02 * np.arange(21)
    assert foot_skate(positions, np.zeros((21, len(NAMES)), dtype=bool), scale=1.0) == 0.0


def _grounded_still(n_frames: int = 12) -> np.ndarray:
    """A motionless pose with both declared feet exactly on the ground."""
    still = np.repeat(_fk_clip(n_frames=1).global_positions(), n_frames, axis=0)
    still[:, list(FEET), 1] = 0.0
    return still


def test_contact_flags_are_the_manifests_rule_on_the_right_frames():
    """The manifest says planted means slower than `max_speed` and lower than
    `max_height`. A motionless foot on the ground is both; a foot crossing 0.5
    units a frame against a 0.045 threshold is neither.

    `FootContact` is a finite difference, so its frame `k` describes animation
    frame `k + 1`. `contact_flags` returns one flag per POSITION frame, and on
    frame 0 there is no previous frame to measure a speed against.
    """
    flags = contact_flags(_grounded_still(), _resolved())
    assert flags.shape == (12, len(NAMES))
    assert not flags[0].any(), "frame 0 has no measurable speed"
    assert flags[1:, list(FEET)].all(), "still, and on the ground, is planted"
    assert not flags[:, [0, 1, 2, 3, 5]].any(), "only declared feet may be planted"

    running = _grounded_still()
    running[:, 4, 0] = 0.5 * np.arange(12)  # >> contact.max_speed
    moving = contact_flags(running, _resolved())
    assert not moving[:, 4].any(), "a foot crossing the floor is not planted"
    assert moving[1:, 6].all(), "the other foot never moved"


# --------------------------------------------------------------------------
# bone-length drift
# --------------------------------------------------------------------------


def test_fk_positions_of_a_rigid_skeleton_have_no_bone_length_drift():
    """The spec's second sanity case, and a geometric identity rather than a
    restatement: a rotation is an isometry, so a joint's distance from its
    parent under forward kinematics IS the rest offset's length."""
    anim = _fk_clip()
    drift = bone_length_drift(anim.global_positions(), anim.parents, anim.offsets)
    assert drift < 1e-12


def test_a_stretched_bone_scores_the_stretch_it_added():
    """One joint pulled a known 0.1 away from its parent, on every frame, in a
    seven-joint rig: six bones carry the mean, one of which is wrong by 0.1.

    `FootR` is deliberately a LEAF, so moving it lengthens exactly one bone.
    """
    anim = _fk_clip()
    positions = anim.global_positions()
    direction = positions[:, 4] - positions[:, 3]
    direction /= np.linalg.norm(direction, axis=-1, keepdims=True)
    positions[:, 4] += 0.1 * direction

    expected = 0.1 / 6 / mean_bone_length(OFFSETS)
    drift = bone_length_drift(positions, anim.parents, anim.offsets)
    assert drift == pytest.approx(expected, rel=1e-6)


def test_drift_defaults_to_the_rigs_own_mean_bone_length():
    """"In bone-length units" is the rig's mean REAL bone, the same quantity
    `ScaleToMeanBoneLength` scales every rig to -- zero-length End Sites are
    not bones and must not enter the mean."""
    with_end_site = np.vstack([OFFSETS, np.zeros((1, 3))])
    assert mean_bone_length(with_end_site) == pytest.approx(mean_bone_length(OFFSETS))
    assert mean_bone_length(OFFSETS) == pytest.approx(
        float(np.linalg.norm(OFFSETS[1:], axis=-1).mean())
    )


# --------------------------------------------------------------------------
# IK residual
# --------------------------------------------------------------------------


def test_a_clip_that_is_already_fk_consistent_survives_the_solve():
    """The spec's third sanity case, run through the REAL reconstruction path.

    The targets handed to `positions_ik` come from a rigid skeleton, so a rigid
    skeleton can hit them exactly and the solver has nowhere to travel. What is
    left is the solver's own convergence plateau, which the module documents as
    ~1% of a bone length -- so the assertion is against that documented figure,
    not against whatever number comes out.
    """
    anim = _fk_clip()
    features, spec = extract_features(anim, _resolved(), FEATURES)
    predicted = positions_from_features(features, spec)
    solved = RECONSTRUCTORS.get("positions_ik")().anim(features, spec, anim).global_positions()

    mean, worst = ik_residual(predicted, solved, scale=mean_bone_length(OFFSETS))
    assert mean < 0.02, f"an exactly-consistent clip should not move; got {mean}"
    assert worst < 0.10, f"no single joint should be dragged far; got {worst}"


def test_a_clip_the_skeleton_cannot_hold_moves_much_further():
    """The contrast that makes the previous number mean something: targets that
    stretch a bone by half its length cannot be reached by a rigid skeleton, so
    the solver must leave them behind."""
    anim = _fk_clip()
    features, spec = extract_features(anim, _resolved(), FEATURES)
    predicted = positions_from_features(features, spec)
    solved = RECONSTRUCTORS.get("positions_ik")().anim(features, spec, anim).global_positions()
    consistent, _ = ik_residual(predicted, solved, scale=mean_bone_length(OFFSETS))

    stretched = predicted.copy()
    stretched[:, 4] += 0.5 * (predicted[:, 4] - predicted[:, 3])
    broken, _ = ik_residual(stretched, solved, scale=mean_bone_length(OFFSETS))
    assert broken > 10 * consistent


def test_the_residual_is_the_mean_and_the_max_of_the_per_joint_distance():
    """Hand arithmetic: one joint on one frame displaced a known 0.4, every
    other joint exactly on target. 24 frames x 7 joints = 168 samples."""
    predicted = np.zeros((24, len(NAMES), 3))
    solved = np.zeros((24, len(NAMES), 3))
    predicted[3, 2, 0] = 0.4

    mean, worst = ik_residual(predicted, solved, scale=1.0)
    assert worst == pytest.approx(0.4)
    assert mean == pytest.approx(0.4 / (24 * len(NAMES)))
    # Halving the bone length doubles both, since both are lengths.
    half = ik_residual(predicted, solved, scale=0.5)
    assert half == pytest.approx((2 * mean, 2 * worst))


# --------------------------------------------------------------------------
# root trajectory error
# --------------------------------------------------------------------------


def test_a_clip_translated_by_a_constant_has_the_same_trajectory():
    """Velocity is invariant under translation. This is also why the metric is
    defined on velocity at all: the feature representation is root-invariant,
    so a retarget always returns to the XZ origin (design spec section 9) and
    an absolute comparison would report that instead of the motion.
    """
    rng = np.random.default_rng(0)
    content = rng.normal(size=(30, 3))
    shifted = content + np.array([12.0, 0.0, -7.5])
    assert root_trajectory_error(shifted, content) == pytest.approx(0.0)


def test_a_clip_that_drifts_by_a_known_step_scores_that_step():
    """A per-frame drift of (0.03, 0, 0.04) is a velocity difference of exactly
    0.05 on every frame -- a 3-4-5 triangle, computed here, not there."""
    rng = np.random.default_rng(1)
    content = rng.normal(size=(30, 3))
    drift = np.arange(30)[:, None] * np.array([0.03, 0.0, 0.04])
    assert root_trajectory_error(content + drift, content) == pytest.approx(0.05)


def test_only_the_horizontal_plane_counts():
    """XZ velocity, per the spec: a clip that rises differs in height, not in
    where it went."""
    rng = np.random.default_rng(2)
    content = rng.normal(size=(30, 3))
    climb = np.arange(30)[:, None] * np.array([0.0, 0.9, 0.0])
    assert root_trajectory_error(content + climb, content) == pytest.approx(0.0)


def test_the_same_physical_walk_on_two_differently_scaled_rigs_agrees():
    """Why the scale factors are there at all, and which way round they go.

    `prepare.npz` records `factor` for the stage that did `prepared = source *
    factor`. One walk, authored once in source units, prepared onto two rigs
    whose factors differ by 4x, is ONE trajectory: the metric must read zero.
    """
    rng = np.random.default_rng(3)
    walk = rng.normal(size=(30, 3))
    generated_scale, content_scale = 0.046, 0.0115
    assert root_trajectory_error(
        walk * generated_scale,
        walk * content_scale,
        generated_scale=generated_scale,
        content_scale=content_scale,
    ) == pytest.approx(0.0, abs=1e-12)


def test_a_length_mismatch_is_refused():
    """`extract_features` costs the last frame, so the generated clip is one
    frame shorter than the content animation. Truncating silently would hide a
    real misalignment; the caller aligns explicitly."""
    with pytest.raises(ValueError, match="frames"):
        root_trajectory_error(np.zeros((29, 3)), np.zeros((30, 3)))


# --------------------------------------------------------------------------
# latent round trip
# --------------------------------------------------------------------------


def test_a_latent_against_itself_is_zero():
    """The spec's fourth sanity case: identity of indiscernibles."""
    z = torch.randn(31, 1, 1, 128)
    assert latent_round_trip(z, z) == pytest.approx(0.0)
    assert latent_round_trip(z.clone(), z) == pytest.approx(0.0)


def test_the_distance_is_the_euclidean_one_per_frame():
    """A known offset: every one of 128 channels moved by 0.5 is a per-frame
    distance of sqrt(128) * 0.5, computed here rather than read off the code."""
    z = torch.zeros(31, 1, 1, 128)
    assert latent_round_trip(z + 0.5, z) == pytest.approx(float(np.sqrt(128) * 0.5), rel=1e-5)


def test_the_leading_rest_token_is_what_lines_the_two_up():
    """`MoDiffAE.encode` prepends the rig's rest frame, so a latent's frame
    axis is T+1; and re-encoding a GENERATED clip costs one more frame to
    `local_vel`. The comparison keeps the frames the two share, counted from
    the front -- the rest token against the rest token."""
    z = torch.randn(31, 1, 1, 128)
    assert latent_round_trip(z[:30], z) == pytest.approx(0.0)
    assert latent_round_trip(z, z[:30]) == pytest.approx(0.0)


def test_a_channel_count_mismatch_is_not_quietly_truncated():
    with pytest.raises(ValueError, match="128"):
        latent_round_trip(torch.zeros(31, 1, 1, 64), torch.zeros(31, 1, 1, 128))


# --------------------------------------------------------------------------
# the orchestrator, on the real corpus
# --------------------------------------------------------------------------

CORPUS = Path("data/truebones")
CONTENT_RIG, CONTENT_ACTION, TARGET_RIG = "Scorpion", "agitated", "Flamingo"


class _Pooled(Denoiser):
    """`MoDiffAE`'s two shape contracts without its cost -- the stub
    `tests/ops/test_retarget_run.py` pins `run_retarget` against."""

    def __init__(self) -> None:
        super().__init__()
        self.gain = torch.nn.Parameter(torch.ones(1))

    def encode(self, clean, cond, masks=None, temporal_valid=None):
        pooled = clean.mean(dim=(1, 2))  # (B, T)
        return pooled.permute(1, 0)[:, :, None, None].repeat(1, 1, 1, 8), {}

    def forward(self, z_t, t, cond, masks=None):
        return Prediction(out=z_t * self.gain, aux={})


@pytest.fixture(scope="module")
def retargeted(tmp_path_factory):
    if not (CORPUS / "index.jsonl").is_file():
        pytest.skip("stage-2 corpus absent; run scripts/build_features.py")
    dataset = MotionDataset(
        index=CorpusIndex.load(CORPUS / "index.jsonl"),
        root=CORPUS,
        manifest_dir=CORPUS / "rigs",
        features=FEATURES,
        window=RandomCrop(length=8),
        conditioners=["topology", "tpose", "norm_stats"],
        split="train",
        seed=0,
    )
    model = _Pooled().eval()
    result = run_retarget(
        model=model,
        process=GaussianDiffusion(num_steps=2),
        sampler=DDPM(),
        dataset=dataset,
        content=f"{CONTENT_RIG}/{CONTENT_ACTION}",
        target_rig=TARGET_RIG,
        out_dir=tmp_path_factory.mktemp("metrics"),
        generator=torch.Generator().manual_seed(0),
    )
    return result, model, dataset


def test_it_reports_every_scalar_the_spec_lists(retargeted):
    result, model, dataset = retargeted
    scores = retarget_metrics(
        result, model, dataset, content_rig=CONTENT_RIG, target_rig=TARGET_RIG
    )
    assert set(scores) == {
        "foot_skate",
        "bone_length_drift",
        "ik_residual_mean",
        "ik_residual_max",
        "root_trajectory_error",
        "latent_round_trip",
    }
    assert all(np.isfinite(v) for v in scores.values()), scores
    assert all(v >= 0.0 for v in scores.values()), scores
    assert scores["ik_residual_max"] >= scores["ik_residual_mean"]


def test_the_drift_reported_is_the_one_before_ik(retargeted):
    """The distinction the whole pair of metrics exists for.

    `positions_ik` returns forward kinematics of a rigid skeleton, so the
    SOLVED output's bone-length drift is zero by construction whatever the
    model predicted. A metric reporting that number would be reporting the
    solver's parameterization; this asserts the reported value is the raw
    prediction's instead.
    """
    result, model, dataset = retargeted
    anim = result.target_anim
    after_ik = bone_length_drift(result.solved_positions, anim.parents, anim.offsets)
    before_ik = bone_length_drift(result.predicted_positions, anim.parents, anim.offsets)
    assert after_ik < 1e-9, "the solve is rigid by construction; this is the trap"
    assert before_ik > 1e-6, "an untrained model should not predict a holdable rig"

    scores = retarget_metrics(
        result, model, dataset, content_rig=CONTENT_RIG, target_rig=TARGET_RIG
    )
    assert scores["bone_length_drift"] == pytest.approx(before_ik)


def test_the_feet_it_watches_are_the_target_rigs_own(retargeted):
    """The metric is scored on the rig that was GENERATED, so the foot joints
    have to be the target's, in the reduced joint order the generated clip is
    actually in -- not the content rig's, and not the unreduced manifest's."""
    result, _, dataset = retargeted
    resolved = target_skeleton(dataset, TARGET_RIG)
    assert resolved.manifest.name == TARGET_RIG
    feet = list(resolved.foot_indices)
    assert feet, f"{TARGET_RIG} declares no feet; pick another target"
    assert max(feet) < result.solved_positions.shape[1]
    names = result.target_anim.names
    assert {names[f] for f in feet} == set(resolved.manifest.foot_joints)

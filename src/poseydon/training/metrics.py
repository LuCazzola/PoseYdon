"""The scalars that say whether a retarget is any good.

One function per row below, plus two that read a
:class:`~poseydon.ops.retarget_run.RetargetResult`: :func:`retarget_metrics`
returns the five numbers measured on the retarget itself (``ik_residual`` is a
mean AND a max), and :func:`reconstruction_metrics` the two that decode again,
``rec_error`` and ``rec_cycle_error``. The validation callback logs all seven.

| foot skate            | horizontal displacement of a foot while its contact flag is set, on the SOLVED output |
| bone-length drift     | ``abs(dist(p_j, p_parent) - norm(offset_j))`` on the RAW PREDICTED positions          |
| IK residual           | ``norm(p_predicted - p_solved)``, mean and max                                        |
| root trajectory error | generated root XZ velocity against the content clip's, each rescaled by its rig       |
| reconstruction error  | content decoded onto its own rig, against the content (no retarget)                   |
| cycle error           | the retarget decoded back onto the content's rig, against the content                 |

Every length is reported **in bone-length units** -- divided by the rig's mean
real bone -- so a scorpion and a flamingo produce numbers that can be averaged
across the three validation pairs and watched on one axis.

**Bone-length drift is measured BEFORE the IK solve, and that is the point.**
``positions_ik`` returns forward kinematics of a rigid skeleton, so on the
SOLVED output the drift is identically zero whatever the model predicted:
measuring it there reports the solver's parameterization, not the model. Pre-IK,
read alongside the residual, the two separate *"the model predicts bad bone
lengths"* from *"the solver had to move joints a long way"* -- the confusion
Phase 1's retrospective named as its recurring bug. `RetargetResult` keeps
``predicted_positions`` and ``solved_positions`` apart for exactly this reason;
do not collapse them.

The functions below take arrays, not a result, so each one can be checked
against a definition that lives outside this file: forward kinematics of a rigid
skeleton has no drift, a foot that does not move cannot slide, velocity is
invariant under translation, and a distance from a thing to itself is zero.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import torch

from poseydon.build.prepare import RigTransform
from poseydon.core.skeleton import ResolvedSkeleton
from poseydon.data.collate import collate
from poseydon.data.dataset import MotionDataset
from poseydon.features import extract_features
from poseydon.features.base import FeatureContext
from poseydon.features.contact import FootContact
from poseydon.ingest.align import facing_quats
from poseydon.models.base import Denoiser
from poseydon.ops.retarget_run import (
    RetargetResult,
    _resolve,
    _whole_clip,
    decode_positions,
    find_clip_of_rig,
)
from poseydon.process.base import Process
from poseydon.sampling.base import Sampler

#: Offsets at or below this are End Sites, not bones. Same constant, and the
#: same reason, as `features.reduce` and `ScaleToMeanBoneLength`: a zero-length
#: bone contributes nothing to the numerator and one to the denominator, so
#: including it scales every metric by a rig-dependent factor.
BONE_TOLERANCE = 1e-8

#: Horizontal plane. Y is up throughout the codebase.
XZ = [0, 2]


def mean_bone_length(offsets: np.ndarray, *, tolerance: float = BONE_TOLERANCE) -> float:
    """The rig's mean REAL bone length -- the unit every length here is in."""
    lengths = np.linalg.norm(np.asarray(offsets)[1:], axis=-1)
    real = lengths[lengths > tolerance]
    if real.size == 0:
        raise ValueError("skeleton has no bone of non-zero length to measure against")
    return float(real.mean())


def contact_flags(positions: np.ndarray, resolved: ResolvedSkeleton) -> np.ndarray:
    """``(F, J)`` bool: is this joint planted on this frame?

    The rule is not restated here -- it is `features.contact.FootContact`, the
    same detector that produces the training feature, so the manifest's
    thresholds mean one thing in the loss and in the metric.

    That feature is a finite difference, so its frame ``k`` describes animation
    frame ``k + 1``. A leading all-false row is prepended to bring the flags
    back onto the position array's own indexing: on frame 0 there is no
    previous frame, so nothing can be KNOWN planted.
    """
    positions = np.asarray(positions, dtype=np.float64)
    ctx = FeatureContext(
        anim=None,  # FootContact reads only `resolved`, `positions` and `n_joints`
        resolved=resolved,
        positions=positions,
        root_quats=facing_quats(
            positions, resolved.facing_indices, resolved.manifest.extra_yaw_deg
        ),
    )
    flags = FootContact()(ctx)[..., 0] > 0.5  # (F-1, J), describing frames 1..F-1
    return np.concatenate([np.zeros((1, flags.shape[1]), dtype=bool), flags], axis=0)


def foot_skate(positions: np.ndarray, contacts: np.ndarray, *, scale: float = 1.0) -> float:
    """Mean horizontal displacement of a foot over the frames it is planted.

    ``contacts[t, j]`` says joint ``j`` is planted on frame ``t``; the slide it
    is charged with is the move from ``t`` to ``t + 1``, which is how
    `losses.footskate.FootSkateLoss` pairs the two as well. Vertical motion is
    not skate -- a foot lifting off is a step.

    Zero when nothing is planted: no contact frames is no evidence of sliding,
    not evidence of none, and the caller can see the clip.
    """
    positions = np.asarray(positions, dtype=np.float64)
    contacts = np.asarray(contacts, dtype=bool)
    if contacts.shape != positions.shape[:2]:
        raise ValueError(
            f"contacts must be one flag per (frame, joint) of positions "
            f"{positions.shape[:2]}, got {contacts.shape}"
        )
    if positions.shape[0] < 2:
        return 0.0

    step = np.linalg.norm(
        positions[1:, :, XZ] - positions[:-1, :, XZ], axis=-1
    )  # (F-1, J)
    planted = contacts[:-1]
    if not planted.any():
        return 0.0
    return float(step[planted].mean() / scale)


def bone_length_drift(
    positions: np.ndarray,
    parents: np.ndarray,
    offsets: np.ndarray,
    *,
    scale: float | None = None,
) -> float:
    """How far the positions are from a skeleton that could hold them.

    ``abs(dist(p_j, p_parent) - norm(offset_j))`` over every non-root joint and
    every frame, in bone-length units. **Give this the RAW PREDICTED positions**
    -- see the module docstring.
    """
    positions = np.asarray(positions, dtype=np.float64)
    parents = np.asarray(parents)
    offsets = np.asarray(offsets, dtype=np.float64)
    if scale is None:
        scale = mean_bone_length(offsets)

    parent_of = parents[1:].astype(np.int64)
    if (parent_of < 0).any():
        raise ValueError("a non-root joint has no parent; this is not one skeleton")
    measured = np.linalg.norm(positions[:, 1:] - positions[:, parent_of], axis=-1)
    rest = np.linalg.norm(offsets[1:], axis=-1)
    return float(np.abs(measured - rest[None, :]).mean() / scale)


def ik_residual(
    predicted: np.ndarray, solved: np.ndarray, *, scale: float = 1.0
) -> tuple[float, float]:
    """``(mean, max)`` distance from the prediction to the solve, per joint.

    How far the solver had to travel -- whether it is cleaning the prediction up
    or inventing a different pose. Read next to `bone_length_drift`, never
    instead of it.
    """
    predicted = np.asarray(predicted, dtype=np.float64)
    solved = np.asarray(solved, dtype=np.float64)
    if predicted.shape != solved.shape:
        raise ValueError(
            f"predicted {predicted.shape} and solved {solved.shape} describe "
            "different clips"
        )
    distance = np.linalg.norm(predicted - solved, axis=-1)
    return float(distance.mean() / scale), float(distance.max() / scale)


def root_trajectory_error(
    generated_root: np.ndarray,
    content_root: np.ndarray,
    *,
    generated_scale: float = 1.0,
    content_scale: float = 1.0,
) -> float:
    """Mean per-frame difference of the two root XZ velocities.

    Velocity, not position: `features.recover.root_trajectory` is deliberately
    root-invariant, so a retarget always comes back at the XZ origin (design
    spec section 9). An absolute comparison would report that and nothing else.

    Each trajectory is divided by its own rig's ``scale_factor``. That is the
    number `ScaleToMeanBoneLength` recorded in ``preproc.npz`` for the step
    ``prepared = source * factor``, so dividing by it is the inverse -- the two
    trajectories are compared in the units the two rigs were authored in rather
    than in each rig's prepared units.
    """
    generated_root = np.asarray(generated_root, dtype=np.float64)
    content_root = np.asarray(content_root, dtype=np.float64)
    if generated_root.shape != content_root.shape:
        raise ValueError(
            f"the two roots cover different frames: {generated_root.shape} against "
            f"{content_root.shape}. Align them at the caller -- `extract_features` "
            "costs the last frame, so a generated clip is one shorter than the "
            "animation it was encoded from."
        )
    if generated_root.ndim != 2 or generated_root.shape[-1] != 3:
        raise ValueError(f"roots must be (frames, 3), got {generated_root.shape}")

    generated = np.diff(generated_root[:, XZ], axis=0) / generated_scale
    content = np.diff(content_root[:, XZ], axis=0) / content_scale
    if generated.shape[0] == 0:
        return 0.0
    return float(np.linalg.norm(generated - content, axis=-1).mean())


def reconstruction_error(generated: np.ndarray, source: np.ndarray, *, scale: float = 1.0) -> float:
    """Mean joint distance between two clips of ONE rig, the root's XZ removed.

    Both are ``(F, J, 3)`` world positions on the same skeleton; the shared
    frames are counted from the front, since every decode costs the frame
    `local_vel` spends. The root's ground-plane position is removed from each
    frame because `features.recover.root_trajectory` starts a decode at the XZ
    origin and the trajectory has its own metric; height and facing stay in.
    """
    generated = np.asarray(generated, dtype=np.float64)
    source = np.asarray(source, dtype=np.float64)
    if generated.shape[1:] != source.shape[1:]:
        raise ValueError(
            f"the two clips are on different rigs: {generated.shape[1:]} against {source.shape[1:]}"
        )
    frames = min(generated.shape[0], source.shape[0])
    if frames == 0:
        raise ValueError("a clip with no frames cannot be compared")
    generated, source = generated[:frames].copy(), source[:frames].copy()
    for clip in (generated, source):
        clip[..., XZ] -= clip[:, :1, XZ]
    return float(np.linalg.norm(generated - source, axis=-1).mean() / scale)


# ---------------------------------------------------------------------------
# Pulling the arguments out of a real run
# ---------------------------------------------------------------------------


def target_skeleton(dataset: MotionDataset, rig: str) -> ResolvedSkeleton:
    """The rig's manifest bound to the joint order a GENERATED clip is in.

    `MotionDataset._resolved` binds the manifest's joint NAMES against the
    reduced animation's names, which is the one remapping in the codebase --
    the same resolved skeleton `augment/topology.py::_reindex_resolved` then
    reindexes when an augmentation edits the rig further. Writing a second map
    from the reduction's `JointEdit` would duplicate it, and would fail with a
    `KeyError` instead of the manifest's "did you mean" if a foot were ever
    reduced away.
    """
    return dataset._resolved(find_clip_of_rig(dataset, rig))


def prepare_scale(dataset: MotionDataset, rig: str) -> float:
    """The rig's ``scale_factor``, as `ScaleToMeanBoneLength` recorded it."""
    transform = RigTransform.load(dataset.manifest_dir / rig / "preproc.npz")
    return float(transform.rig_params["scale"]["factor"])


def _batch_of_generated(dataset: MotionDataset, result: RetargetResult, target_rig: str):
    """The clip that was actually written, as a batch on the target rig.

    Re-extracted from `result.target_anim` rather than from the sampler's
    output tensor, so what is scored is the artifact -- reconstruction and all
    -- rather than an intermediate nobody can open. The conditioning is the
    TARGET rig's, taken from a real item of it the way `run_retarget` does:
    topology, rest pose and statistics describe the rig, not the clip.
    """
    item = _whole_clip(dataset, find_clip_of_rig(dataset, target_rig))
    raw, spec = extract_features(
        result.target_anim, target_skeleton(dataset, target_rig), dataset.features
    )
    if spec != item.spec:
        raise ValueError(
            f"the generated clip extracts as {spec} but the rig's items are "
            f"{item.spec}; the two cannot share an encoder"
        )
    features = dataset._normalizer(target_rig, spec).normalize(raw)
    return collate(
        [dataclasses.replace(item, features=features, source_length=features.shape[0])]
    )


def reconstruction_metrics(
    result: RetargetResult,
    model: Denoiser,
    process: Process,
    sampler: Sampler,
    dataset: MotionDataset,
    *,
    content: str,
    target_rig: str,
    device: torch.device | str = "cpu",
    generator: torch.Generator | None = None,
    guidance_scale: float = 1.0,
) -> dict[str, float]:
    """The two retarget checks that have a right answer: the source clip itself.

    ``rec_error`` decodes the content onto its OWN rig; ``rec_cycle_error``
    decodes the retarget `result` (content on the target rig) back onto the
    content's rig. Both compare against the content clip, in its rig's bone
    lengths. Neither says what the target-rig motion should look like -- no
    ground truth exists for that -- only that the latent carried the content
    there and back. Each costs one more decode.
    """
    record = _resolve(dataset, content)
    source = result.content_anim.global_positions()
    scale = mean_bone_length(result.content_anim.offsets)

    def decode(batch) -> np.ndarray:
        return decode_positions(
            model, process, sampler, dataset, batch, record.skeleton,
            device=device, generator=generator, reconstruct=result.reconstruct,
            guidance_scale=guidance_scale,
        )

    own = decode(collate([_whole_clip(dataset, record)]))
    back = decode(_batch_of_generated(dataset, result, target_rig))
    return {
        "rec_error": reconstruction_error(own, source, scale=scale),
        "rec_cycle_error": reconstruction_error(back, source, scale=scale),
    }


def retarget_metrics(
    result: RetargetResult,
    dataset: MotionDataset,
    *,
    content_rig: str,
    target_rig: str,
) -> dict[str, float]:
    """The five scalars measured on one retargeted pair, ready to log.

    ``content_rig`` and ``target_rig`` are passed rather than recovered from the
    written filename: pairs are a QUERY the caller resolved, and nothing
    downstream parses a filename (design spec section 7).
    """
    anim = result.target_anim
    resolved = target_skeleton(dataset, target_rig)
    scale = mean_bone_length(anim.offsets)

    # Planted according to the RAW PREDICTION -- the model's own claim that this
    # foot is down -- and slid according to the SOLVED output, which is what was
    # written. Detecting contact on the solved output instead would bound the
    # metric by the manifest's own speed threshold and make it unable to report
    # a bad clip.
    flags = contact_flags(result.predicted_positions, resolved)
    residual_mean, residual_max = ik_residual(
        result.predicted_positions, result.solved_positions, scale=scale
    )

    frames = result.solved_positions.shape[0]
    return {
        "foot_skate": foot_skate(result.solved_positions, flags, scale=scale),
        "bone_length_drift": bone_length_drift(
            result.predicted_positions, anim.parents, anim.offsets, scale=scale
        ),
        "ik_residual_mean": residual_mean,
        "ik_residual_max": residual_max,
        # The content animation keeps the frame `extract_features` spends on
        # `local_vel`, so it is one longer than what came out of the model.
        "root_trajectory_error": root_trajectory_error(
            result.solved_positions[:, 0],
            result.content_anim.root_pos[:frames],
            generated_scale=prepare_scale(dataset, target_rig),
            content_scale=prepare_scale(dataset, content_rig),
        ),
    }

"""Running a retarget end to end, once, for everyone who needs one.

`poseydon retarget` and the training-time `RetargetValidation` callback do the
same thing and must keep doing the same thing: what is watched across a
three-day run has to be the code path that ships, or the run reports on
something nobody can reproduce afterwards. So the whole operation lives here as
one function, and both callers are thin.

What it fixes, and why, is spec section 6:

* **DDPM**, not DDIM -- retargeting has no transition zone to anchor, so DDIM
  inversion buys nothing. The sampler is an argument all the same, because the
  choice belongs to the caller's config, not to this module.
* **The content clip's own length.** Not a configured `n_frames`, and not the
  training window: the latent is per-frame, so the decode is exactly as long as
  what was encoded.
* **The recipe's reconstruction**, defaulting to
  `training.recipe.DEFAULT_RECONSTRUCT` -- read through the module rather than
  bound at import, so `sample`, `retarget` and the callback cannot end up
  disagreeing about what a generated tensor means.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from poseydon.build.index import ClipRecord
from poseydon.core.animation import RigidBodyAnimation
from poseydon.core.batch import Masks
from poseydon.data.collate import collate
from poseydon.data.dataset import ClipView, Item, MotionDataset
from poseydon.features import RECONSTRUCTORS, extract_features, positions_from_features
from poseydon.io.bvh import BVH
from poseydon.models.base import Denoiser
from poseydon.ops.retarget import Retarget
from poseydon.process.base import Process
from poseydon.sampling.base import Sampler
from poseydon.training import recipe

#: `<content_rig>_<action>__to__<target>`. Nothing parses this back; it is for
#: the human reading a validation directory of nine files.
SEPARATOR = "__to__"


@dataclass(frozen=True)
class RetargetResult:
    """One retarget: what was written, and what the metrics are measured on.

    `predicted_positions` and `solved_positions` are BOTH kept, and must stay
    that way. Bone-length drift is measured on the raw prediction, because on
    IK-solved output it is identically zero by construction -- that would
    measure the solver's parameterization rather than the model.
    """

    npz: Path
    bvh: Path
    mp4: Path

    predicted_positions: np.ndarray   # (F, J, 3) as the model predicted them
    solved_positions: np.ndarray      # (F, J, 3) after reconstruction
    latent: torch.Tensor              # (F, 1, 1, C) the z_sem that was injected
    target_anim: RigidBodyAnimation   # what was generated, on the target rig
    content_anim: RigidBodyAnimation  # the clip it was encoded from
    reconstruct: str                  # the method actually used

    @property
    def paths(self) -> tuple[Path, Path, Path]:
        return self.npz, self.bvh, self.mp4


def find_clip(dataset: MotionDataset, rig: str, action: str) -> ClipRecord:
    """The one indexed clip of `rig` performing `action`.

    Pairs are a QUERY -- same action, different skeleton -- not a materialized
    list of filenames, so both callers resolve them the same way.
    """
    matching = [
        record
        for record in dataset.records
        if record.skeleton == rig and record.action == action
    ]
    if not matching:
        raise LookupError(
            f"no clip `{rig}/{action}` in this dataset. `poseydon list` does not "
            "enumerate clips; check the index and the split."
        )
    return matching[0]


def find_clip_of_rig(dataset: MotionDataset, rig: str) -> ClipRecord:
    """Any clip of `rig`: what is wanted from it is the RIG's conditioning."""
    matching = [record for record in dataset.records if record.skeleton == rig]
    if not matching:
        raise LookupError(f"no clips for skeleton `{rig}` in this dataset")
    return matching[0]


def _resolve(dataset: MotionDataset, content: ClipRecord | str) -> ClipRecord:
    if isinstance(content, ClipRecord):
        return content
    rig, _, action = str(content).partition("/")
    if not action:
        raise ValueError(f"expected `<rig>/<action>`, got `{content}`")
    return find_clip(dataset, rig, action)


def _whole_clip(dataset: MotionDataset, record: ClipRecord) -> Item:
    """One clip at its FULL length, with its rig's conditioning.

    `dataset[i]` would apply the training window and the augmentation pipeline:
    a crop is exactly wrong here (the output length is the content's) and an
    augmentation is a different rig than the one being reported on. The pieces
    are the dataset's own, so the clip still arrives in the shape the statistics
    were fitted on -- rigid body, reduced, normalized.
    """
    anim = dataset._anim(record)
    resolved = dataset._resolved(record)
    raw, spec = extract_features(anim, resolved, dataset.features)
    normalizer = dataset._normalizer(record.skeleton, spec)
    view = ClipView(
        record=record,
        anim=anim,
        resolved=resolved,
        normalizer=normalizer,
        rest_frame=dataset._rest_frame_of(record.skeleton),
        clip=record.clip_id,
        rig=record.skeleton,
    )
    return Item(
        features=normalizer.normalize(raw),
        spec=spec,
        start=0,
        source_length=raw.shape[0],
        cond={c.name: c.extract(view) for c in dataset.conditioners},
    )


def run_retarget(
    model: Denoiser,
    process: Process,
    sampler: Sampler,
    dataset: MotionDataset,
    content: ClipRecord | str,
    target_rig: str,
    out_dir: str | Path,
    device: torch.device | str = "cpu",
    generator: torch.Generator | None = None,
    reconstruct: str | None = None,
) -> RetargetResult:
    """Decode one clip's semantics onto another rig, and write the evidence.

    `content` is a `ClipRecord` or a `"<rig>/<action>"` string. `model` is
    expected to be on `device` already -- the caller owns it, and during
    training it is the model being trained.
    """
    method = reconstruct or recipe.DEFAULT_RECONSTRUCT
    reconstructor = RECONSTRUCTORS.get(method)()
    if not reconstructor.produces_rotations:
        # Checked BEFORE the sampler runs: a hundred denoising steps followed by
        # "this method cannot write a BVH" is a hundred steps spent to learn
        # something knowable up front, and two of the three files written is
        # worse than none.
        raise ValueError(
            f"`{method}` produces no rotations, so a retarget cannot be written as "
            "BVH. The recipe records `positions_ik` for exactly this reason."
        )

    record = _resolve(dataset, content)
    target_record = find_clip_of_rig(dataset, target_rig)

    content_item = _whole_clip(dataset, record)
    target_item = _whole_clip(dataset, target_record)
    content_batch = collate([content_item])
    # One real item supplies the target's conditioning: topology, rest pose and
    # normalization statistics all describe the RIG, not the clip. Its frames
    # are not used at all -- the length below is the content's.
    target_batch = collate([target_item])

    frames = content_batch.n_frames
    joints = target_batch.n_joints
    masks = Masks(
        frames=torch.ones(1, frames, dtype=torch.bool),
        joints=target_batch.masks.joints[:1],
    ).to(device)

    operation = Retarget(content_batch.x, content_batch.cond)
    with torch.no_grad():
        out = operation.run(
            model=model,
            process=process,
            sampler=sampler,
            shape=(1, joints, target_batch.spec.dim, frames),
            cond=target_batch.cond.to(device),
            device=device,
            generator=generator,
            masks=masks,
        )

    normalizer = dataset._normalizer(target_rig, target_batch.spec)
    features = normalizer.denormalize(
        out[0].permute(2, 0, 1).cpu().numpy().astype("float64")
    )

    content_anim = dataset._anim(record)
    # The target rig, timed by the content clip: the motion's duration is the
    # content's, so writing it at the target clip's rate would misstate it.
    template = dataclasses.replace(dataset._anim(target_record), fps=content_anim.fps)

    predicted = positions_from_features(features, target_batch.spec)
    # `reconstruct(...)` would run the solve TWICE -- once for the positions,
    # once for the animation. One solve, and the positions read back off the
    # animation it produced.
    anim = reconstructor.anim(features, target_batch.spec, template)
    solved = anim.global_positions()

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / f"{record.skeleton}_{record.action}{SEPARATOR}{target_rig}"
    # The op keeps the latent it pinned; the metrics compare `encode(generated)`
    # against exactly this tensor, so it has to be the injected one rather than
    # a second encode of the same content.
    latent = operation._latent.detach().cpu()

    npz = stem.with_suffix(".npz")
    np.savez_compressed(
        npz,
        predicted_positions=predicted,
        solved_positions=solved,
        latent=latent.numpy(),
        parents=template.parents,
        offsets=template.offsets,
        names=np.array(template.names),
        fps=np.array(template.fps),
        reconstruct=np.array(method),
        content_clip=np.array(record.clip_id),
        target_rig=np.array(target_rig),
    )

    bvh = stem.with_suffix(".bvh")
    BVH.from_animation(anim).write(bvh)

    mp4 = stem.with_suffix(".mp4")
    # Imported here, not at the top: rendering is an optional extra
    # (`poseydon[render]`), and importing it eagerly would make matplotlib a
    # hard dependency of every `poseydon.ops` import.
    from poseydon.io.render import render_skeleton

    render_skeleton(
        mp4,
        template.parents,
        solved,
        fps=round(template.fps),
        title=f"{record.clip_id}{SEPARATOR}{target_rig}  ({method})",
    )

    return RetargetResult(
        npz=npz,
        bvh=bvh,
        mp4=mp4,
        predicted_positions=predicted,
        solved_positions=solved,
        latent=latent,
        target_anim=anim,
        content_anim=content_anim,
        reconstruct=method,
    )


"""Everything recorded about one rig, and moving its clips between spaces.

A clip lives in one of three spaces: the model view (reduced, canonical),
prepared (every joint, canonical) and source (the user's rig, as supplied).
`RigRecord.expand` and `RigRecord.restore` move a clip between them; writing
it is then the file format's job, `BVH.from_animation` or
`FBX.from_animation`. The two things a file needs that a clip does not carry
come from the record too: a source BVH's CHANNELS layout
(`RigRecord.source_channels`) and an FBX's asset and calibration
(`RigRecord.calibration(space)`).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from poseydon.build.authority import ArtifactUnavailableError
from poseydon.build.index import action_slug, is_all_takes_bundle, strip_skeleton_prefix
from poseydon.build.prepare import PrepareChain, prepare_chain
from poseydon.core.animation import Animation, RigidBodyAnimation
from poseydon.core.rotations import QUAT_IDENTITY, quat_to_matrix
from poseydon.core.skeleton import SkeletonManifest
from poseydon.features.reduce import build_reduction, invert_reduction
from poseydon.io.bvh import BVH

#: Largest disagreement, as a fraction of the rig's extent, between the vendor
#: FBX and the raw BVH of the same clip once calibrated. Above it the two files
#: are not the same motion and nothing can be exported onto the FBX.
CALIBRATION_TOL = 1e-4


def world_matrices(anim: Animation) -> np.ndarray:
    """``(F, J, 4, 4)`` world transforms of every joint."""
    positions, rotations = anim.global_transforms()
    world = np.zeros((*positions.shape[:2], 4, 4))
    world[..., :3, :3] = quat_to_matrix(rotations)
    world[..., :3, 3] = positions
    world[..., 3, 3] = 1.0
    return world


def _similarity(source: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """``s, R, t`` minimising ``|s R x + t - y|`` over paired ``(N, 3)`` points (Umeyama)."""
    mu_x, mu_y = source.mean(0), target.mean(0)
    x, y = source - mu_x, target - mu_y
    u, d, vt = np.linalg.svd(y.T @ x / len(x))
    fix = np.diag([1.0, 1.0, np.sign(np.linalg.det(u @ vt))])
    rotation = u @ fix @ vt
    scale = float(np.trace(np.diag(d) @ fix) / (x**2).sum(1).mean())
    return scale, rotation, mu_y - scale * rotation @ mu_x


@dataclass(frozen=True)
class FbxCalibration:
    """How a rig's BVH joints sit in its vendor FBX, measured, not assumed.

    The vendor's FBX and BVH of one clip are the same motion in two
    conventions: a global similarity (units, axes, placement) and, per joint,
    a constant offset between the BVH joint frame and the FBX node frame. Both
    are solved from that pair, so neither the bind poses nor the bone axes of
    the two files have to agree, which is what refused 13 rigs before.
    """

    asset: Path  # the vendor FBX: the asset exports are written onto
    clip: Path  # the vendor BVH of the same clip
    joints: tuple[str, ...]
    scale: float
    rotation: np.ndarray
    translation: np.ndarray
    offsets: np.ndarray  # (J, 3, 3): BVH joint frame -> FBX node linear part
    residual: float  # worst world-position disagreement, fraction of extent
    # 4x4 applied to the whole driven scene, mesh included: identity leaves
    # the asset in its own space, `inverse()` puts it in the clip's.
    placement: np.ndarray = field(default_factory=lambda: np.eye(4))

    @classmethod
    def fit(cls, bvh_path: Path, fbx_path: Path) -> FbxCalibration:
        from poseydon.io.fbx import FBX

        source = BVH.read(bvh_path).to_animation()
        scene = FBX.read(fbx_path)
        joints = tuple(n for n in source.names if n in scene.names)
        b = world_matrices(source)[:, [source.names.index(n) for n in joints]]
        f = scene.world()[:, [scene.names.index(n) for n in joints]]
        if len(b) != len(f):
            raise ValueError(
                f"{bvh_path.name} has {len(b)} frames but {fbx_path.name} has {len(f)}"
            )
        scale, rotation, translation = _similarity(
            b[..., :3, 3].reshape(-1, 3), f[..., :3, 3].reshape(-1, 3)
        )
        frame = scale * rotation @ b[0, :, :3, :3]
        offsets = np.linalg.solve(frame, f[0, :, :3, :3])
        mapped = scale * b[..., :3, 3] @ rotation.T + translation
        extent = np.ptp(f[0, :, :3, 3], axis=0).max()
        linear = np.abs(scale * rotation @ b[..., :3, :3] @ offsets - f[..., :3, :3]).max() / scale
        residual = max(
            float(np.linalg.norm(mapped - f[..., :3, 3], axis=-1).max() / extent), float(linear)
        )
        return cls(fbx_path, bvh_path, joints, scale, rotation, translation, offsets, residual)

    def drive(self, anim: Animation) -> dict[str, np.ndarray]:
        """World matrices, in the asset's space, that pose its nodes as ``anim``."""
        missing = [n for n in self.joints if n not in anim.names]
        if missing:
            raise ValueError(
                f"the clip lacks {len(missing)} joint(s) the asset is calibrated on, "
                f"e.g. {missing[:3]}; is it in the space this calibration was fitted for?"
            )
        world = world_matrices(anim)[:, [anim.names.index(n) for n in self.joints]]
        out = np.zeros_like(world)
        out[..., :3, :3] = self.scale * self.rotation @ world[..., :3, :3] @ self.offsets
        out[..., :3, 3] = self.scale * world[..., :3, 3] @ self.rotation.T + self.translation
        out[..., 3, 3] = 1.0
        return {name: out[:, j] for j, name in enumerate(self.joints)}

    def inverse(self) -> np.ndarray:
        """4x4 similarity taking the asset's space back into the BVH's."""
        out = np.eye(4)
        out[:3, :3] = self.rotation.T / self.scale
        out[:3, 3] = -self.rotation.T @ self.translation / self.scale
        return out

    def to_source(self, world: np.ndarray) -> np.ndarray:
        """Asset-space world positions ``(..., 3)`` mapped back into the BVH's space."""
        return (world - self.translation) @ self.rotation / self.scale


def same_clip_pairs(source: Path, rig: str, first: str | None) -> list[tuple[Path, Path]]:
    """Raw BVH/FBX files of the same action, the rest pose's pair first.

    Truebones names one action differently in the two formats
    (``__BigMouth.bvh`` against ``AlligatorALL-BigMouth.fbx``,
    ``__chase_bite.bvh`` against ``T-Rex-chase bite.fbx``), so names are
    reduced to letters and digits and an FBX goes to the BVH whose name is the
    longest suffix of its own. A wrong match cannot slip through calibration:
    the calibration it yields is checked against every frame.
    """
    def key(path: Path) -> str:
        return strip_skeleton_prefix(action_slug(path.stem), rig).replace("_", "")

    bvhs = sorted(source.glob("*.bvh"), key=lambda p: -len(key(p)))
    best: dict[Path, Path] = {}
    for fbx in sorted(source.glob("*.fbx")):
        if is_all_takes_bundle(fbx.stem, rig):
            continue
        match = next((b for b in bvhs if key(fbx).endswith(key(b))), None)
        # Several FBX can end in one BVH's name (Monkey's `die.fbx`,
        # `B01 die.fbx` and `B02 die.fbx` against `__Die.bvh`); the one with
        # the least left over is that clip, the others are other takes.
        if match is not None and (match not in best or len(key(fbx)) < len(key(best[match]))):
            best[match] = fbx
    return sorted(best.items(), key=lambda pair: (pair[0].name != first, pair[0].name))


@dataclass
class RigRecord:
    """One rig as the corpus recorded it: its reduction, its preparation
    parameters and, for FBX, its vendor asset and calibrations.

    It holds no file: ``expand`` and ``restore`` return plain animations, and
    `BVH.from_animation` / `FBX.from_animation` write them.
    """

    corpus: Path
    raw: Path
    rig: str
    chain: PrepareChain
    params: dict
    reduction: object
    clip_params: dict = field(default_factory=dict)
    _calibrations: dict = field(default_factory=dict)

    @classmethod
    def load(
        cls, corpus: str | Path, raw: str | Path, rig: str, layout: str = "rot6d_joint"
    ) -> RigRecord:
        """``layout`` is the rotation block the model's features use: it
        decides which joints the model view drops (`build_reduction`)."""
        corpus, raw = Path(corpus), Path(raw)
        skeleton = corpus / "rigs" / rig / "skeleton.npz"
        if not skeleton.is_file():
            raise FileNotFoundError(skeleton)
        with np.load(skeleton, allow_pickle=True) as data:
            names = tuple(str(x) for x in data["names"])
            rest = RigidBodyAnimation.from_root_motion(
                rotations=np.tile(QUAT_IDENTITY, (1, len(names), 1)),
                root_pos=np.zeros((1, 3)), offsets=data["offsets"],
                parents=data["parents"].astype(np.int32), names=names, fps=1.0)
            reduction = build_reduction(rest, layout=layout)
        prep = corpus / "rigs" / rig / "preproc.npz"
        params = {}
        clip_params = {}
        if prep.is_file():
            from poseydon.build.prepare import RigTransform
            transform = RigTransform.load(prep)
            params, clip_params = transform.rig_params, transform.clip_params
        # Inversion reads the persisted parameters, so the chain needs neither
        # this rig's channel layout nor its promotion target -- only the ORDER,
        # which `prepare_chain` is the single definition of.
        return cls(corpus, raw, rig, prepare_chain(), params, reduction, clip_params)

    @property
    def _model_names(self) -> tuple[str, ...]:
        return tuple(self.reduction.source_names[i] for i in self.reduction.source_of)

    def expand(self, anim: Animation) -> Animation:
        """The model view -> the prepared clip, every joint, still canonical."""
        if tuple(anim.names) != self._model_names:
            raise ValueError(f"{self.rig}: animation does not match the reduced model topology")
        return invert_reduction(anim, self.reduction)

    def restore(self, anim: Animation, facing_from: str | None = None) -> Animation:
        """The model view or a prepared clip -> the source rig, in source space.

        ``facing_from=<action>`` restores that clip's recorded yaw; without it,
        the rig-level facing (the rest pose's).
        """
        if tuple(anim.names) == self._model_names:
            current = self.expand(anim)
        elif tuple(anim.names) == tuple(self.reduction.source_names):
            current = anim
        else:
            raise ValueError(f"{self.rig}: animation is neither the model view nor the prepared rig")
        params = {name: dict(value) for name, value in self.params.items()}
        if facing_from is not None:
            if facing_from not in self.clip_params:
                raise KeyError(f"unknown facing clip `{facing_from}` for rig `{self.rig}`")
            params.update(self.clip_params[facing_from])
        running = list(reversed(self.chain.stages))
        if facing_from is None and "face_axis" not in params:
            raise KeyError(
                f"`{self.rig}`'s preproc.npz records no rig-level facing, so a "
                "generated clip has no frame to be restored into; pass "
                "`facing_from` to borrow a recorded clip's yaw, or rebuild the "
                "rig with scripts/process_dataset_truebones.py"
            )
        # An inverse is only as good as its recorded parameters, and a missing
        # one used to fall back to `{}` -- which for `enforce_rigid` is a silent
        # no-op and for the rest an opaque KeyError from inside the stage. Name
        # what is absent instead: the usual cause is a corpus prepared by an
        # older chain than the one being inverted with.
        missing = [stage.name for stage in running if stage.name not in params]
        if missing:
            raise KeyError(
                f"`{self.rig}` has no recorded parameters for preparation "
                f"stage(s) {', '.join(missing)}; this corpus was prepared by a "
                "different chain than the one being inverted with, so rebuild "
                "it with scripts/process_dataset_truebones.py"
            )
        for stage in running:
            current = stage.invert(current, params[stage.name])
        return current

    @property
    def source_channels(self) -> tuple[tuple[str, ...], ...] | None:
        """The source BVH's CHANNELS layout, for writing a restored clip.

        `EnforceRigid.invert` is the identity on the animation precisely
        because the declaration is the writer's job: a joint the source gave
        position channels gets them back, carrying its rest offset, so the file
        is structurally the rig the user supplied and the bones still do not
        stretch.
        """
        recorded = self.params.get("enforce_rigid", {}).get("source_channels")
        return None if recorded is None else tuple(tuple(spec) for spec in recorded)

    def calibration(self, space: str = "source") -> FbxCalibration:
        """The rig's FBX calibration for clips in ``space``, "source" or "prepared".

        Source: fitted once from a vendor BVH/FBX pair of one clip; the first
        pair whose two files agree on every frame wins, trying the rest pose's
        first. The manifest's `artifacts` pair is not used, because its two
        files are generally different clips.

        Prepared: refitted on that clip as the corpus prepared it. Preparation
        re-frames every joint (`RestRelative`) and moves, turns and scales the
        whole clip, so the asset's nodes sit in a prepared clip under their own
        constant frame changes. Its placement puts the asset, mesh included, in
        canonical space. Its residual is not gated: `EnforceRigid` changes bone
        lengths by design, which the similarity can't absorb.
        """
        if space not in ("source", "prepared"):
            raise ValueError(f"space must be 'source' or 'prepared', not {space!r}")
        if space not in self._calibrations:
            self._calibrations[space] = (
                self._fit_source() if space == "source" else self._fit_prepared()
            )
        return self._calibrations[space]

    def _fit_source(self) -> FbxCalibration:
        manifest = SkeletonManifest.load(self.corpus / "rigs" / self.rig / "manifest.yaml")
        source = self.raw / "source" / self.rig
        tried = []
        for bvh, fbx in same_clip_pairs(source, self.rig, manifest.rest_pose):
            try:
                calibration = FbxCalibration.fit(bvh, fbx)
            except ValueError as error:
                tried.append(f"{fbx.name}: {error}")
                continue
            if calibration.residual <= CALIBRATION_TOL:
                return calibration
            tried.append(f"{fbx.name}: off by {calibration.residual:.2e} of the extent")
        raise ArtifactUnavailableError(
            "corpus", self.rig, "source", "fbx",
            "no vendor BVH/FBX pair of one clip agrees"
            + (f" ({'; '.join(tried[:3])})" if tried else " (no pair found)"),
        )

    def _fit_prepared(self) -> FbxCalibration:
        source = self.calibration("source")
        action = strip_skeleton_prefix(action_slug(source.clip.stem), self.rig)
        clip = self.corpus / "clips" / self.rig / f"{action}.bvh"
        if not clip.is_file():
            raise ArtifactUnavailableError(
                "corpus", self.rig, "prepared", "fbx",
                f"the calibration clip `{action}` was not prepared ({clip} missing)",
            )
        fitted = FbxCalibration.fit(clip, source.asset)
        return replace(fitted, placement=fitted.inverse())

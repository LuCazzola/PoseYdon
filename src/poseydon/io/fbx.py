"""FBX reading and writing through the Autodesk FBX SDK.

Blender's FBX exporter is not used for anything that must round-trip: it
converts every bone rotation to XYZ Euler in float32 with a "compatible Euler"
step that, near gimbal lock, returns a rotation that is not the one it was
given. On Flamingo's root (FBX-local Y at 90 degrees) it wrote frames 4.76
degrees off its own input matrix. This module keys rotations itself, in
float64, and leaves the file format to Autodesk's own SDK.

Everything here stays in the FILE's native space (its units and axes). Mapping
that space onto a BVH's is a calibration, owned by `poseydon.export`.

The SDK is proprietary and x86-64 only: it is imported here and nowhere else,
and only the `fbxsdk` container has it (see the Dockerfile).

Supported transform model: per node, ``T @ R @ S`` with XYZ rotation order and
no pre/post rotation, offsets or pivots -- which is every node of every
Truebones file. A file using anything else is refused rather than mis-read:
those are the parts no file here could ever check.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial.transform import Rotation

if TYPE_CHECKING:
    from poseydon.core.animation import Animation
    from poseydon.export.rig import FbxCalibration

try:
    import fbx
except ImportError as error:  # pragma: no cover - depends on the image
    raise ImportError(
        "poseydon.io.fbx needs the Autodesk FBX SDK, which only the `fbxsdk` "
        "container has: docker compose run --rm fbxsdk ..."
    ) from error



def _matrix(m) -> np.ndarray:
    """An FbxAMatrix (row-vector convention) as a column-vector 4x4."""
    return np.array([[m.Get(r, c) for c in range(4)] for r in range(4)]).T


def _polar(linear: np.ndarray) -> np.ndarray:
    """The rotation part of ``(..., 3, 3)`` rotation-times-scale matrices."""
    u, _, vt = np.linalg.svd(linear)
    fix = np.ones(linear.shape[:-1])
    fix[..., -1] = np.sign(np.linalg.det(u @ vt))
    return (u * fix[..., None, :]) @ vt


def _frame(frame: int, mode):
    # Whole frames only: `SetSecondDouble` is not exact (it put 2.2 s at
    # 2.1999978 s), and a key a few microseconds off is interpolated, not hit.
    time = fbx.FbxTime()
    time.SetFrame(int(frame), mode)
    return time



def _time_mode(fps: float):
    """The explicit FBX time mode for ``fps``.

    Not `ConvertFrameRateToTimeMode`: for 30 fps it returns `eDefaultMode`,
    which the SDK reads as 30 but Blender reads as 25, so the clip played at
    the wrong speed there.
    """
    for mode in fbx.FbxTime.EMode.__members__.values():
        if mode in (fbx.FbxTime.EMode.eDefaultMode, fbx.FbxTime.EMode.eCustom,
                    fbx.FbxTime.EMode.eModesCount):
            continue
        # Relative: a BVH stores its frame time to six decimals, so 30 fps reads 30.00003.
        if abs(fbx.FbxTime.GetFrameRate(mode) - fps) < 1e-4 * fps:
            return mode
    raise ValueError(f"{fps} fps has no FBX time mode; only standard rates are written")

@dataclass
class FBX:
    """One loaded FBX scene, its node hierarchy and its single take."""

    manager: object
    scene: object
    nodes: list  # every transform node below the scene root, parents first
    parents: np.ndarray

    @classmethod
    def read(cls, path: str | Path) -> FBX:
        manager = fbx.FbxManager.Create()
        manager.SetIOSettings(fbx.FbxIOSettings.Create(manager, fbx.IOSROOT))
        scene = fbx.FbxScene.Create(manager, "")
        importer = fbx.FbxImporter.Create(manager, "")
        if not importer.Initialize(str(path), -1, manager.GetIOSettings()):
            raise ValueError(f"{path}: {importer.GetStatus().GetErrorString()}")
        importer.Import(scene)
        importer.Destroy()
        return cls._wrap(manager, scene)

    @classmethod
    def _wrap(cls, manager, scene) -> FBX:
        nodes, parents = [], []

        def walk(node, parent):
            for i in range(node.GetChildCount()):
                child = node.GetChild(i)
                nodes.append(child)
                parents.append(parent)
                walk(child, len(nodes) - 1)

        walk(scene.GetRootNode(), -1)
        return cls(manager, scene, nodes, np.array(parents, dtype=np.int32))

    @classmethod
    def from_animation(cls, anim: Animation, calibration: FbxCalibration) -> FBX:
        """``anim`` on the calibration's asset: its nodes, mesh, skin and materials.

        ``calibration`` must be the one fitted for ``anim``'s space
        (`RigRecord.calibration`); a clip in another space still writes, at the
        wrong place and scale, which nothing in the clip can reveal.
        """
        asset = cls.read(calibration.asset)
        asset.animate(calibration.drive(anim), anim.fps, space=calibration.placement)
        return asset

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(node.GetName() for node in self.nodes)

    def is_skeleton(self, index: int) -> bool:
        attribute = self.nodes[index].GetNodeAttribute()
        return attribute is not None and (
            attribute.GetAttributeType() == fbx.FbxNodeAttribute.EType.eSkeleton
        )

    def _stack(self):
        count = self.scene.GetSrcObjectCount(fbx.FbxCriteria.ObjectType(fbx.FbxAnimStack.ClassId))
        if count == 0:
            return None
        return self.scene.GetSrcObject(fbx.FbxCriteria.ObjectType(fbx.FbxAnimStack.ClassId), 0)

    @property
    def _mode(self):
        return self.scene.GetGlobalSettings().GetTimeMode()

    @property
    def fps(self) -> float:
        """The declared frame rate. A file declaring the SDK's "default" mode is
        refused: the SDK reads it as 30 fps and Blender as 25."""
        if self._mode == fbx.FbxTime.EMode.eDefaultMode:
            raise ValueError("this FBX declares no explicit frame rate")
        return fbx.FbxTime.GetFrameRate(self._mode)

    def frames(self) -> np.ndarray:
        """Every frame number of the first take (frame 0 alone if there is none)."""
        stack = self._stack()
        if stack is None:
            return np.zeros(1, dtype=int)
        span = stack.GetLocalTimeSpan()
        start = span.GetStart().GetFrameCount(self._mode)
        stop = span.GetStop().GetFrameCount(self._mode)
        return np.arange(start, stop + 1)

    def world(self, frames: np.ndarray | None = None) -> np.ndarray:
        """``(F, N, 4, 4)`` world matrices of every node, evaluated by the SDK."""
        if self._stack() is not None:
            self.scene.SetCurrentAnimationStack(self._stack())
        frames = self.frames() if frames is None else frames
        evaluator = self.scene.GetAnimationEvaluator()
        out = np.empty((len(frames), len(self.nodes), 4, 4))
        for f, frame in enumerate(frames):
            time = _frame(frame, self._mode)
            for j, node in enumerate(self.nodes):
                out[f, j] = _matrix(evaluator.GetNodeGlobalTransform(node, time))
        return out

    def _check_transform_model(self, node) -> None:
        if node.RotationOrder.Get() != fbx.EFbxRotationOrder.eEulerXYZ:
            raise ValueError(f"{node.GetName()}: rotation order other than XYZ")
        props = [node.RotationOffset, node.RotationPivot, node.ScalingOffset, node.ScalingPivot]
        if node.RotationActive.Get():
            props += [node.PreRotation, node.PostRotation]
        for prop in props:
            if np.abs(np.array(list(prop.Get()))).max() > 1e-9:
                raise ValueError(f"{node.GetName()}: non-zero {prop.GetName()} is not supported")

    def animate(
        self, world: dict[str, np.ndarray], fps: float, space: np.ndarray | None = None
    ) -> None:
        """Replace the scene's animation so the named nodes follow ``world``.

        ``world`` maps node name -> ``(F, 4, 4)`` world matrices. Every other
        node keeps the local transform it had on the scene's first frame, so a
        helper bone rides along with its parent. Every take is removed and one
        is written, keyed on every frame, rotations as Euler angles solved in
        float64.

        ``space`` (4x4, a similarity) moves the whole scene, mesh included:
        every world matrix, given or kept, is composed onto it.
        """
        space = np.eye(4) if space is None else space
        world = {name: space @ matrices for name, matrices in world.items()}
        n_frames = len(next(iter(world.values())))
        first = self.world(self.frames()[:1])[0]
        local0 = np.empty_like(first)
        for j, parent in enumerate(self.parents):
            local0[j] = space @ first[j] if parent < 0 else np.linalg.inv(first[parent]) @ first[j]

        # Forward kinematics over the whole hierarchy: driven nodes take their
        # target, the rest compose their fixed local transform on the parent.
        index = {name: j for j, name in enumerate(self.names)}
        unknown = sorted(set(world) - set(index))
        if unknown:
            raise ValueError(f"no node named {unknown[:3]} in this file")
        target = np.empty((n_frames, len(self.nodes), 4, 4))
        local = np.empty_like(target)
        for j, parent in enumerate(self.parents):
            above = np.eye(4) if parent < 0 else target[:, parent]
            if self.names[j] in world:
                target[:, j] = world[self.names[j]]
                local[:, j] = np.linalg.inv(above) @ target[:, j]
            else:
                local[:, j] = local0[j]
                target[:, j] = above @ local0[j]

        for stack_index in reversed(range(1, self.scene.GetSrcObjectCount(
            fbx.FbxCriteria.ObjectType(fbx.FbxAnimStack.ClassId)
        ))):
            self.scene.GetSrcObject(
                fbx.FbxCriteria.ObjectType(fbx.FbxAnimStack.ClassId), stack_index
            ).Destroy()
        stack = self._stack() or fbx.FbxAnimStack.Create(self.scene, "Take 001")
        layer = stack.GetMember(fbx.FbxCriteria.ObjectType(fbx.FbxAnimLayer.ClassId), 0)
        if layer is None:
            layer = fbx.FbxAnimLayer.Create(self.scene, "Base Layer")
            stack.AddMember(layer)

        settings = self.scene.GetGlobalSettings()
        settings.SetTimeMode(_time_mode(fps))
        mode = settings.GetTimeMode()
        frames = np.arange(n_frames)
        span = fbx.FbxTimeSpan(_frame(0, mode), _frame(n_frames - 1, mode))
        stack.SetLocalTimeSpan(span)
        stack.SetReferenceTimeSpan(span)
        settings.SetTimelineDefaultTimeSpan(span)

        for j, node in enumerate(self.nodes):
            driven = self.names[j] in world
            for prop in (node.LclTranslation, node.LclRotation, node.LclScaling):
                curve_node = prop.GetCurveNode(layer)
                if curve_node is not None:
                    curve_node.Destroy()
            if not driven:
                self._set_static(node, local0[j])
                continue
            self._check_transform_model(node)
            euler = self._euler(local[:, j, :3, :3])
            self._key(node.LclRotation, layer, frames, mode, euler)
            self._key(node.LclTranslation, layer, frames, mode, local[:, j, :3, 3])
            scale = np.linalg.norm(local[0, j, :3, :3], axis=0)
            node.LclScaling.Set(fbx.FbxDouble3(*map(float, scale)))

    @staticmethod
    def _euler(linear: np.ndarray) -> np.ndarray:
        """``(F, 3)`` LclRotation degrees giving these local linear parts."""
        euler = Rotation.from_matrix(_polar(linear)).as_euler("xyz")
        # Any branch reproduces the rotation exactly; unwrapping only keeps
        # neighbouring keys close so that an interpolating reader does not spin.
        return np.degrees(np.unwrap(euler, axis=0))

    def _set_static(self, node, local: np.ndarray) -> None:
        self._check_transform_model(node)
        euler = self._euler(local[None, :3, :3])[0]
        node.LclRotation.Set(fbx.FbxDouble3(*map(float, euler)))
        node.LclTranslation.Set(fbx.FbxDouble3(*map(float, local[:3, 3])))
        node.LclScaling.Set(fbx.FbxDouble3(*map(float, np.linalg.norm(local[:3, :3], axis=0))))

    @staticmethod
    def _key(prop, layer, frames: np.ndarray, mode, values: np.ndarray) -> None:
        prop.GetCurveNode(layer, True)
        linear = fbx.FbxAnimCurveDef.EInterpolationType.eInterpolationLinear
        for axis, channel in enumerate("XYZ"):
            curve = prop.GetCurve(layer, channel, True)
            curve.KeyModifyBegin()
            curve.KeyClear()
            for frame, value in zip(frames, values[:, axis]):
                time = _frame(frame, mode)
                curve.KeySet(curve.KeyAdd(time)[0], time, float(value), linear)
            curve.KeyModifyEnd()

    def write(self, path: str | Path, embed_textures: bool = False) -> None:
        """Write the scene. ``embed_textures`` packs image files into the FBX,
        so a skinned asset written away from its source folder still finds them."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.manager.GetIOSettings().SetBoolProp(fbx.EXP_FBX_EMBEDDED, embed_textures)
        exporter = fbx.FbxExporter.Create(self.manager, "")
        writer = self.manager.GetIOPluginRegistry().GetNativeWriterFormat()
        if not exporter.Initialize(str(path), writer, self.manager.GetIOSettings()):
            raise ValueError(f"{path}: {exporter.GetStatus().GetErrorString()}")
        if not exporter.Export(self.scene):
            raise ValueError(f"{path}: {exporter.GetStatus().GetErrorString()}")
        exporter.Destroy()

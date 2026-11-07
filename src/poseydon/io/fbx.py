"""FBX reading and writing through the Autodesk FBX SDK.

Blender's FBX exporter is not used for anything that must round-trip: it
converts every bone rotation to XYZ Euler in float32 with a "compatible Euler"
step that, near gimbal lock, returns a rotation that is not the one it was
given. On Flamingo's root (FBX-local Y at 90 degrees) it wrote frames 4.76
degrees off its own input matrix. This module keys rotations itself, in
float64, and leaves the file format to Autodesk's own SDK.

Everything here stays in the FILE's native space (its units and axes). A clip
read from an FBX (`poseydon.build.sources.FbxCache`) is in that space already,
so writing it back is by node name (`FBX.from_animation`); `FBX.pose` holds the
file in any pose, on its own bind and mesh.

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

from poseydon.core.rotations import rotation_part

if TYPE_CHECKING:
    from poseydon.core.animation import Animation

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

@dataclass(frozen=True)
class Skin:
    """One skin deformer's weights, per node of the file.

    ``max_weight`` and ``weight_sum`` are ``(N,)`` over every node: the largest
    single weight and the summed weight a node's cluster gives its vertices
    (0 for a node with no cluster). Enough to tell a bone that moves the mesh
    from one that is only a bind-time joint, without keeping per-vertex weights.
    """

    mesh: str
    n_vertices: int
    max_weight: np.ndarray
    weight_sum: np.ndarray


@dataclass(frozen=True)
class SkinData:
    """One skin deformer read whole: the mesh, and per cluster its link, bind
    matrices and weights, enough to skin it without the SDK.

    ``points`` keeps every control point (indices match the file's), ``faces``
    are the polygons fanned into triangles, ``link_bind`` and ``mesh_bind`` are
    ``(C, 4, 4)`` column-vector matrices, ``indices``/``weights`` one array per cluster.
    """

    mesh: str
    points: np.ndarray
    faces: np.ndarray
    links: tuple[str, ...]
    link_bind: np.ndarray
    mesh_bind: np.ndarray
    indices: tuple[np.ndarray, ...]
    weights: tuple[np.ndarray, ...]


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
    def from_animation(cls, anim: Animation, asset: str | Path, space: np.ndarray | None = None,
                       fixed: tuple[str, ...] = ()) -> FBX:
        """``anim`` on ``asset``: its nodes, mesh, skin and materials, each joint
        driving the node of its name, except those in ``fixed``, which keep the
        asset's own local transform (`RigRecord.fixed`: joints a restored clip
        holds by structure only).

        ``anim`` must be in the asset's own space, as a clip read from it is
        (`RigRecord.restore` gives a corpus clip back in it). Each node keeps its
        own world scale on the asset's first frame (Crow's armature is scaled 5),
        which an `Animation` does not carry. ``space`` moves the whole scene, mesh
        included (`FBX.animate`): `RigRecord.placement` puts it in canonical space.
        """
        scene = cls.read(asset)
        names = [n for n in anim.names if n in scene.names and n not in fixed]
        if not names:
            raise ValueError(f"{asset}: no node shares a name with the clip's joints")
        positions, rotations = anim.global_transforms()
        scale = np.linalg.norm(scene.world(scene.frames()[:1])[0, :, :3, :3], axis=-2)
        world = {}
        for name in names:
            j, node = anim.names.index(name), scene.names.index(name)
            m = np.zeros((anim.n_frames, 4, 4))
            m[:, :3, :3] = Rotation.from_quat(rotations[:, j]).as_matrix() * scale[node]
            m[:, :3, 3], m[:, 3, 3] = positions[:, j], 1.0
            world[name] = m
        scene.animate(world, anim.fps, space=space)
        return scene

    def close(self) -> None:
        """Free the SDK manager and everything it owns. The object is unusable after."""
        self.manager.Destroy()

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(node.GetName() for node in self.nodes)

    def is_skeleton(self, index: int) -> bool:
        attribute = self.nodes[index].GetNodeAttribute()
        return attribute is not None and (
            attribute.GetAttributeType() == fbx.FbxNodeAttribute.EType.eSkeleton
        )

    def _stacks(self) -> list:
        kind = fbx.FbxCriteria.ObjectType(fbx.FbxAnimStack.ClassId)
        found = (self.scene.GetSrcObject(kind, s) for s in range(self.scene.GetSrcObjectCount(kind)))
        # The criteria is not a strict filter: Eagle's and Horse's files return
        # FbxNodes through it, which `animate` would otherwise have destroyed.
        return [o for o in found if isinstance(o, fbx.FbxAnimStack)]

    def _stack(self):
        stacks = self._stacks()
        return stacks[0] if stacks else None

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

    def node_types(self) -> tuple[str, ...]:
        """Attribute type name per node (``"eSkeleton"``, ``"eMesh"``, ...; ``""`` if none)."""
        return tuple(
            "" if (a := node.GetNodeAttribute()) is None else a.GetAttributeType().name
            for node in self.nodes
        )

    def bind_matrices(self) -> np.ndarray:
        """``(N, 4, 4)`` world bind matrix per node, NaN where the file has none.

        A skin cluster's ``TransformLinkMatrix`` wins (it is what the skin was
        bound against); else the scene's bind `FbxPose`. Read straight from the
        file: no animation evaluated, no transform-model check.
        """
        out = np.full((len(self.nodes), 4, 4), np.nan)
        # Python proxies of one SDK object are not identical: match by unique id.
        by_uid = {node.GetUniqueID(): j for j, node in enumerate(self.nodes)}
        for pose_i in range(self.scene.GetPoseCount()):
            pose = self.scene.GetPose(pose_i)
            if not pose.IsBindPose():
                continue
            for k in range(pose.GetCount()):
                node = pose.GetNode(k)
                at = by_uid.get(node.GetUniqueID()) if node is not None else None
                if at is not None and np.isnan(out[at, 0, 0]):
                    out[at] = _matrix(pose.GetMatrix(k))
        for skin in self._skins():
            for c in range(skin.GetClusterCount()):
                cluster = skin.GetCluster(c)
                at = by_uid.get(cluster.GetLink().GetUniqueID())
                if at is not None:
                    link = fbx.FbxAMatrix()
                    cluster.GetTransformLinkMatrix(link)
                    out[at] = _matrix(link)
        return out

    def bind_or_first(self) -> np.ndarray:
        """``(N, 4, 4)`` `bind_matrices`, the first frame's world where a node has no bind."""
        bind = self.bind_matrices()
        missing = np.isnan(bind).any(axis=(-2, -1))
        if missing.any():
            bind[missing] = self.world(self.frames()[:1])[0][missing]
        return bind

    def pose(self, world: dict[str, np.ndarray], fps: float) -> np.ndarray:
        """Hold this scene in the pose ``world`` (node name -> 4x4 world matrix).

        A node not named keeps its local bind transform under its parent. The
        scene's animation becomes a one-frame take at the pose; mesh, skin and
        bind pose stay the file's, so the mesh there is the vendor's, skinned
        into the pose. Returns every node's world matrix in that pose, in float64
        (the take's keys are float32).
        """
        bind = self.bind_or_first()
        index = {name: j for j, name in enumerate(self.names)}
        unknown = sorted(set(world) - set(index))
        if unknown:
            raise ValueError(f"no node named {unknown[:3]} in this file")
        posed = np.empty_like(bind)
        for j, parent in enumerate(self.parents):
            if self.names[j] in world:
                posed[j] = world[self.names[j]]
            else:
                posed[j] = bind[j] if parent < 0 else posed[parent] @ np.linalg.inv(bind[parent]) @ bind[j]
        self.animate({name: posed[j][None] for name, j in index.items()}, fps)
        return posed

    def key_range(self) -> tuple[float, float]:
        """First and last key time, in frames, over every curve of the first take.

        What a Blender import calls the action's frame range. ``(0, 0)`` for a
        file with no keys.
        """
        stack = self._stack()
        if stack is None:
            return 0.0, 0.0
        times = []
        layers = stack.GetMemberCount(fbx.FbxCriteria.ObjectType(fbx.FbxAnimLayer.ClassId))
        kind = fbx.FbxCriteria.ObjectType(fbx.FbxAnimCurveNode.ClassId)
        for i in range(layers):
            layer = stack.GetMember(fbx.FbxCriteria.ObjectType(fbx.FbxAnimLayer.ClassId), i)
            if not isinstance(layer, fbx.FbxAnimLayer):
                continue
            for n in range(layer.GetSrcObjectCount(kind)):
                curve_node = layer.GetSrcObject(kind, n)
                # The criteria also lets other objects through on some files
                # (FbxNodes on 349 Truebones clips).
                if not isinstance(curve_node, fbx.FbxAnimCurveNode):
                    continue
                for channel in range(curve_node.GetChannelsCount()):
                    for c in range(curve_node.GetCurveCount(channel)):
                        curve = curve_node.GetCurve(channel, c)
                        if curve.KeyGetCount() == 0:
                            continue
                        for k in (0, curve.KeyGetCount() - 1):
                            times.append(curve.KeyGetTime(k).GetFrameCountPrecise(self._mode))
        return (min(times), max(times)) if times else (0.0, 0.0)

    def up_axis(self) -> str:
        """The file's up axis from its global axis system: ``"+Y"``, ``"-Z"``, ..."""
        up, sign = self.scene.GetGlobalSettings().GetAxisSystem().GetUpVector()
        return ("+" if sign >= 0 else "-") + up.name[1]

    def _skins(self) -> list:
        kind = fbx.FbxCriteria.ObjectType(fbx.FbxSkin.ClassId)
        found = (self.scene.GetSrcObject(kind, s) for s in range(self.scene.GetSrcObjectCount(kind)))
        return [o for o in found if isinstance(o, fbx.FbxSkin)]  # the criteria is not a strict filter

    def skins(self) -> list[Skin]:
        """Every skin deformer's per-node weight summary (see `Skin`)."""
        by_uid = {node.GetUniqueID(): j for j, node in enumerate(self.nodes)}
        out = []
        for skin in self._skins():
            geometry = skin.GetGeometry()
            top = np.zeros(len(self.nodes))
            total = np.zeros(len(self.nodes))
            for c in range(skin.GetClusterCount()):
                cluster = skin.GetCluster(c)
                at = by_uid.get(cluster.GetLink().GetUniqueID())
                weight = np.array(cluster.GetControlPointWeights(), dtype=float)
                if at is not None and weight.size:
                    top[at], total[at] = max(top[at], weight.max()), total[at] + weight.sum()
            node = geometry.GetNode()
            out.append(Skin(node.GetName() if node else "", geometry.GetControlPointsCount(), top, total))
        return out

    def skin_data(self) -> list[SkinData]:
        """Every skin deformer read whole (see `SkinData`)."""
        out = []
        for skin in self._skins():
            geometry = skin.GetGeometry()
            node = geometry.GetNode()
            points = np.array([[p[0], p[1], p[2]] for p in geometry.GetControlPoints()])
            faces = []
            for p in range(geometry.GetPolygonCount()):
                ring = [geometry.GetPolygonVertex(p, i) for i in range(geometry.GetPolygonSize(p))]
                faces += [(ring[0], ring[i], ring[i + 1]) for i in range(1, len(ring) - 1)]
            links, link_bind, mesh_bind, indices, weights = [], [], [], [], []
            for c in range(skin.GetClusterCount()):
                cluster = skin.GetCluster(c)
                a, b = fbx.FbxAMatrix(), fbx.FbxAMatrix()
                cluster.GetTransformLinkMatrix(a)
                cluster.GetTransformMatrix(b)
                links.append(cluster.GetLink().GetName())
                link_bind.append(_matrix(a))
                mesh_bind.append(_matrix(b))
                indices.append(np.array(cluster.GetControlPointIndices(), dtype=int))
                weights.append(np.array(cluster.GetControlPointWeights(), dtype=float))
            out.append(SkinData(node.GetName() if node else "", points, np.array(faces, dtype=int).reshape(-1, 3),
                                tuple(links), np.array(link_bind).reshape(-1, 4, 4),
                                np.array(mesh_bind).reshape(-1, 4, 4), tuple(indices), tuple(weights)))
        return out

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

    def skinned_vertices(self, frame: int = 0) -> np.ndarray:
        """``(V, 3)`` world positions of every skinned mesh's vertices on ``frame``.

        Linear blend skinning as the file declares it: each cluster moves its
        vertices by its link's world matrix on that frame times the inverse of
        the link's bind matrix (the mesh's own bind matrix applied first).
        Weights are normalised per vertex; a vertex no cluster weighs is dropped.
        Meshes without a skin deformer are not included: a prop or helper
        geometry is not the body.
        """
        if self._stack() is not None:
            self.scene.SetCurrentAnimationStack(self._stack())
        evaluator = self.scene.GetAnimationEvaluator()
        time = _frame(frame, self._mode)
        out = []
        for skin in self._skins():
            points = np.array([[p[0], p[1], p[2], 1.0] for p in skin.GetGeometry().GetControlPoints()])
            moved = np.zeros((len(points), 4))
            total = np.zeros(len(points))
            for c in range(skin.GetClusterCount()):
                cluster = skin.GetCluster(c)
                link_bind, mesh_bind = fbx.FbxAMatrix(), fbx.FbxAMatrix()
                cluster.GetTransformLinkMatrix(link_bind)
                cluster.GetTransformMatrix(mesh_bind)
                deform = (
                    _matrix(evaluator.GetNodeGlobalTransform(cluster.GetLink(), time))
                    @ np.linalg.inv(_matrix(link_bind)) @ _matrix(mesh_bind)
                )
                index = np.array(cluster.GetControlPointIndices(), dtype=int)
                weight = np.array(cluster.GetControlPointWeights(), dtype=float)
                if index.size:
                    np.add.at(moved, index, weight[:, None] * (points[index] @ deform.T))
                    np.add.at(total, index, weight)
            weighted = total > 0
            out.append(moved[weighted, :3] / total[weighted, None])
        if not out:
            raise ValueError("this FBX has no skinned mesh")
        return np.concatenate(out)

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

        for extra in self._stacks()[1:]:
            extra.Destroy()
        stack = self._stack() or fbx.FbxAnimStack.Create(self.scene, "Take 001")
        layer = stack.GetMember(fbx.FbxCriteria.ObjectType(fbx.FbxAnimLayer.ClassId), 0)
        if not isinstance(layer, fbx.FbxAnimLayer):
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
        # The evaluator caches what `world` read above (the old take's first
        # frame) and keeps returning it for the new keys: Bird's posed mesh
        # came back 70 units up in its attack pose until this reset.
        self.scene.GetAnimationEvaluator().Reset()

    @staticmethod
    def _euler(linear: np.ndarray) -> np.ndarray:
        """``(F, 3)`` LclRotation degrees giving these local linear parts."""
        euler = Rotation.from_matrix(rotation_part(linear)).as_euler("xyz")
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

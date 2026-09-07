"""FBX reading and writing, the counterpart to :mod:`poseydon.io.bvh`.

One class, :class:`FBX`, holds an imported scene -- armature, skinned meshes
and animation -- and converts both ways, plus the operations a preprocessing
pass needs (turn the rig, write the bind-pose mesh).

Unlike :mod:`poseydon.io.bvh`, this module needs Blender: FBX is a binary
format with no usable pure-Python reader, and Blender's importer is the only
mature one available. It therefore imports ``bpy`` and is importable ONLY
inside the ``fbx`` container (see ``Dockerfile.fbx``); nothing in
``poseydon.io`` imports it, so the rest of the package is unaffected.

Two conventions are stated once here and used throughout:

* **Axes.** Blender is Z-UP; Truebones, BVH and PoseYdon are Y-UP. So
  PoseYdon's "+Z forward" is Blender's +Y, and world up is Blender's +Z.
* **Bones vs joints.** FBX stores node TRANSFORMS, not segments, so a bone's
  tail is not in the file and Blender must invent one. Left to itself it
  makes a stub along the node's own axis, and the skeleton renders as
  disconnected fragments (median gap 1.414x bone length, 0 of 38
  parent-child pairs touching). :meth:`FBX.read` asks for
  ``automatic_bone_orientation`` instead, which points each bone at its
  child (gap 0.020) without moving a single joint -- verified, agreement
  with the BVH pipeline is unchanged to the digit.

Two things the format makes impossible, worth knowing before filing them as
bugs:

**Not every bone connects, and leaf bones point oddly.** A bone's tail is
not in an FBX at all. A LEAF has no child to aim at (10 of BrownBear's 39
bones), and a BRANCH joint's parent can only aim at one of its children, so
roughly 21 of 38 pairs touch and the rest visibly do not. A BVH renders
better here purely because it stores explicit End Site offsets, which give
every tip a real direction. The two importer options that appear to fix
this were measured and both destroy data:
``force_connect_children=True`` snaps each child's head onto its parent's
tail and MOVES JOINTS by up to 0.709 against a 1.079 extent (66% of the
character); ``ignore_leaf_bones=True`` deletes 7 joints outright. Neither is
used. The disconnection is cosmetic -- joint positions are exact.

**The exported armature is directly movable**, which took getting the
export settings right rather than working around them -- see
:meth:`FBX.write`. Blender's exporter writes a keyframed object transform
whenever animation baking keeps constant curves, and a keyframed object
snaps back to its keys on every frame change.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from poseydon.core.skeleton import HML_MEAN_BONE_LENGTH

try:
    import bpy
    from mathutils import Matrix, Vector
except ImportError as error:  # pragma: no cover - depends on the image
    raise ImportError(
        "poseydon.io.fbx requires Blender's Python API, which only exists in the "
        "`fbx` container. Run it with: docker compose run --rm fbx blender "
        "--background --python <script>"
    ) from error

# Blender's frame: world up is +Z, and PoseYdon's +Z forward is +Y.
UP = Vector((0.0, 0.0, 1.0))
FORWARD = Vector((0.0, 1.0, 0.0))

_AXES = {"X": Vector((1.0, 0.0, 0.0)), "Y": FORWARD, "Z": UP}

# Blender world (x, y, z) -> PoseYdon Y-up (-x, z, y). Measured against the
# OBJ exporter and cross-checked against the BVH pipeline: forward.+Z = 1.0000
# and across-body vectors agree to 1.0000. The negated X is not a mirror --
# swapping two axes flips handedness and negating the third restores it.
TO_Y_UP = Matrix(((-1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0)))


def _to_y_up(vector: Vector) -> np.ndarray:
    """Blender world-space vector -> a PoseYdon Y-up ``(3,)`` array."""
    converted = TO_Y_UP @ vector
    return np.array((converted.x, converted.y, converted.z), dtype=np.float64)


class FBX:
    """An imported FBX scene: one armature, its skinned meshes, its animation."""

    def __init__(self, armature, meshes):
        self.armature = armature
        self.meshes = list(meshes)

    # ------------------------------------------------------------------ read

    @classmethod
    def read(cls, path: str | Path) -> FBX:
        """Import an FBX into a fresh scene.

        ``automatic_bone_orientation`` makes bones point at their children so
        the rig reads as a skeleton rather than a scatter of stubs; see the
        module docstring for why FBX needs this and BVH does not. It changes
        only how bones are DRAWN -- joint positions are identical either way.
        """
        bpy.ops.wm.read_factory_settings(use_empty=True)
        bpy.ops.import_scene.fbx(filepath=str(path), automatic_bone_orientation=True)
        armatures = [o for o in bpy.data.objects if o.type == "ARMATURE"]
        if not armatures:
            raise ValueError(f"{path}: no armature in this file")
        return cls(armatures[0], [o for o in bpy.data.objects if o.type == "MESH"])

    # -------------------------------------------------------------- geometry

    @property
    def joint_names(self) -> list[str]:
        """Bone names, in the armature's own order."""
        return [b.name for b in self.armature.data.bones]

    @property
    def joint_parents(self) -> list[int]:
        """Parent index per joint, ``-1`` for a root -- the BVH convention."""
        index = {name: i for i, name in enumerate(self.joint_names)}
        return [
            index[bone.parent.name] if bone.parent is not None else -1
            for bone in self.armature.data.bones
        ]

    @property
    def joint_offsets(self) -> np.ndarray:
        """``(J, 3)`` bind-pose parent-relative offsets, in PoseYdon's Y-up frame.

        FBX/Blender expose only absolute bind-pose joint positions, never an
        explicit offset block the way BVH does, so this derives one: each
        joint's offset is its bind position minus its parent's, and a root's
        offset is its own bind position.
        """
        names = self.joint_names
        parents = self.joint_parents
        positions = np.stack(
            [_to_y_up(self.joint_position(name, rest=True)) for name in names]
        )
        offsets = np.empty_like(positions)
        for joint, parent in enumerate(parents):
            offsets[joint] = (
                positions[joint] if parent < 0 else positions[joint] - positions[parent]
            )
        return offsets

    @property
    def action(self):
        data = self.armature.animation_data
        return data.action if data else None

    def frame_range(self) -> tuple[int, int]:
        action = self.action
        if action is None:
            return (bpy.context.scene.frame_current,) * 2
        return round(action.frame_range[0]), round(action.frame_range[1])

    def set_frame(self, frame: int) -> None:
        bpy.context.scene.frame_set(frame)
        bpy.context.view_layer.update()

    def joint_position(self, name: str, rest: bool = False) -> Vector:
        """World-space position of one joint, in Blender's frame."""
        bones = self.armature.data.bones
        if rest:
            return self.armature.matrix_world @ bones[name].head_local
        return self.armature.matrix_world @ self.armature.pose.bones[name].head

    # ---------------------------------------------------------------- facing

    def forward(self, facing_pairs, rest: bool = False) -> Vector:
        """The character's forward axis, by the rule in :mod:`poseydon.ingest.align`.

        ``across`` is the summed right-minus-left over the facing pairs, and
        forward is ``up x across`` -- so forward is always horizontal.
        """
        bones = self.armature.data.bones
        missing = [n for pair in facing_pairs for n in pair if n not in bones]
        if missing:
            raise ValueError(f"facing joints not in this rig: {', '.join(missing)}")

        across = Vector((0.0, 0.0, 0.0))
        for right, left in facing_pairs:
            across += self.joint_position(right, rest) - self.joint_position(left, rest)
        if across.length < 1e-8:
            raise ValueError("facing joints are coincident; forward is undefined")
        across.normalize()

        forward = UP.cross(across)
        if forward.length < 1e-8:
            raise ValueError("across-body axis is parallel to world up")
        return forward.normalized()

    def facing_rotation(self, facing_pairs, axis: str = "+Z", rest: bool = False) -> Matrix:
        """Rotation taking the character's forward onto ``axis``.

        A pure YAW about world up, never a shortest arc. Both vectors are
        horizontal, so a yaw is what is meant -- and it is the only form that
        stays defined when the character faces exactly BACKWARDS, where
        shortest-arc is degenerate and Blender picks an arbitrary
        perpendicular axis (for Trex it chose (0.707, 0, 0.707), tipping the
        animal out of the horizontal plane across all 72 of its files).
        """
        spec = str(axis).strip().upper()
        name = spec[1:] if spec[:1] in "+-" else spec
        if name not in _AXES:
            raise ValueError(f"target axis must be X, Y or Z with an optional sign, got `{axis}`")
        sign = -1.0 if spec.startswith("-") else 1.0
        # The caller names an axis in PoseYdon's Y-up frame; convert to Blender's.
        target = sign * (TO_Y_UP.transposed() @ _AXES[name])

        forward = self.forward(facing_pairs, rest)
        angle = math.atan2(forward.cross(target).dot(UP), forward.dot(target))
        return Matrix.Rotation(angle, 4, UP)

    def facing_dot(self, facing_pairs, rest: bool = False) -> float:
        """How well the forward axis agrees with +Z. 1.0 is exact."""
        return self.forward(facing_pairs, rest).dot(FORWARD)

    # -------------------------------------------------------------- rotating

    def rotate(self, rotation: Matrix) -> None:
        """Turn the whole rig: object transform first, then baked into the data.

        The object transform is what actually turns both the rest pose and the
        animation. Rotating the armature DATA instead does NOT work: Blender
        compensates the pose to keep the posed character where it was, so the
        rest pose turns and the animation stays put (measured: rest forward
        (1, 0, 0) -> (0.108, 0.994, 0) while the pose stayed at
        (0.994, 0.108, 0)).

        ``transform_apply`` then bakes that rotation down into the armature's
        bones and the meshes' vertices, returning the object transform to
        identity. That is not what stops the exported character being pinned
        in place -- Blender's FBX exporter bakes object transform curves
        whenever it bakes animation, no matter what, so the export gains
        ``location``/``rotation_euler``/``scale`` curves either way (an
        untouched import-export round trip of a source file gains exactly the
        same 9). What it does buy is that those baked curves are IDENTITY
        rather than a rotation, so the facing lives in the skeleton where it
        belongs, and a viewer who clears the object's animation gets a
        character that is still correctly faced.
        """
        self.armature.matrix_world = rotation @ self.armature.matrix_world
        for mesh in self.meshes:
            if mesh.parent is None:
                mesh.matrix_world = rotation @ mesh.matrix_world
        bpy.context.view_layer.update()

        bpy.ops.object.select_all(action="DESELECT")
        for obj in (self.armature, *self.meshes):
            obj.select_set(True)
        bpy.context.view_layer.objects.active = self.armature
        bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)
        bpy.context.view_layer.update()

    # ----------------------------------------------------------------- write

    def write(self, path: str | Path) -> None:
        """Export to FBX, with full animation fidelity and a movable object.

        Two export settings, both load-bearing, and they interact:

        ``bake_anim_simplify_factor`` controls the exporter's curve
        decimation. The default 1.0 is far too aggressive: it drops keys
        wherever it judges a curve simple, leaving an error of roughly
        constant size that reads as a visible tremor exactly where the motion
        is calmest -- on BrownBear's RiseSwat it raised frame-to-frame jitter
        over frames 110-140 to 1.49x the source while leaving the busy first
        100 frames untouched. But 0.0 is not the answer either: the exporter
        reads it as "keep every key of every curve", which includes the
        armature OBJECT's constant transform, and a keyframed object cannot
        be dragged in Blender -- move it and the next frame snaps it back.

        0.01 is the value that satisfies both. Bone curves keep their keys
        (jitter 1.00x, max positional error 0.00004 -- identical to 0.0),
        while the object's transform, whose every sample equals the previous
        one, is simplified away to nothing. That needs
        ``bake_anim_force_startend_keying=False`` as well, since the default
        True re-adds a start and end key to exactly those constant curves
        (which is why decimated files showed curves with 2 keyframes).

        Result: 0 object-level curves, so the armature is directly selectable
        and movable, with animation indistinguishable from the source.
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        bpy.ops.export_scene.fbx(
            filepath=str(path),
            add_leaf_bones=False,
            bake_anim_simplify_factor=0.01,
            bake_anim_force_startend_keying=False,
        )

    def bind_pose_arrays(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[str, ...]]:
        """Gather the character's BIND-pose mesh, skin weights and joint order.

        ``pose_position = 'REST'`` restores the bind pose, so this reads the
        undeformed character rather than whichever frame happened to be
        current -- an Armature modifier only ever deforms the EVALUATED mesh,
        never ``obj.data``, but ``pose_position`` still selects which pose
        matrices a later evaluation would use, so this keeps that consistent.
        Every mesh object on the armature is merged into one set of arrays,
        because a rig's skin is one character, not one per object. Vertices
        are converted to PoseYdon's Y-up, +Z-forward frame the same way
        :attr:`joint_offsets` is, so mesh and skeleton land in the same space.

        Returns ``(vertices, faces, weights, joints)``:

        * ``vertices`` -- ``(V, 3)`` float64 bind-pose vertex positions.
        * ``faces`` -- ``(F, 3)`` int32 triangle indices.
        * ``weights`` -- ``(V, J)`` float64 per-vertex weight per joint, each
          row normalized to sum to 1 (a vertex with no vertex-group weight at
          all is left as all zeros rather than divided by zero).
        * ``joints`` -- the ``J`` joint names ``weights`` is indexed by, in
          the same order as :attr:`joint_names`.
        """
        armature = self.armature
        previous = armature.data.pose_position
        armature.data.pose_position = "REST"
        bpy.context.view_layer.update()

        try:
            names = self.joint_names
            joint_index = {name: i for i, name in enumerate(names)}
            n_joints = len(names)

            all_vertices: list[np.ndarray] = []
            all_faces: list[np.ndarray] = []
            all_weights: list[np.ndarray] = []
            offset = 0
            for obj in self.meshes:
                mesh = obj.data
                if not mesh.vertices:
                    continue
                mesh.calc_loop_triangles()
                matrix = obj.matrix_world

                verts = np.stack([_to_y_up(matrix @ v.co) for v in mesh.vertices])
                all_vertices.append(verts)

                tris = np.array(
                    [list(tri.vertices) for tri in mesh.loop_triangles], dtype=np.int32
                ).reshape(-1, 3)
                all_faces.append(tris + offset)

                groups = obj.vertex_groups
                rows = np.zeros((len(mesh.vertices), n_joints), dtype=np.float64)
                for vi, vertex in enumerate(mesh.vertices):
                    for element in vertex.groups:
                        joint = joint_index.get(groups[element.group].name)
                        if joint is not None:
                            rows[vi, joint] = element.weight
                sums = rows.sum(axis=1)
                nonzero = sums > 1e-12
                rows[nonzero] /= sums[nonzero, None]
                all_weights.append(rows)

                offset += len(mesh.vertices)

            if not all_vertices:
                raise ValueError("this FBX has no mesh to gather a bind pose from")

            return (
                np.concatenate(all_vertices, axis=0),
                np.concatenate(all_faces, axis=0),
                np.concatenate(all_weights, axis=0),
                tuple(names),
            )
        finally:
            armature.data.pose_position = previous
            bpy.context.view_layer.update()

    def write_bind_pose_obj(self, path: str | Path) -> None:
        """Write the character's mesh, in its BIND pose, as a single OBJ.

        Shares its geometry with :meth:`write_mesh_npz` via
        :meth:`bind_pose_arrays`, so the two can never disagree. Kept for
        inspection -- the skin weights only survive in the npz.
        """
        vertices, faces, _weights, _joints = self.bind_pose_arrays()
        lines = [f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in vertices]
        lines += [f"f {a + 1} {b + 1} {c + 1}" for a, b, c in faces]
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text("\n".join(lines) + "\n")

    def write_mesh_npz(
        self, path: str | Path, target_mean_bone_length: float = HML_MEAN_BONE_LENGTH
    ) -> None:
        """Mesh, skinning and rest skeleton in one file.

        One load gives a consumer everything needed to drive the skin:
        vertices and faces, the per-vertex weight matrix and the joint
        ordering it is indexed by, and the rest skeleton those weights were
        bound against. Splitting them across an OBJ and something else
        guarantees they drift apart.

        FBX/Blender units are whatever the source file declares -- centimeters
        for most of this corpus's 3ds Max exports -- while the BVH pipeline's
        ``ScaleToMeanBoneLength`` rescales every rig so its mean bone length
        equals ``target_mean_bone_length``. Left unscaled, a mesh.npz would sit
        in a wholly different numeric range from its BVH counterpart (measured
        ratios of 0.01-0.6x across three rigs, not a constant conversion
        factor -- these corpora do not even share a unit convention).
        Rescaling here by that SAME target, computed from this rig's OWN bind
        pose the same way ``ScaleToMeanBoneLength`` computes it from a BVH
        T-pose -- including that stage's exclusion of zero-length bones from
        the mean, which this method must mirror exactly or the two computed
        factors disagree by a rig-dependent (J_all-1)/(J_real-1) -- is what
        makes the two corpora comparable at all; the agreement test in
        ``tests/build/test_bvh_fbx_agreement.py`` depends on it, confirmed to
        2e-5 of a bone length against Crab, the one sample rig whose FBX
        skeleton happens to have no zero-length bone (so the exclusion is a
        no-op for it and it would have agreed either way -- it is not general
        evidence the unfiltered mean was fine).
        """
        vertices, faces, weights, joints = self.bind_pose_arrays()
        offsets = self.joint_offsets
        # Bones at or below this tolerance are zero-length End-Site-equivalent
        # joints, not real bones; including them in the mean inflates the
        # factor. Matches ScaleToMeanBoneLength.tolerance in
        # poseydon.build.prepare, which this method must stay in step with.
        lengths = np.linalg.norm(offsets[1:], axis=-1)
        real_lengths = lengths[lengths > 1e-8]
        mean_length = float(real_lengths.mean()) if real_lengths.size else 0.0
        factor = target_mean_bone_length / mean_length if mean_length > 1e-12 else 1.0

        np.savez_compressed(
            Path(path),
            vertices=vertices * factor,
            faces=faces,
            skin_weights=weights,
            skin_joints=np.array(joints, dtype=object),
            joint_names=np.array(self.joint_names, dtype=object),
            joint_parents=np.asarray(self.joint_parents, dtype=np.int32),
            joint_offsets=np.asarray(offsets * factor, dtype=np.float64),
        )

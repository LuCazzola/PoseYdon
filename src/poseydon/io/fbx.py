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

import hashlib
import json
import math
from pathlib import Path

import numpy as np

from poseydon.core.animation import Animation
from poseydon.core.rotations import QUAT_IDENTITY, matrix_to_quat, quat_apply, quat_inverse, quat_mul, quat_to_matrix
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

#: Tail length, as a fraction of the rig's mean bone length, given to a bone
#: whose children are all coincident with it. See `FBX.from_animation`.
_STUB_FRACTION = 0.05


def _to_y_up(vector: Vector) -> np.ndarray:
    """Blender world-space vector -> a PoseYdon Y-up ``(3,)`` array."""
    converted = TO_Y_UP @ vector
    return np.array((converted.x, converted.y, converted.z), dtype=np.float64)


def _from_y_up(vector: np.ndarray) -> Vector:
    """A PoseYdon Y-up ``(3,)`` array -> a Blender world-space vector.

    The inverse of :func:`_to_y_up`. ``TO_Y_UP`` is a rotation (det +1, see its
    definition), so the inverse is the transpose.
    """
    return TO_Y_UP.transposed() @ Vector(
        (float(vector[0]), float(vector[1]), float(vector[2]))
    )


def _rest_positions(anim: Animation) -> np.ndarray:
    """``(J, 3)`` joint positions of ``anim`` with every rotation set to identity.

    The rest skeleton is ``translations`` at frame 0, NOT ``offsets``: forward
    kinematics walks translations, and after an inverse preparation the two are
    no longer the same array (`RestRelative.invert` restores the source's OFFSET
    block while rotating the translations back into the source's bind frame).
    `EnforceRigid` is what makes frame 0 representative -- every non-root
    translation is constant over time.

    In the REST-RELATIVE space this pose is the rig's T-pose, which is what
    `FBX.apply_animation` aligns against a bind pose. After
    `RestRelative.invert` it is instead the source's OFFSET-block pose, which is
    the frame `FBX.from_animation` builds its edit bones in.
    """
    local = anim.translations[0]
    positions = np.empty_like(local)
    for joint, parent in enumerate(int(p) for p in anim.parents):
        positions[joint] = (
            local[joint] if parent < 0 else positions[parent] + local[joint]
        )
    return positions


def _similarity(source: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Least-squares ``(scale, rotation, translation)`` taking ``source`` onto ``target``.

    Umeyama's closed form, with a UNIFORM scale. Fitting the transform rather
    than composing a known one is what lets `apply_animation` drive a rig whose
    units, axis convention and facing it knows nothing about: an FBX in
    centimeters and a BVH in whatever its exporter chose differ by a rotation, a
    factor and an offset, which is exactly what this recovers. It also absorbs
    the Y-up/Z-up change, so nothing on that path needs `TO_Y_UP`.

    Uniform, not per-axis, on purpose: a non-uniform fit would silently absorb a
    genuine skeleton disagreement into a stretch instead of reporting it, and
    the caller's residual check is the only thing standing between a stale rig
    and a mangled mesh.
    """
    source_mean, target_mean = source.mean(axis=0), target.mean(axis=0)
    centred_source, centred_target = source - source_mean, target - target_mean

    u, singular, vt = np.linalg.svd(centred_target.T @ centred_source / len(source))
    # Reflections are not rotations: if the raw solution flips handedness, the
    # smallest singular direction is the cheapest one to flip back.
    correction = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        correction[2, 2] = -1.0
    rotation = u @ correction @ vt

    variance = float((centred_source**2).sum() / len(source))
    scale = float((singular * np.diag(correction)).sum() / variance) if variance > 0 else 1.0
    return scale, rotation, target_mean - scale * (rotation @ source_mean)


def _quat_matrix(quaternion: np.ndarray) -> Matrix:
    """A scalar-last quaternion -> a Blender ``Matrix``, frame unchanged."""
    return Matrix([list(row) for row in quat_to_matrix(quaternion)])


def _rotation_to_blender(quaternion: np.ndarray) -> Matrix:
    """A PoseYdon Y-up world rotation -> the same rotation in Blender's frame.

    A change of basis, not a rotation of a vector: ``TO_Y_UP`` carries Blender
    coordinates into Y-up ones, so a map expressed in Y-up is conjugated by it
    to be expressed in Blender's.

    `FBX.apply_animation` deliberately does NOT use this: the change of basis it
    needs is the one `_similarity` fitted, which already contains this one.
    Applying both would rotate the character twice.
    """
    return TO_Y_UP.transposed() @ _quat_matrix(quaternion) @ TO_Y_UP


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

    def to_animation(self) -> Animation:
        """Sample the evaluated armature into PoseYdon's Y-up animation model.

        FBX files occasionally contain a locator/object root beside the
        character root.  PoseYdon's mathematical animation type requires one
        root, so those roots are joined under a deterministic synthetic root;
        single-root files retain their original names and hierarchy exactly.
        """
        source_names = tuple(self.joint_names)
        source_parents = np.asarray(self.joint_parents, dtype=np.int32)
        first, last = self.frame_range()
        frame_numbers = tuple(range(first, last + 1))
        n_source = len(source_names)
        positions = np.empty((len(frame_numbers), n_source, 3), dtype=np.float64)
        global_rotations = np.empty((len(frame_numbers), n_source, 4), dtype=np.float64)
        for sample, frame in enumerate(frame_numbers):
            self.set_frame(frame)
            for joint, name in enumerate(source_names):
                pose_matrix = self.armature.matrix_world @ self.armature.pose.bones[name].matrix
                positions[sample, joint] = _to_y_up(pose_matrix.translation)
                blender_rotation = pose_matrix.to_quaternion().to_matrix()
                y_up_rotation = TO_Y_UP @ blender_rotation @ TO_Y_UP.transposed()
                global_rotations[sample, joint] = matrix_to_quat(np.asarray(y_up_rotation))

        roots = np.flatnonzero(source_parents < 0).tolist()
        synthetic = len(roots) > 1
        if synthetic:
            names = ("__FBX_ROOT__",) + source_names
            parents = np.concatenate((np.array([-1], dtype=np.int32), source_parents + 1))
            for root in roots:
                parents[root + 1] = 0
            offsets = np.vstack((np.zeros((1, 3)), self.joint_offsets))
            positions = np.concatenate((np.zeros((len(frame_numbers), 1, 3)), positions), axis=1)
            global_rotations = np.concatenate(
                (np.tile(QUAT_IDENTITY, (len(frame_numbers), 1, 1)), global_rotations), axis=1
            )
        else:
            names = source_names
            parents = source_parents
            offsets = self.joint_offsets

        rotations = global_rotations.copy()
        translations = positions.copy()
        for joint, parent in enumerate(parents):
            if parent < 0:
                continue
            inverse_parent = quat_inverse(global_rotations[:, parent])
            rotations[:, joint] = quat_mul(inverse_parent, global_rotations[:, joint])
            translations[:, joint] = quat_apply(
                inverse_parent, positions[:, joint] - positions[:, parent]
            )
        return Animation(
            rotations=rotations, translations=translations,
            offsets=offsets, parents=parents, names=names,
            fps=float(bpy.context.scene.render.fps / bpy.context.scene.render.fps_base),
        )

    @classmethod
    def from_bare_animation(cls, anim: Animation, name: str = "Armature") -> FBX:
        """Build a keyed armature from an :class:`~poseydon.core.animation.Animation`.

        The counterpart to :meth:`poseydon.io.bvh.BVH.from_animation`, and what
        `scripts/anim_2_fbx.py` exports a generated clip through. The scene it
        leaves behind is a fresh one holding a single armature and no mesh, so
        :meth:`write` exports exactly the skeleton and its animation.

        **Childless joints do not become bones.** A BVH End Site is a childless
        joint with an offset, synthesized under every real leaf on read; FBX has
        no such concept, so writing one would hand the user joints their own
        asset never had. They are consumed as their parent's TAIL direction
        instead, which is the only thing an End Site's offset ever meant, and
        leaves the bone count equal to `build/pipeline.py::_n_real_joints` --
        the count that module already defines as "the count an FBX skeleton is
        comparable against".

        **Bones are keyed through `matrix_basis`, not `matrix`.** The world-space
        `pose_bone.matrix` setter needs the parent's pose already evaluated, so
        driving it means a `view_layer.update()` between every bone of every
        frame -- a full depsgraph evaluation some ten thousand times for a
        200-frame clip. The basis is the same quantity algebraically,

            basis = rest^-1 @ parent_rest @ parent_world^-1 @ world

        computed from static rest matrices and world transforms this module
        already has, so no evaluation is needed at all.

        Rotations are carried through world space rather than joint-locally, for
        the reason `scripts/process_dataset_truebones_fbx.py` states about the
        two corpora: BVH and FBX each express a joint's rotation in its own
        local bone frame, differing by a constant per-joint offset. World space
        is where they are the same thing.
        """
        parents = [int(p) for p in anim.parents]
        children: list[list[int]] = [[] for _ in parents]
        for joint, parent in enumerate(parents):
            if parent >= 0:
                children[parent].append(joint)

        bones = [joint for joint in range(anim.n_joints) if children[joint]]
        if not bones:
            raise ValueError(
                "this animation has no joint with a child, so there is no bone to "
                "build; an armature cannot be made from a single point"
            )

        local = anim.translations[0]
        rest = _rest_positions(anim)
        head = [_from_y_up(rest[joint]) for joint in range(anim.n_joints)]

        # Blender deletes an edit bone of zero length, and a rig that reached
        # here through `invert_reduction` is full of coincident joints -- every
        # one the reduction removed had a zero offset by definition. Such a bone
        # gets a stub tail instead: its length is cosmetic (the module docstring
        # says as much about leaf bones), while its HEAD is the joint position
        # and is exact either way.
        lengths = np.linalg.norm(local[1:], axis=-1)
        real = lengths[lengths > 1e-8]
        stub = max(float(real.mean()) * _STUB_FRACTION, 1e-4) if real.size else 1e-4

        def tail_of(joint: int) -> Vector:
            for child in children[joint]:
                if (head[child] - head[joint]).length > 1e-8:
                    return head[child]
            # Every child sits exactly on top of this joint, so the data names
            # no direction. Continue this joint's own bone, or world up at a
            # root that has none.
            parent = parents[joint]
            along = head[joint] - head[parent] if parent >= 0 else Vector((0.0, 0.0, 0.0))
            if along.length <= 1e-8:
                along = UP
            return head[joint] + along.normalized() * stub

        bpy.ops.wm.read_factory_settings(use_empty=True)
        scene = bpy.context.scene
        scene.render.fps = max(1, round(anim.fps))
        scene.frame_start = 1
        scene.frame_end = anim.n_frames

        armature = bpy.data.armatures.new(name)
        obj = bpy.data.objects.new(name, armature)
        scene.collection.objects.link(obj)
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)

        bpy.ops.object.mode_set(mode="EDIT")
        for joint in bones:
            bone = armature.edit_bones.new(anim.names[joint])
            if bone.name != anim.names[joint]:
                raise ValueError(
                    f"Blender renamed joint `{anim.names[joint]}` to `{bone.name}`; "
                    "the exported rig would not carry the names the source did"
                )
            bone.head = head[joint]
            bone.tail = tail_of(joint)
        for joint in bones:
            # A bone's parent always has a child -- this one -- so it is itself
            # a bone. No ancestor search is needed.
            if parents[joint] >= 0:
                armature.edit_bones[anim.names[joint]].parent = armature.edit_bones[
                    anim.names[parents[joint]]
                ]
        bpy.ops.object.mode_set(mode="OBJECT")

        positions, rotations = anim.global_transforms()
        rest_matrix = {
            joint: armature.bones[anim.names[joint]].matrix_local.copy()
            for joint in bones
        }
        inverse_rest = {joint: matrix.inverted() for joint, matrix in rest_matrix.items()}
        pose = {joint: obj.pose.bones[anim.names[joint]] for joint in bones}
        for bone in pose.values():
            bone.rotation_mode = "QUATERNION"

        for frame in range(anim.n_frames):
            world = {
                joint: Matrix.Translation(_from_y_up(positions[frame, joint]))
                @ _rotation_to_blender(rotations[frame, joint]).to_4x4()
                for joint in bones
            }
            for joint in bones:
                parent = parents[joint]
                if parent < 0:
                    basis = inverse_rest[joint] @ world[joint]
                else:
                    basis = (
                        inverse_rest[joint]
                        @ rest_matrix[parent]
                        @ world[parent].inverted()
                        @ world[joint]
                    )
                bone = pose[joint]
                bone.matrix_basis = basis
                bone.keyframe_insert("location", frame=frame + 1)
                bone.keyframe_insert("rotation_quaternion", frame=frame + 1)

        bpy.context.view_layer.update()
        return cls(obj, [])

    @classmethod
    def from_animation(cls, anim: Animation, name: str = "Armature") -> FBX:
        return cls.from_bare_animation(anim, name=name)

    def align_animation_to_bind(self, anim: Animation) -> Animation:
        """Reframe source-space motion onto this imported bind pose.

        Original export must preserve the vendor bind pose even when paired BVH
        and FBX files have small rest-geometry discrepancies.  We therefore
        map frame-to-frame displacements through the best uniform similarity,
        anchor frame zero exactly on the FBX bind heads, and express rotations
        as deltas from the vendor bind orientation.
        """
        names = tuple(name for name in anim.names if name in self.armature.data.bones)
        if not names:
            raise ValueError("animation has no joints in the FBX armature")
        indices = [anim.names.index(name) for name in names]
        source_positions, source_rotations = anim.global_transforms()
        source_positions = source_positions[:, indices]
        source_rotations = source_rotations[:, indices]
        target_positions = np.stack([
            _to_y_up(self.joint_position(name, rest=True)) for name in names
        ])
        scale, rotation, _offset = _similarity(source_positions[0], target_positions)
        mapped = target_positions[None] + scale * (
            (source_positions - source_positions[0]) @ rotation.T
        )
        rotation_matrix = Matrix([list(row) for row in rotation])
        mapped_global_rotations = np.empty_like(source_rotations)
        for frame in range(anim.n_frames):
            for joint in range(len(names)):
                delta = quat_mul(
                    quat_inverse(source_rotations[0, joint]), source_rotations[frame, joint]
                )
                mapped_global_rotations[frame, joint] = matrix_to_quat(
                    np.asarray(rotation_matrix @ _quat_matrix(delta) @ rotation_matrix.transposed())
                )
        parent_indices = []
        name_to_local = {name: i for i, name in enumerate(names)}
        for name in names:
            parent = self.armature.data.bones[name].parent
            parent_indices.append(-1 if parent is None or parent.name not in name_to_local else name_to_local[parent.name])
        parents = np.asarray(parent_indices, dtype=np.int32)
        local_rotations = mapped_global_rotations.copy()
        translations = mapped.copy()
        offsets = np.empty((len(names), 3), dtype=np.float64)
        for joint, parent in enumerate(parents):
            if parent < 0:
                offsets[joint] = mapped[0, joint]
                continue
            inverse_parent = quat_inverse(mapped_global_rotations[:, parent])
            local_rotations[:, joint] = quat_mul(inverse_parent, mapped_global_rotations[:, joint])
            translations[:, joint] = quat_apply(
                inverse_parent, mapped[:, joint] - mapped[:, parent]
            )
            offsets[joint] = mapped[0, joint] - mapped[0, parent]
        return Animation(
            rotations=local_rotations, translations=translations,
            offsets=offsets, parents=parents, names=names, fps=anim.fps,
        )

    def apply_animation(self, anim: Animation, max_residual: float = 0.05, allow_undriven: bool = False) -> float:
        """Replace this scene's animation with ``anim``, keeping its mesh and skin.

        The skinned counterpart to :meth:`from_animation`. That one builds a bare
        armature and so returns a skeleton; this one drives the armature the user
        already has -- mesh, skin weights, materials and all -- so what comes out
        of :meth:`write` afterwards is their own asset performing generated
        motion.

        ``anim`` must be REST-RELATIVE: identity rotation on every joint has to
        reproduce the rig's T-pose, which is what `RestRelative` establishes and
        what an FBX bind pose already is ("the FBX bind pose IS the T-pose: 0.00
        degrees median", `scripts/process_dataset_truebones_fbx.py`). That shared
        meaning of "zero" is the whole reason the two can be married at all:
        joint rotations become deltas from the bind, and neither side's local
        bone-frame convention -- which genuinely differ, by a constant per-joint
        offset -- ever has to be converted.

        Everything else is FITTED rather than assumed. ``anim``'s T-pose is
        aligned onto this rig's bind pose by :func:`_similarity`, recovering the
        rotation, uniform scale and translation between them, so units (Scorpion:
        a factor of 100.2783), axis convention and facing need no agreement in
        advance and no recorded parameters. Joints are matched BY NAME, and only
        those this armature has are driven -- a BVH's End Sites have no bone here
        and are not expected to.

        Returns the fit residual as a fraction of this rig's mean bone length,
        and raises above ``max_residual``: past a point the two are not the same
        skeleton, and draping a skin over the difference deforms the character
        rather than animating it. Measured at 1.15e-06 for Scorpion.
        """
        bones = self.armature.data.bones
        driven = [name for name in anim.names if name in bones]
        missing = [bone.name for bone in bones if bone.name not in set(anim.names)]
        if missing and allow_undriven:
            weighted = [name for name in missing if self.vertex_group_weight(name) > 1e-12]
            if weighted:
                raise ValueError(
                    f"animation does not drive weighted armature bone(s): {', '.join(weighted[:3])}"
                )
            missing = []
        if missing:
            raise ValueError(
                f"this armature has {len(missing)} bone(s) the animation does not "
                f"drive ({', '.join(missing[:3])}), so they would stay in the bind "
                "pose while the rest of the character moves"
            )
        index_of = {name: i for i, name in enumerate(anim.names)}

        # Bind heads in ARMATURE-OBJECT space, which is the space `matrix_local`
        # and `pose_bone.matrix` both live in. Doing the whole fit there means
        # the armature object's own transform never enters it.
        bind = {name: bones[name].matrix_local.copy() for name in driven}
        target = np.array([tuple(bind[name].translation) for name in driven])
        source = _rest_positions(anim)[[index_of[name] for name in driven]]
        scale, rotation, offset = _similarity(source, target)

        lengths = np.array(
            [
                (bind[name].translation - bind[bones[name].parent.name].translation).length
                for name in driven
                if bones[name].parent is not None and bones[name].parent.name in bind
            ]
        )
        real = lengths[lengths > 1e-8]
        bone_length = float(real.mean()) if real.size else 1.0
        residual = float(
            np.abs(scale * (source @ rotation.T) + offset - target).max() / bone_length
        )
        if residual > max_residual:
            raise ValueError(
                f"this rig's bind pose and the animation's T-pose disagree by "
                f"{residual:.3g} bone lengths, past the {max_residual:g} bound -- "
                "they are not the same skeleton, and skinning across the "
                "difference would deform the character rather than pose it"
            )

        positions, rotations = anim.global_transforms()
        # Both mapped through the fitted similarity: a point by `s R x + t`, and
        # a rotation by conjugation, which is what a change of basis does to one.
        posed = scale * (positions @ rotation.T) + offset
        blender_rotation = Matrix([list(row) for row in rotation])

        scene = bpy.context.scene
        scene.render.fps = max(1, round(anim.fps))
        scene.frame_start = 1
        scene.frame_end = anim.n_frames
        # The imported action is this clip's ORIGINAL motion. Left in place, its
        # keys survive on every channel the loop below does not write.
        if self.armature.animation_data is not None:
            self.armature.animation_data_clear()
        self.armature.animation_data_create()

        pose = {name: self.armature.pose.bones[name] for name in driven}
        for bone in pose.values():
            bone.rotation_mode = "QUATERNION"
        inverse_bind = {name: bind[name].inverted() for name in driven}

        for frame in range(anim.n_frames):
            world = {}
            for name in driven:
                joint = index_of[name]
                delta = (
                    blender_rotation
                    @ _quat_matrix(rotations[frame, joint])
                    @ blender_rotation.transposed()
                )
                world[name] = Matrix.Translation(
                    Vector(tuple(posed[frame, joint]))
                ) @ (delta @ bind[name].to_quaternion().to_matrix()).to_4x4()
            for name in driven:
                parent = bones[name].parent
                if parent is None or parent.name not in world:
                    basis = inverse_bind[name] @ world[name]
                else:
                    basis = (
                        inverse_bind[name]
                        @ bind[parent.name]
                        @ world[parent.name].inverted()
                        @ world[name]
                    )
                bone = pose[name]
                bone.matrix_basis = basis
                bone.keyframe_insert("location", frame=frame + 1)
                bone.keyframe_insert("rotation_quaternion", frame=frame + 1)

        bpy.context.view_layer.update()
        return residual

    def clear_animation(self) -> None:
        if self.armature.animation_data:
            self.armature.animation_data_clear()

    def apply_scene_transform(self, transform: Matrix) -> None:
        for obj in (self.armature, *self.meshes):
            obj.matrix_world = transform @ obj.matrix_world
        bpy.context.view_layer.update()
        bpy.ops.object.select_all(action="DESELECT")
        for obj in (self.armature, *self.meshes):
            obj.select_set(True)
        bpy.context.view_layer.objects.active = self.armature
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
        bpy.context.view_layer.update()

    def prepare_model_topology(self, names, parents, skinning_plan=None) -> None:
        """Reduce the imported rig to an exact model topology.

        Bones are removed only after their surviving children are explicitly
        reparented to the supplied model hierarchy.  Vertex-group influence is
        transferred according to the skinning plan (or to the removed bone's
        source parent for the legacy zero-offset reduction) before obsolete
        groups are deleted, so reducing a rig never silently drops mesh
        deformation.
        """
        names = tuple(names)
        parents = tuple(int(parent) for parent in parents)
        keep = set(names)
        source_parents = {name: parent for name, parent in zip(self.joint_names, self.joint_parents)}
        decisions = {
            decision.joint: decision.destination
            for decision in (getattr(skinning_plan, "decisions", ()) or ())
        }
        if self.armature.mode != "EDIT":
            bpy.context.view_layer.objects.active = self.armature
            self.armature.select_set(True)
            bpy.ops.object.mode_set(mode="EDIT")
        bones = self.armature.data.edit_bones
        missing = [name for name in names if name not in bones]
        if missing:
            bpy.ops.object.mode_set(mode="OBJECT")
            raise ValueError(f"model topology joints missing from FBX: {', '.join(missing)}")
        for bone in list(bones):
            if bone.name not in keep:
                bones.remove(bone)
        for name, parent_index in zip(names, parents):
            bones[name].parent = None if parent_index < 0 else bones[names[parent_index]]
        bpy.ops.object.mode_set(mode="OBJECT")

        # Transfer removed influences before deleting their groups.  The
        # zero-offset reduction makes parent transfer the exact deformation
        # preserving operation; an explicit SkinningPlan can override it.
        for mesh in self.meshes:
            groups = {group.name: group for group in mesh.vertex_groups}
            for removed, source_parent in source_parents.items():
                if removed in keep or removed not in groups:
                    continue
                destination = decisions.get(removed, source_parent)
                while destination is not None and destination not in keep:
                    destination = source_parents.get(destination)
                if destination is not None:
                    target = groups.get(destination) or mesh.vertex_groups.new(name=destination)
                    source = groups[removed]
                    for vertex in mesh.data.vertices:
                        try:
                            weight = source.weight(vertex.index)
                        except RuntimeError:
                            continue
                        if weight:
                            try:
                                old = target.weight(vertex.index)
                            except RuntimeError:
                                old = 0.0
                            target.add([vertex.index], old + weight, "REPLACE")
                mesh.vertex_groups.remove(groups[removed])

    def validate_topology(self, names, parents, offsets=None) -> None:
        actual_names = tuple(self.joint_names)
        if actual_names != tuple(names):
            raise ValueError(f"model topology names differ: expected {tuple(names)!r}, got {actual_names!r}")
        actual_parents = tuple(self.joint_parents)
        if actual_parents != tuple(parents):
            raise ValueError(f"model topology parents differ: expected {tuple(parents)!r}, got {actual_parents!r}")
        if offsets is not None:
            expected = np.asarray(offsets)
            # BVH's root OFFSET is a channel/rest-pose declaration, while an
            # FBX root has no parent-relative offset and its bind head is the
            # armature origin.  Compare the actual parent-relative geometry
            # for all non-root joints; root placement is validated by the
            # canonical scene transform itself.
            if expected.shape != self.joint_offsets.shape or not np.allclose(
                self.joint_offsets[1:], expected[1:], atol=1e-4
            ):
                raise ValueError("model topology offsets differ from the prepared skeleton")

    def vertex_group_weight(self, name: str) -> float:
        values = []
        for mesh in self.meshes:
            group = mesh.vertex_groups.get(name)
            if group is None:
                continue
            for vertex in mesh.data.vertices:
                try:
                    values.append(group.weight(vertex.index))
                except RuntimeError:
                    pass
        return max(values, default=0.0)

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

    def scale_to_mean_bone_length(self, target: float = HML_MEAN_BONE_LENGTH) -> float:
        """Uniformly scale the rig so its BIND-pose mean bone length equals ``target``.

        Modelled on :meth:`rotate`, for the same reason: the object transform
        is what actually moves the rest pose and the animation together, and
        ``transform_apply`` then bakes the factor into the armature's bones
        and the meshes' vertices rather than leaving it on the object
        transform, which is what a viewer sees and what the exporter would
        otherwise re-derive as a scale curve.

        Without this, ``write`` exports the clip in raw Blender/FBX units
        (centimeters for most of this corpus) while ``write_mesh_npz``
        rescales the mesh to ``target`` -- so a rig's exported ``.fbx`` clips
        and its ``mesh.npz`` sit in different unit systems, and both disagree
        with the BVH pipeline's own ``ScaleToMeanBoneLength``, which scales
        every BVH clip to the same target. BVH is the primary corpus; this
        brings FBX into the same units rather than the reverse.

        The factor is computed from the bind pose (:attr:`joint_offsets`),
        excluding zero-length bones the same way ``ScaleToMeanBoneLength``
        and :meth:`write_mesh_npz` do -- so a rig scaled once here and then
        fed through ``write_mesh_npz`` gets a second factor of ~1.0, not a
        double scale.

        Returns the factor applied, ``1.0`` if the bind pose has no
        non-degenerate bone.
        """
        offsets = self.joint_offsets
        lengths = np.linalg.norm(offsets[1:], axis=-1)
        real_lengths = lengths[lengths > 1e-8]
        mean_length = float(real_lengths.mean()) if real_lengths.size else 0.0
        factor = target / mean_length if mean_length > 1e-12 else 1.0

        scale = Matrix.Scale(factor, 4)
        self.armature.matrix_world = scale @ self.armature.matrix_world
        for mesh in self.meshes:
            if mesh.parent is None:
                mesh.matrix_world = scale @ mesh.matrix_world
        bpy.context.view_layer.update()

        bpy.ops.object.select_all(action="DESELECT")
        for obj in (self.armature, *self.meshes):
            obj.select_set(True)
        bpy.context.view_layer.objects.active = self.armature
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        bpy.context.view_layer.update()

        return factor

    # ----------------------------------------------------------------- write

    def write(self, path: str | Path, embed_textures: bool = False) -> None:
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

        ``embed_textures`` packs the image files into the FBX itself. The
        default `path_mode="AUTO"` writes a texture path RELATIVE to the output
        file, and that path is wrong whenever the export does not sit beside the
        source: exporting Scorpion to ``out/`` rewrote
        ``/app/data/_raw/truebones/source/Scorpion/tex/radscorpion.JPG`` as
        ``out/app/data/.../radscorpion.JPG``, which resolves to nothing, so the
        character loads with no texture and renders MAGENTA -- in a preview
        render and equally in whatever tool the file was handed to.

        Off by default all the same, because the two callers want opposite
        things. `scripts/process_dataset_truebones_fbx.py` writes one FBX per
        clip of 73 rigs into the corpus, and packing a rig's textures into every
        one of its clips multiplies them by the clip count for files that are
        intermediate artefacts nothing textures. A single user-facing export
        (`scripts/anim_2_fbx.py --skin`) wants to be self-contained and pays the
        size once.
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        bpy.ops.export_scene.fbx(
            filepath=str(path),
            add_leaf_bones=False,
            bake_anim_simplify_factor=0.01,
            bake_anim_force_startend_keying=False,
            **(
                {"path_mode": "COPY", "embed_textures": True}
                if embed_textures
                else {}
            ),
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

"""An FBX -> an MP4 of the character performing it.

The skinned counterpart to `poseydon.io.render.render_skeleton`, which draws a
stick figure with matplotlib. This one builds a Blender scene around whatever
the file contains -- a skinned character from `scripts/anim_2_fbx.py --skin`, a
bare armature from the same script without it, or any asset of your own -- and
renders it.

    docker compose run --rm fbx blender -b --python scripts/render_fbx.py -- \\
        out/Flamingo_onelegbent__to__Scorpion_skinned.fbx --out out/scorpion.mp4

It takes an FBX rather than a generated `.npz` on purpose: the two scripts then
compose, and this one is useful on files PoseYdon never produced.

**The shot is the same shot.** Camera placement comes from
`poseydon.io.framing.camera` -- the very function the stick-figure renderer
uses, which is why that module exists -- so `--zoom`, `--elev` and `--azim` mean
here exactly what they mean there, and the two renders of one clip are directly
comparable. The camera FOLLOWS the character at a fixed reach rather than
framing the whole trajectory, and the floor is a checkerboard laid out in world
space so travel is visible; both are that module's decisions, not this one's.

**EEVEE, always.** This is a verification tool -- its job is to show that a
mesh follows its skeleton coherently -- and a path tracer is the wrong price for
that. EEVEE rasterizes instead, on llvmpipe (Mesa's software rasterizer), which
is why the `fbx` image installs `libegl1` and `libgl1-mesa-dri`: without a GL
context Blender aborts the whole process with `Couldn't open libEGL.so.1`
before drawing a frame. There is deliberately no Cycles fallback -- an engine
that silently swaps itself for a path tracer when something is missing is how a
preview ends up eating every core on the machine.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

from poseydon.io.framing import GROUND_MARGIN, TILE, camera

#: Matches `render_skeleton`'s defaults, so the skinned render and the stick
#: figure are the same shot of the same clip.
DEFAULT_ELEV = 14.0
DEFAULT_AZIM = -70.0
#: Slightly wider than `render_skeleton`'s 1.15. A stick figure is a few lines
#: and wants filling the frame; a skinned character has bulk, and a tail or a
#: claw swinging out of shot is exactly the thing this render exists to catch.
DEFAULT_ZOOM = 0.95

#: Sun elevation and bearing, in degrees. Placed high and off to one side of
#: the camera rather than on its axis: a light on the camera axis flattens the
#: character, since every surface the camera sees is lit head-on.
SUN_ELEV = 50.0
SUN_AZIM_OFFSET = 35.0

#: The two floor tones. Deliberately faint and close together, the same
#: judgement `TILE_COLOURS` records for the matplotlib floor: the ground is a
#: motion cue, not something to read. Kept clearly darker than the white
#: backdrop so the character still has a surface to sit on rather than floating.
TILE_TONES = ((0.55, 0.55, 0.58, 1.0), (0.72, 0.72, 0.75, 1.0))

#: Ambient fill from the world. The backdrop is composited white rather than
#: lit white (see `configure_render`), so this is free to be whatever reads
#: best on the character -- and against a white field it has to be reasonably
#: bright, or the subject renders as a silhouette.
WORLD_LIGHT = (0.28, 0.28, 0.30, 1.0)

#: The backdrop, composited behind the render.
BACKDROP = (1.0, 1.0, 1.0, 1.0)



class _Scene:
    """The imported FBX as Blender holds it: one armature, its meshes, its action.

    Rendering is the one FBX job left on Blender, and only as a renderer:
    reading and writing go through `poseydon.io.fbx` (the Autodesk SDK).
    """

    def __init__(self, path: Path):
        import bpy

        bpy.ops.wm.read_factory_settings(use_empty=True)
        bpy.ops.import_scene.fbx(filepath=str(path), automatic_bone_orientation=True)
        armatures = [o for o in bpy.data.objects if o.type == "ARMATURE"]
        if not armatures:
            raise ValueError(f"{path}: no armature in this file")
        self.armature = armatures[0]
        self.meshes = [o for o in bpy.data.objects if o.type == "MESH"]

    @property
    def joint_names(self) -> list[str]:
        return [b.name for b in self.armature.data.bones]

    def frame_range(self) -> tuple[int, int]:
        import bpy

        data = self.armature.animation_data
        if data is None or data.action is None:
            return (bpy.context.scene.frame_current,) * 2
        return round(data.action.frame_range[0]), round(data.action.frame_range[1])

    def set_frame(self, frame: int) -> None:
        import bpy

        bpy.context.scene.frame_set(frame)
        bpy.context.view_layer.update()

    def joint_position(self, name: str, rest: bool = False):
        if rest:
            return self.armature.matrix_world @ self.armature.data.bones[name].head_local
        return self.armature.matrix_world @ self.armature.pose.bones[name].head

    def forward(self, facing_pairs, rest: bool = False):
        """Horizontal forward, ``up x (right - left)`` summed over the pairs."""
        from mathutils import Vector

        across = Vector((0.0, 0.0, 0.0))
        for right, left in facing_pairs:
            across += self.joint_position(right, rest) - self.joint_position(left, rest)
        return Vector((0.0, 0.0, 1.0)).cross(across.normalized()).normalized()


def joint_paths(scene) -> np.ndarray:
    """``(F, J, 3)`` world joint positions over the scene's frame range.

    In Blender's own frame, not PoseYdon's: the file is already in it, the
    camera being placed is a Blender object, and `poseydon.io.framing` only
    asks that the vertical axis be the third -- which is true of both.
    """
    start, end = scene.frame_range()
    names = scene.joint_names
    frames = []
    for frame in range(start, end + 1):
        scene.set_frame(frame)
        frames.append([tuple(scene.joint_position(name)) for name in names])
    return np.array(frames, dtype=np.float64)


def facing_pairs(corpus: Path, scene, rig: str | None = None):
    """The rig's right/left facing pairs, or ``None`` if they cannot be resolved.

    Needed because ``--azim`` has to be measured from the CHARACTER's forward,
    not from a world axis. A clip exported with `--skin` is aligned onto its
    source asset's own bind pose, and Truebones assets are authored facing
    whichever way their exporter chose -- so a fixed world azimuth shows one rig
    its front and the next one its back.

    The rig is identified by its facing joints when not named: those are
    specific enough that a rig whose skeleton carries all of another's is not a
    thing this corpus contains, and an ambiguous match returns ``None`` rather
    than guessing. Cheap enough to do over every manifest -- they are small YAML
    and there are 73.
    """
    from poseydon.core.skeleton import SkeletonManifest

    bones = set(scene.joint_names)

    def pairs_of(name: str):
        manifest = SkeletonManifest.load(corpus / "rigs" / name / "manifest.yaml")
        return [(pair.right, pair.left) for pair in manifest.facing]

    if rig is not None:
        return pairs_of(rig)

    matches = []
    for path in sorted((corpus / "rigs").glob("*/manifest.yaml")):
        try:
            pairs = pairs_of(path.parent.name)
        except Exception as error:  # noqa: BLE001 - one bad manifest is not fatal
            print(f"note: skipping {path.parent.name}: {error}", file=sys.stderr)
            continue
        if all(joint in bones for pair in pairs for joint in pair):
            matches.append(pairs)
    return matches[0] if len(matches) == 1 else None


def _orbit(elev: float, azim: float, distance: float):
    """A camera offset and the Euler rotation that aims it back at the origin.

    ``elev``/``azim`` are matplotlib's, which is what `render_skeleton` takes,
    so the two renderers are steered by one pair of numbers.
    """
    from mathutils import Vector

    theta, phi = math.radians(elev), math.radians(azim)
    offset = Vector(
        (
            distance * math.cos(theta) * math.cos(phi),
            distance * math.cos(theta) * math.sin(phi),
            distance * math.sin(theta),
        )
    )
    # Blender's camera looks down its own -Z with +Y up, so aiming it is a
    # rotation from that axis onto the direction back to the target.
    rotation = (-offset).to_track_quat("-Z", "Y").to_euler()
    return offset, rotation


def world_azimuth(fbx, azim: float, pairs) -> float:
    """``azim`` measured from the character's forward, as a world angle.

    `render_skeleton` draws motion that faces +Z, which its `to_view` rotation
    puts at -Y in the drawing frame -- world angle -90 degrees. So its `azim` is
    really an angle RELATIVE to the character, offset by 90, and that relative
    angle is the thing worth preserving: the default -70 is a near-frontal
    three-quarter view of the subject, whichever way the subject happens to
    point. With no facing pairs to resolve, ``azim`` stays a world angle, which
    is the old behaviour and is right for an asset this corpus knows nothing
    about.
    """
    if pairs is None:
        return azim
    forward = fbx.forward(pairs, rest=True)
    return math.degrees(math.atan2(forward.y, forward.x)) + azim + 90.0


def build_scene(
    fbx,
    zoom: float = DEFAULT_ZOOM,
    elev: float = DEFAULT_ELEV,
    azim: float = DEFAULT_AZIM,
) -> None:
    """Add a following camera, a sun and a checkerboard floor around ``fbx``.

    ``azim`` is a WORLD angle here; `world_azimuth` is what turns the
    character-relative one the CLI takes into it.
    """
    import bpy
    from mathutils import Vector

    scene = bpy.context.scene
    view = joint_paths(fbx)
    track, reach, reach_z, floor = camera(view, zoom)

    # A camera parented to an empty that rides the smoothed path: the empty
    # carries the follow, the camera holds a fixed offset, so the framing is
    # decided once rather than re-derived every frame.
    target = bpy.data.objects.new("CameraTarget", None)
    scene.collection.objects.link(target)

    # Far enough back that the whole `reach` cube fits the narrower of the two
    # field-of-view angles; `reach_z` is its own quantity (see `camera`), so the
    # half-extent that must fit is whichever is larger.
    lens = bpy.data.cameras.new("Camera")
    half = max(reach, reach_z)
    distance = half / math.tan(lens.angle / 2.0) + half
    offset, rotation = _orbit(elev, azim, distance)

    eye = bpy.data.objects.new("Camera", lens)
    eye.location = offset
    eye.rotation_euler = rotation
    eye.parent = target
    scene.collection.objects.link(eye)
    scene.camera = eye

    for frame, centre in enumerate(track, start=scene.frame_start):
        # Vertically the camera holds the clip's middle rather than following:
        # tracking height would cancel every jump, which is the motion.
        target.location = Vector((centre[0], centre[1], floor + reach_z))
        target.keyframe_insert("location", frame=frame)

    sun_data = bpy.data.lights.new("Sun", type="SUN")
    sun_data.energy = 4.0
    sun_data.angle = math.radians(3.0)  # a soft edge on the shadows
    sun = bpy.data.objects.new("Sun", sun_data)
    sun.location, sun.rotation_euler = _orbit(
        SUN_ELEV, azim + SUN_AZIM_OFFSET, distance
    )
    sun.parent = target
    scene.collection.objects.link(sun)

    world = bpy.data.worlds.new("World")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs[0].default_value = WORLD_LIGHT
    scene.world = world

    _add_floor(view, reach, floor)


def _add_floor(view: np.ndarray, reach: float, floor: float) -> None:
    """A checkerboard plane covering everywhere the character goes, plus margin.

    Sized from the PATH, not from the camera, and textured in world coordinates
    -- so the tiles stay put on the ground while the view moves over them. With
    a camera that follows, a floor that travelled with it would give no cue that
    anything is moving at all; `poseydon.io.framing._ground` states the same
    reasoning for the matplotlib floor, and `TILE`/`GROUND_MARGIN` are its
    numbers, used here so the two renders lay out the same tiles.
    """
    import bpy

    tile = max(reach * TILE, 1e-6)
    flat = view.reshape(-1, 3)[:, :2]
    low = flat.min(axis=0) - tile * GROUND_MARGIN
    high = flat.max(axis=0) + tile * GROUND_MARGIN
    centre = (low + high) / 2.0
    size = float(max(high - low))

    bpy.ops.mesh.primitive_plane_add(
        size=size, location=(centre[0], centre[1], floor)
    )
    plane = bpy.context.object
    plane.name = "Ground"

    material = bpy.data.materials.new("Checker")
    material.use_nodes = True
    nodes, links = material.node_tree.nodes, material.node_tree.links
    shader = nodes["Principled BSDF"]
    shader.inputs["Roughness"].default_value = 0.9
    # Texture coordinates in OBJECT space, scaled so one checker square is one
    # `tile`: the generated coordinates a plane carries are normalized to its
    # own size, which would rescale the tiles with the character's range.
    coordinate = nodes.new("ShaderNodeTexCoord")
    mapping = nodes.new("ShaderNodeMapping")
    # One checker square per `tile` of WORLD space. Blender's checker alternates
    # on integer boundaries of its input, so the scale is 1/tile -- not
    # size/tile, which is a dimensionless ratio and tiled the floor about as
    # finely as the character was large.
    mapping.inputs["Scale"].default_value = (1.0 / tile,) * 3
    checker = nodes.new("ShaderNodeTexChecker")
    checker.inputs["Color1"].default_value = TILE_TONES[0]
    checker.inputs["Color2"].default_value = TILE_TONES[1]
    checker.inputs["Scale"].default_value = 1.0
    links.new(coordinate.outputs["Object"], mapping.inputs["Vector"])
    links.new(mapping.outputs["Vector"], checker.inputs["Vector"])
    links.new(checker.outputs["Color"], shader.inputs["Base Color"])
    plane.data.materials.append(material)


def repair_materials(fbx) -> list[str]:
    """Unlink image textures that did not load, so nothing renders magenta.

    Cycles draws a missing image as magenta, which is a useless preview and, on
    a character, an alarming one. `FBX.write(embed_textures=True)` stops PoseYdon's
    own exports losing their textures, but this renderer is pointed at arbitrary
    files -- an asset whose textures were never shipped, or moved since -- and a
    flat surface colour shows the motion, which is what a preview is for.

    Returns the names of the images dropped, so the caller can say so rather
    than silently changing what the file looked like.
    """
    dropped = []
    for mesh in fbx.meshes:
        for slot in mesh.material_slots:
            material = slot.material
            if material is None or not material.use_nodes:
                continue
            tree = material.node_tree
            for node in [n for n in tree.nodes if n.type == "TEX_IMAGE"]:
                if node.image is not None and node.image.has_data:
                    continue
                dropped.append(node.image.name if node.image else node.name)
                # The NODE goes, not its links. `tree.links.remove(link)`
                # reports success here and leaves the link in place (measured:
                # the link list is identical before and after), so the socket
                # stays fed by a dead image. Removing the node takes its links
                # with it and lets the consumer fall back to its own default --
                # the Principled BSDF's base colour, or a flat normal.
                tree.nodes.remove(node)
    return dropped


def configure_render(
    path: Path,
    resolution: tuple[int, int],
    samples: int,
    title: str,
    threads: int = 0,
) -> None:
    """Point the scene at an H.264 MP4 under EEVEE.

    ``threads`` of 0 leaves Blender's default of one per core. It is exposed
    because "one per core" on a workstation that is also training a model is how
    a preview render becomes the thing making the machine hot.
    """
    import bpy

    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE"
    # EEVEE's anti-aliasing samples, which is what `samples` means for a
    # rasterizer -- there is no path-tracing sample count to set.
    scene.eevee.taa_render_samples = samples
    # Soft shadows and ambient occlusion: on llvmpipe both are cheap, and
    # without them a character reads as a flat silhouette against the floor,
    # which is precisely the depth cue this render exists to give.
    scene.eevee.use_soft_shadows = True
    scene.eevee.use_gtao = True
    scene.render.resolution_x, scene.render.resolution_y = resolution
    scene.render.resolution_percentage = 100

    # A pure white backdrop, composited rather than lit. Setting the WORLD to
    # white would do both at once -- it is the background the camera sees AND
    # the ambient light on the character -- which blows the subject out to get
    # the backdrop. `film_transparent` drops the world from camera rays only,
    # leaving it lighting the scene, and the compositor lays the render over
    # flat white. The two then tune independently.
    scene.render.film_transparent = True
    scene.use_nodes = True
    tree = scene.node_tree
    tree.nodes.clear()
    layers = tree.nodes.new("CompositorNodeRLayers")
    backdrop = tree.nodes.new("CompositorNodeRGB")
    backdrop.outputs[0].default_value = BACKDROP
    over = tree.nodes.new("CompositorNodeAlphaOver")
    output = tree.nodes.new("CompositorNodeComposite")
    # AlphaOver's inputs are (Fac, bottom, top).
    tree.links.new(backdrop.outputs[0], over.inputs[1])
    tree.links.new(layers.outputs["Image"], over.inputs[2])
    tree.links.new(over.outputs[0], output.inputs["Image"])

    scene.render.image_settings.file_format = "FFMPEG"
    scene.render.ffmpeg.format = "MPEG4"
    scene.render.ffmpeg.codec = "H264"
    scene.render.ffmpeg.constant_rate_factor = "HIGH"
    scene.render.filepath = str(path)

    # Blender's own stamp, rather than a text object in the scene: it is drawn
    # over the film, so it cannot be occluded by the character or leave the
    # frame when the camera moves. `render_skeleton` draws the same two facts.
    scene.render.use_stamp = True
    scene.render.use_stamp_frame = True
    scene.render.use_stamp_note = bool(title)
    scene.render.stamp_note_text = title
    scene.render.use_stamp_date = False
    scene.render.use_stamp_time = False
    scene.render.use_stamp_render_time = False
    scene.render.use_stamp_filename = False
    # Blender stamps the camera, scene, lens and more by DEFAULT; left on they
    # print "Frame 001 Camera Camera ... Scene Scene" across the film.
    for field in ("camera", "lens", "scene", "marker", "sequencer_strip",
                  "hostname", "memory", "frame_range"):
        if hasattr(scene.render, f"use_stamp_{field}"):
            setattr(scene.render, f"use_stamp_{field}", False)
    scene.render.stamp_font_size = 16
    # Dark text: the backdrop is white, and Blender stamps in white by default,
    # which is invisible against it.
    scene.render.stamp_foreground = (0.1, 0.1, 0.1, 1.0)
    scene.render.stamp_background = (1.0, 1.0, 1.0, 0.0)

    if threads > 0:
        scene.render.threads_mode = "FIXED"
        scene.render.threads = threads


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else sys.argv[1:]

    parser = argparse.ArgumentParser(
        description="Render an FBX to an MP4, framed like `render_skeleton`."
    )
    parser.add_argument("input", type=Path, help="the .fbx to render")
    parser.add_argument("--out", type=Path, required=True, help="the .mp4 to write")
    parser.add_argument("--zoom", type=float, default=DEFAULT_ZOOM)
    parser.add_argument("--elev", type=float, default=DEFAULT_ELEV)
    parser.add_argument("--azim", type=float, default=DEFAULT_AZIM)
    parser.add_argument(
        "--samples", type=int, default=16, help="EEVEE anti-aliasing samples"
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=0,
        help="cap the render threads (0 = one per core, Blender's default)",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        nargs=2,
        default=(640, 480),
        metavar=("W", "H"),
    )
    parser.add_argument(
        "--title", default=None, help="stamped on the film (default: the file's stem)"
    )
    parser.add_argument(
        "--rig",
        default=None,
        help="name the rig, so --azim is measured from its forward (default: "
        "resolve it from the corpus by the facing joints in the file)",
    )
    parser.add_argument("--corpus", type=Path, default=Path("data/truebones"))
    parser.add_argument(
        "--frames",
        type=int,
        default=None,
        help="render only the first N frames, for a quick look",
    )
    args = parser.parse_args(argv)

    import bpy

    scene = _Scene(args.input)
    start, end = scene.frame_range()
    if args.frames is not None:
        end = min(end, start + args.frames - 1)
    bpy.context.scene.frame_start, bpy.context.scene.frame_end = start, end

    pairs = facing_pairs(args.corpus, scene, args.rig)
    if pairs is None:
        print(
            "note: could not resolve this rig's facing joints, so --azim is a "
            "world angle rather than an angle from the character's forward",
            file=sys.stderr,
        )
    dropped = repair_materials(scene)
    if dropped:
        print(
            f"note: dropped {len(dropped)} texture(s) that did not load "
            f"({', '.join(sorted(set(dropped))[:3])}); any that did load are kept",
            file=sys.stderr,
        )

    build_scene(
        scene,
        zoom=args.zoom,
        elev=args.elev,
        azim=world_azimuth(scene, args.azim, pairs),
    )
    configure_render(
        args.out,
        tuple(args.resolution),
        args.samples,
        args.title if args.title is not None else args.input.stem,
        threads=args.threads,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.render.render(animation=True)
    print(
        f"wrote {args.out}  ({end - start + 1} frames @ "
        f"{bpy.context.scene.render.fps} fps, {args.resolution[0]}x{args.resolution[1]}, "
        f"eevee {args.samples} samples)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

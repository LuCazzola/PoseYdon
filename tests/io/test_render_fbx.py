"""`scripts/render_fbx.py`: the scene it builds, and that it writes a film.

Blender-gated, like `tests/build/test_roundtrip_fbx.py` and for the same
reason -- the test image ships no Blender, so this runs in the `fbx` service:

    docker compose run --rm fbx blender -b --python-expr \\
        "import sys, pytest; sys.exit(pytest.main(['tests/io/test_render_fbx.py','-v','-rs']))"

The framing itself is NOT retested here. It is `poseydon.io.framing.camera`,
which `tests/io/test_render_framing.py` already covers over every joint of every
frame without needing Blender at all; that is the whole reason the framing was
split out of `poseydon.io.render`. What is left to check is the part that is
Blender's: that a scene gets a camera pointed at the character's front, that a
missing texture cannot reach the film, and that a file comes out.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from scripts.render_fbx import (
    DEFAULT_AZIM,
    build_scene,
    configure_render,
    facing_pairs,
    repair_materials,
    world_azimuth,
)

from poseydon.build.index import is_all_takes_bundle
from tests.conftest import CORPUS

bpy = pytest.importorskip("bpy", reason="rendering an FBX needs Blender")

from poseydon.io.fbx import FBX

#: One rig, not every rig. Nothing here is rig-specific -- the framing is
#: covered without Blender, and the facing rule is the manifest's rather than
#: this rig's -- so six cases would buy six times the render for one claim.
RIG = "Scorpion"


def _source_fbx(rig: str) -> Path:
    directory = Path("data/_raw/truebones/source") / rig
    if not directory.is_dir():
        pytest.skip(f"{rig}: no source directory")
    candidates = sorted(
        path for path in directory.glob("*.fbx") if not is_all_takes_bundle(path.stem, rig)
    )
    if not candidates:
        pytest.skip(f"{rig}: no per-clip source FBX")
    return candidates[0]


def test_azimuth_is_measured_from_the_character_forward():
    """The camera must end up in front of the character, whichever way it faces.

    A Truebones asset is authored facing whatever direction its exporter chose,
    and `--skin` lands a clip in that asset's own orientation -- so a fixed
    world azimuth showed Scorpion its BACK. `--azim` is therefore an angle from
    the character's own forward, and the default -70 (matching
    `render_skeleton`) is a near-frontal three-quarter.
    """
    scene = FBX.read(_source_fbx(RIG))
    pairs = facing_pairs(CORPUS, scene)
    assert pairs is not None, f"{RIG}: could not resolve facing joints from the corpus"

    forward = scene.forward(pairs, rest=True)
    angle = math.radians(world_azimuth(scene, DEFAULT_AZIM, pairs))
    # Where the camera sits, horizontally, relative to what it looks at.
    towards_camera = np.array([math.cos(angle), math.sin(angle)])
    facing = np.array([forward.x, forward.y])
    facing /= np.linalg.norm(facing)

    alignment = float(towards_camera @ facing)
    assert alignment > 0.5, (
        f"{RIG}: the camera sits at {math.degrees(math.acos(max(min(alignment, 1), -1))):.0f} "
        "degrees from the character's forward -- it is not looking at the front"
    )


def test_facing_is_not_guessed_for_an_unknown_rig(tmp_path):
    """An asset the corpus does not describe must return None, not a wrong rig.

    `world_azimuth` falls back to a world angle in that case, which is the only
    honest answer. Silently matching the nearest manifest would point the camera
    using another character's anatomy.
    """
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.object.armature_add()
    scene = FBX(bpy.context.object, [])
    assert facing_pairs(CORPUS, scene) is None
    assert world_azimuth(scene, DEFAULT_AZIM, None) == DEFAULT_AZIM


def test_repair_materials_unlinks_every_image_that_did_not_load():
    """Blender renders a missing image as magenta, which is a useless preview.

    Scorpion's own asset is the case: its normal map `radscorpion_n.JPG` does
    not load in this Blender, while its diffuse map does. The dead one has to go
    and the live one has to stay -- dropping both would throw away the texture
    that works.
    """
    scene = FBX.read(_source_fbx(RIG))
    repair_materials(scene)

    for mesh in scene.meshes:
        for slot in mesh.material_slots:
            material = slot.material
            if material is None or not material.use_nodes:
                continue
            tree = material.node_tree
            linked = {link.from_node for link in tree.links}
            for node in tree.nodes:
                if node.type == "TEX_IMAGE" and node in linked:
                    assert node.image is not None and node.image.has_data, (
                        f"{RIG}: `{node.name}` has no image data but is still "
                        "linked, so it will render magenta"
                    )


def test_render_writes_a_film(tmp_path):
    """End to end, at the smallest size that still exercises the real path."""
    scene = FBX.read(_source_fbx(RIG))
    start, _end = scene.frame_range()
    bpy.context.scene.frame_start, bpy.context.scene.frame_end = start, start + 2

    repair_materials(scene)
    build_scene(scene, azim=world_azimuth(scene, DEFAULT_AZIM, facing_pairs(CORPUS, scene)))

    out_path = tmp_path / "preview.mp4"
    configure_render(out_path, (96, 72), 2, RIG, threads=2)

    # EEVEE, always. A preview that quietly becomes a path trace is how this
    # tool ends up saturating every core of a machine that is also training.
    assert bpy.context.scene.render.engine == "BLENDER_EEVEE"
    assert bpy.context.scene.render.threads_mode == "FIXED"
    assert bpy.context.scene.render.threads == 2

    bpy.ops.render.render(animation=True)

    # Blender appends the frame range to a movie path when it is not already a
    # movie extension; either way exactly one file must land and carry data.
    written = sorted(tmp_path.glob("preview*"))
    assert written, f"no film written to {tmp_path}"
    assert written[0].stat().st_size > 0, f"{written[0].name} is empty"


def test_the_scene_gets_a_camera_a_sun_and_a_floor():
    """The three things a render needs and an FBX does not carry."""
    scene = FBX.read(_source_fbx(RIG))
    start, _end = scene.frame_range()
    bpy.context.scene.frame_start, bpy.context.scene.frame_end = start, start + 1
    build_scene(scene)

    objects = bpy.context.scene.objects
    assert bpy.context.scene.camera is not None, "no camera was added"
    assert any(o.type == "LIGHT" and o.data.type == "SUN" for o in objects), (
        "no sun was added, so the render would be lit only by the world"
    )
    ground = [o for o in objects if o.name == "Ground"]
    assert ground, "no floor was added, so there is no depth cue and no shadow"
    assert ground[0].data.materials, "the floor has no checkerboard material"

    # The camera follows: its parent is keyframed, so the shot tracks the
    # character rather than framing the whole trajectory.
    target = bpy.context.scene.camera.parent
    assert target is not None, "the camera is not parented to a follow target"
    assert target.animation_data and target.animation_data.action, (
        "the follow target carries no keyframes, so the camera is static"
    )

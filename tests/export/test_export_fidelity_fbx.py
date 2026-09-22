"""What the FBX representations must preserve.

`apply_animation` states its own precondition: the animation it is given must
be REST-RELATIVE, so that identity means the bind pose. `original_fbx` used to
break it -- it restored the whole chain, undoing `RestRelative`, and then
repaired the damage with a bind-alignment pass that anchored frame 0 onto the
bind heads and rewrote every rotation as a delta from it. Two consequences,
one test each: the clip's own first pose was replaced by the T-pose, and the
deltas were composed in global space, so the skeleton stopped being rigid.

A third test covers what that repair pass cost on top: by refitting the
animation onto the bind it made `apply_animation`'s residual guard fit
trivially, so a frame-mismatched restoration sailed through instead of being
rejected.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("bpy")

from poseydon.core.kinematics import forward_kinematics
from poseydon.export import CharacterExporter
from poseydon.features.reduce import apply_reduction
from poseydon.io.bvh import BVH
from poseydon.io.fbx import FBX

CORPUS = Path("data/truebones")
RAW = CORPUS.parent / "_raw" / "truebones"
RIGS = ("Fox", "Bear")


def _case(rig: str):
    exporter = CharacterExporter.load(CORPUS, RAW, rig)
    clips = [p for p in sorted((CORPUS / "clips" / rig).glob("*.bvh")) if p.stem != "tpose"]
    if not clips:
        pytest.skip(f"no prepared clips for {rig}")
    prepared = BVH.read(clips[0]).to_animation().as_rigid_body(joint_translation="drop")
    return exporter, clips[0].stem, apply_reduction(prepared, exporter.reduction)


SYNTHETIC_ROOT = "__FBX_ROOT__"


def _bone_length_drift(animation) -> float:
    """Largest fractional change in any bone's length across the clip.

    A file with more than one root (Fox's presentation asset carries a stray
    skeleton node beside `Hips`) is joined under `FBX.to_animation`'s synthetic
    root, which sits at the origin. The segments from it to each real root are
    root TRAJECTORIES, not bones, so they are skipped -- measuring them reports
    the character walking as a stretching rig.
    """
    positions, _ = forward_kinematics(
        animation.rotations, animation.translations, animation.parents
    )
    synthetic = animation.names[0] == SYNTHETIC_ROOT
    worst = 0.0
    for joint, parent in enumerate(animation.parents):
        if parent < 0 or (synthetic and parent == 0):
            continue
        lengths = np.linalg.norm(positions[:, joint] - positions[:, parent], axis=-1)
        if lengths.mean() > 1e-6:
            worst = max(worst, float((lengths.max() - lengths.min()) / lengths.mean()))
    return worst


@pytest.mark.parametrize("rig", RIGS)
@pytest.mark.parametrize("representation", ["original_fbx", "presentation_fbx"])
def test_exported_fbx_keeps_the_skeleton_rigid(tmp_path, rig, representation, request):
    exporter, clip, model = _case(rig)
    path = tmp_path / f"{rig}.fbx"
    getattr(exporter, representation)(model, facing_from=clip).write(path)
    assert _bone_length_drift(FBX.read(path).to_animation()) < 1e-3


@pytest.mark.parametrize("rig", RIGS)
def test_original_fbx_does_not_replace_frame_zero_with_the_bind_pose(tmp_path, rig):
    """The clip's first pose is the clip's, not the rig's T-pose.

    Measured as: the exported frame 0 must sit no closer to the bind pose than
    the motion it came from does. The bind-alignment pass made it land exactly
    ON the bind, which is a different animation.
    """
    exporter, clip, model = _case(rig)
    path = tmp_path / f"{rig}.fbx"
    exporter.original_fbx(model, facing_from=clip).write(path)

    scene = FBX.read(path)
    exported = scene.to_animation()
    restored = exporter._restore(model, "rest_relative", facing_from=clip)
    driven = [
        n
        for n in exported.names
        if n in scene.armature.data.bones and n in set(restored.names)
    ]
    bind = np.stack([np.asarray(scene.joint_position(n, rest=True)) for n in driven])
    posed, _ = forward_kinematics(
        exported.rotations, exported.translations, exported.parents
    )
    posed = posed[0][[exported.names.index(n) for n in driven]]

    reference, _ = forward_kinematics(
        restored.rotations, restored.translations, restored.parents
    )
    truth = reference[0][[restored.names.index(n) for n in driven]]

    def spread(a, b):
        ca, cb = a - a.mean(0), b - b.mean(0)
        scale = np.sqrt((ca**2).sum() / (cb**2).sum())
        u, _s, vt = np.linalg.svd(ca.T @ (cb * scale))
        rotation = u @ np.diag([1.0, 1.0, float(np.sign(np.linalg.det(u @ vt)))]) @ vt
        return float(
            np.linalg.norm((cb * scale) @ rotation.T - ca, axis=-1).max()
            / np.linalg.norm(ca, axis=-1).max()
        )

    # Whatever the clip's own first pose says about the bind, the export has
    # to say the same. Landing exactly on the bind (0.0) is the failure.
    assert spread(bind, posed) == pytest.approx(spread(bind, truth), abs=5e-3)


@pytest.mark.parametrize("rig", RIGS)
def test_original_fbx_tracks_the_restored_motion_on_every_frame(tmp_path, rig):
    """Not just frame 0: the whole clip, not bind-plus-delta-from-frame-zero."""
    exporter, clip, model = _case(rig)
    path = tmp_path / f"{rig}.fbx"
    exporter.original_fbx(model, facing_from=clip).write(path)

    exported = FBX.read(path).to_animation()
    restored = exporter._restore(model, "rest_relative", facing_from=clip)
    shared = [n for n in restored.names if n in set(exported.names)]
    pe, _ = forward_kinematics(
        exported.rotations, exported.translations, exported.parents
    )
    pr, _ = forward_kinematics(
        restored.rotations, restored.translations, restored.parents
    )
    pe = pe[:, [exported.names.index(n) for n in shared]]
    pr = pr[:, [restored.names.index(n) for n in shared]]

    frames = min(pe.shape[0], pr.shape[0])
    worst = 0.0
    for frame in range(frames):
        a, b = pr[frame], pe[frame]
        ca, cb = a - a.mean(0), b - b.mean(0)
        scale = np.sqrt((ca**2).sum() / (cb**2).sum())
        u, _s, vt = np.linalg.svd(ca.T @ (cb * scale))
        rotation = u @ np.diag([1.0, 1.0, float(np.sign(np.linalg.det(u @ vt)))]) @ vt
        worst = max(
            worst,
            float(
                np.linalg.norm((cb * scale) @ rotation.T - ca, axis=-1).max()
                / np.linalg.norm(ca, axis=-1).max()
            ),
        )
    assert worst < 1e-3


@pytest.mark.parametrize("rig", ["Bear"])
def test_original_fbx_rejects_a_frame_mismatched_restoration(rig):
    """The bind-residual guard must be reachable on this path.

    Refitting the animation onto the bind before applying it made the guard
    fit trivially, so a restoration whose body sat in one frame and whose
    offsets sat in another was exported instead of refused.
    """
    exporter, _clip, model = _case(rig)
    exporter.params = {
        name: (
            {"rotation": np.array([1.0, 0.0, 0.0, 0.0])}
            if name == "face_axis"
            else value
        )
        for name, value in exporter.params.items()
    }
    with pytest.raises(ValueError, match="not the same skeleton"):
        exporter.original_fbx(model)

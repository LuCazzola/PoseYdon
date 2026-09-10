"""Do the BVH and FBX arms of the corpus describe the same motion?

FBX is a MEASUREMENT in this project, not a training input: nothing on the
training path reads ``mesh.npz``. So this is a sidecheck, not a build step --
it writes nothing, and its only output is a number per rig saying how far the
two arms have drifted apart.

**Compare through WORLD space, never raw quaternions.** BVH and FBX both store
a joint's rotation in that joint's OWN local bone frame, and the two frames
differ by a constant per-joint offset (Blender's importer re-orients bones to
point at their children; BVH's frame comes from the OFFSET block). Two
identical motions therefore hold completely different quaternion values, and a
component-wise quaternion comparison measures the frame convention rather than
the motion. Composing down the hierarchy to world positions cancels that
offset exactly, which is why every number below is a POSITION.

Two whole-skeleton conventions are normalised away first, because neither is a
disagreement about the motion:

* **Scale**, as the ratio of the two arms' mean non-degenerate rest bone
  length. Raw Truebones BVH and FBX of the same clip are authored in different
  units -- the FBX side is consistently ~1/100 of the BVH side.
* **World placement and orientation**, as one rigid (rotation + translation)
  alignment per FRAME, never per clip: it must not be able to absorb a drift
  that accumulates over time. Both are needed. A raw BVH and its raw FBX do
  not share an up axis or a facing -- that is what the ingest pipeline's
  facing step exists to fix -- so removing only the translation leaves 1.4
  (Flamingo), 2.9 (Goat) and 3.5 (Crab) bone lengths of pure rotation, while
  removing the rotation too leaves 1.3e-4, 1.4e-1 and 7.6e-3. The residual
  after alignment is the honest measure of whether the two describe the same
  POSE, and the size of what was removed is printed beside it rather than
  hidden: a prepared clip, already faced +Z on both arms, should need an
  alignment near 0 degrees.

Errors are reported in BONE LENGTHS (the BVH arm's mean non-degenerate rest
bone length), because the rigs run from a Crab to a Mammoth and a millimetre
means nothing across them. ``extent`` is the ratio of the two arms' POSED
sizes; it is 1 when they agree and is the diagnostic that separates "the pose
differs" from "the whole skeleton is the wrong size in this file".

Reads the prepared corpus (``data/truebones/clips/<Rig>/<action>.{bvh,fbx}``)
when a rig has both arms there, and otherwise falls back to pairing the RAW
files (``data/truebones/source/<Rig>/``), which every rig has -- most rigs have
no prepared FBX, since writing those needs this same Blender container. Both
sides of a comparison always come from the same stage; the two are never mixed.

**Measured, 2026-09-10** (10 frames per clip, this script's own output):

    Flamingo: 5/5 clips (raw)    median 1.37e-04  p90 3.02e-04  [extent x1.00]
    Scorpion: 14/14 clips (raw)  median 6.41e-03  p90 1.26e-02  [extent x1.00]
    Crab:     11/11 clips (raw)  median 7.92e-03  p90 1.48e-02  [extent x1.00]
    Goat:     10/10 clips (raw)  median 1.37e-01  p90 2.34e-01  [extent x0.93]
    Camel:    17/17 clips (prepared)  median 1.28e+01  p90 3.18e+01  [extent x5.50]

The raw arms agree: Flamingo to 1e-4 bone lengths, and Goat's 1.4e-1 is the
one raw rig with a real (small) disagreement, visible as its extent ratio of
0.93 rather than 1.00. The PREPARED Camel clips -- the corpus's only prepared
FBX -- are three orders of magnitude worse, and the extent ratio says why: at
x5.50 the posed skeleton is five times the size of the BVH arm's while its
REST bone lengths match (scale x0.98). ``FBX.scale_to_mean_bone_length`` is
what does that. It scales the object transform (correct, the posed skeleton
scales with it) and then calls ``transform_apply(scale=True)``, which bakes
the factor into the armature data but does NOT rescale the action's pose-bone
``location`` channels, so the pose inflates relative to its own rest.
Isolated on `Camel-IdleLoop.fbx`, posed extent in bone lengths: 2.869 as
imported, 2.869 after an object scale of x2, 13.464 after
``transform_apply``. Every prepared FBX clip written since that call was added
carries this. Reported, not fixed: it is an `io/fbx` bug that needs the FBX
corpus rewritten, both outside this script's remit.

Blender-gated: run it in the `fbx` container.

    docker compose run --rm fbx blender --background \\
        --python scripts/check_fbx_coherence.py -- --rigs Goat Crab Flamingo
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from poseydon.build.index import action_slug, is_all_takes_bundle, strip_skeleton_prefix
from poseydon.ingest.pipeline import available_rigs
from poseydon.io.bvh import BVH

DEFAULT_ROOT = Path("data/truebones")
DEFAULT_FRAMES = 10


def _action_of(path: Path, rig: str) -> str:
    """The basename both pipelines derive for a clip, raw stem or prepared."""
    return strip_skeleton_prefix(action_slug(path.stem), rig)


def paired_clips(rig: str, root: Path = DEFAULT_ROOT) -> tuple[str, list[tuple[str, Path, Path]]]:
    """``(stage, [(action, bvh_path, fbx_path)])`` for one rig, both from one stage.

    Prepared clips are preferred; a rig with no prepared FBX falls back to its
    raw files. Pairing is by the action name both pipelines derive from the
    stem, which is how the two corpora come to share basenames at all -- the
    raw stems themselves do not match (`__Attack.bvh` against `Goat-Attack.fbx`).
    """
    for stage, directory in (("prepared", root / "clips" / rig), ("raw", root / "source" / rig)):
        if not directory.is_dir():
            continue
        bvhs = {_action_of(p, rig): p for p in sorted(directory.glob("*.bvh"))}
        fbxs = {
            _action_of(p, rig): p
            for p in sorted(directory.glob("*.fbx"))
            if not is_all_takes_bundle(p.stem, rig)
        }
        shared = sorted(set(bvhs) & set(fbxs))
        if shared:
            return stage, [(action, bvhs[action], fbxs[action]) for action in shared]
    return "none", []


def rigid_residual(target: np.ndarray, source: np.ndarray) -> tuple[np.ndarray, float]:
    """Kabsch: fit rotation + translation taking ``source`` onto ``target``.

    Returns the per-point residual and the rotation's angle in degrees. The
    reflection guard (the ``det`` term) matters: without it a mirrored fit can
    beat the true one and report agreement where there is none.
    """
    target_centre, source_centre = target.mean(axis=0), source.mean(axis=0)
    u, _, vt = np.linalg.svd((source - source_centre).T @ (target - target_centre))
    flip = np.sign(np.linalg.det(vt.T @ u.T))
    rotation = vt.T @ np.diag([1.0, 1.0, flip]) @ u.T
    aligned = (source - source_centre) @ rotation.T + target_centre
    cosine = np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0)
    return np.linalg.norm(target - aligned, axis=-1), float(np.degrees(np.arccos(cosine)))


@dataclass
class ClipMeasurement:
    """One clip's agreement, everything already in bone-length units."""

    errors: np.ndarray  # (frames, shared joints) residual after alignment
    scale: float  # rest bone-length ratio fbx/bvh, divided out
    angle: float  # median rotation the alignment removed, degrees
    extent: float  # median posed-size ratio fbx/bvh; 1 if they agree


def measure_clip(bvh_path: Path, fbx_path: Path, n_frames: int) -> ClipMeasurement:
    """Compare one BVH against one FBX over the first ``n_frames`` frames."""
    # Imported here, not at module scope, so the pairing above stays importable
    # (and testable) in the CPU image, which ships no Blender. `_to_y_up` is
    # `poseydon.io.fbx`'s own Blender-frame -> PoseYdon-Y-up conversion:
    # `joint_position` returns a raw Blender (Z-up) vector and the BVH arm is
    # Y-up, so the two are not comparable without it.
    from poseydon.io.fbx import FBX, _to_y_up

    anim = BVH.read(bvh_path).to_animation()
    fbx = FBX.read(fbx_path)

    fbx_names = fbx.joint_names
    shared = [name for name in anim.names if name in set(fbx_names)]
    if len(shared) < 3:
        raise ValueError(f"only {len(shared)} shared joints; a rigid fit needs 3")

    bvh_index = [anim.names.index(name) for name in shared]
    bone_length = _mean_bone_length(anim.offsets)
    scale = _mean_bone_length(fbx.joint_offsets) / bone_length

    start, end = fbx.frame_range()
    count = min(n_frames, anim.n_frames, end - start + 1)
    if count < 1:
        raise ValueError("no overlapping frames")

    bvh_positions = anim.global_positions()[:count][:, bvh_index]
    errors = np.empty(bvh_positions.shape[:2])
    angles, extents = [], []
    for frame in range(count):
        fbx.set_frame(start + frame)
        posed = np.stack([_to_y_up(fbx.joint_position(name)) for name in shared]) / scale
        errors[frame], angle = rigid_residual(bvh_positions[frame], posed)
        angles.append(angle)
        extents.append(_extent(posed) / _extent(bvh_positions[frame]))

    return ClipMeasurement(
        errors=errors / bone_length,
        scale=scale,
        angle=float(np.median(angles)),
        extent=float(np.median(extents)),
    )


def _mean_bone_length(offsets: np.ndarray) -> float:
    """Mean rest bone length, ignoring the root and any zero-length joint.

    Zero-length entries are End Sites and their FBX equivalents; averaging over
    them inflates the mean by a rig-dependent factor and would leave the two
    arms disagreeing about scale for no reason but their joint counts.
    """
    lengths = np.linalg.norm(offsets[1:], axis=-1)
    real = lengths[lengths > 1e-8]
    return float(real.mean()) if real.size else 1.0


def _extent(positions: np.ndarray) -> float:
    """Mean distance from the skeleton's centroid: its posed size."""
    return float(np.linalg.norm(positions - positions.mean(axis=0), axis=-1).mean())


def check_rig(rig: str, root: Path, n_frames: int) -> tuple[str, list[str]]:
    """One report line for a rig, plus any per-clip warnings."""
    stage, pairs = paired_clips(rig, root)
    if not pairs:
        return f"{rig}: no clip pairs in either stage", []

    warnings: list[str] = []
    measured: list[ClipMeasurement] = []
    for action, bvh_path, fbx_path in pairs:
        # One unreadable file must not cost the rig: Truebones ships clips
        # whose joint sets differ from their own rig's, and a rig with twenty
        # comparable clips is still worth a number.
        try:
            measured.append(measure_clip(bvh_path, fbx_path, n_frames))
        except Exception as error:  # noqa: BLE001 - collect, don't abort the rig
            warnings.append(f"{rig}/{action}: {type(error).__name__}: {error}")

    if not measured:
        return f"{rig}: {len(pairs)} pairs ({stage}), none comparable", warnings

    errors = np.concatenate([m.errors.ravel() for m in measured])
    line = (
        f"{rig}: {len(measured)}/{len(pairs)} clips ({stage})  "
        f"median {np.median(errors):.2e}  p90 {np.percentile(errors, 90):.2e} bone lengths  "
        f"[scale x{np.median([m.scale for m in measured]):.4f}, "
        f"align {np.median([m.angle for m in measured]):.1f}deg, "
        f"extent x{np.median([m.extent for m in measured]):.2f}]"
    )
    return line, warnings


def main() -> None:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--rigs", nargs="*", default=None, help="Rig names (default: every rig with a manifest)"
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=DEFAULT_FRAMES,
        help=(
            f"Frames per clip to compare (default {DEFAULT_FRAMES}); Blender steps them "
            "one at a time, so this is the runtime knob."
        ),
    )
    args = parser.parse_args(argv)

    rigs = available_rigs(args.root / "rigs")
    if args.rigs is not None:
        wanted = set(args.rigs)
        rigs = [rig for rig in rigs if rig in wanted]

    all_warnings: list[str] = []
    for rig in sorted(rigs):
        line, warnings = check_rig(rig, args.root, args.frames)
        print(line, flush=True)
        all_warnings.extend(warnings)

    if all_warnings:
        print(f"\n{len(all_warnings)} warning(s):")
        for warning in all_warnings:
            print(f"  - {warning}")


if __name__ == "__main__":
    main()

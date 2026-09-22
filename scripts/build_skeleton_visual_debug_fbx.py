"""Generate CharacterExporter roundtrip files for visual-debug bundles.

Run this companion in the headless Blender container after
``build_skeleton_visual_debug.py`` has copied the deterministic inputs.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from poseydon.core.animation import Animation
from poseydon.export import CharacterExporter

def _drop(bundle: Path, *names: str) -> None:
    """Remove artifacts this run could not rebuild, so none is left stale."""
    for name in names:
        path=bundle/name
        if path.exists():
            path.unlink()


def _action(bundle: Path) -> str | None:
    """The clip this bundle was built from, as `build_skeleton_visual_debug` recorded it."""
    metadata=bundle/"metadata.json"
    if not metadata.is_file():
        return None
    return json.loads(metadata.read_text()).get("action")


def main(argv=None):
    if argv is None:
        # Blender puts its own flags on sys.argv and swallows everything after
        # `--`; parsing sys.argv directly makes `blender -b --python this.py`
        # fail on "unrecognized arguments: -b --python ...". Same convention as
        # scripts/anim_2_fbx.py.
        argv=sys.argv[sys.argv.index("--")+1:] if "--" in sys.argv else []
    parser=argparse.ArgumentParser()
    parser.add_argument("--corpus",type=Path,default=Path("data/truebones"))
    parser.add_argument("--raw",type=Path,default=Path("data/_raw/truebones"))
    parser.add_argument("--out",type=Path,default=Path("tmp/skel_visual_debug"))
    args=parser.parse_args(argv)
    ok=0
    for bundle in sorted(args.out.iterdir()):
        training=bundle/"training.anim.npz"
        if not bundle.is_dir() or not training.is_file():
            continue
        animation=Animation.load(training)
        try:
            exporter=CharacterExporter.load(args.corpus,args.raw,bundle.name)
        except Exception as error:
            _drop(bundle,"roundtrip.bvh","roundtrip.fbx")
            print(f"{bundle.name}: unavailable ({type(error).__name__}: {error})",flush=True)
            continue
        # These bundles reconstruct a REAL clip, so its recorded yaw is on
        # hand and is the exact answer; the rig-level rest facing that a
        # generated clip falls back to would turn the roundtrip against
        # the original it sits beside.
        action=_action(bundle)
        facing=action if action in exporter.clip_params else None
        written=[]
        # Each artifact stands or falls on its own: a rig whose vendor FBX is
        # not the same skeleton as its BVH still has a valid BVH roundtrip, and
        # one that cannot be rebuilt must not be left behind from a previous
        # run beside a fresh sibling. Stale is worse than absent -- it is the
        # shape a bundle takes when half of it silently did not regenerate.
        for name,build in (("roundtrip.bvh",lambda: exporter.original_bvh(animation,facing_from=facing).write(bundle/"roundtrip.bvh")),
                           ("roundtrip.fbx",lambda: exporter.original_fbx(animation,facing_from=facing).write(bundle/"roundtrip.fbx",embed_textures=True))):
            try:
                build(); written.append(name)
            except Exception as error:
                _drop(bundle,name)
                print(f"{bundle.name}: {name} unavailable ({type(error).__name__}: {error})",flush=True)
        if written:
            ok += 1
            print(f"{bundle.name}: wrote {', '.join(written)}",flush=True)
    print(f"wrote {ok} roundtrip bundles")
if __name__ == "__main__": main()

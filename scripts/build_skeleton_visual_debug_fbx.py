"""Generate CharacterExporter roundtrip files for visual-debug bundles.

Run this companion in the headless Blender container after
``build_skeleton_visual_debug.py`` has copied the deterministic inputs.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
from poseydon.core.animation import Animation
from poseydon.export import CharacterExporter

def _action(bundle: Path) -> str | None:
    """The clip this bundle was built from, as `build_skeleton_visual_debug` recorded it."""
    metadata=bundle/"metadata.json"
    if not metadata.is_file():
        return None
    return json.loads(metadata.read_text()).get("action")


def main(argv=None):
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
        try:
            animation=Animation.load(training)
            exporter=CharacterExporter.load(args.corpus,args.raw,bundle.name)
            # These bundles reconstruct a REAL clip, so its recorded yaw is on
            # hand and is the exact answer; the rig-level rest facing that a
            # generated clip falls back to would turn the roundtrip against
            # the original it sits beside.
            action=_action(bundle)
            facing=action if action in exporter.clip_params else None
            exporter.original_bvh(animation,facing_from=facing).write(bundle/"roundtrip.bvh")
            exporter.original_fbx(animation,facing_from=facing).write(bundle/"roundtrip.fbx",embed_textures=True)
            ok += 1
            print(f"{bundle.name}: wrote roundtrip.bvh and roundtrip.fbx",flush=True)
        except Exception as error:
            print(f"{bundle.name}: unavailable ({type(error).__name__}: {error})",flush=True)
    print(f"wrote {ok} roundtrip bundles")
if __name__ == "__main__": main()

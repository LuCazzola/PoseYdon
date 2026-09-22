"""Generate CharacterExporter roundtrip files for visual-debug bundles.

Run this companion in the headless Blender container after
``build_skeleton_visual_debug.py`` has copied the deterministic inputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from poseydon.core.animation import Animation
from poseydon.export import CharacterExporter

#: metadata.json field name for each artifact this script owns.
FIELDS={"roundtrip.bvh":"roundtrip_bvh","roundtrip.fbx":"roundtrip_fbx"}

def _drop(bundle: Path, *names: str) -> None:
    """Remove artifacts this run could not rebuild, so none is left stale."""
    for name in names:
        path=bundle/name
        if path.exists():
            path.unlink()


def _record(bundle: Path, name: str, reason: str | None) -> None:
    """Write this artifact's outcome into the bundle's own metadata.

    `build_skeleton_visual_debug.py` writes metadata before this companion
    runs, so the two roundtrip entries are necessarily null there. Left that
    way the bundle under-reports itself: the file is on disk with no hash
    beside it, and a rig whose FBX was REFUSED is indistinguishable from one
    whose vendor asset is simply absent. Both are filled in here, by the run
    that actually knows.
    """
    metadata=bundle/"metadata.json"
    if not metadata.is_file():
        return
    data=json.loads(metadata.read_text())
    field=FIELDS[name]
    path=bundle/name
    if reason is None and path.is_file():
        digest=hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1<<20), b""): digest.update(block)
        data[field]={"path":str(path),"sha256":digest.hexdigest()}
        data.setdefault("unavailable",{}).pop(field,None)
    else:
        data[field]=None
        data.setdefault("unavailable",{})[field]=reason or "not_generated"
    metadata.write_text(json.dumps(data,indent=2,sort_keys=True)+"\n")


def _reason(error: Exception) -> str:
    """The typed reason where the exporter gives one, else the exception."""
    return getattr(error,"reason",None) or f"{type(error).__name__}: {error}"


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
        except Exception as error:  # noqa: BLE001 - one rig must not cost the corpus
            _drop(bundle,"roundtrip.bvh","roundtrip.fbx")
            for name in FIELDS: _record(bundle,name,_reason(error))
            print(f"{bundle.name}: unavailable ({_reason(error)})",flush=True)
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
        for name in FIELDS:
            try:
                if name.endswith(".bvh"):
                    exporter.original_bvh(animation,facing_from=facing).write(bundle/name)
                else:
                    exporter.original_fbx(animation,facing_from=facing).write(bundle/name,embed_textures=True)
                written.append(name); _record(bundle,name,None)
            except Exception as error:  # noqa: BLE001 - one artifact must not cost the other
                _drop(bundle,name)
                _record(bundle,name,_reason(error))
                print(f"{bundle.name}: {name} unavailable ({_reason(error)})",flush=True)
        if written:
            ok += 1
            print(f"{bundle.name}: wrote {', '.join(written)}",flush=True)
    print(f"wrote {ok} roundtrip bundles")
if __name__ == "__main__": main()

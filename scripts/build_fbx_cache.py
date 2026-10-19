"""Read UniMate's Truebones clips once with the FBX SDK and cache what later stages need.

    docker compose run --rm -T --entrypoint python -e HOME=/tmp fbxsdk scripts/build_fbx_cache.py \
        [--species Alligator Horse ...] [--clips Alligator-Big_Mouth ...] [--print] [--force]

One ``.npz`` per ``status == linked`` row of UniMate's ``clips.csv``, at
``data/unimate/cache/fbx/<Species>/<Action>.npz`` (``normalized_name`` is
``<Species>-<Action>.fbx``): node names, parents, attribute types, the world
matrix of EVERY node at every integer frame of the clip's key range (floor(start
+ 1e-3) .. ceil(stop - 1e-3), UniMate's ``action_frame_range``; which nodes are
bones is decided later), the world bind, the key range, fps, up axis and a
per-node skin weight summary. Everything after this runs natively, with no SDK.

Resumable: a clip whose cache holds the source's size and mtime is skipped.

KingCobra is read from repaired copies under ``data/unimate/cache/repaired/``
(raw files are never touched; the probes write in place, so they get copies):
all 12 clips keep only the ``A01`` colourway (``fbx_strip_variant.py``), and
``walk.fbx``, exported without geometry, gets the mesh and skin of ``run.fbx``
(``fbx_graft_mesh.py --bind donor``). Donor choice: ``walk.fbx`` has the
``run.fbx`` / ``walk fast.fbx`` export's material order (A01 first, the
``Cobra-*`` files list A02 first) and its 19 bones are those clips' 20 minus
a ``Null`` root; ``run.fbx`` is the first of the two alphabetically.
The Cat texture and KingCobra material repairs are skipped: they touch neither
skeleton nor skin.
"""

from __future__ import annotations

import argparse
import csv
import math
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from poseydon.build.corpus import cache_dir
from poseydon.io.fbx import FBX

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "_raw"
PROBES = RAW / "UniML3D" / "truebones_annotations" / "scripts" / "probes"
CLIPS = RAW / "UniML3D" / "truebones_annotations" / "clips.csv"
CACHE = cache_dir(ROOT / "data" / "unimate")
FRAME_RANGE_EPS = 1e-3  # UniMate's action_frame_range rounding
DONOR = "run.fbx"  # for KingCobra's walk.fbx, see the module docstring


def repaired(source: Path) -> Path:
    """``source``, or a repaired copy of it when its species needs one."""
    if source.parent.name != "KingCobra":
        return source
    out = CACHE / "repaired" / "KingCobra" / source.name
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(source, out)
        if source.name == "walk.fbx":
            probe("fbx_graft_mesh.py", "--donor", str(source.parent / DONOR), "--bind", "donor", str(out))
            define_types(out, source.parent / DONOR)
        probe("fbx_strip_variant.py", "--keep", "A01", str(out))
    return out


def define_types(target: Path, donor: Path) -> None:
    """Declare in ``target``'s Definitions the object types the graft added.

    The graft inserts Geometry and Pose objects but leaves them undeclared, and
    the SDK then drops them (``walk.fbx`` read back with no mesh and skin
    geometry None) though the probes' own parser sees them. The declarations are
    copied from the donor, which has them.
    """
    sys.path.insert(0, str(PROBES))
    from fbx_graft_mesh import copy_node, parse, serialize, sval, top

    tree, source = parse(str(target)), parse(str(donor))
    have = {sval(c.props[0]) for c in top(tree, b"Definitions").children if c.name == b"ObjectType"}
    for c in top(source, b"Definitions").children:
        if c.name == b"ObjectType" and sval(c.props[0]) in {b"Geometry", b"Pose"} - have:
            top(tree, b"Definitions").children.append(copy_node(c))
    target.write_bytes(serialize(tree))


def probe(script: str, *args: str) -> None:
    subprocess.run([sys.executable, str(PROBES / script), *args], check=True, capture_output=True)


def read(path: Path) -> dict:
    fbx = FBX.read(path)
    start, stop = fbx.key_range()
    frames = np.arange(math.floor(start + FRAME_RANGE_EPS), math.ceil(stop - FRAME_RANGE_EPS) + 1)
    skins = fbx.skins()
    n = len(fbx.nodes)
    return dict(
        names=np.array(fbx.names), parents=fbx.parents, types=np.array(fbx.node_types()),
        frames=frames, world=fbx.world(frames), key_range=np.array([start, stop]),
        bind=fbx.bind_matrices(), fps=fbx.fps, up_axis=fbx.up_axis(),
        skin_mesh=np.array([s.mesh for s in skins], dtype=str),
        skin_vertices=np.array([s.n_vertices for s in skins], dtype=int),
        skin_max=np.array([s.max_weight for s in skins]).reshape(len(skins), n),
        skin_sum=np.array([s.weight_sum for s in skins]).reshape(len(skins), n),
    )


def summary(d) -> str:
    skeleton = int((d["types"] == "eSkeleton").sum())
    return (f"{skeleton} nodes ({len(d['names'])} total), {len(d['frames'])} frames, "
            f"{d['fps']:g} fps, up {d['up_axis']}\n{len(d['skin_mesh'])} skin"
            + "".join(f"\n  {m}: {v} vertices" for m, v in zip(d["skin_mesh"], d["skin_vertices"])))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--species", nargs="*", default=[])
    ap.add_argument("--clips", nargs="*", default=[], help="normalized names, e.g. Alligator-Big_Mouth")
    ap.add_argument("--print", action="store_true", help="read and print each clip's summary; write nothing")
    ap.add_argument("--force", action="store_true", help="recache clips that are up to date")
    args = ap.parse_args()

    rows = [r for r in csv.DictReader(open(CLIPS)) if r["status"] == "linked"]
    rows = [r for r in rows
            if (not args.species or r["group"] in args.species)
            and (not args.clips or r["normalized_name"].removesuffix(".fbx") in args.clips)]
    done = failed = skipped = 0
    for r in rows:
        source = RAW / "truebones" / "source" / r["group"] / Path(r["file"]).name
        species, action = r["normalized_name"].removesuffix(".fbx").split("-", 1)
        out = CACHE / "fbx" / species / f"{action}.npz"
        stat = source.stat()
        if not args.print and not args.force and out.exists():
            with np.load(out) as old:
                if int(old["source_size"]) == stat.st_size and float(old["source_mtime"]) == stat.st_mtime:
                    skipped += 1
                    continue
        t0 = time.time()
        try:
            d = read(repaired(source))
        except Exception as error:  # one bad clip must not stop a 3 h run
            failed += 1
            print(f"FAILED {species}-{action}: {error!r}", flush=True)
            continue
        if args.print:
            print(f"{species}-{action}: {summary(d)}")
        else:
            out.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(out, **d, source=str(source), source_size=stat.st_size,
                                source_mtime=stat.st_mtime)
        done += 1
        print(f"{species}-{action}: {time.time() - t0:.0f}s", flush=True)
    print(f"{done} cached, {skipped} skipped, {failed} failed")
    return int(failed > 0)


if __name__ == "__main__":
    sys.exit(main())

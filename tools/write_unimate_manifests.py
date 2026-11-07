"""Writes ``data/unimate/rigs/<Species>/manifest.yaml`` for ``data=unimate``.

The manifests come from UniMate's own annotation files, nothing is authored:
the facing pair of each species is its ``face_joint_names.json`` entry (raw
joint names), ``up_axis`` is read from the species' cached FBX clips. There is
no ``rest_pose:`` (``fbx_cache`` takes the rest from the clips' bind poses).

``body_axis: true`` (Anaconda alone) means UniMate's pair is the head and tail
joints, the body's forward axis, not a left/right pair. UniMate's
``get_root_facing_quat`` (``data_process/utils/skeleton.py``) turns that vector
by -90 deg about Y after taking ``forward = Y x across``, then rotates the result
onto +Z with ``between``. Call the unturned forward ``f``: the rotation it applies
is R = Ry(theta) with ``R Ry(-90) f = +Z``, so on the real joints ``R f =
Ry(+90) +Z = +X``. PoseYdon's ``facing_quats`` returns ``yaw * between(f, +Z)``
with ``yaw = Ry(extra_yaw_deg)``, i.e. ``f -> Ry(extra_yaw_deg) +Z``. The two
agree for ``extra_yaw_deg = +90``. Measured through stage 1 (``data=unimate``): in
all 18 Anaconda clips, frame 0's head-minus-tail direction projected on XZ has
dot 1.0000 with +Z, as UniMate's snake faces +Z along its body.

A species with no cached clip yet is reported and skipped; rerun once
``scripts/build_fbx_cache.py`` has cached it.

    docker compose run --rm -T test python tools/write_unimate_manifests.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from poseydon.build.sources import FbxCache

BODY_AXIS_YAW_DEG = 90.0   # derivation in the module docstring


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    raw = "data/_raw/UniML3D"
    parser.add_argument("--index", default=f"{raw}/truebones_annotations/clips.csv")
    parser.add_argument("--faces", default=f"{raw}/export/truebones/face_joint_names.json")
    parser.add_argument("--skip", default=f"{raw}/export/truebones/filtered_clips.txt")
    parser.add_argument("--cache", default="data/unimate/cache/fbx")
    parser.add_argument("--out", default="data/unimate/rigs")
    args = parser.parse_args()

    faces = json.loads(Path(args.faces).read_text())
    source = FbxCache(args.index, args.cache, args.skip)
    pending = []
    for species in source.rigs():
        cached = sorted((Path(args.cache) / species).glob("*.npz"))
        if species not in faces or not cached:
            pending.append(species)
            continue
        with np.load(cached[0]) as z:
            up_axis = str(z["up_axis"])
        pair, facing = faces[species], {}
        facing["pair"] = {"right": pair["r_hip"]["raw"], "left": pair["l_hip"]["raw"]}
        if pair.get("body_axis"):
            facing["extra_yaw_deg"] = BODY_AXIS_YAW_DEG
        manifest = {"skeleton": species, "rest_pose": None, "up_axis": up_axis, "facing": facing}
        path = Path(args.out) / species / "manifest.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(manifest, sort_keys=False))
    print(f"wrote {len(source.rigs()) - len(pending)} manifests -> {args.out}")
    if pending:
        print(f"skipped, no face pair or no cached clip yet ({len(pending)}): {', '.join(pending)}")


if __name__ == "__main__":
    main()

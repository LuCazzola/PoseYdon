"""Retarget through the EXACT reference implementation, rendered BEFORE fitting.

`transfer_compare.py` already runs the reference's own `p_sample_loop` and saves
its denormalized output. Its MP4s, though, go through PoseYdon's
`features_to_anim`, which SOLVES a skeleton -- bone lengths are enforced and the
result is a fitted animation. That hides what the model itself claimed.

This renders the step before that: the reference's own
`recover_from_bvh_ric_np`, which reads positions straight out of the `ric_pos`
block and un-rotates them by the recovered root, with no skeleton fitting at
all. It is exactly what `sample/mix.py` plots. Bone lengths are whatever the
model produced, so limbs stretch where it was unsure -- which is the point.

Both implementations are rendered, from the same experiment and the same noise,
so the pre-fitting outputs can be compared directly.

Run:
    docker compose run --rm --entrypoint python compat \
        tools/reference/reference_pre_ik_render.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REFERENCE = Path("external/neural_motion_blending").resolve()
sys.path.insert(0, str(REFERENCE))

from data_loaders.truebones.truebones_utils.motion_process import (
    recover_from_bvh_ric_np,
)

from poseydon.io.render import render_skeleton

SOURCE, TARGET = "Flamingo", "Scorpion"
FEATURES = Path("artifacts/headtohead-full/truebones_attnpool/ddpm")
OUT = Path("artifacts/pre-ik")
FPS = 30


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    with np.load(Path("data/truebones/reference") / f"{TARGET}.npz", allow_pickle=True) as d:
        parents = np.asarray(d["parents"]).astype(int)

    for label in ("reference", "poseydon"):
        path = FEATURES / f"{label}_{SOURCE}_to_{TARGET}.npy"
        if not path.exists():
            raise SystemExit(
                f"{path} is missing -- run transfer_compare.py --frames 200 first"
            )
        raw = np.load(path)                      # (T, J, 13), denormalized
        positions = recover_from_bvh_ric_np(raw)  # (T, J, 3), NO fitting
        print(f"{label:10s} features {raw.shape} -> positions {positions.shape}")

        # What the fitting would have had to correct: how far the model's own
        # bone lengths drift from the rig's, frame by frame.
        live = np.linalg.norm(
            positions[:, 1:] - positions[:, parents[1:]], axis=-1
        )
        rest = live.mean(axis=0)
        drift = np.abs(live - rest[None]) / np.maximum(rest[None], 1e-6)
        print(f"           bone-length drift about its own mean: "
              f"mean {100 * drift.mean():.1f}%  max {100 * drift.max():.1f}%")

        out = OUT / f"{label}_{SOURCE}_to_{TARGET}.PRE-IK.mp4"
        render_skeleton(
            out, parents, positions, FPS,
            f"{label} (exact NMB impl), PRE-fitting: {SOURCE} -> {TARGET}",
        )
        print(f"           wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

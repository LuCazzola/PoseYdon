"""Encode joint names with the reference's own T5 conditioner.

Both shipped checkpoints were trained with `skip_t5: False`, so reproducing
their outputs needs the same joint-name embeddings. The reference computes these
at dataset load and caches them next to cond.npy; that directory is read-only
here, so the cache is written into the PoseYdon tree instead.

The reference's name preprocessing is non-trivial -- prefix stripping,
camel-case splitting, a Japanese word table -- so its class is used directly
rather than reimplemented.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import torch

REFERENCE = Path("external/neural_motion_blending").resolve()
sys.path.insert(0, str(REFERENCE))

# The reference imports spacy at module scope for a text normalizer that only
# runs when normalize_text=True, which this does not use. A stub avoids pulling
# in spacy and its language models for a code path that never executes.
if "spacy" not in sys.modules:
    stub = types.ModuleType("spacy")
    stub.load = lambda *a, **k: None  # type: ignore[attr-defined]
    stub.blank = lambda *a, **k: None  # type: ignore[attr-defined]
    sys.modules["spacy"] = stub

from model.modules.conditioners import T5Conditioner

REFERENCE_DIR = Path("data/truebones/reference")
OUT = REFERENCE_DIR / "t5_cache.npz"
T5_NAME = "t5-base"


def main() -> int:
    names: set[str] = set()
    for path in sorted(REFERENCE_DIR.glob("*.npz")):
        if path.name == OUT.name:
            continue
        with np.load(path, allow_pickle=True) as data:
            names.update(str(n) for n in data["joints_names"])

    ordered = sorted(names)
    print(f"{len(ordered)} unique joint names across {len(list(REFERENCE_DIR.glob('*.npz')))} skeletons")

    conditioner = T5Conditioner(
        name=T5_NAME, finetune=False, word_dropout=0.0, normalize_text=False, device="cpu"
    )
    with torch.no_grad():
        embeddings = conditioner(conditioner.tokenize(ordered))

    array = embeddings.detach().cpu().numpy()
    print(f"embeddings: {array.shape} dtype={array.dtype}")
    if array.ndim != 2:
        raise SystemExit(f"expected (N, D) embeddings, got {array.shape}")

    np.savez_compressed(OUT, **{name: array[i] for i, name in enumerate(ordered)})
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

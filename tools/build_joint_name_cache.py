"""Cache T5 embeddings of every joint name in the PoseYdon corpus.

The reference trains with `skip_t5: False` and `cond_mask_prob: 0.0`, so joint
name embeddings are on EVERY forward pass, never masked. `MoDiffAE` treats them
as optional conditioning and skips the term silently when absent, which is why
their absence never raised anything -- it just trained a different model.

Run once, in the `compat` image, at build time: `transformers` lives only there,
and the `test`/`train` images must never need it.

    docker compose run --rm --entrypoint python compat tools/build_joint_name_cache.py

Names are taken POST-REDUCTION -- `skeleton.npz`'s `names` indexed by
`reduction_source_of` -- because that is the joint set the read path yields and
therefore the set the model is conditioned on. Preprocessing (prefix stripping,
camel-case splitting, the Japanese word table) is the reference's own, applied
inside `T5Conditioner`, so its class is used directly rather than reimplemented.
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
# runs when normalize_text=True, which this does not use.
if "spacy" not in sys.modules:
    stub = types.ModuleType("spacy")
    stub.load = lambda *a, **k: None  # type: ignore[attr-defined]
    stub.blank = lambda *a, **k: None  # type: ignore[attr-defined]
    sys.modules["spacy"] = stub

from model.modules.conditioners import T5Conditioner

RIGS = Path("data/truebones/rigs")
OUT = Path("data/truebones/joint_names_t5.npz")
T5_NAME = "t5-base"


def reduced_names(rig: Path) -> list[str]:
    with np.load(rig / "skeleton.npz", allow_pickle=True) as data:
        full = [str(n) for n in data["names"]]
        return [full[int(i)] for i in data["reduction_source_of"]]


def main() -> int:
    rigs = sorted(p for p in RIGS.iterdir() if p.is_dir())
    names: set[str] = set()
    for rig in rigs:
        names.update(reduced_names(rig))

    ordered = sorted(names)
    print(f"{len(ordered)} unique joint names across {len(rigs)} rigs")

    conditioner = T5Conditioner(
        name=T5_NAME, finetune=False, word_dropout=0.0, normalize_text=False, device="cpu"
    )
    with torch.no_grad():
        embeddings = conditioner(conditioner.tokenize(ordered))

    array = embeddings.detach().cpu().numpy().astype("float32")
    print(f"embeddings: {array.shape} dtype={array.dtype}")
    if array.ndim != 2 or array.shape[0] != len(ordered):
        raise SystemExit(f"expected ({len(ordered)}, D) embeddings, got {array.shape}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT, names=np.array(ordered), embeddings=array)
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

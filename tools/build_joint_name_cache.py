"""Cache T5 embeddings of every joint name in the PoseYdon corpus.

The reference trains with `skip_t5: False` and `cond_mask_prob: 0.0`, so joint
name embeddings are on EVERY forward pass, never masked. `MoDiffAE` treats them
as optional conditioning and skips the term silently when absent, which is why
their absence never raised anything -- it just trained a different model.

Run once per corpus rebuild, in the `compat` image: it carries `transformers`
(the `text` extra), and the `test`/`train` images must never need it.

    docker compose run --rm --entrypoint python compat tools/build_joint_name_cache.py

Names are taken POST-REDUCTION -- `skeleton.npz`'s `names` indexed by
`reduction_source_of` -- because that is the joint set the read path yields and
therefore the set the model is conditioned on. The encoding is
`poseydon.conditioners.text`, the reference's own rules ported.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from poseydon.conditioners.skeleton import DEFAULT_JOINT_NAME_CACHE
from poseydon.conditioners.text import encode_joint_names

RIGS = Path("data/truebones/rigs")


def reduced_names(rig: Path) -> list[str]:
    with np.load(rig / "skeleton.npz", allow_pickle=True) as data:
        full = [str(n) for n in data["names"]]
        return [full[int(i)] for i in data["reduction_source_of"]]


def main() -> int:
    rigs = sorted(p for p in RIGS.iterdir() if p.is_dir())
    ordered = sorted({name for rig in rigs for name in reduced_names(rig)})
    print(f"{len(ordered)} unique joint names across {len(rigs)} rigs")

    array = encode_joint_names(ordered)
    print(f"embeddings: {array.shape} dtype={array.dtype}")

    out = DEFAULT_JOINT_NAME_CACHE
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, names=np.array(ordered), embeddings=array)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

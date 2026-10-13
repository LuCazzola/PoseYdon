# ingest

Canonical alignment math, and the original BVH-corpus ingest behind
`poseydon ingest`. `align.py` holds the primitives (facing rotations, rotating
a whole rig, XZ centring, mean-bone-length scaling, grounding), applied in the
reference's `process_anim` order, which is part of the contract: reordering
changes the result. Scale and grounding are per skeleton, from the rig's
reference clip, so every clip of a character shares one frame.
[build: preparation](../build/preparation.md) reuses these primitives inside
its stages; `features` and `training/metrics.py` use them too.

`pipeline.py` holds rig discovery (`available_rigs`, `infer_skeleton`, used by
`scripts/process_dataset_truebones.py`) and the legacy `ingest_corpus`, which
writes aligned clips plus an index that nothing reads: the model-ready corpus
comes from `scripts/process_dataset_truebones.py` then
`scripts/build_features.py` ([build](../build/index.md)). Why this package
stays separate from `build`: **rationale not recorded**.

## Limitations

- **Input is assumed Y-up.** Nothing detects or converts the up axis, so a
  Z-up source is mis-prepared without an error.

## Reference

::: poseydon.ingest
    options:
      members: false

::: poseydon.ingest.align

::: poseydon.ingest.pipeline

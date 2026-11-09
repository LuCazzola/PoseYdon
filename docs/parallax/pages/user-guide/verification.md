# Verification

Code: `tests/test_roundtrip.py`, `tests/test_verify.py`, `tools/verify`, `scripts/verify_motion.py`, `configs/verify`

Both checks below are tooling. They judge the library's output against something the
library did not produce.

## The round-trip test compares against the raw source

`tests/test_roundtrip.py` takes every prepared clip of one dataset through the reduction
leg and the features leg, writes a file, and compares that file with the **raw source**
read through the dataset's own `source:`. The docstring calls the raw source "the one
reference the pipeline did not produce". It asserts exactly what the trip promises and
leaves out what is lost by design (see [The round trip](round-trip.md#what-the-trip-loses-by-design)).
There is one test per clip, so `-k Rig/action` narrows a run.

## The motion verifier judges FBX output against the vendor rig

`scripts/verify_motion.py` builds the stage files of one round trip (`build`), judges them
(`judge`), or judges a single FBX (`file`). It writes `report.json` and `report.txt` and
exits non-zero unless every check is `PASS` or `ALLOWED`. A `NOT RUN` line counts against
that, and `C4-symmetry` is always `NOT RUN` (not implemented), so a built run currently
always exits 1 (`tools/verify/checks.py`, `tools/verify/report.py`). `scripts/verify_motion.sh` runs
the whole sequence from the host: building in the `fbxsdk` container, reading in Blender
(`fbx`), then judging.

The pipeline's own code cannot serve as the reference, because that would be circular:

- The **loss laws** (`tools/verify/laws.py`) restate what each lossy stage is defined to
  do, written from each stage's stated definition and never imported from
  `poseydon.build` or `poseydon.features`. `tests/test_verify.py` guards that rule.
- The **reference skin** (`tools/verify/skin.py`) deforms the vendor mesh by plain linear
  blend skinning, independently of the SDK's own evaluation.
- **Blender** is a second, independent reader of the written files
  (`tools/verify/blender_read.py`).

Most tolerances in `configs/verify/verify.yaml` carry a comment giving the date, the run
and the measured floor they were set from; `inventory_tol` and `stretch_margin` do not.
Each check records what it rests on (its `Evidence`). The `Evidence` docstring says
pipeline evidence is never proof on its own, but the verdict does not read it: a
`PIPELINE` line that passes counts as a pass (`tools/verify/report.py`).

`tests/test_verify.py` turns the verifier on itself. It mutates a built round trip and
expects the verifier to go red on the check that owns each mutation.

# Verification

Code: `tests/test_roundtrip.py`, `tests/test_verify.py`, `tools/verify`, `tools/verify/blender_read.py`, `scripts/verify_motion.py`, `scripts/verify_motion.sh`, `configs/verify`

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
(`judge`), or judges a single FBX (`file`). It writes `<run>/report.json` and `<run>/report.txt` and
exits non-zero unless every check is `PASS` or `ALLOWED`. A `NOT RUN` line counts against
that, and `C4-symmetry` is always `NOT RUN` (not implemented), so a built run currently
always exits 1 (`tools/verify/checks.py`, `tools/verify/report.py`). `scripts/verify_motion.sh` runs
the whole sequence from the host: building in the `fbxsdk` container, reading in Blender
(`fbx`), then judging.

**C6** reads each file the pipeline wrote in Blender (`tools/verify/blender_read.py`, `fbx` service, run by
`scripts/verify_motion.sh`) and compares Blender's evaluated vertices with the SDK's, index to index (the importer keeps
the control-point order), after one fitted similarity per file, within `blender_tol` of extent (`independent` evidence).
The fit is judged too, so it can absorb units and axes but not a real resize or move: its scale must match the file's
unit factor (`UnitScaleFactor`/100, 0.01 for cm) and its shift must stay within `blender_tol` of extent; the note gives the
fitted scale, the expected one and the shift. A mesh Blender drops or an import that fails is `FAIL`; a skin Blender loses
is caught only through the vertex gap (nothing checks for it directly). No Blender output is `NOT RUN`, which counts
against the verdict. `blender_tol` is ten times the worst gap measured on the pipeline's files (1e-5, eight rigs). The
vendor file is read too but only noted, never judged: the readers differ on it by 7e-4 to 5.8e-3 (Blender's importer).
`tests/test_verify.py` runs in `fbxsdk`, where Blender cannot, so it ignores a `C6` `NOT RUN`.

Known disagreement, Crow (`ascend`): C6 FAILs on `2_prepared`, `3_model` and `4_roundtrip` (4.9e-3 of extent; fitted
scale 0.0039 against 0.01) while `5_zero` agrees at 3e-8. The fault is Blender's reading, not the pipeline's: the SDK's
joints on the vendor FBX match the vendor's own `__Ascend.bvh` to 2.2e-6 of extent, Blender's gap on the vendor file is the
same (5.8e-3), and Blender keeps the mesh rigid, so it poses the joints differently. Crow has a `Null` node scaled 5
with `eInheritRrSs` on every node; the exact cause is not pinned. No allowance is set: allowances are the owner's.
Proposed for the owner: a C6 `ALLOWED` entry for Crow in `configs/verify/allowances.yaml`, citing the BVH agreement.
(`tmp/verify/findings.md`)
**Sheets** (`tools/verify/sheet.py`, into `<run>/sheets/`, named in `report.json` under `run.sheets`): the reference mesh
above and the file's below, in front, side, top and 3/4 views, at the first sampled frame (standing in for the rest
pose) and at the frame of worst error. Four kinds: C2a draws each stage's own skin against the reference under the
file's joints (`<stage>_*`: skin faults only); C2b draws `4_roundtrip` against the expected pose, no fit; C3 draws
`2_prepared` and `3_model` against the expected pose put in place by the *recorded* placement; C4 draws `5_zero`
against the rig at rest, likewise (`<stage>-vs-expected_*`, `5_zero-vs-rest_*`). So a file written in the wrong place
looks it. The camera is fixed from the reference's bounds, never the file's. Vertices are coloured by their distance
from the reference as a fraction of extent on one fixed log scale (1e-6 blue to 1e-1 red, clipped beyond), printed on
the sheet. Same files give the same bytes. They are for reading and never change a verdict, except that a run which
should have written some and wrote none gets `C7-sheets: NOT RUN`; a rig with no mesh is listed under
`run.not_applicable`. `judge(run, sheets=False)` skips them (the mutation rows do).
`scripts/verify_motion.sh --smoke` carries on past a rig that fails to build.

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

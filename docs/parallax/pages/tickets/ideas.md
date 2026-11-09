# Ideas

Proposed changes and things worth investigating; nothing is broken yet. How tickets
work is in [Tickets](index.md). Newest first.

| ID | Noticed | Where | Issue |
|---|---|---|---|
| `wbjk` | 2026-11-09 | `src/poseydon/core/animation.py`, `src/poseydon/build/pipeline.py` | Every library call of Animation.as_rigid_body passes joint_translation='drop', so the checked rigid conversion never fails; consider failing (or reporting) when a non-rigid source reaches stage 2. |
| `qwsy` | 2026-11-09 | `tools/verify/report.py` | Let Report.verdict read Evidence: a PIPELINE-evidence check that passes currently counts as PASS, although the Evidence docstring says pipeline evidence is never proof alone. |
| `s8ia` | 2026-11-09 | `tools/verify/case.py`, `tools/verify/checks.py`, `tools/verify/skin.py` | Make the verifier's references independent of pipeline inputs: laws read RigRecord/skeleton.npz decisions (pruned, reduced, still joints), C3/C4 placement reuses poseydon.build.prepare.similarity, and the reference skin copies FBX.skinned_vertices; a wrongly recorded decision is then exempted by the law too. |

---
name: verify-motion
description: Judge an FBX or BVH file against the user's rig with the motion verifier. Use before changing, and before calling done, any work on src/poseydon/io, build, export or reduction, rest poses, rig manifests or tools/build_rest_fbx.py; and before any claim that a pose, mesh, skin, reference frame or the round trip is right.
---

# Verify motion

The returned FBX or BVH file is the contract. The verifier reads it as a user's tool would and compares it
with the user's rig, the source of truth. Its report is the only evidence a claim about motion or a mesh rests on.

## Steps

1. **Baseline.** Before you change anything, run the verifier on the rigs your change touches and keep the report.
   Lines already red are known; your change is judged by what moves.
2. **Change, then run again** on the same rigs and clips. Run `--smoke` before you call a change to shared code done.
3. **Read the sheets.** Open every `sheets/*.png` the report names for a line you are about to report on, with
   your Read tool. A mesh coloured above the blue end, or placed off the grey reference, is a finding even when
   the numbers pass. Iterate until the sheets look right; the owner is not the feedback loop.
4. **Report.** Quote the report: rig, clip, check id, status, value. "Done" names exactly what ran; a sample of
   rigs is a sample, never the corpus.

Done means: every line the change could affect is PASS or ALLOWED on the rigs you name, the sheets were read, and
every remaining red is reported as a finding with its rig, clip, check and value.

## Commands

All run on the host and call docker compose; nothing needs host Python.

```bash
scripts/verify_motion.sh --rig Camel [--clip bite] [--data truebones]   # build, read in Blender, judge
scripts/verify_motion.sh --smoke [--data unimate]                       # the smoke set in configs/verify/verify.yaml
scripts/verify_motion.sh file OUT.fbx --rig Camel --data truebones [--clip GEN.npz]   # a file with no source (generated motion)
scripts/verify_motion.sh file OUT.bvh --reference USER.bvh             # a BVH, with or without a mesh
```

The exit status is 0 only when every line is PASS or ALLOWED. Each run directory holds `report.txt`,
`report.json` and `sheets/`.

## Reading a report

- **Statuses:** `PASS`, `FAIL`, `ALLOWED` (an owner-set allowance covers it), `NOT RUN` (the check could not
  run; the verdict is then not PASS). Checks that cannot apply to the input, such as mesh checks on a meshless
  BVH, are listed under `not_applicable` instead.
- **Evidence**, strongest first: `independent` (vendor files, Blender), `derived` (vendor-calibrated limits, an
  authored T-pose), `pipeline` (built by the code under test: never proof on its own), `driven` (a BVH driving a
  skin through the writer).
- **C2b traced vs untraced:** traced loss is what the stage laws in `tools/verify/laws.py` explain, and is fine;
  untraced error is the verdict. A lossy stage with no law gives `NOT RUN`, and the fix is a law, not a tolerance.
- What each check covers, its blind spots and its tolerances: `docs/parallax/pages/user-guide/verification.md`.

## Red is a finding

A red line means the file or the verifier is wrong; find out which and report it. Checks, tolerances and the
mutation rows in `tests/test_verify.py` change only when the requirement changes and the owner was told.
Allowances (`configs/verify/allowances.yaml`) and symmetry limits (`configs/verify/limits.yaml`) are the owner's:
propose them with the evidence, and leave the files as they are.

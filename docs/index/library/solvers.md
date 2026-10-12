# solvers

`src/poseydon/solvers/`, library. Inverse kinematics: fitting joint rotations
to desired positions, batched on any device.

## Purpose

`solvers` owns the IK seam (`Solver`, `IKTerm`), the batched gradient-descent
solver, and its weighted objective terms.

It does **not** decide when IK runs. That's `positions_ik` in
`src/poseydon/features/reconstruct.py`, see [features](features.md). It also
doesn't own forward kinematics (`core.torch_kinematics`, see [core](core.md)).

## Key files and types

| file | read it for |
|---|---|
| `src/poseydon/solvers/base.py` | `Solver`, `SolverSkeleton` (what a solver needs to know about the rig), `IKTerm`; the `SOLVERS` and `IK_TERMS` registries |
| `src/poseydon/solvers/gradient_ik.py` | `GradientIK` (`gradient_ik`): every frame solved at once, on whatever device the tensors live on |
| `src/poseydon/solvers/terms.py` | `position`, `smoothness` (kills Jacobian-IK jitter), `joint_limits`, `contact_pin` (holds a planted joint still) |

## How it connects

Consumes `core.torch_kinematics` (differentiable FK) and `core.rotations`.

Produces solved rotations for `features/reconstruct.py` (`SolvedPositions`),
which `poseydon sample reconstruct.name=positions_ik`, retargeting and
validation use.

Called by: `features/reconstruct.py`; tuned by `tools/ik_grid_search.py` and
`tools/ik_contact_compare.py`; compared against the reference by
`tools/reference/ik_compare.py`.

## Why it is shaped this way

- **A seam, not a fixed dependency.** The reference calls one fixed-iteration
  CPU Jacobian solver with no joint limits or weighting, at both ingest and
  export, and the export call determines the visual quality of every BVH it
  produces. Source: `src/poseydon/solvers/base.py` docstring.
- **Rotation-space parameterization.** Bone lengths are preserved by
  construction, so no term has to defend them. Source:
  `src/poseydon/solvers/base.py` docstring.
- **Batched gradient descent on the GPU** solves every frame at once instead of
  iterating per frame. Source: commit `46d0fb5`.
- **The term weights in use**: chosen by grid search, foot-contact term
  included (`tools/ik_grid_search.py`). The chosen values themselves are
  **rationale not recorded** outside that tool's output.

## Gotchas

- **IK can't recover twist.** Rotation about a bone's own axis doesn't move its
  children, so the solve plateaus around 1% of a bone length on average (about
  15% for the worst joint), not at zero. Don't chase it as a bug. Source:
  `src/poseydon/features/reconstruct.py` docstring.
- **Recovery from features doesn't need IK.** Rotations are in the
  representation (`features/recover.py`). IK is only for outputs with
  positions but no consistent rotations.
- **Bone-length drift must be measured before the solve.** After it, drift is
  zero by construction. See [training](training.md).

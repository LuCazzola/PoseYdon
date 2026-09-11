# Testing

Real data is primary. The suite runs against seven Truebones clips spanning
bipeds, quadrupeds and millipeds at 31 to 63 joints.

The decisive test is **golden parity**: features extracted by PoseYdon reproduce
the reference's own `.npy` arrays for all seven skeletons, every block, with foot
contact bit-exact. Tolerance is bounded by the file format — BVH stores six
decimals — not by float64.

Tests requiring those fixtures skip when the reference checkout is absent, so a
fresh clone still runs everything else.

### Passing is not the same as covering

A green suite proves nothing about a test that cannot fail. The parity and unit
tests are periodically checked by **mutation**: deliberately breaking a
behaviour and confirming something goes red.

That has already earned its place twice. The camera tests reimplemented the
framing formula instead of calling it, so they exercised a copy and stayed green
through the exact regression they were written for. And the whole diffusion
core — `corrupt`, the posterior, both sampler loops — survived eleven separate
mutations untouched, including signal and noise swapped in the forward process.
The math was right, verified against the reference to 3e-06; nothing in CI would
have noticed it breaking.

If you add a test here, break the thing it covers and watch it fail first.


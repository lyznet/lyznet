# Development

This file is the tool-independent development record for humans and coding
agents. It identifies the current task, its acceptance criteria, and the
ordered queue. Keep it short and update it when a task begins or ends.

## Working rules

- Work on one coherent capability at a time.
- Include focused tests with each capability.
- Keep formal mathematical conditions visible in `verification/core/`.
- Distinguish empirical evidence from formal verification.
- Do not migrate generated results, checkpoints, paper artifacts, or WIP code.
- Do not expand a task beyond its acceptance criteria without approval.
- Record the tests run before marking a task complete.

## Release target

The first public release is `0.1.0`.

## Current task

### M06 — Optional CROWN verification of quadratic funnels

Status: pending

Objective: add an optional CROWN backend for the same two-stage quadratic
funnel theorem without bundling alpha-beta-CROWN.

Acceptance criteria:

- The CROWN theorem and every bounded expression are visible in one core
  module.
- PyTorch and alpha-beta-CROWN remain optional external dependencies.
- The backend requires an exact equilibrium and reports unsupported dynamics
  explicitly.
- A minimal polynomial example exercises the backend.
- Focused tests cover verified and unsupported outcomes.

## Queue

| ID | Capability |
|---|---|
| M07 | Global and polynomial Lyapunov verification |
| M08 | Quadratic control-Lyapunov verification |
| M09 | Empirical screening and plotting |
| M10 | Plain and anchored neural Lyapunov workflows |
| M11 | Viability-aware state constraints |
| M12 | Documentation, packaging, and the `0.1.0` release |

Automatic search policies, homogeneous certificates, compositional
verification, benchmark campaigns, and paper artifacts are deferred until the
corresponding general interfaces are ready.

## Completed work

### M01 — Package foundation

Created the installable package skeleton, generated development versions,
Linux CI, and the first import test.

Validation:

- `.venv/bin/python -m pip install -e '.[test]'`
- `.venv/bin/python -c 'import lyznet; print(lyznet.__version__)'`
- `.venv/bin/python -m pytest -q` (`1 passed`)
- `git diff --check`

### M02 — Continuous-time systems and sets

Added immutable autonomous and control-affine systems, numerical views, basic
sets, and a runnable Van der Pol definition.

Validation: `.venv/bin/python -m pytest -q` (`45 passed`).

### M03 — Quadratic Lyapunov construction

Added system linearization and the numerically checked construction
`A.T P + P A = -Q` for a Hurwitz linearization.

Validation: `.venv/bin/python -m pytest -q` (`53 passed`).

### M04 — Z3 verification of quadratic Lyapunov sublevels

Added exact-rational Z3 conditions for positivity, strict sublevel decrease,
and box-boundary exclusion, together with `find_lyapunov` and a runnable Van
der Pol workflow.

Validation:

- `.venv/bin/python -m pytest -q` (`60 passed`)
- `vdp_roa.py`: `VERIFIED`, level `2.3044`

### M05 — dReal verification of quadratic Lyapunov funnels

Added a two-stage dReal proof: local contraction and invariance followed by
strict decrease on an outer annulus. Numerical equilibrium evidence is
retained, and approximate centers receive a separate invariance proof.

Validation:

- `.venv/bin/python -m pytest -q` (`72 passed`)
- `pendulum_roa.py`: `VERIFIED`, local `1.7371`, outer `3.3332`

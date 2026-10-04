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

### M02 — Continuous-time systems and sets

Status: pending

Objective: migrate the continuous-time system definitions and mathematical
set value objects needed by later analysis and verification code.

Acceptance criteria:

- Autonomous and control-affine continuous-time systems are immutable and
  validate their dimensions and symbols.
- Boxes, Euclidean balls, whole space, and symbolic sublevel sets have small,
  explicit value-object interfaces.
- The public package API exposes only the types needed to define systems and
  sets.
- Focused tests cover valid construction and rejected inputs.
- No analysis, learning, plotting, or verification code is included.

## Queue

| ID | Capability |
|---|---|
| M03 | Quadratic Lyapunov and control-Lyapunov analysis |
| M04 | Z3 verification of quadratic Lyapunov sublevels |
| M05 | dReal verification of quadratic Lyapunov funnels |
| M06 | Optional CROWN verification of quadratic funnels |
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

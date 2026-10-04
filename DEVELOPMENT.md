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

### M01 — Package foundation

Status: pending

Objective: create a small, installable Python package without scientific
functionality.

Acceptance criteria:

- `pip install -e .` succeeds in a fresh Python 3.11 environment.
- `import lyznet` succeeds.
- The package exposes its generated development version.
- A package-import test passes.
- Continuous integration runs the test on Linux.

Expected files:

- `.github/workflows/tests.yml`
- `.gitignore`
- `LICENSE`
- `pyproject.toml`
- `src/lyznet/__init__.py`
- `tests/test_package.py`

## Queue

| ID | Capability |
|---|---|
| M02 | Continuous-time systems and sets |
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

No implementation task has been completed.

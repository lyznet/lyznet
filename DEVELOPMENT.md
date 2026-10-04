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

M07--M11 are complete and awaiting review. M12 has not started.

## Queue

| ID | Capability |
|---|---|
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

### M06 — Optional CROWN verification of quadratic funnels

Added an optional external alpha-beta-CROWN backend for directly proposed
local and outer levels. The local proof bounds the nonlinear Jacobian
remainder; the outer proof checks strict annular decrease.

Validation:

- Base environment: `.venv/bin/python -m pytest -q`
  (`76 passed, 1 skipped`).
- External CROWN environment: `python -m pytest -q`
  (`77 passed`).
- `vdp_roa_crown.py`: `VERIFIED`, local `0.4`, outer `2.3`.

### M07 — Global and polynomial Lyapunov verification

Added whole-space quadratic verification with Z3 and dReal, an immutable
polynomial-certificate representation, exact Z3 polynomial synthesis, and an
independent fixed-candidate verification step. The proof core displays the
positivity, strict-decrease, and radial-unboundedness conditions directly.

Validation:

- `.venv/bin/python -m pytest -q` (`80 passed, 1 skipped`).
- `inverted_pendulum.py`: global quadratic certificate `VERIFIED` by Z3.
- `polynomial_z3.py`: synthesized `V=(x1^2+x2^2)/2`, then independently
  `VERIFIED` by Z3.
- `git diff --check`.

### M08 — Quadratic control-Lyapunov verification

Added immutable LQR quadratic CLF candidates, global and bounded Artstein
proofs with Z3, and symmetric two-stage dReal/CROWN funnel proofs. The formal
core shows the Artstein implication and local contraction conditions directly.

Validation:

- Base environment: `.venv/bin/python -m pytest -q`
  (`85 passed, 2 skipped`).
- CROWN environment: `python -m pytest -q tests/test_control_lyapunov.py`
  (`6 passed`).
- `vdp_global.py`: global quadratic CLF `VERIFIED` by Z3.
- `vdp_roa_dreal.py`: local `2.8708`, outer `42.096`, both `VERIFIED`.
- `linear_roa_crown.py`: local `0.1`, outer `0.2`, both `VERIFIED`.

### M09 — Empirical screening and plotting

Added deterministic Sobol screening for the quadratic LF and CLF predicates,
automatic conservative CROWN proposals, and a compact optional Matplotlib
plotter. Empirical proposal records explicitly report that they are not formal
evidence; fixed proposed levels are still checked by CROWN.

Validation:

- `.venv/bin/python -m pytest -q` (`88 passed, 2 skipped`).
- Automatic CROWN replay: screened local `0.3804`, outer `2.0837`; both
  formally `VERIFIED`.
- `vdp_roa_crown.py` saved a two-panel PDF with local and verified contours.
- `git diff --check`.

### M10 — Plain and anchored neural Lyapunov workflows

Added deterministic trajectory data, compact plain and quadratic-anchored
neural candidates, two-stage PINN training, empirical level screening, and a
CROWN reach-and-stay proof bootstrapped by a verified quadratic sublevel. The
anchored proof works with its latent function and sign-equivalent derivative,
so CROWN need not bound the final tanh layer.

Validation:

- Base environment: `.venv/bin/python -m pytest -q`
  (`88 passed, 3 skipped`).
- CROWN environment: `python -m pytest -q tests/test_neural_lyapunov.py`
  (`6 passed`).
- `vdp_mu_1.py`: trained in the low effort setting, screened levels `0.2073`
  and `0.7158`, formally `VERIFIED` by CROWN, and saved its plot.
- `git diff --check`.

### M11 — Viability-aware state constraints

Added the admissible-set convention ``K=intersection_i {h_i<=1}`` to neural
data generation, collocation sampling, empirical screening, formal proof, and
plotting. The CROWN core visibly proves decrease only in ``K``, strict Nagumo
conditions near every active ``h_i=1`` boundary, and either separation or
strict inward flow on each box face. Formal results now represent the actual
restricted set ``D intersect K intersect {W<=c}``.

Validation:

- Base environment: `.venv/bin/python -m pytest -q`
  (`89 passed, 3 skipped`).
- Complete CROWN environment: `python -m pytest -q` (`99 passed`).
- `linear_disk.py`: screened local/outer levels `0.0455` and `0.9`, formally
  `VERIFIED` the state constraint and all four viable faces, estimated
  constrained volume `3.80444`, and saved the masked constraint plot.
- `git diff --check`.

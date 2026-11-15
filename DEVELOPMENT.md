# Development

This is the tool-independent handoff for the current development task. Keep
release history in `CHANGELOG.md` and quantitative research gates in
`examples/reproductions/README.md`.

## Working rules

- Work on one coherent capability at a time.
- Keep formal conditions visible in `src/lyznet/verification/core/`.
- Distinguish empirical evidence from formal verification.
- Include focused tests and record validation before completing a task.
- Do not commit generated data, models, plots, or solver ledgers.

## Current task

Prepare the first public release, `0.1.0`.

Acceptance criteria:

- Every minimal workflow completes in its documented environment; workflows
  requesting a proof return `VERIFIED`.
- The Automatica and homogeneous reproduction gates pass from fresh models.
- The base and CROWN test suites pass.
- CI builds and smoke-tests the wheel and source distribution.
- Generated artifacts remain outside Git and both distributions.

Current validation:

- Base tests: `94 passed, 4 skipped`.
- CROWN tests: `116 passed` with one upstream PyTorch warning.
- All 15 minimal workflow scripts complete in their documented environments;
  every workflow requesting a proof returns `VERIFIED`.
- Cumulative mid effort on the VDP example retained the low checkpoint,
  added `6,500` updates, screened both candidates, and formally verified only
  the stronger candidate. Its sampled area was `13.2211`; the cached-data
  validation took `49.56 s` on an H100. The added high checkpoint screened at
  `13.2254` without displacing either lower-effort checkpoint.
- Fresh H100 runs formally verified the two-machine candidate with sampled
  area `8.54462` in `75.78 s` and the exact-basin candidate with area
  `11.5884` in `90.37 s`.
- The exact `0.1.0` wheel and source distribution passed `twine check`.
- A clean Python 3.11 wheel installation passed `pip check` and a Z3 proof.

Remaining release steps:

1. Review and commit the release candidate.
2. Make the GitHub repository public and configure PyPI publishing.
3. Tag the release commit as `v0.1.0` and publish it.

## Next

Develop neural control-Lyapunov workflows and reproduce the ACC 2025 results.

# LyZNet

LyZNet supports the analysis, learning, and formal verification of
certificates for dynamical systems.

This repository contains the new implementation and has not yet produced a
public release. See [DEVELOPMENT.md](DEVELOPMENT.md) for the current task and
release plan.

The previous implementation is preserved in
[lyznet-legacy](https://github.com/lyznet/lyznet-legacy).

## Current minimal workflow

Install the package with its test, Z3, and dReal extras:

```bash
python -m pip install -e '.[test,z3,dreal]'
```

To review system construction, run:

```bash
python examples/minimal_workflows/system_definition/vdp.py
```

This script defines autonomous and control-affine Van der Pol systems and
evaluates their dynamics.

To construct and formally verify a quadratic Lyapunov sublevel, run:

```bash
python examples/minimal_workflows/quadratic_lyapunov/vdp_roa.py
```

To run the two-stage dReal proof for a nonpolynomial pendulum, run:

```bash
python examples/minimal_workflows/quadratic_lyapunov/pendulum_roa.py
```

The optional CROWN backend uses empirical screening to propose local and outer
levels, then verifies those fixed levels without formal bisection:

```bash
python -m pip install -e '.[crown]'
python examples/minimal_workflows/quadratic_lyapunov/vdp_roa_crown.py
```

Install alpha-beta-CROWN 0.7 separately from its
[upstream repository](https://github.com/Verified-Intelligence/alpha-beta-CROWN);
LyZNet does not bundle the verifier.

The returned assessment records the sampled proposals as empirical evidence;
only the subsequent CROWN obligations establish a formal result. Install the
`plot` extra to save the example's PDF:

```bash
python -m pip install -e '.[plot]'
```

To prove global stability with a quadratic certificate, or synthesize and
independently verify a polynomial certificate with Z3, run:

```bash
python examples/minimal_workflows/global_lyapunov/inverted_pendulum.py
python examples/minimal_workflows/global_lyapunov/polynomial_z3.py
```

For a control-affine system, construct an LQR quadratic control-Lyapunov
function and verify either a global Artstein condition or a bounded two-stage
funnel:

```bash
python examples/minimal_workflows/quadratic_clf/vdp_global.py
python examples/minimal_workflows/quadratic_clf/vdp_roa_dreal.py
```

The optional CROWN environment can run
`examples/minimal_workflows/quadratic_clf/linear_roa_crown.py`.

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

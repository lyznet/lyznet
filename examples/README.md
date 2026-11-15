# Examples

The examples have two purposes:

- `minimal_workflows/` demonstrates one public LyZNet workflow at a time
  with ordinary defaults and little configuration.
- `reproductions/` records the explicit data, training, verification, and
  regression settings used to reproduce published results.

Reproduction recipes may be tailored to a published experiment. They keep
every tailored parameter visible and do not change the package defaults.
Generated data, model checkpoints, plots, and proof ledgers are written to
local `results/` directories and are not tracked by Git.

The neural examples include both plain and anchored architectures. The
two-machine and exact-basin scripts fix a reported random seed because neural
optimization is not deterministic across seeds.

```bash
.venv-crown/bin/python examples/minimal_workflows/pinn_lyapunov/vdp_mu_1.py
.venv-crown/bin/python examples/minimal_workflows/pinn_lyapunov/vdp_mu_3.py
.venv-crown/bin/python examples/minimal_workflows/pinn_lyapunov/polynomial_2d.py
.venv-crown/bin/python examples/minimal_workflows/pinn_lyapunov/two_machine.py
.venv-crown/bin/python examples/minimal_workflows/pinn_lyapunov/exact_basin.py
```

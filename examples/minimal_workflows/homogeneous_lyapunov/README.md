# Global homogeneous Lyapunov examples

These workflows reproduce the autonomous examples from:

> M. Fitzsimmons and J. Liu, “A Neural Network Approach to Finding Global
> Lyapunov Functions for Homogeneous Vector Fields,” *IEEE Control Systems
> Letters*, vol. 8, pp. 670–675, 2024.
> [doi:10.1109/LCSYS.2024.3402142][paper]

[paper]: https://doi.org/10.1109/LCSYS.2024.3402142

Both examples use the same deterministic positive architecture,

```text
V(x) = ||x||^p exp(a^T tanh(W x/||x|| + b)).
```

The hidden angular features are fixed. Two linear programs select the output
coefficients: the first maximizes sampled decrease margin, and the second
minimizes their one-norm while retaining part of that margin. This numerical
construction is only a candidate search. dReal independently proves
positivity and strict decrease on the complete unit sphere, which implies the
global Lyapunov conditions by homogeneity.

Run the published degree-five and irrational benchmarks from the repository
root:

```bash
.venv-crown/bin/python examples/minimal_workflows/homogeneous_lyapunov/degree_five.py
.venv-crown/bin/python examples/minimal_workflows/homogeneous_lyapunov/irrational.py
```

The irrational formula has a removable singularity at the equilibrium. Its
system definition explicitly assigns the continuous value zero at the origin;
the verifier separately proves that every denominator and even-root radicand
is uniformly well defined on the unit sphere.

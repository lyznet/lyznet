# Automatica 2025 reproduction

These examples learn Lyapunov functions from trajectory values and the Zubov
PDE, propose level sets by deterministic sampling, and formally verify the
fixed proposals with CROWN.

The three scripts reproduce the benchmark systems from:

> J. Liu, Y. Meng, M. Fitzsimmons, and R. Zhou, “Physics-Informed Neural
> Network Lyapunov Functions: PDE Characterization, Learning, and
> Verification,” *Automatica*, vol. 175, 112193, 2025.
> [doi:10.1016/j.automatica.2025.112193][automatica-doi]

[automatica-doi]: https://doi.org/10.1016/j.automatica.2025.112193

| Script | Paper result | Release requirement |
| --- | ---: | ---: |
| `vdp_mu_1.py` | 95.64% of the simulated ROA | verified area at least 13.1032 |
| `vdp_mu_3.py` | 85.12% of the simulated ROA | verified area at least 17.5327 |
| `two_machine.py` | 82.52% of the simulated ROA | verified area at least 8.2025 |

The percentages use the trajectory-normalized measure reported in the paper.
The area requirements use eight fixed, independently scrambled Sobol banks
and are therefore more suitable for regression testing. Every replicate must
reach the stated threshold. Area remains an empirical estimate; the CROWN
result formally verifies the corresponding level set.

Run a benchmark from the repository root with the optional CROWN and plotting
dependencies installed:

```bash
.venv-crown/bin/python examples/reproductions/automatica_2025/vdp_mu_1.py
```

Generated data, model checkpoints, and plots are written to the local
`results/` directory, which is not tracked by Git. A compatible checkpoint is
loaded unless model overwriting is requested through the public workflow.

The fresh `0.1.0` candidate replay and exact model fingerprints are recorded
in the [reproduction gates](../README.md). The reproduction files keep their
workload and loss settings explicit so the published result does not depend
on an undocumented system-name-specific preset. These paper-specific settings
do not define LyZNet's public defaults. Each script deliberately omits
`effort`, so its explicit learner options define one fixed checkpoint rather
than a cumulative low, mid, and high search.

# PINN Lyapunov functions

These workflows learn neural Lyapunov functions from trajectory data and a
Zubov PDE, screen candidate level sets, and formally verify the selected
candidate with CROWN.

`vdp_mu_1.py` retains the low-effort anchored Van der Pol workflow;
`vdp_mu_3.py` uses the published fixed plain-network recipe on the larger
domain. `polynomial_2d.py` is the polynomial saddle oscillator from the CDC
2023 paper. `two_machine.py` and `exact_basin.py` provide two further
nonlinear examples.

The method and the Van der Pol and two-machine benchmarks are developed in:

> J. Liu, Y. Meng, M. Fitzsimmons, and R. Zhou, “Physics-Informed Neural
> Network Lyapunov Functions: PDE Characterization, Learning, and
> Verification,” *Automatica* 175, 112193, 2025.
> [doi:10.1016/j.automatica.2025.112193](https://doi.org/10.1016/j.automatica.2025.112193)

The polynomial example first appeared in:

> J. Liu, Y. Meng, M. Fitzsimmons, and R. Zhou, “Towards Learning and
> Verifying Maximal Neural Lyapunov Functions,” *IEEE Conference on Decision
> and Control*, 8012–8019, 2023.
> [doi:10.1109/CDC49753.2023.10383299](https://doi.org/10.1109/CDC49753.2023.10383299)

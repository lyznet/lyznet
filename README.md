# LyZNet: Learning and formally verifying Lyapunov functions for nonlinear dynamical systems

LyZNet is an open-source Python toolbox for constructing, learning, and
formally verifying Lyapunov certificates and regions of attraction. It
combines classical Lyapunov analysis, physics-informed neural networks,
satisfiability modulo theories (SMT) solvers, and neural network verifiers to
obtain rigorous stability certificates.

## Installation

LyZNet 0.1.0 is tested with Python 3.11 on Linux.

```bash
python -m pip install lyznet
```

See the [installation guide](INSTALL.md) for optional dependencies, GPU
support, dReal, and CROWN.

## Quick start

```python
import sympy as sp
import lyznet as lz

x1, x2 = sp.symbols("x1 x2")
system = lz.System.continuous(
    dynamics=(-x2, x1 - (1 - x1**2) * x2),
    domain=((-2.5, 2.5), (-3.5, 3.5)),
)

lz.find_lyap(system, verify="z3").require_verified()
```

This constructs the quadratic Lyapunov function

```text
V(x) = 1.5 x1^2 - x1 x2 + x2^2
```

and uses Z3 to verify that its sublevel set `V(x) <= 2.3044` is a region of
attraction within the specified domain.

To learn a neural Lyapunov function from trajectory data and the Zubov PDE,
then verify it with CROWN, use the same system:

```python
lz.find_lyap(system, family="neural", verify="crown", plot=True).require_verified()
```

See [`examples/minimal_workflows`](examples/minimal_workflows) for concise
workflows and [`examples/reproductions`](examples/reproductions) for published
results. The previous implementation is preserved in
[lyznet-legacy](https://github.com/lyznet/lyznet-legacy) [1, 2].

## References

If you use LyZNet in your research, please consider citing the relevant
papers:

1. J. Liu, Y. Meng, M. Fitzsimmons, and R. Zhou, “LyZNet: A
   Lightweight Python Tool for Learning and Verifying Neural Lyapunov
   Functions and Regions of Attraction,” *Proceedings of the 27th ACM
   International Conference on Hybrid Systems: Computation and Control
   (HSCC)*, pp. 1–8, 2024.
   [doi:10.1145/3641513.3650134](https://doi.org/10.1145/3641513.3650134)
2. J. Liu, Y. Meng, M. Fitzsimmons, and R. Zhou, “Physics-Informed
   Neural Network Lyapunov Functions: PDE Characterization, Learning, and
   Verification,” *Automatica*, vol. 175, 112193, 2025.
   [doi:10.1016/j.automatica.2025.112193](https://doi.org/10.1016/j.automatica.2025.112193)
3. J. Liu, Y. Meng, M. Fitzsimmons, and R. Zhou, “Towards Learning and
   Verifying Maximal Neural Lyapunov Functions,” *2023 62nd IEEE Conference
   on Decision and Control (CDC)*, pp. 8012–8019, 2023.
   [doi:10.1109/CDC49753.2023.10383299](https://doi.org/10.1109/CDC49753.2023.10383299)

```bibtex
@inproceedings{liu2024lyznet,
  author={Liu, Jun and Meng, Yiming and Fitzsimmons, Maxwell and Zhou, Ruikun},
  title={{LyZNet}: A Lightweight Python Tool for Learning and Verifying
    Neural Lyapunov Functions and Regions of Attraction},
  booktitle={Proceedings of the 27th ACM International Conference on
    Hybrid Systems: Computation and Control},
  pages={1--8},
  year={2024},
  publisher={ACM},
  doi={10.1145/3641513.3650134}
}

@article{liu2025physics,
  author={Liu, Jun and Meng, Yiming and Fitzsimmons, Maxwell and Zhou, Ruikun},
  title={Physics-Informed Neural Network Lyapunov Functions: {PDE}
    Characterization, Learning, and Verification},
  journal={Automatica},
  volume={175},
  pages={112193},
  year={2025},
  doi={10.1016/j.automatica.2025.112193}
}

@inproceedings{liu2023towards,
  author={Liu, Jun and Meng, Yiming and Fitzsimmons, Maxwell and Zhou, Ruikun},
  title={Towards Learning and Verifying Maximal Neural Lyapunov Functions},
  booktitle={2023 62nd IEEE Conference on Decision and Control ({CDC})},
  pages={8012--8019},
  year={2023},
  publisher={IEEE},
  doi={10.1109/CDC49753.2023.10383299}
}
```

## Contact

Jun Liu — [j.liu@uwaterloo.ca](mailto:j.liu@uwaterloo.ca)

## License

LyZNet is released under the [Apache License 2.0](LICENSE).

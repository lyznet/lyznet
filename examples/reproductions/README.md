# Reproductions and release gates

This file records the published capabilities that every future LyZNet release
must retain. Formal status and empirical geometry are deliberately separate:
a sampled area is not a proof, while a formal proof does not determine area
without an additional geometric calculation.

## Automatica 2025: neural regions of attraction

All three candidates must be formally verified. Their verified sublevel sets
must also meet the following deterministic sampled-area thresholds on the
paper domains.

| Example | Domain | Published ROA coverage | Minimum verified-set area |
| --- | --- | ---: | ---: |
| Van der Pol, mu=1 | `[-2.5,2.5] x [-3.5,3.5]` | 95.64% | 13.1032 |
| Van der Pol, mu=3 | `[-3,3] x [-6,6]` | 85.12% | 17.5327 |
| Two-machine power system | `[-2,3] x [-3,1.5]` | 82.52% | 8.2025 |

The published percentages normalize by a simulation-based estimate of the
ROA. The area thresholds reconstruct the corresponding physical areas and
avoid using the arbitrary numerical scale of a Lyapunov function. The release
gate uses eight fixed, independently scrambled Sobol replicates of `2**17`
points. Every replicate must reach the threshold. These samples are separate
from the bank used to propose the verified level.

### Fresh `0.1.0` candidate replay

A clean H100 replay on 2026-10-04 used PyTorch 2.11, the documented seed 123,
fresh trajectory integration, fresh training, and CROWN verification. All
three release gates passed:

| Example | CROWN-verified level | Held-out area | Total time | Training fingerprint |
| --- | ---: | ---: | ---: | --- |
| Van der Pol, mu=1 | 0.916 | 13.1710 +/- 0.0011 | 154.24 s | `15145a3689bac382` |
| Van der Pol, mu=3 | 0.839 | 17.9931 +/- 0.0017 | 145.79 s | `83a7adab6a699d0b` |
| Two-machine power system | 0.745 | 8.2155 +/- 0.0009 | 218.14 s | `43d34f43b9049519` |

The area columns evaluate the exact formally verified candidates on the
independent eight-replicate bank. They remain empirical geometry estimates;
the CROWN result, rather than the area calculation, establishes the Lyapunov
conditions.

Reference: J. Liu, Y. Meng, M. Fitzsimmons, and R. Zhou,
“Physics-Informed Neural Network Lyapunov Functions: PDE Characterization,
Learning, and Verification,” *Automatica* 175, 112193, 2025.
[doi:10.1016/j.automatica.2025.112193](https://doi.org/10.1016/j.automatica.2025.112193)

## HSCC 2024: global nonlinear stability

`examples/minimal_workflows/quadratic_lyapunov/inverted_pendulum_global.py`
must return a formally `VERIFIED` global quadratic Lyapunov certificate. This
is a binary regression requirement; the numerical scaling of the quadratic
matrix is not part of the requirement.

Reference: J. Liu, Y. Meng, M. Fitzsimmons, and R. Zhou, “LyZNet: A
Lightweight Python Tool for Learning and Verifying Neural Lyapunov Functions
and Regions of Attraction,” *HSCC*, 2024.
[doi:10.1145/3641513.3650134](https://doi.org/10.1145/3641513.3650134)

## IEEE Control Systems Letters 2024: homogeneous systems

The degree-five polynomial and irrational homogeneous examples must each
produce a global homogeneous neural Lyapunov function followed by a formally
`VERIFIED` dReal proof. This is a binary requirement: homogeneity makes every
positive level set a scaled copy, so a bounded-set volume is not an intrinsic
performance measure.

A clean `0.1.0` candidate replay on 2026-10-04 used the common deterministic
width-50 feature basis. Minimum-one-norm fitting retained only coefficients
needed by each candidate:

| Example | Active angular features | Synthesis | dReal proof | Status |
| --- | ---: | ---: | ---: | --- |
| Degree-five polynomial | 24 | 2.65 s | 54.59 s | `VERIFIED` |
| Irrational homogeneous | 0 | 9.78 s | 0.01 s | `VERIFIED` |

The irrational proof includes a separate obligation showing that all
denominators and even-root radicands are uniformly well defined on the unit
sphere. The zero active-feature count means the convex search correctly found
that a radial homogeneous candidate suffices; its stored network retains one
zero-weight feature for a uniform artifact format.

Reference: M. Fitzsimmons and J. Liu, “A Neural Network Approach to Finding
Global Lyapunov Functions for Homogeneous Vector Fields,” *IEEE Control
Systems Letters*, vol. 8, pp. 670–675, 2024.
[doi:10.1109/LCSYS.2024.3402142](https://doi.org/10.1109/LCSYS.2024.3402142)

## Release procedure

Before a release:

1. Retrain the Automatica candidates with their documented seeds.
2. Formally verify the resulting fixed candidates with CROWN.
3. Confirm the held-out sampled-area thresholds above.
4. Re-synthesize and formally verify both homogeneous examples.
5. Record the environment, device, elapsed times, and resulting metrics in
   the release notes without changing these acceptance thresholds.

# Changelog

Public development releases increment the middle number (`0.1.0`, `0.2.0`,
`0.3.0`, and so on). The last number is reserved for corrections to an
already published release. Before version 1.0, release notes identify every
incompatible interface change.

## [Unreleased]

## [0.1.0] - 2026-11-15

### Added

- Immutable continuous-time autonomous and control-affine systems, box and
  ball domains, sublevel sets, and numerical equilibrium solving.
- Quadratic Lyapunov and control-Lyapunov construction from linearization.
- Bounded and global verification with Z3, dReal, and optional CROWN
  backends.
- Global polynomial Lyapunov synthesis with independent fixed-candidate
  verification.
- Plain and quadratic-anchored neural Lyapunov learning from trajectory data
  and the Zubov PDE.
- Cumulative low, mid, and high neural effort checkpoints with common
  empirical screening and selective formal verification.
- Empirical screening separated explicitly from formal verification.
- State constraints and viability-aware neural verification.
- Global homogeneous neural Lyapunov synthesis and dReal verification on the
  unit sphere.
- Minimal workflows and exact Automatica 2025 reproduction recipes.

[Unreleased]: https://github.com/lyznet/lyznet/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/lyznet/lyznet/releases/tag/v0.1.0

"""Reproduce the Automatica neural-ROA benchmark for Van der Pol, mu=1."""

from pathlib import Path

import sympy as sp

import lyznet as lz
from _regression import require_verified_area


x1, x2 = sp.symbols("x1 x2")
system = lz.System.continuous(
    states=(x1, x2),
    dynamics=(-x2, x1 - (1 - x1**2) * x2),
    domain=((-2.5, 2.5), (-3.5, 3.5)),
    name="van_der_pol_mu_1",
)

MINIMUM_VERIFIED_AREA = 13.1032

if __name__ == "__main__":
    result = lz.find_lyapunov(
        system,
        family="neural",
        architecture="plain",
        verify="crown",
        plot=True,
        results_dir=Path(__file__).parent / "results",
        seed=123,
        data_options={
            "sample_count": 3_000,
            "sampling": "random",
            "include_equilibrium": False,
            "terminate_on_domain_exit": False,
            "step_size": 0.02,
            "max_time": 100.0,
            "convergence_radius": 1e-5,
            "require_termination": True,
        },
        learner_options={
            "hidden_layers": 2,
            "width": 30,
            "sampling_seed": 123,
            "compile_training": True,
            "warmup_steps": 20_000,
            "warmup_learning_rate": 1e-3,
            "data_batch_size": None,
            "collocation_points": 300_000,
            "collocation_batch_size": 4_096,
            "correction_points": 2**19,
            "correction_batch_size": 4_096,
            "correction_steps": 15_000,
            "correction_learning_rate": 1.5e-5,
            "warmup_data_weight": 1.0,
            "data_weight": 100.0,
            "pde_weight": 1.0,
            "positivity_weight": 10.0,
            "decrease_weight": 500.0,
            "correction_level_bounds": (0.22, 0.93),
            "range_weight": 10.0,
            "plateau_gradient_weight": 1.0,
            "warmup_range_weight": 10.0,
            "warmup_plateau_gradient_weight": 1.0,
            "boundary_points": 2**14,
        },
        verification_options={
            "screening_samples": 2**18,
            "screening_seed": 20260824,
            "search_precision": 1e-3,
        },
        learner_verbose=True,
        verifier_verbose=True,
    )
    require_verified_area(result, MINIMUM_VERIFIED_AREA)

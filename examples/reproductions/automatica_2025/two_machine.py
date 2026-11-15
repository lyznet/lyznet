"""Reproduce the Automatica neural-ROA benchmark for two machines."""

from pathlib import Path

import sympy as sp

import lyznet as lz
from _regression import require_verified_area


x1, x2 = sp.symbols("x1 x2")
delta = sp.Float("3.141592653589793") / 3
system = lz.System.continuous(
    states=(x1, x2),
    dynamics=(
        x2,
        -sp.Rational(1, 2) * x2
        - (sp.sin(x1 + delta) - sp.sin(delta)),
    ),
    domain=((-2.0, 3.0), (-3.0, 1.5)),
    name="two_machine",
)

MINIMUM_VERIFIED_AREA = 8.2025

CORRECTION_STAGES = (
    lz.PINNCorrectionStage(
        steps=10_000,
        learning_rate=5e-4,
        pde_weight=0.0,
        decrease_weight=0.0,
        condition_source="uniform",
        condition_seed=987,
        data_batch_size=8_192,
    ),
    lz.PINNCorrectionStage(
        steps=5_000,
        learning_rate=2e-5,
        data_weight=10.0,
        decrease_weight=30.0,
        condition_source="target_data",
        condition_target_bounds=(0.5, 0.99),
        data_batch_size=8_192,
    ),
    lz.PINNCorrectionStage(
        steps=3_000,
        learning_rate=1e-5,
        data_weight=20.0,
        decrease_weight=80.0,
        condition_source="target_data",
        condition_target_bounds=(0.5, 0.99),
        data_batch_size=8_192,
    ),
    lz.PINNCorrectionStage(
        steps=2_000,
        learning_rate=1e-5,
        data_weight=20.0,
        decrease_weight=80.0,
        boundary_weight=10.0,
        boundary_lower_bound=0.78,
        boundary_seed=654,
        condition_source="target_data",
        condition_target_bounds=(0.5, 0.99),
        data_batch_size=8_192,
    ),
    lz.PINNCorrectionStage(
        steps=1_000,
        learning_rate=5e-6,
        data_weight=30.0,
        decrease_weight=80.0,
        boundary_weight=10.0,
        boundary_lower_bound=0.82,
        boundary_seed=654,
        condition_source="target_data",
        condition_target_bounds=(0.5, 0.99),
        data_batch_size=8_192,
    ),
)

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
            "correction": {"sample_count": 100_000, "seed": 888},
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
            "collocation_batch_size": 256,
            "correction_points": 2**19,
            "correction_batch_size": 4_096,
            "correction_steps": 0,
            "correction_stages": CORRECTION_STAGES,
            "pde_weight": 1.0,
            "positivity_weight": 10.0,
            "correction_level_bounds": (0.15, 0.96),
            "range_weight": 0.0,
            "plateau_gradient_weight": 0.001,
            "warmup_range_weight": 0.0,
            "warmup_plateau_gradient_weight": 0.0,
            "boundary_points": 2**14,
        },
        verification_options={
            "bootstrap_backend": "dreal",
            "screening_samples": 2**18,
            "screening_seed": 20260824,
            "search_precision": 1e-3,
        },
        learner_verbose=True,
        verifier_verbose=True,
    )
    require_verified_area(result, MINIMUM_VERIFIED_AREA)

"""Learn and verify a PINN Lyapunov function for Van der Pol, mu=3."""

import sympy as sp

import lyznet as lz


x1, x2 = sp.symbols("x1 x2")
system = lz.System.continuous(
    dynamics=(-x2, x1 - 3 * (1 - x1**2) * x2),
    domain=((-3.0, 3.0), (-6.0, 6.0)),
    name="minimal_vdp_mu_3",
)


if __name__ == "__main__":
    lz.find_lyap(
        system,
        family="neural",
        architecture="plain",
        verify="crown",
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
            "correction_steps": 10_000,
            "correction_learning_rate": 2e-5,
            "warmup_data_weight": 1.0,
            "data_weight": 10.0,
            "pde_weight": 1.0,
            "positivity_weight": 10.0,
            "decrease_weight": 100.0,
            "correction_level_bounds": (0.07, 0.96),
            "range_weight": 3.0,
            "plateau_gradient_weight": 0.3,
            "plateau_gradient_level": 0.98,
            "warmup_range_weight": 0.0,
            "warmup_plateau_gradient_weight": 0.0,
            "boundary_points": 2**14,
            "boundary_weight": 5.0,
            "warmup_boundary_weight": 0.0,
        },
        verification_options={
            "screening_samples": 2**18,
            "screening_seed": 20260824,
            "search_precision": 1e-3,
        },
        plot=True,
        learner_verbose=True,
        verifier_verbose=True,
    ).require_verified()

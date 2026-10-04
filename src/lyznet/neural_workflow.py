"""The small public workflow joining neural data, learning, and proof stages."""

from pathlib import Path
from time import perf_counter

from .analysis import quadratic_lyapunov
from .constraints import normalise_state_constraints, normalise_viable_faces
from .neural_assessment import screen_neural_lyapunov
from .neural_learning import (
    effort_configuration,
    generate_trajectory_data,
    train_neural_lyapunov,
)
from .verification import verify_quadratic_roa
from .verification.neural_crown import verify_neural_lyapunov_crown


def find_neural_lyapunov(
    system,
    *,
    architecture,
    effort,
    verify,
    device,
    results_dir,
    seed,
    data_overwrite,
    model_overwrite,
    data_options,
    learner_options,
    verification_options,
    learner_verbose,
    verifier_verbose,
    plot,
    plot_options,
    state_constraints,
    viable_faces,
):
    """Run the four explicit stages of the neural Lyapunov workflow."""

    from .workflows import LyapunovSearchResult, LyapunovSearchTimings

    if system.domain is None:
        raise ValueError("Neural Lyapunov learning requires a bounded domain.")
    if architecture not in ("plain", "anchored"):
        raise ValueError("architecture must be 'plain' or 'anchored'.")
    if verify not in (None, False, "crown"):
        raise ValueError("Neural Lyapunov verification currently uses CROWN.")
    configuration = effort_configuration(effort)
    data_options = {} if data_options is None else dict(data_options)
    learner_options = {} if learner_options is None else dict(learner_options)
    verification_options = (
        {} if verification_options is None else dict(verification_options)
    )
    state_constraints = normalise_state_constraints(
        system, state_constraints
    )
    viable_faces = normalise_viable_faces(system, viable_faces)
    result_directory = Path(results_dir).expanduser().resolve()
    total_started = perf_counter()

    if learner_verbose or verifier_verbose:
        print("Neural Lyapunov workflow: {}".format(system.name or "system"))
        print(
            "  architecture: {}; effort: {}; device: {}"
            .format(architecture, effort, device)
        )

    # Stage 1: establish the independent local theorem used by the neural
    # reach-and-stay proof.
    if verifier_verbose:
        print("\n--- Quadratic bootstrap ---")
    bootstrap_backend = verification_options.pop("bootstrap_backend", "z3")
    bootstrap_options = verification_options.pop("bootstrap_options", {})
    quadratic = quadratic_lyapunov(system)
    bootstrap = verify_quadratic_roa(
        system,
        certificate=quadratic,
        backend=bootstrap_backend,
        **bootstrap_options,
    )
    if not bootstrap.is_verified:
        raise RuntimeError(
            "The quadratic bootstrap was not formally verified (status: {})."
            .format(bootstrap.status.value)
        )
    if verifier_verbose:
        print(
            "  verified {} level: {:.6g}"
            .format(bootstrap.backend, bootstrap.verified_level)
        )

    # Stage 2: generate or load empirical trajectory values.
    if learner_verbose:
        print("\n--- Trajectory data ---")
    data = generate_trajectory_data(
        system,
        sample_count=data_options.pop(
            "sample_count", configuration["samples"]
        ),
        results_dir=result_directory,
        device=data_options.pop("device", device),
        seed=data_options.pop("seed", seed),
        overwrite=data_overwrite,
        verbose=learner_verbose,
        state_constraints=state_constraints,
        **data_options,
    )

    # Stage 3: fit values, then correct the Zubov PDE residual.
    if learner_verbose:
        print("\n--- Candidate training ---")
    candidate_started = perf_counter()
    training = train_neural_lyapunov(
        system,
        quadratic,
        data,
        architecture=architecture,
        results_dir=result_directory,
        device=learner_options.pop("device", device),
        seed=learner_options.pop("seed", seed),
        width=learner_options.pop("width", configuration["width"]),
        warmup_steps=learner_options.pop(
            "warmup_steps", configuration["warmup_steps"]
        ),
        correction_steps=learner_options.pop(
            "correction_steps", configuration["correction_steps"]
        ),
        collocation_points=learner_options.pop(
            "collocation_points", configuration["collocation_points"]
        ),
        overwrite=model_overwrite,
        verbose=learner_verbose,
        state_constraints=state_constraints,
        **learner_options,
    )
    candidate_time = perf_counter() - candidate_started

    # Stage 4a: empirical proposal. The result explicitly remains nonformal.
    if verifier_verbose:
        print("\n--- Empirical screening ---")
    state_constraint_band = verification_options.pop(
        "state_constraint_band", 1e-3
    )
    state_constraint_viability_margin = verification_options.pop(
        "state_constraint_viability_margin", 1e-5
    )
    face_viability_margin = verification_options.pop(
        "face_viability_margin", 1e-5
    )
    screening = screen_neural_lyapunov(
        system,
        training.candidate,
        bootstrap.verified_set,
        sample_count=verification_options.pop("screening_samples", 2**14),
        seed=verification_options.pop("screening_seed", seed),
        device=verification_options.pop("screening_device", device),
        precision=verification_options.pop("search_precision", 1e-4),
        proof_margin=verification_options.get("crown_proof_margin", 1e-6),
        state_constraints=state_constraints,
        viable_faces=viable_faces,
        state_constraint_band=state_constraint_band,
        state_constraint_viability_margin=(
            state_constraint_viability_margin
        ),
        face_viability_margin=face_viability_margin,
    )
    if verifier_verbose:
        print(
            "  proposed levels: local={:.6g}, outer={:.6g}; "
            "sampled volume={:.6g} (not a proof)"
            .format(
                screening.local_level,
                screening.outer_level,
                screening.estimated_volume,
            )
        )

    # Stage 4b: one independent formal check of the fixed proposed levels.
    verification = None
    verification_time = 0.0
    if verify == "crown":
        crown_device = verification_options.pop("crown_device", device)
        if verifier_verbose:
            print("\n--- Formal CROWN verification ---")
            import torch

            from .verification._crown import normalise_device

            print(
                "  device: {}"
                .format(normalise_device(torch, crown_device))
            )
        verification_started = perf_counter()
        verification = verify_neural_lyapunov_crown(
            system,
            training.candidate,
            bootstrap,
            local_level=screening.local_level,
            outer_level=screening.outer_level,
            assessment=screening,
            state_constraints=state_constraints,
            viable_faces=viable_faces,
            state_constraint_band=state_constraint_band,
            state_constraint_viability_margin=(
                state_constraint_viability_margin
            ),
            face_viability_margin=face_viability_margin,
            timeout=verification_options.pop("crown_timeout", None),
            device=crown_device,
            batch_size=verification_options.pop("crown_batch_size", 64),
            proof_margin=verification_options.pop(
                "crown_proof_margin", 1e-6
            ),
            verbose=verification_options.pop(
                "crown_verbose", False
            ),
            **verification_options,
        )
        verification_time = perf_counter() - verification_started

    result = LyapunovSearchResult(
        system=system,
        family="neural",
        candidate=training.candidate,
        verification=verification,
        timings=LyapunovSearchTimings(
            candidate=candidate_time,
            verification=verification_time,
            total=perf_counter() - total_started,
        ),
        assessment=screening,
        data_path=data.path,
        model_path=training.model_path,
        state_constraints=state_constraints,
        viable_faces=viable_faces,
    )
    if verifier_verbose:
        status = "not run" if verification is None else verification.status.value
        print("\n=== Neural Lyapunov outcome ===")
        print("  formal status: {}".format(status))
        print("  model: {}".format(training.model_path))
        print("  elapsed: {:.2f} s".format(result.timings.total))
    if plot:
        options = {} if plot_options is None else dict(plot_options)
        options.setdefault("show", False)
        options.setdefault(
            "save",
            result_directory / (
                "{}-pinn-{}-{}.pdf".format(
                    system.name or "system",
                    architecture,
                    training.fingerprint,
                )
            ),
        )
        result.plot(**options)
        if verifier_verbose:
            print("  plot: {}".format(options["save"]))
    return result


__all__ = ["find_neural_lyapunov"]

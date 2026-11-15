"""The small public workflow joining neural data, learning, and proof stages."""

from dataclasses import replace
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


def _find_single_neural_lyapunov(
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
    """Train, screen, and optionally prove one fixed-effort candidate."""

    from .workflows import LyapunovSearchResult, LyapunovSearchTimings

    if system.domain is None:
        raise ValueError("Neural Lyapunov learning requires a bounded domain.")
    if architecture not in ("plain", "anchored"):
        raise ValueError("architecture must be 'plain' or 'anchored'.")
    if verify not in (None, False, "crown"):
        raise ValueError("Neural Lyapunov verification currently uses CROWN.")
    configuration_effort = "low" if effort == "fixed" else effort
    configuration = effort_configuration(configuration_effort, architecture)
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
    correction_data_options = data_options.pop("correction", None)
    if architecture == "anchored" and correction_data_options is None:
        correction_data_options = {
            "sample_count": configuration["correction_samples"],
            "seed": int(seed) + 765,
        }
    primary_data_options = dict(data_options)
    primary_data_options.setdefault(
        "max_time", configuration.get("max_time", 30.0)
    )
    data_transform_mu = float(
        primary_data_options.get("transform_mu", 0.1)
    )
    data = generate_trajectory_data(
        system,
        sample_count=primary_data_options.pop(
            "sample_count", configuration["samples"]
        ),
        results_dir=result_directory,
        device=primary_data_options.pop("device", device),
        seed=primary_data_options.pop("seed", seed),
        overwrite=data_overwrite,
        verbose=learner_verbose,
        state_constraints=state_constraints,
        **primary_data_options,
    )
    correction_data = None
    if correction_data_options is not None:
        if learner_verbose:
            print("\n--- Correction trajectory data ---")
        correction_options = dict(data_options)
        correction_options.setdefault(
            "max_time", configuration.get("max_time", 30.0)
        )
        correction_options.update(correction_data_options)
        correction_data = generate_trajectory_data(
            system,
            sample_count=correction_options.pop("sample_count"),
            results_dir=result_directory,
            device=correction_options.pop("device", device),
            seed=correction_options.pop("seed", seed),
            overwrite=data_overwrite,
            verbose=learner_verbose,
            state_constraints=state_constraints,
            **correction_options,
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
        correction_stages=learner_options.pop(
            "correction_stages",
            configuration.get("correction_stages", ()),
        ),
        collocation_points=learner_options.pop(
            "collocation_points", configuration["collocation_points"]
        ),
        collocation_batch_size=learner_options.pop(
            "collocation_batch_size",
            configuration.get("collocation_batch_size"),
        ),
        correction_points=learner_options.pop(
            "correction_points", configuration.get("correction_points")
        ),
        correction_batch_size=learner_options.pop(
            "correction_batch_size",
            configuration.get("correction_batch_size"),
        ),
        boundary_points=learner_options.pop(
            "boundary_points", configuration.get("boundary_points", 1024)
        ),
        compile_training=learner_options.pop(
            "compile_training", architecture == "anchored"
        ),
        transform_mu=learner_options.pop(
            "transform_mu", data_transform_mu
        ),
        anchor_scale=learner_options.pop(
            "anchor_scale", data_transform_mu
        ),
        quadratic_anchored_data_loss=learner_options.pop(
            "quadratic_anchored_data_loss",
            "log_latent" if architecture == "anchored" else "value",
        ),
        initial_model_path=learner_options.pop(
            "initial_model_path", None
        ),
        correction_data=correction_data,
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
        local_max_level=verification_options.pop("local_max_level", 0.3),
        maximum_level=verification_options.pop("maximum_level", 1.0),
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
            batch_size=verification_options.pop("crown_batch_size", 128),
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
        _plot_result(result, plot_options, verifier_verbose)
    return result


def _plot_result(result, plot_options, verbose):
    """Plot one selected neural result without rerunning its workflow."""

    options = {} if plot_options is None else dict(plot_options)
    options.setdefault("show", False)
    options.setdefault("save", Path(result.model_path).with_suffix(".pdf"))
    result.plot(**options)
    if verbose:
        print("  plot: {}".format(options["save"]))


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
    """Train a fixed recipe or selectively verify nested effort checkpoints.

    When ``effort`` is omitted, explicit learner options define one fixed
    candidate; without learner options the built-in low recipe applies.
    ``mid`` retains the low checkpoint before adding its own updates.
    ``high`` likewise retains both lower-effort checkpoints. All retained
    candidates use the same empirical screen. Formal verification starts
    from the largest screened candidate and tries a fallback only when the
    preceding proof does not succeed.
    """

    if effort is None:
        effort = "fixed" if learner_options is not None else "low"
    if effort in ("fixed", "low"):
        return _find_single_neural_lyapunov(
            system,
            architecture=architecture,
            effort=effort,
            verify=verify,
            device=device,
            results_dir=results_dir,
            seed=seed,
            data_overwrite=data_overwrite,
            model_overwrite=model_overwrite,
            data_options=data_options,
            learner_options=learner_options,
            verification_options=verification_options,
            learner_verbose=learner_verbose,
            verifier_verbose=verifier_verbose,
            plot=plot,
            plot_options=plot_options,
            state_constraints=state_constraints,
            viable_faces=viable_faces,
        )
    if effort not in ("mid", "high"):
        raise ValueError(
            "effort must be omitted or set to 'low', 'mid', or 'high'."
        )
    if verify not in (None, False, "crown"):
        raise ValueError("Neural Lyapunov verification currently uses CROWN.")

    from .workflows import LyapunovSearchTimings

    started = perf_counter()
    candidate_efforts = (
        ("low", "mid")
        if effort == "mid"
        else ("low", "mid", "high")
    )
    base_learner_options = (
        {} if learner_options is None else dict(learner_options)
    )
    previous_model_path = None
    candidates = []
    if learner_verbose or verifier_verbose:
        print(
            "{} effort: cumulative training with {} retained checkpoints."
            .format(effort.capitalize(), len(candidate_efforts))
        )
    for index, candidate_effort in enumerate(candidate_efforts, 1):
        candidate_learner_options = dict(base_learner_options)
        if previous_model_path is not None:
            candidate_learner_options["initial_model_path"] = (
                previous_model_path
            )
        if learner_verbose or verifier_verbose:
            print(
                "\n=== Checkpoint {}/{}: {} ==="
                .format(index, len(candidate_efforts), candidate_effort)
            )
        candidate = _find_single_neural_lyapunov(
            system,
            architecture=architecture,
            effort=candidate_effort,
            verify=False,
            device=device,
            results_dir=results_dir,
            seed=seed,
            data_overwrite=data_overwrite,
            model_overwrite=model_overwrite,
            data_options=data_options,
            learner_options=candidate_learner_options,
            verification_options=verification_options,
            learner_verbose=learner_verbose,
            verifier_verbose=verifier_verbose,
            plot=False,
            plot_options=None,
            state_constraints=state_constraints,
            viable_faces=viable_faces,
        )
        candidates.append(
            (candidate_effort, candidate_learner_options, candidate)
        )
        previous_model_path = candidate.model_path

    ranked = sorted(
        candidates,
        key=lambda item: item[2].assessment.estimated_volume,
        reverse=True,
    )
    if learner_verbose or verifier_verbose:
        print("\n=== Common-screen ranking ===")
        for rank, (candidate_effort, _, candidate) in enumerate(ranked, 1):
            print(
                "  {}. {}: sampled volume {:.6g}"
                .format(
                    rank,
                    candidate_effort,
                    candidate.assessment.estimated_volume,
                )
            )

    selected = ranked[0][2]
    proof_results = []
    if verify == "crown":
        for candidate_effort, candidate_learner_options, _ in ranked:
            if verifier_verbose:
                print(
                    "\n=== Formal proof attempt: {} checkpoint ==="
                    .format(candidate_effort)
                )
            proved = _find_single_neural_lyapunov(
                system,
                architecture=architecture,
                effort=candidate_effort,
                verify="crown",
                device=device,
                results_dir=results_dir,
                seed=seed,
                data_overwrite=False,
                model_overwrite=False,
                data_options=data_options,
                learner_options=candidate_learner_options,
                verification_options=verification_options,
                learner_verbose=False,
                verifier_verbose=verifier_verbose,
                plot=False,
                plot_options=None,
                state_constraints=state_constraints,
                viable_faces=viable_faces,
            )
            proof_results.append(proved)
            if proved.is_verified:
                selected = proved
                break
        else:
            selected = proof_results[0]

    selected = replace(
        selected,
        timings=LyapunovSearchTimings(
            candidate=sum(
                candidate.timings.candidate
                for _, _, candidate in candidates
            ),
            verification=sum(
                result.timings.verification for result in proof_results
            ),
            total=perf_counter() - started,
        ),
    )
    if learner_verbose or verifier_verbose:
        print("\n=== Cumulative-effort outcome ===")
        print("  model: {}".format(selected.model_path))
        print(
            "  formal status: {}"
            .format(
                "not run"
                if selected.verification is None
                else selected.verification.status.value
            )
        )
        print(
            "  sampled volume: {:.6g}"
            .format(selected.assessment.estimated_volume)
        )
        print("  total elapsed: {:.2f} s".format(selected.timings.total))
    if plot:
        _plot_result(selected, plot_options, verifier_verbose)
    return selected


__all__ = ["find_neural_lyapunov"]

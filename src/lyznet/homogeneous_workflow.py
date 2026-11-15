"""Public workflow for global homogeneous neural Lyapunov functions."""

from pathlib import Path
from time import perf_counter

from .homogeneous_learning import synthesize_homogeneous_neural_lyapunov
from .verification.homogeneous_dreal import (
    verify_homogeneous_neural_lyapunov_dreal,
)


def find_homogeneous_neural_lyapunov(
    system,
    *,
    search,
    scope,
    verify,
    results_dir,
    model_overwrite,
    data_options,
    data_overwrite,
    learner_options,
    verification_options,
    learner_verbose,
    verifier_verbose,
    plot,
    plot_options,
    state_constraints,
    viable_faces,
):
    """Synthesize one candidate and optionally prove its global conditions."""

    from .workflows import LyapunovSearchResult, LyapunovSearchTimings

    if scope != "global":
        raise ValueError(
            "Homogeneous neural Lyapunov functions require scope='global'."
        )
    if search not in (None, "convex"):
        raise ValueError("Homogeneous neural synthesis uses search='convex'.")
    if verify not in (None, False, "dreal"):
        raise ValueError("Homogeneous neural verification currently uses dReal.")
    if data_options or data_overwrite:
        raise ValueError(
            "Homogeneous convex synthesis does not use trajectory data."
        )
    if state_constraints or viable_faces:
        raise ValueError(
            "Global homogeneous verification does not use bounded state "
            "constraints or viable faces."
        )
    learner_values = {} if learner_options is None else dict(learner_options)
    verification_values = (
        {} if verification_options is None else dict(verification_options)
    )
    total_started = perf_counter()
    synthesis = synthesize_homogeneous_neural_lyapunov(
        system,
        results_dir=results_dir,
        overwrite=model_overwrite,
        verbose=learner_verbose,
        **learner_values,
    )
    verification = None
    verification_time = 0.0
    if verify == "dreal":
        if verifier_verbose:
            print("\n--- Global homogeneous dReal verification ---")
        verification_started = perf_counter()
        verification = verify_homogeneous_neural_lyapunov_dreal(
            system,
            synthesis.candidate,
            **verification_values,
        )
        verification_time = perf_counter() - verification_started
        if verifier_verbose:
            for obligation in verification.obligations:
                print(
                    "  {}: {} ({:.2f} s)".format(
                        obligation.name,
                        obligation.status.value.upper(),
                        obligation.elapsed_time,
                    )
                )
    result = LyapunovSearchResult(
        system=system,
        family="neural",
        candidate=synthesis.candidate,
        verification=verification,
        timings=LyapunovSearchTimings(
            candidate=synthesis.elapsed_time,
            verification=verification_time,
            total=perf_counter() - total_started,
        ),
        scope="global",
        synthesis=synthesis,
        model_path=synthesis.model_path,
    )
    if verifier_verbose:
        status = "not run" if verification is None else verification.status.value
        print("\n=== Homogeneous Lyapunov outcome ===")
        print("  formal status: {}".format(status))
        print("  model: {}".format(synthesis.model_path))
        print("  elapsed: {:.2f} s".format(result.timings.total))
    if plot:
        options = {} if plot_options is None else dict(plot_options)
        options.setdefault("show", False)
        options.setdefault(
            "save",
            Path(results_dir).expanduser().resolve()
            / "{}-homogeneous-{}.pdf".format(
                system.name or "system", synthesis.fingerprint
            ),
        )
        result.plot(**options)
        if verifier_verbose:
            print("  plot: {}".format(options["save"]))
    return result


__all__ = ["find_homogeneous_neural_lyapunov"]

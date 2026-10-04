"""Small user-facing workflows composed from analysis and verification."""

from dataclasses import dataclass
from time import perf_counter

from .analysis import (
    QuadraticLyapunov,
    quadratic_control_lyapunov,
    quadratic_lyapunov,
)
from .sets import SublevelSet
from .synthesis import synthesize_global_polynomial_lyapunov
from .verification import (
    QuadraticVerificationResult,
    VerificationStatus,
    verify_quadratic_roa,
)


@dataclass(frozen=True)
class LyapunovSearchTimings:
    candidate: float
    verification: float
    total: float


@dataclass(frozen=True)
class VerifiedLyapunovCertificate:
    system: object
    candidate: object
    local_set: SublevelSet | None
    verified_set: object
    verification: object


@dataclass(frozen=True)
class LyapunovSearchResult:
    system: object
    family: str
    candidate: object | None
    verification: object | None
    timings: LyapunovSearchTimings
    scope: str = "roa"
    synthesis: object | None = None
    assessment: object | None = None
    data_path: object | None = None
    model_path: object | None = None

    @property
    def is_verified(self):
        return (
            self.verification is not None
            and self.verification.status is VerificationStatus.VERIFIED
        )

    @property
    def is_global(self):
        return self.scope == "global"

    @property
    def verified_set(self):
        return None if self.verification is None else self.verification.verified_set

    def require_verified(self):
        if not self.is_verified:
            status = (
                "not run"
                if self.verification is None
                else self.verification.status.value
            )
            raise RuntimeError(
                "A verified Lyapunov certificate is required; status: {}."
                .format(status)
            )
        return VerifiedLyapunovCertificate(
            system=self.system,
            candidate=self.candidate,
            local_set=getattr(self.verification, "local_set", None),
            verified_set=self.verification.verified_set,
            verification=self.verification,
        )

    def plot(self, **options):
        from .plotting import plot_lyapunov_result

        return plot_lyapunov_result(self, **options)


def find_lyapunov(
    system,
    *,
    family="quadratic",
    search=None,
    scope="roa",
    verify=None,
    verification_options=None,
    terms=None,
    search_options=None,
    architecture="anchored",
    effort="low",
    device="auto",
    results_dir="results",
    seed=20260823,
    data_overwrite=False,
    model_overwrite=False,
    data_options=None,
    learner_options=None,
    learner_verbose=False,
    verifier_verbose=False,
    plot=False,
    plot_options=None,
):
    """Construct a Lyapunov candidate and optionally verify it formally."""

    if family == "neural":
        if scope != "roa":
            raise ValueError("Neural Lyapunov learning currently uses scope='roa'.")
        if search not in (None, "pinn"):
            raise ValueError("Neural Lyapunov learning uses search='pinn'.")
        from .neural_workflow import find_neural_lyapunov

        return find_neural_lyapunov(
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
        )
    if family not in ("quadratic", "polynomial"):
        raise ValueError(
            "family must be 'quadratic', 'polynomial', or 'neural'."
        )
    if scope not in ("roa", "global"):
        raise ValueError("scope must be 'roa' or 'global'.")
    if verify not in (None, False, "z3", "dreal", "crown"):
        raise ValueError(
            "verify must be None, False, 'z3', 'dreal', or 'crown'."
        )
    options = {} if verification_options is None else dict(verification_options)
    synthesis_options = {} if search_options is None else dict(search_options)

    total_started = perf_counter()
    candidate_started = perf_counter()
    synthesis = None
    if family == "quadratic":
        if search not in (None, "lyapunov"):
            raise ValueError("Quadratic construction uses search='lyapunov'.")
        candidate = quadratic_lyapunov(system)
    else:
        if scope != "global" or search != "z3" or verify != "z3":
            raise ValueError(
                "Polynomial synthesis requires search='z3', "
                "scope='global', and verify='z3'."
            )
        if terms is None:
            raise ValueError("Polynomial synthesis requires explicit terms.")
        synthesis = synthesize_global_polynomial_lyapunov(
            system, terms, **synthesis_options
        )
        candidate = synthesis.candidate
    candidate_time = perf_counter() - candidate_started

    verification = None
    verification_time = 0.0
    if candidate is not None and verify in ("z3", "dreal", "crown"):
        verification_started = perf_counter()
        if scope == "global":
            if verify == "crown":
                raise ValueError("CROWN currently verifies bounded ROA funnels.")
            if family == "quadratic":
                from .verification.global_quadratic import (
                    verify_global_quadratic,
                )
                verification = verify_global_quadratic(
                    system,
                    certificate=candidate,
                    backend=verify,
                    **options,
                )
            else:
                from .verification.global_lyapunov import (
                    verify_global_lyapunov,
                )
                verification = verify_global_lyapunov(
                    system,
                    candidate,
                    **options,
                )
        else:
            if family != "quadratic":
                raise ValueError(
                    "Bounded polynomial funnels are not in this stage."
                )
            verification = verify_quadratic_roa(
                system,
                certificate=candidate,
                backend=verify,
                **options,
            )
        verification_time = perf_counter() - verification_started

    return LyapunovSearchResult(
        system=system,
        family=family,
        candidate=candidate,
        verification=verification,
        timings=LyapunovSearchTimings(
            candidate=candidate_time,
            verification=verification_time,
            total=perf_counter() - total_started,
        ),
        scope=scope,
        synthesis=synthesis,
    )


def find_control_lyapunov(
    system,
    *,
    family="quadratic",
    search="lqr",
    scope="roa",
    verify=None,
    verification_options=None,
):
    """Construct an LQR quadratic CLF and optionally verify it formally."""

    if family != "quadratic":
        raise ValueError("This stage supports family='quadratic'.")
    if search != "lqr":
        raise ValueError("Quadratic CLF construction uses search='lqr'.")
    if scope not in ("roa", "global"):
        raise ValueError("scope must be 'roa' or 'global'.")
    if verify not in (None, False, "z3", "dreal", "crown"):
        raise ValueError(
            "verify must be None, False, 'z3', 'dreal', or 'crown'."
        )
    if scope == "global" and verify in ("dreal", "crown"):
        raise ValueError(
            "Global CLF verification uses Z3; use scope='roa' for "
            "dReal or CROWN."
        )
    options = (
        {} if verification_options is None else dict(verification_options)
    )
    total_started = perf_counter()
    candidate_started = perf_counter()
    candidate = quadratic_control_lyapunov(system)
    candidate_time = perf_counter() - candidate_started
    verification = None
    verification_time = 0.0
    if verify in ("z3", "dreal", "crown"):
        from .verification.control_lyapunov import (
            verify_global_quadratic_clf,
            verify_quadratic_clf,
        )

        verification_started = perf_counter()
        if scope == "global":
            verification = verify_global_quadratic_clf(
                system,
                candidate,
                backend=verify,
                **options,
            )
        else:
            verification = verify_quadratic_clf(
                system,
                candidate,
                backend=verify,
                **options,
            )
        verification_time = perf_counter() - verification_started
    return LyapunovSearchResult(
        system=system,
        family=family,
        candidate=candidate,
        verification=verification,
        timings=LyapunovSearchTimings(
            candidate=candidate_time,
            verification=verification_time,
            total=perf_counter() - total_started,
        ),
        scope=scope,
    )


find_lyap = find_lyapunov


__all__ = [
    "LyapunovSearchResult",
    "LyapunovSearchTimings",
    "VerifiedLyapunovCertificate",
    "find_control_lyapunov",
    "find_lyap",
    "find_lyapunov",
]

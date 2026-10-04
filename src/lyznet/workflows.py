"""Small user-facing workflows composed from analysis and verification."""

from dataclasses import dataclass
from time import perf_counter

from .analysis import QuadraticLyapunov, quadratic_lyapunov
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
):
    """Construct a Lyapunov candidate and optionally verify it formally."""

    if family not in ("quadratic", "polynomial"):
        raise ValueError("family must be 'quadratic' or 'polynomial'.")
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


find_lyap = find_lyapunov


__all__ = [
    "LyapunovSearchResult",
    "LyapunovSearchTimings",
    "VerifiedLyapunovCertificate",
    "find_lyap",
    "find_lyapunov",
]

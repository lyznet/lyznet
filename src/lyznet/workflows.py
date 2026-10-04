"""Small user-facing workflows composed from analysis and verification."""

from dataclasses import dataclass
from time import perf_counter

from .analysis import QuadraticLyapunov, quadratic_lyapunov
from .sets import SublevelSet
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
    candidate: QuadraticLyapunov
    verified_set: SublevelSet
    verification: QuadraticVerificationResult


@dataclass(frozen=True)
class LyapunovSearchResult:
    system: object
    family: str
    candidate: QuadraticLyapunov
    verification: QuadraticVerificationResult | None
    timings: LyapunovSearchTimings

    @property
    def is_verified(self):
        return (
            self.verification is not None
            and self.verification.status is VerificationStatus.VERIFIED
        )

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
            verified_set=self.verification.verified_set,
            verification=self.verification,
        )


def find_lyapunov(
    system,
    *,
    family="quadratic",
    verify=None,
    verification_options=None,
):
    """Construct a Lyapunov candidate and optionally verify it formally."""

    if family != "quadratic":
        raise ValueError("This stage supports only family='quadratic'.")
    if verify not in (None, False, "z3"):
        raise ValueError("verify must be None, False, or 'z3'.")
    options = {} if verification_options is None else dict(verification_options)

    total_started = perf_counter()
    candidate_started = perf_counter()
    candidate = quadratic_lyapunov(system)
    candidate_time = perf_counter() - candidate_started

    verification = None
    verification_time = 0.0
    if verify == "z3":
        verification_started = perf_counter()
        verification = verify_quadratic_roa(
            system,
            certificate=candidate,
            backend="z3",
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
    )


find_lyap = find_lyapunov


__all__ = [
    "LyapunovSearchResult",
    "LyapunovSearchTimings",
    "VerifiedLyapunovCertificate",
    "find_lyap",
    "find_lyapunov",
]

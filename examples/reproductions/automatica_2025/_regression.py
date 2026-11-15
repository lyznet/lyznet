"""Independent geometry check for the published planar benchmarks."""

import math

import numpy as np
from scipy.stats import qmc


SOBOL_POWER = 17
SOBOL_SEEDS = tuple(range(20261017, 20261025))
_BATCH_SIZE = 65_536


def _replicate_areas(system, candidate, level):
    areas = []
    for seed in SOBOL_SEEDS:
        unit = qmc.Sobol(
            d=system.state_dimension, scramble=True, seed=seed
        ).random_base2(SOBOL_POWER)
        points = qmc.scale(
            unit,
            np.asarray(system.domain.lower),
            np.asarray(system.domain.upper),
        )
        included = 0
        for begin in range(0, len(points), _BATCH_SIZE):
            values = candidate.value(points[begin:begin + _BATCH_SIZE])
            included += int(np.count_nonzero(values <= level))
        areas.append(system.domain.volume * included / len(points))
    return np.asarray(areas, dtype=float)


def require_verified_area(result, minimum_area):
    """Require a formal proof and the held-out sampled-area regression."""

    result.require_verified()
    areas = _replicate_areas(
        result.system,
        result.candidate,
        result.verification.verified_level,
    )
    mean = float(np.mean(areas))
    standard_error = float(np.std(areas, ddof=1) / math.sqrt(len(areas)))
    if float(np.min(areas)) < minimum_area:
        raise RuntimeError(
            "Verified-area regression: minimum replicate {:.4f} < {:.4f}."
            .format(float(np.min(areas)), minimum_area)
        )
    print(
        "Estimated area of the formally verified set: {:.4f} +/- {:.4f}"
        .format(mean, standard_error)
    )
    return mean


__all__ = ["require_verified_area"]

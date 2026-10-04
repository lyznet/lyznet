"""Level-search bookkeeping shared by formal sublevel verifiers."""

from .quadratic import (
    VerificationStatus,
    _floor_to_precision,
)


def all_verified(obligations):
    return all(
        item.status is VerificationStatus.VERIFIED
        for item in obligations
    )


def search_verified_level(upper, precision, check, *, lower=0.0):
    """Bisect a monotone proof condition and recheck the reported level."""

    low = float(lower)
    high = float(upper)
    if low > 0 and not all_verified(check(low)):
        return None, check(low)
    while high - low > precision:
        level = (low + high) / 2
        if all_verified(check(level)):
            low = level
        else:
            high = level
    level = max(float(lower), _floor_to_precision(low, precision))
    if level <= 0:
        return None, check(high)
    obligations = check(level)
    if not all_verified(obligations):
        return None, obligations
    return level, obligations


__all__ = ["all_verified", "search_verified_level"]

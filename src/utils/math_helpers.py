"""
Mathematical helper functions for the volatility infrastructure platform.

Provides normal distribution functions, robust statistics, and
numerical utilities used across analytics modules.
"""

import math
from typing import List, Tuple, Optional


# ─── Normal distribution ──────────────────────────────────────────────

def norm_cdf(x: float) -> float:
    """
    Standard normal cumulative distribution function.

    Uses the Abramowitz & Stegun rational approximation (accuracy ~1e-7).
    This avoids importing scipy for a single function used everywhere.
    """
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def norm_pdf(x: float) -> float:
    """Standard normal probability density function."""
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def norm_inv(p: float) -> float:
    """
    Inverse of the standard normal CDF (quantile function).

    Uses the Beasley-Springer-Moro algorithm for p in (0, 1).
    """
    if p <= 0.0 or p >= 1.0:
        raise ValueError(f"norm_inv requires 0 < p < 1, got {p}")

    # Rational approximation coefficients
    a = [
        -3.969683028665376e+01, 2.209460984245205e+02,
        -2.759285104469687e+02, 1.383577518672690e+02,
        -3.066479806614716e+01, 2.506628277459239e+00,
    ]
    b = [
        -5.447609879822406e+01, 1.615858368580409e+02,
        -1.556989798598866e+02, 6.680131188771972e+01,
        -1.328068155288572e+01,
    ]
    c = [
        -7.784894002430293e-03, -3.223964580411365e-01,
        -2.400758277161838e+00, -2.549732539343734e+00,
        4.374664141464968e+00, 2.938163982698783e+00,
    ]
    d = [
        7.784695709041462e-03, 3.224671290700398e-01,
        2.445134137142996e+00, 3.754408661907416e+00,
    ]

    p_low = 0.02425
    p_high = 1.0 - p_low

    if p < p_low:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
               ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1.0)
    elif p <= p_high:
        q = p - 0.5
        r = q * q
        return (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5]) * q / \
               (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1.0)
    else:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
                ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1.0)


# ─── Robust statistics ────────────────────────────────────────────────

def median(values: List[float]) -> float:
    """Compute the median of a list of values."""
    if not values:
        raise ValueError("Cannot compute median of empty list")
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    if n % 2 == 1:
        return sorted_vals[n // 2]
    return (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2.0


def median_absolute_deviation(values: List[float]) -> float:
    """
    Compute the Median Absolute Deviation (MAD).

    MAD = median(|x_i - median(x)|)

    This is a robust measure of statistical dispersion,
    resistant to outliers. Recommended by the PDF for
    forward candidate filtering and IV outlier detection.
    """
    if not values:
        raise ValueError("Cannot compute MAD of empty list")
    med = median(values)
    abs_devs = [abs(v - med) for v in values]
    return median(abs_devs)


def robust_z_score(value: float, values: List[float]) -> float:
    """
    Compute a robust z-score using MAD.

    z_robust = 0.6745 * (x - median) / MAD

    The constant 0.6745 normalizes MAD to be consistent with
    standard deviation for normal distributions.
    (Equation 24 from the PDF)
    """
    med = median(values)
    mad = median_absolute_deviation(values)
    if mad < 1e-15:
        return 0.0 if abs(value - med) < 1e-15 else float('inf')
    return 0.6745 * (value - med) / mad


def reject_outliers_mad(
    values: List[float],
    threshold: float = 3.0,
) -> Tuple[List[float], List[float]]:
    """
    Split values into accepted and rejected using robust z-scores.

    Args:
        values: List of values to filter.
        threshold: Maximum robust z-score for acceptance.

    Returns:
        Tuple of (accepted_values, rejected_values).
    """
    if len(values) < 3:
        return values[:], []

    accepted = []
    rejected = []
    for v in values:
        z = abs(robust_z_score(v, values))
        if z <= threshold:
            accepted.append(v)
        else:
            rejected.append(v)
    return accepted, rejected


def weighted_average(
    values: List[float],
    weights: List[float],
) -> float:
    """
    Compute a weighted average.

    Args:
        values: Values to average.
        weights: Corresponding weights (must be non-negative).

    Returns:
        Weighted average.
    """
    if not values or not weights:
        raise ValueError("Cannot compute weighted average of empty lists")
    if len(values) != len(weights):
        raise ValueError("values and weights must have equal length")

    total_weight = sum(weights)
    if total_weight < 1e-15:
        raise ValueError("Total weight is effectively zero")

    return sum(v * w for v, w in zip(values, weights)) / total_weight


# ─── Numerical helpers ─────────────────────────────────────────────────

def clamp(value: float, lower: float, upper: float) -> float:
    """Clamp a value to the interval [lower, upper]."""
    return max(lower, min(upper, value))


def safe_log(x: float, floor: float = 1e-15) -> float:
    """Natural logarithm with floor protection against log(0)."""
    return math.log(max(x, floor))


def safe_sqrt(x: float) -> float:
    """Square root with protection against negative inputs."""
    return math.sqrt(max(x, 0.0))


def is_close(a: float, b: float, rel_tol: float = 1e-9, abs_tol: float = 1e-12) -> bool:
    """Check if two floats are approximately equal."""
    return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

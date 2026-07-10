"""
Cross-maturity surface interpolation for the volatility infrastructure.

Interpolates between fitted maturity slices in total variance space
to produce a regularized surface grid.
"""

import math
from typing import List, Optional, Tuple

from src.storage.schemas import VolatilitySurface, SurfaceSlice


def interpolate_vol(
    surface: VolatilitySurface,
    log_moneyness: float,
    maturity_years: float,
) -> Optional[float]:
    """
    Interpolate implied volatility at an arbitrary (k, T) point.

    Uses linear interpolation in total variance across maturities,
    and the slice grid within each maturity.

    Args:
        surface: Fitted volatility surface.
        log_moneyness: Log-moneyness k = ln(K/F).
        maturity_years: Time to maturity T.

    Returns:
        Interpolated implied volatility, or None if outside grid.
    """
    if not surface.slices:
        return None

    # Sort slices by maturity
    sorted_slices = sorted(
        [s for s in surface.slices if s.grid_moneyness and s.grid_total_variance],
        key=lambda s: s.maturity_years,
    )

    if not sorted_slices:
        return None

    # Find bracketing maturities
    if maturity_years <= sorted_slices[0].maturity_years:
        # Extrapolate from shortest maturity
        w = _interpolate_within_slice(sorted_slices[0], log_moneyness)
        if w is None:
            return None
        T = sorted_slices[0].maturity_years
        return math.sqrt(max(w / T, 0.0)) if T > 0 else None

    if maturity_years >= sorted_slices[-1].maturity_years:
        # Extrapolate from longest maturity
        w = _interpolate_within_slice(sorted_slices[-1], log_moneyness)
        if w is None:
            return None
        T = sorted_slices[-1].maturity_years
        return math.sqrt(max(w / T, 0.0)) if T > 0 else None

    # Find bracket
    for i in range(len(sorted_slices) - 1):
        T1 = sorted_slices[i].maturity_years
        T2 = sorted_slices[i + 1].maturity_years
        if T1 <= maturity_years <= T2:
            w1 = _interpolate_within_slice(sorted_slices[i], log_moneyness)
            w2 = _interpolate_within_slice(sorted_slices[i + 1], log_moneyness)
            if w1 is None or w2 is None:
                return None

            # Linear interpolation in total variance
            alpha = (maturity_years - T1) / (T2 - T1) if T2 > T1 else 0.5
            w = w1 + alpha * (w2 - w1)
            return math.sqrt(max(w / maturity_years, 0.0)) if maturity_years > 0 else None

    return None


def _interpolate_within_slice(
    slice: SurfaceSlice,
    log_moneyness: float,
) -> Optional[float]:
    """
    Interpolate total variance within a maturity slice.

    Uses linear interpolation between grid points.
    """
    if not slice.grid_moneyness or not slice.grid_total_variance:
        return None

    k_grid = slice.grid_moneyness
    w_grid = slice.grid_total_variance

    if log_moneyness <= k_grid[0]:
        return w_grid[0]
    if log_moneyness >= k_grid[-1]:
        return w_grid[-1]

    # Find bracket
    for i in range(len(k_grid) - 1):
        if k_grid[i] <= log_moneyness <= k_grid[i + 1]:
            alpha = (
                (log_moneyness - k_grid[i]) / (k_grid[i + 1] - k_grid[i])
                if k_grid[i + 1] > k_grid[i]
                else 0.5
            )
            return w_grid[i] + alpha * (w_grid[i + 1] - w_grid[i])

    return None


def build_surface_grid(
    surface: VolatilitySurface,
    moneyness_range: Tuple[float, float] = (-0.5, 0.5),
    moneyness_step: float = 0.025,
    maturity_range: Tuple[float, float] = (0.02, 1.0),
    maturity_step: float = 0.02,
) -> dict:
    """
    Build a regular 2D grid of implied volatilities.

    Returns a dict with keys: moneyness_grid, maturity_grid, vol_grid.
    vol_grid is a list of lists (maturity x moneyness).
    """
    k_lo, k_hi = moneyness_range
    T_lo, T_hi = maturity_range

    moneyness_pts = []
    k = k_lo
    while k <= k_hi + 1e-10:
        moneyness_pts.append(round(k, 4))
        k += moneyness_step

    maturity_pts = []
    T = T_lo
    while T <= T_hi + 1e-10:
        maturity_pts.append(round(T, 4))
        T += maturity_step

    vol_grid = []
    for T in maturity_pts:
        row = []
        for k in moneyness_pts:
            vol = interpolate_vol(surface, k, T)
            row.append(round(vol, 6) if vol is not None else None)
        vol_grid.append(row)

    return {
        "moneyness_grid": moneyness_pts,
        "maturity_grid": maturity_pts,
        "vol_grid": vol_grid,
    }

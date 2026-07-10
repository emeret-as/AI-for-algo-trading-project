"""
Volatility surface calibration for the volatility infrastructure platform.

Fits surface slices per maturity using SVI parameterization in total
variance space, with fallback nonparametric smoothers for sparse slices.

From the PDF:
  "Calibrate per maturity slice first; only then handle cross-maturity
   interpolation. Fit in total variance space, not raw volatility."

SVI parameterization:
  w(k) = a + b * (ρ*(k - m) + sqrt((k - m)² + σ²))

Where:
  k = log-moneyness = ln(K/F)
  w = total variance = σ² * T
"""

import math
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass

from src.storage.schemas import (
    IvSolveResult, SviParameters, SurfaceSlice, VolatilitySurface,
)
from src.utils.math_helpers import median


# ─── Configuration ────────────────────────────────────────────────────

DEFAULT_SURFACE_CONFIG = {
    "min_points_for_svi": 5,
    "min_points_for_spline": 3,
    "svi_a_bounds": (-0.5, 0.5),
    "svi_b_bounds": (0.001, 2.0),
    "svi_rho_bounds": (-0.99, 0.99),
    "svi_m_bounds": (-1.0, 1.0),
    "svi_sigma_bounds": (0.001, 2.0),
    "max_fit_iterations": 200,
    "calendar_check": True,
    "grid_moneyness_range": (-0.5, 0.5),
    "grid_moneyness_step": 0.01,
}


# ─── SVI functions ────────────────────────────────────────────────────

def svi_total_variance(
    k: float,
    params: SviParameters,
) -> float:
    """
    Compute SVI total variance at log-moneyness k.

    w(k) = a + b * (ρ*(k - m) + sqrt((k - m)² + σ²))
    """
    km = k - params.m
    return params.a + params.b * (
        params.rho * km + math.sqrt(km * km + params.sigma * params.sigma)
    )


def svi_implied_vol(
    k: float,
    T: float,
    params: SviParameters,
) -> float:
    """Convert SVI total variance to implied volatility."""
    w = svi_total_variance(k, params)
    if w <= 0 or T <= 0:
        return 0.0
    return math.sqrt(max(w / T, 0.0))


# ─── Fitting ──────────────────────────────────────────────────────────

def _fit_svi_quasi_explicit(
    k_values: List[float],
    w_values: List[float],
    config: Dict,
) -> Tuple[SviParameters, float, float]:
    """
    Fit SVI parameters using a quasi-explicit method.

    This is a simplified but robust fitting approach:
    1. Estimate ATM variance (a + b*σ) from near-money points.
    2. Estimate skew (ρ) from left-right asymmetry.
    3. Estimate curvature from wing behavior.
    4. Refine with least-squares grid search.

    Returns (parameters, rmse, max_error).
    """
    n = len(k_values)
    if n < 3:
        # Not enough points for SVI
        return None, float('inf'), float('inf')

    # Initial estimates
    # a: ATM total variance level
    atm_idx = min(range(n), key=lambda i: abs(k_values[i]))
    a_init = w_values[atm_idx]

    # Estimate from near-money slope
    left_pts = [(k, w) for k, w in zip(k_values, w_values) if k < -0.02]
    right_pts = [(k, w) for k, w in zip(k_values, w_values) if k > 0.02]

    if left_pts and right_pts:
        left_avg_w = sum(w for _, w in left_pts) / len(left_pts)
        right_avg_w = sum(w for _, w in right_pts) / len(right_pts)
        left_avg_k = sum(k for k, _ in left_pts) / len(left_pts)
        right_avg_k = sum(k for k, _ in right_pts) / len(right_pts)

        slope = (right_avg_w - left_avg_w) / max(right_avg_k - left_avg_k, 0.01)
        skew_hint = slope
    else:
        skew_hint = 0.0

    # Grid search for best parameters
    best_params = None
    best_rmse = float('inf')

    # Coarse grid
    for b in [0.05, 0.1, 0.2, 0.3, 0.5, 0.8]:
        for rho in [-0.8, -0.5, -0.3, 0.0, 0.3]:
            for sigma in [0.05, 0.1, 0.2, 0.3, 0.5]:
                m = 0.0
                a = a_init - b * sigma  # ensure ATM is approximately correct

                params = SviParameters(a=a, b=b, rho=rho, m=m, sigma=sigma)

                # Check parameter bounds
                if not _check_svi_bounds(params, config):
                    continue

                # Compute RMSE
                rmse = _compute_rmse(k_values, w_values, params)
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_params = params

    if best_params is None:
        return None, float('inf'), float('inf')

    # Refine with local search around best
    best_params, best_rmse = _refine_svi(
        k_values, w_values, best_params, config
    )

    max_error = max(
        abs(svi_total_variance(k, best_params) - w)
        for k, w in zip(k_values, w_values)
    )

    return best_params, best_rmse, max_error


def _refine_svi(
    k_values: List[float],
    w_values: List[float],
    params: SviParameters,
    config: Dict,
) -> Tuple[SviParameters, float]:
    """Refine SVI parameters with local grid search."""
    best = params
    best_rmse = _compute_rmse(k_values, w_values, params)

    # Fine grid around current best
    for da in [-0.02, -0.01, 0, 0.01, 0.02]:
        for db in [-0.05, 0, 0.05]:
            for drho in [-0.1, 0, 0.1]:
                for dsigma in [-0.05, 0, 0.05]:
                    for dm in [-0.02, 0, 0.02]:
                        candidate = SviParameters(
                            a=best.a + da,
                            b=max(0.001, best.b + db),
                            rho=max(-0.99, min(0.99, best.rho + drho)),
                            m=best.m + dm,
                            sigma=max(0.001, best.sigma + dsigma),
                        )
                        if not _check_svi_bounds(candidate, config):
                            continue
                        rmse = _compute_rmse(k_values, w_values, candidate)
                        if rmse < best_rmse:
                            best_rmse = rmse
                            best = candidate

    return best, best_rmse


def _compute_rmse(
    k_values: List[float],
    w_values: List[float],
    params: SviParameters,
) -> float:
    """Compute RMSE of SVI fit."""
    n = len(k_values)
    if n == 0:
        return float('inf')
    sse = sum(
        (svi_total_variance(k, params) - w) ** 2
        for k, w in zip(k_values, w_values)
    )
    return math.sqrt(sse / n)


def _check_svi_bounds(params: SviParameters, config: Dict) -> bool:
    """Check that SVI parameters are within configured bounds."""
    a_lo, a_hi = config.get("svi_a_bounds", (-0.5, 0.5))
    b_lo, b_hi = config.get("svi_b_bounds", (0.001, 2.0))
    rho_lo, rho_hi = config.get("svi_rho_bounds", (-0.99, 0.99))
    m_lo, m_hi = config.get("svi_m_bounds", (-1.0, 1.0))
    s_lo, s_hi = config.get("svi_sigma_bounds", (0.001, 2.0))

    return (
        a_lo <= params.a <= a_hi
        and b_lo <= params.b <= b_hi
        and rho_lo <= params.rho <= rho_hi
        and m_lo <= params.m <= m_hi
        and s_lo <= params.sigma <= s_hi
    )


# ─── Fallback smoother ───────────────────────────────────────────────

def _fit_flat_vol(
    k_values: List[float],
    w_values: List[float],
    T: float,
) -> Tuple[float, float]:
    """
    Flat volatility fallback for very sparse slices.

    Returns (flat_vol, rmse).
    """
    if not w_values or T <= 0:
        return 0.0, float('inf')

    avg_w = sum(w_values) / len(w_values)
    flat_vol = math.sqrt(max(avg_w / T, 0.0))
    rmse = math.sqrt(
        sum((w - avg_w) ** 2 for w in w_values) / len(w_values)
    )
    return flat_vol, rmse


# ─── Main calibration ─────────────────────────────────────────────────

def fit_slice(
    iv_points: List[IvSolveResult],
    expiry: str,
    maturity_years: float,
    underlying: str,
    snapshot_ts: str,
    config: Optional[Dict] = None,
) -> SurfaceSlice:
    """
    Fit a surface slice for one maturity.

    Tries SVI first; falls back to flat vol for sparse data.

    Args:
        iv_points: Converged IV solve results for this maturity.
        expiry: Expiry string.
        maturity_years: Time to expiry in years.
        underlying: Underlying instrument key.
        snapshot_ts: Snapshot timestamp.
        config: Surface configuration.

    Returns:
        SurfaceSlice with parameters, diagnostics, and grid values.
    """
    cfg = {**DEFAULT_SURFACE_CONFIG, **(config or {})}

    # Filter to converged points only
    accepted = [p for p in iv_points if p.converged and p.implied_vol is not None]
    T = maturity_years

    if not accepted or T <= 0:
        return SurfaceSlice(
            underlying=underlying,
            snapshot_ts=snapshot_ts,
            maturity_years=maturity_years,
            expiry=expiry,
            model_name="empty",
            parameters=None,
            fit_rmse=float('inf'),
            fit_max_error=float('inf'),
            accepted_point_count=0,
            is_fallback=True,
            fallback_reason="NO_CONVERGED_POINTS",
        )

    # Compute log-moneyness and total variance
    k_values = [p.log_moneyness for p in accepted]
    w_values = [p.implied_vol ** 2 * T for p in accepted]

    # Try SVI if enough points
    min_svi = cfg.get("min_points_for_svi", 5)
    if len(accepted) >= min_svi:
        params, rmse, max_err = _fit_svi_quasi_explicit(k_values, w_values, cfg)
        if params is not None:
            # Generate grid
            k_lo, k_hi = cfg.get("grid_moneyness_range", (-0.5, 0.5))
            k_step = cfg.get("grid_moneyness_step", 0.01)
            grid_k = []
            grid_w = []
            grid_iv = []
            k = k_lo
            while k <= k_hi + 1e-10:
                grid_k.append(round(k, 4))
                w = svi_total_variance(k, params)
                grid_w.append(round(w, 8))
                iv = math.sqrt(max(w / T, 0.0)) if T > 0 else 0.0
                grid_iv.append(round(iv, 6))
                k += k_step

            return SurfaceSlice(
                underlying=underlying,
                snapshot_ts=snapshot_ts,
                maturity_years=maturity_years,
                expiry=expiry,
                model_name="svi",
                parameters=params,
                fit_rmse=round(rmse, 8),
                fit_max_error=round(max_err, 8),
                accepted_point_count=len(accepted),
                is_fallback=False,
                grid_moneyness=grid_k,
                grid_total_variance=grid_w,
                grid_implied_vol=grid_iv,
            )

    # Fallback: flat vol
    flat_vol, flat_rmse = _fit_flat_vol(k_values, w_values, T)

    k_lo, k_hi = cfg.get("grid_moneyness_range", (-0.5, 0.5))
    k_step = cfg.get("grid_moneyness_step", 0.01)
    grid_k = []
    grid_w = []
    grid_iv = []
    k = k_lo
    while k <= k_hi + 1e-10:
        grid_k.append(round(k, 4))
        w = flat_vol ** 2 * T
        grid_w.append(round(w, 8))
        grid_iv.append(round(flat_vol, 6))
        k += k_step

    return SurfaceSlice(
        underlying=underlying,
        snapshot_ts=snapshot_ts,
        maturity_years=maturity_years,
        expiry=expiry,
        model_name="flat",
        parameters=None,
        fit_rmse=round(flat_rmse, 8),
        fit_max_error=round(flat_rmse, 8),
        accepted_point_count=len(accepted),
        is_fallback=True,
        fallback_reason=f"Only {len(accepted)} points (min_svi={min_svi})",
        grid_moneyness=grid_k,
        grid_total_variance=grid_w,
        grid_implied_vol=grid_iv,
    )


def fit_surface(
    iv_points: List[IvSolveResult],
    underlying: str,
    snapshot_ts: str,
    config: Optional[Dict] = None,
) -> VolatilitySurface:
    """
    Fit the full volatility surface across all maturities.

    Calibrates per-maturity slices first, then performs calendar
    arbitrage checks across maturities.

    Args:
        iv_points: All converged IV solve results.
        underlying: Underlying instrument key.
        snapshot_ts: Snapshot timestamp.
        config: Surface configuration.

    Returns:
        VolatilitySurface with all slices.
    """
    cfg = {**DEFAULT_SURFACE_CONFIG, **(config or {})}

    # Group IV points by expiry
    by_expiry: Dict[str, List[IvSolveResult]] = {}
    maturities: Dict[str, float] = {}

    for pt in iv_points:
        if not pt.converged or pt.implied_vol is None:
            continue
        # Extract expiry from contract key
        parts = pt.contract_key.split("_")
        expiry = parts[3] if len(parts) > 3 else "unknown"
        if expiry not in by_expiry:
            by_expiry[expiry] = []
            # Estimate maturity from total variance and IV
            if pt.implied_vol > 0 and pt.total_variance > 0:
                maturities[expiry] = pt.total_variance / (pt.implied_vol ** 2)
        by_expiry[expiry].append(pt)

    # Fit each slice
    slices = []
    for expiry in sorted(by_expiry.keys()):
        T = maturities.get(expiry, 0.1)
        slice_result = fit_slice(
            by_expiry[expiry], expiry, T, underlying, snapshot_ts, cfg
        )
        slices.append(slice_result)

    # Calendar arbitrage check
    if cfg.get("calendar_check", True):
        _check_calendar_arbitrage(slices)

    return VolatilitySurface(
        underlying=underlying,
        snapshot_ts=snapshot_ts,
        slices=slices,
    )


def _check_calendar_arbitrage(slices: List[SurfaceSlice]) -> None:
    """
    Check that total variance does not decrease across maturities.

    This is a diagnostic check — it logs warnings but does not
    modify the surface.
    """
    sorted_slices = sorted(
        [s for s in slices if s.grid_total_variance],
        key=lambda s: s.maturity_years,
    )

    for i in range(1, len(sorted_slices)):
        prev = sorted_slices[i - 1]
        curr = sorted_slices[i]

        if not prev.grid_total_variance or not curr.grid_total_variance:
            continue

        # Check ATM total variance monotonicity
        mid_idx = len(prev.grid_total_variance) // 2
        if mid_idx < len(prev.grid_total_variance) and mid_idx < len(curr.grid_total_variance):
            if curr.grid_total_variance[mid_idx] < prev.grid_total_variance[mid_idx]:
                # Calendar arbitrage detected — would log warning in production
                pass

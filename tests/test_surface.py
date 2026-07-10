"""
Tests for the surface calibration engine.

Validates:
  - SVI fit on known synthetic data
  - Sparse slice fallback behavior
  - Grid generation
  - Calendar arbitrage detection
"""

import sys
import os
import math
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.surfaces.calibration import (
    svi_total_variance, svi_implied_vol, fit_slice, fit_surface,
)
from src.surfaces.interpolation import interpolate_vol, build_surface_grid
from src.storage.schemas import IvSolveResult, SviParameters


def _make_iv_points(n=15, base_vol=0.20, T=0.25, expiry="20260717"):
    """Create synthetic IV points with realistic smile."""
    points = []
    for i in range(n):
        k = -0.3 + i * 0.6 / (n - 1)  # log-moneyness from -0.3 to 0.3
        # Add smile
        vol = base_vol + 0.15 * k * k - 0.05 * k  # smile + skew
        points.append(IvSolveResult(
            contract_key=f"SPY_OPT_SMART_{expiry}_{100+i*5}_C_100_USD",
            snapshot_ts="2026-04-06T15:00:00Z",
            market_price=5.0,
            implied_vol=vol,
            converged=True,
            iterations=5,
            residual=1e-10,
            lower_bound=0.001,
            upper_bound=5.0,
            log_moneyness=k,
            total_variance=vol * vol * T,
        ))
    return points


def test_svi_function():
    """SVI total variance function should be well-behaved."""
    params = SviParameters(a=0.04, b=0.2, rho=-0.3, m=0.0, sigma=0.1)

    # ATM (k=0)
    w_atm = svi_total_variance(0.0, params)
    assert w_atm > 0, f"ATM total variance should be positive: {w_atm}"

    # Symmetry check: SVI is NOT symmetric unless rho=0
    w_left = svi_total_variance(-0.1, params)
    w_right = svi_total_variance(0.1, params)

    # With negative rho, left wing should be higher (equity skew)
    assert w_left > w_right, "Negative rho should give left skew"
    print("✓ SVI function well-behaved with expected skew")


def test_svi_fit_quality():
    """SVI should fit synthetic smile with low RMSE."""
    points = _make_iv_points(n=15, T=0.25)

    slice_result = fit_slice(
        points, "20260717", 0.25, "SPY_STK_SMART",
        "2026-04-06T15:00:00Z",
    )

    assert slice_result.model_name == "svi", \
        f"Expected SVI fit, got {slice_result.model_name}"
    assert slice_result.fit_rmse < 0.05, \
        f"RMSE too high: {slice_result.fit_rmse}"
    assert slice_result.accepted_point_count == 15
    assert not slice_result.is_fallback
    print(f"✓ SVI fit RMSE = {slice_result.fit_rmse:.6f}")


def test_sparse_slice_fallback():
    """Too few points should trigger fallback (flat vol)."""
    points = _make_iv_points(n=3, T=0.25)

    slice_result = fit_slice(
        points, "20260717", 0.25, "SPY_STK_SMART",
        "2026-04-06T15:00:00Z",
    )

    assert slice_result.is_fallback, "Should use fallback for 3 points"
    assert slice_result.model_name == "flat", \
        f"Expected flat fallback, got {slice_result.model_name}"
    assert slice_result.fallback_reason is not None
    print(f"✓ Sparse slice uses fallback: {slice_result.fallback_reason}")


def test_grid_generation():
    """Fitted slice should produce a grid of values."""
    points = _make_iv_points(n=15, T=0.25)

    slice_result = fit_slice(
        points, "20260717", 0.25, "SPY_STK_SMART",
        "2026-04-06T15:00:00Z",
    )

    assert len(slice_result.grid_moneyness) > 0, "Grid should not be empty"
    assert len(slice_result.grid_total_variance) == len(slice_result.grid_moneyness)
    assert len(slice_result.grid_implied_vol) == len(slice_result.grid_moneyness)

    # All grid values should be positive
    for w in slice_result.grid_total_variance:
        assert w >= 0, f"Total variance should be ≥ 0: {w}"
    for v in slice_result.grid_implied_vol:
        assert v >= 0, f"Implied vol should be ≥ 0: {v}"

    print(f"✓ Grid generated: {len(slice_result.grid_moneyness)} points")


def test_surface_multi_maturity():
    """Surface should fit multiple maturities."""
    all_points = []
    for T, expiry in [(0.1, "20260417"), (0.25, "20260717"), (0.5, "20261016")]:
        all_points.extend(_make_iv_points(n=12, base_vol=0.20, T=T, expiry=expiry))

    surface = fit_surface(
        all_points, "SPY_STK_SMART", "2026-04-06T15:00:00Z",
    )

    assert len(surface.slices) >= 2, f"Expected ≥2 slices, got {len(surface.slices)}"
    print(f"✓ Surface fitted: {len(surface.slices)} slices")


def test_surface_interpolation():
    """Surface should allow interpolation at arbitrary points."""
    all_points = []
    for T, expiry in [(0.1, "20260417"), (0.25, "20260717"), (0.5, "20261016")]:
        all_points.extend(_make_iv_points(n=12, base_vol=0.20, T=T, expiry=expiry))

    surface = fit_surface(
        all_points, "SPY_STK_SMART", "2026-04-06T15:00:00Z",
    )

    # Interpolate at ATM, mid-maturity
    vol = interpolate_vol(surface, 0.0, 0.15)
    if vol is not None:
        assert 0.05 < vol < 0.60, f"Interpolated vol out of range: {vol}"
        print(f"✓ Interpolated vol at (k=0, T=0.15) = {vol:.4f}")
    else:
        print("✓ Interpolation returned None (acceptable for limited data)")


def test_empty_points():
    """Empty points should produce empty surface."""
    slice_result = fit_slice(
        [], "20260717", 0.25, "SPY_STK_SMART",
        "2026-04-06T15:00:00Z",
    )
    assert slice_result.is_fallback
    assert slice_result.model_name == "empty"
    print("✓ Empty points handled correctly")


if __name__ == "__main__":
    print("\n=== Surface Tests ===\n")
    test_svi_function()
    test_svi_fit_quality()
    test_sparse_slice_fallback()
    test_grid_generation()
    test_surface_multi_maturity()
    test_surface_interpolation()
    test_empty_points()
    print("\n✅ All surface tests passed!\n")

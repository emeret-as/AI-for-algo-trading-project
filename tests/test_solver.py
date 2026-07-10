"""
Tests for the implied volatility solver.

Validates:
  - Convergence on standard quotes
  - Structured failure on bad quotes (below intrinsic)
  - Round-trip: price → IV → price consistency
  - Bounds behavior
  - Diagnostics completeness
"""

import sys
import os
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.iv.solver import solve_implied_vol, DEFAULT_SOLVER_CONFIG
from src.pricing.european import black76_price


def test_round_trip_atm():
    """Price → IV → Price should round-trip for ATM options."""
    F, K, T, true_vol, r = 100.0, 100.0, 0.5, 0.25, 0.05
    market_price = black76_price(F, K, T, true_vol, r, True)

    result = solve_implied_vol(market_price, F, K, T, r, True)

    assert result.converged, f"Failed to converge: {result.failure_reason}"
    assert abs(result.implied_vol - true_vol) < 1e-6, \
        f"IV mismatch: {result.implied_vol:.8f} vs {true_vol:.8f}"
    print("✓ ATM round-trip OK")


def test_round_trip_otm():
    """Round-trip for OTM options."""
    F, T, r = 100.0, 0.5, 0.05
    for K in [80, 90, 110, 120]:
        for vol in [0.15, 0.25, 0.40]:
            price = black76_price(F, K, T, vol, r, K > F)
            if price < 0.01:
                continue
            result = solve_implied_vol(price, F, K, T, r, K > F)
            assert result.converged, \
                f"Failed at K={K}, vol={vol}: {result.failure_reason}"
            assert abs(result.implied_vol - vol) < 1e-4, \
                f"IV mismatch at K={K}: {result.implied_vol:.6f} vs {vol:.6f}"
    print("✓ OTM round-trips OK")


def test_below_intrinsic_fails():
    """Price below intrinsic should produce structured failure."""
    F, K, T, r = 100.0, 90.0, 0.5, 0.05
    intrinsic = math.exp(-r * T) * max(F - K, 0)
    bad_price = intrinsic * 0.5  # Below intrinsic

    result = solve_implied_vol(bad_price, F, K, T, r, True)

    assert not result.converged, "Should not converge for below-intrinsic"
    assert result.failure_reason == "BELOW_INTRINSIC", \
        f"Wrong failure reason: {result.failure_reason}"
    assert result.implied_vol is None, "Should not return an IV"
    print("✓ Below-intrinsic produces structured failure")


def test_negative_price_fails():
    """Negative price should fail with structured diagnostic."""
    result = solve_implied_vol(-1.0, 100, 100, 0.5, 0.05, True)

    assert not result.converged
    assert result.failure_reason == "NON_POSITIVE_PRICE"
    print("✓ Negative price → structured failure")


def test_zero_maturity_fails():
    """Zero maturity should fail gracefully."""
    result = solve_implied_vol(5.0, 100, 95, 0.0, 0.05, True)

    assert not result.converged
    assert result.failure_reason == "NON_POSITIVE_MATURITY"
    print("✓ Zero maturity → structured failure")


def test_diagnostics_always_present():
    """Diagnostics should be present for both success and failure."""
    # Success case
    F, K, T, r = 100.0, 100.0, 0.5, 0.05
    price = black76_price(F, K, T, 0.25, r, True)
    result = solve_implied_vol(price, F, K, T, r, True, "TEST_KEY", "2026-04-06T00:00:00Z")

    assert result.contract_key == "TEST_KEY"
    assert result.snapshot_ts == "2026-04-06T00:00:00Z"
    assert result.market_price == price
    assert result.lower_bound > 0
    assert result.upper_bound > result.lower_bound
    assert result.model_name == "black76"

    # Failure case
    fail_result = solve_implied_vol(-1.0, 100, 100, 0.5, 0.05, True, "FAIL_KEY")
    assert fail_result.contract_key == "FAIL_KEY"
    assert fail_result.failure_reason is not None
    print("✓ Diagnostics complete for success and failure")


def test_high_vol():
    """Solver should handle high volatilities."""
    F, K, T, r = 100.0, 100.0, 0.25, 0.05
    high_vol = 2.0
    price = black76_price(F, K, T, high_vol, r, True)

    result = solve_implied_vol(price, F, K, T, r, True)
    assert result.converged, f"Failed on high vol: {result.failure_reason}"
    assert abs(result.implied_vol - high_vol) < 0.01, \
        f"High vol mismatch: {result.implied_vol:.4f} vs {high_vol}"
    print("✓ High vol solved correctly")


def test_log_moneyness_computed():
    """Log-moneyness should be computed in the result."""
    F, K, T, r = 100.0, 110.0, 0.5, 0.05
    price = black76_price(F, K, T, 0.25, r, True)
    result = solve_implied_vol(price, F, K, T, r, True)

    expected_lm = math.log(K / F)
    assert abs(result.log_moneyness - expected_lm) < 1e-10, \
        f"Log moneyness: {result.log_moneyness:.8f} vs {expected_lm:.8f}"
    print("✓ Log-moneyness computed correctly")


if __name__ == "__main__":
    print("\n=== IV Solver Tests ===\n")
    test_round_trip_atm()
    test_round_trip_otm()
    test_below_intrinsic_fails()
    test_negative_price_fails()
    test_zero_maturity_fails()
    test_diagnostics_always_present()
    test_high_vol()
    test_log_moneyness_computed()
    print("\n✅ All solver tests passed!\n")

"""
Tests for European and American pricing modules.

Validates:
  - Put-call parity
  - Limiting cases (deep ITM/OTM, short-dated, zero vol)
  - American ≥ European price
  - American → European convergence for zero-dividend case
  - Greek consistency (analytic vs finite-difference)
"""

import sys
import os
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pricing.european import black76_price, price_european, bs_vega
from src.pricing.american import crr_tree_price
from src.risk.greeks import finite_difference_greeks
from src.utils.math_helpers import is_close


# ─── Put-Call Parity ──────────────────────────────────────────────────

def test_put_call_parity():
    """Black-76 put-call parity: C - P = exp(-rT) * (F - K)"""
    F, K, T, vol, r = 100.0, 100.0, 0.5, 0.25, 0.05

    call = black76_price(F, K, T, vol, r, is_call=True)
    put = black76_price(F, K, T, vol, r, is_call=False)
    df = math.exp(-r * T)
    expected = df * (F - K)

    assert is_close(call - put, expected, rel_tol=1e-8), \
        f"Put-call parity failed: C-P={call-put:.8f}, exp={expected:.8f}"
    print("✓ Put-call parity holds")


def test_put_call_parity_otm():
    """Put-call parity for OTM options."""
    for K in [80, 90, 100, 110, 120]:
        F, T, vol, r = 100.0, 1.0, 0.30, 0.03

        call = black76_price(F, K, T, vol, r, True)
        put = black76_price(F, K, T, vol, r, False)
        df = math.exp(-r * T)
        expected = df * (F - K)

        assert is_close(call - put, expected, rel_tol=1e-7), \
            f"PCP failed at K={K}: C-P={call-put:.8f}, exp={expected:.8f}"
    print("✓ Put-call parity holds across strikes")


# ─── Limiting Cases ──────────────────────────────────────────────────

def test_deep_itm_call():
    """Deep ITM call should approach discounted intrinsic."""
    F, K, T, vol, r = 200.0, 100.0, 1.0, 0.20, 0.05
    call = black76_price(F, K, T, vol, r, True)
    intrinsic = math.exp(-r * T) * (F - K)
    assert call >= intrinsic - 0.01, f"Deep ITM call below intrinsic: {call} < {intrinsic}"
    print("✓ Deep ITM call ≥ intrinsic")


def test_deep_otm_call():
    """Deep OTM call should be near zero."""
    F, K, T, vol, r = 100.0, 300.0, 0.5, 0.20, 0.05
    call = black76_price(F, K, T, vol, r, True)
    assert call < 0.01, f"Deep OTM call too high: {call}"
    print("✓ Deep OTM call ≈ 0")


def test_zero_vol():
    """Zero vol should give discounted intrinsic."""
    F, K, T, r = 110.0, 100.0, 1.0, 0.05
    call = black76_price(F, K, T, 1e-12, r, True)
    expected = math.exp(-r * T) * max(F - K, 0)
    assert is_close(call, expected, abs_tol=0.01), \
        f"Zero vol: {call} vs {expected}"
    print("✓ Zero vol gives intrinsic")


def test_zero_maturity():
    """Zero maturity should give immediate exercise value."""
    call = black76_price(100, 95, 0.0, 0.20, 0.05, True)
    assert is_close(call, 5.0, abs_tol=0.01), f"T=0 call: {call} vs 5.0"

    put = black76_price(100, 110, 0.0, 0.20, 0.05, False)
    assert is_close(put, 10.0, abs_tol=0.01), f"T=0 put: {put} vs 10.0"
    print("✓ Zero maturity gives immediate exercise")


def test_short_dated():
    """Short-dated option should be close to intrinsic (low time value)."""
    F, K, T, vol, r = 100.0, 100.0, 1.0 / 365.0, 0.20, 0.05
    call = black76_price(F, K, T, vol, r, True)
    assert 0 < call < 2.0, f"1-day ATM call unexpected: {call}"
    print("✓ Short-dated ATM call is small")


# ─── Greek Consistency ────────────────────────────────────────────────

def test_greeks_analytic_vs_fd():
    """Analytic Greeks should match finite-difference within tolerance."""
    F, K, T, vol, r = 100.0, 100.0, 0.5, 0.25, 0.05
    S = F * math.exp(-r * T)

    analytic = price_european(F, K, T, vol, r, True, S)
    fd = finite_difference_greeks(F, K, T, vol, r, True, S)

    assert abs(analytic.delta - fd["fd_delta"]) < 0.01, \
        f"Delta mismatch: {analytic.delta:.6f} vs {fd['fd_delta']:.6f}"
    assert abs(analytic.gamma - fd["fd_gamma"]) < 0.005, \
        f"Gamma mismatch: {analytic.gamma:.8f} vs {fd['fd_gamma']:.8f}"
    print("✓ Analytic Greeks match finite-difference")


def test_delta_bounds():
    """Delta should be in [0,1] for calls and [-1,0] for puts."""
    F, K, T, vol, r = 100.0, 100.0, 0.5, 0.25, 0.05
    S = F * math.exp(-r * T)

    call_result = price_european(F, K, T, vol, r, True, S)
    put_result = price_european(F, K, T, vol, r, False, S)

    assert 0 <= call_result.delta <= 1.0, f"Call delta out of bounds: {call_result.delta}"
    assert -1.0 <= put_result.delta <= 0, f"Put delta out of bounds: {put_result.delta}"
    print("✓ Delta within bounds")


def test_vega_positive():
    """Vega should be positive for both calls and puts."""
    F, K, T, vol, r = 100.0, 100.0, 0.5, 0.25, 0.05
    vega = bs_vega(F, K, T, vol, r)
    assert vega > 0, f"Vega should be positive: {vega}"
    print("✓ Vega is positive")


# ─── American Pricing ────────────────────────────────────────────────

def test_american_geq_european():
    """American price should be ≥ European price."""
    for K in [80, 90, 100, 110, 120]:
        result = crr_tree_price(100, K, 0.5, 0.25, 0.05, 0.02, False, 200)
        assert result.price >= result.european_price - 0.01, \
            f"American < European at K={K}: {result.price} < {result.european_price}"
    print("✓ American ≥ European for puts")


def test_american_converges_to_european():
    """American call on non-dividend stock → European call."""
    result = crr_tree_price(100, 100, 0.5, 0.25, 0.05, 0.0, True, 300)
    assert abs(result.early_exercise_premium) < 0.05, \
        f"Early exercise premium too large: {result.early_exercise_premium}"
    print("✓ American call (q=0) converges to European")


def test_american_put_exercise_premium():
    """Deep ITM American put should have positive early exercise premium."""
    result = crr_tree_price(100, 130, 1.0, 0.30, 0.05, 0.0, False, 200)
    assert result.price > result.european_price, \
        f"No early exercise premium for deep ITM put"
    print("✓ Deep ITM American put has early exercise premium")


# ─── Run all tests ────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n=== Pricing Tests ===\n")
    test_put_call_parity()
    test_put_call_parity_otm()
    test_deep_itm_call()
    test_deep_otm_call()
    test_zero_vol()
    test_zero_maturity()
    test_short_dated()
    test_greeks_analytic_vs_fd()
    test_delta_bounds()
    test_vega_positive()
    test_american_geq_european()
    test_american_converges_to_european()
    test_american_put_exercise_premium()
    print("\n✅ All pricing tests passed!\n")

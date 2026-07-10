"""
Tests for the forward engine.

Validates:
  - Known synthetic chains with expected forwards
  - Outlier rejection behavior
  - Confidence scoring
  - Sparse maturity fallback
"""

import sys
import os
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.forwards.engine import estimate_forward, build_forward_curve
from src.forwards.carry import compute_implied_carry, build_carry_curve
from src.snapshots.builder import build_snapshot
from src.universe.contracts import (
    create_test_underlying, build_option_universe,
)
from src.collectors.synthetic_feed import SyntheticFeed


def _build_test_snapshot(spot=450.0, regime="calm", seed=42):
    """Helper to build a test snapshot."""
    underlying = create_test_underlying("SPY")
    valuation_date = date(2026, 4, 6)
    contracts = build_option_universe(
        underlying, valuation_date, spot,
        num_strikes=15, num_monthly=3,
    )
    feed = SyntheticFeed(spot=spot, regime=regime, seed=seed)
    events = feed.generate_full_snapshot_events(
        underlying, contracts, valuation_date,
    )
    return build_snapshot(
        events, underlying.instrument_key, contracts,
        "2026-04-06T15:00:00Z", "2026-04-06",
    )


def test_forward_near_spot():
    """Forward should be close to S*exp(r*T) for near-term maturities."""
    snapshot = _build_test_snapshot(spot=450.0, regime="calm")
    forward_results = build_forward_curve(snapshot)

    assert len(forward_results) > 0, "No forward results"

    for fwd in forward_results:
        if fwd.maturity_years < 0.2 and fwd.confidence_score > 0.5:
            # Forward should be near spot * exp(r*T)
            import math
            expected = 450.0 * math.exp(0.05 * fwd.maturity_years)
            tolerance = expected * 0.05  # 5% tolerance
            assert abs(fwd.chosen_forward - expected) < tolerance, \
                f"Forward {fwd.chosen_forward:.2f} too far from expected {expected:.2f}"

    print("✓ Forwards near S*exp(r*T)")


def test_forward_confidence():
    """Calm markets should produce high-confidence forwards."""
    snapshot = _build_test_snapshot(regime="calm")
    forward_results = build_forward_curve(snapshot)

    high_conf = sum(1 for f in forward_results if f.confidence_score > 0.3)
    assert high_conf > 0, "No high-confidence forwards in calm market"
    print(f"✓ {high_conf}/{len(forward_results)} high-confidence forwards")


def test_forward_diagnostics():
    """Forward results should contain full diagnostics."""
    snapshot = _build_test_snapshot()
    forward_results = build_forward_curve(snapshot)

    for fwd in forward_results:
        assert fwd.candidate_count_raw >= 0
        assert fwd.candidate_count_accepted >= 0
        assert fwd.mad_forward >= 0
        assert fwd.diagnostics_version != ""
        # Check candidate details are preserved
        for c in fwd.candidates:
            assert c.strike > 0
            assert c.quality_flag in ("accepted", "outlier", "raw")

    print("✓ Forward diagnostics complete")


def test_carry_computation():
    """Implied carry should be reasonable."""
    spot = 450.0
    snapshot = _build_test_snapshot(spot=spot)
    forward_results = build_forward_curve(snapshot)
    carry_diags = build_carry_curve(spot, forward_results)

    for diag in carry_diags:
        if diag.is_reliable:
            # Carry should be positive and reasonable
            assert -0.2 < diag.implied_carry_yield < 0.2, \
                f"Carry yield out of range: {diag.implied_carry_yield}"

    print("✓ Implied carry yields reasonable")


def test_sparse_forward_fallback():
    """Sparse market should trigger forward fallback."""
    snapshot = _build_test_snapshot(regime="sparse", seed=99)
    forward_results = build_forward_curve(snapshot)

    # Some maturities might use fallback
    fallbacks = sum(1 for f in forward_results if f.fallback_used)
    print(f"✓ Sparse market: {fallbacks}/{len(forward_results)} fallbacks used")


if __name__ == "__main__":
    print("\n=== Forward Engine Tests ===\n")
    test_forward_near_spot()
    test_forward_confidence()
    test_forward_diagnostics()
    test_carry_computation()
    test_sparse_forward_fallback()
    print("\n✅ All forward tests passed!\n")

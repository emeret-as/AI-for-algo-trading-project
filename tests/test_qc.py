"""
Tests for quote QC and system-level validation checks.

Validates:
  - Individual named checks work independently
  - Overall QC classification (usable/caution/reject)
  - Threshold version consistency
  - QC report generation
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.qc.quote_quality import (
    check_spread_pct, check_bid_positivity, check_quote_age,
    check_intrinsic_value, check_crossed_market, check_monotonicity,
    run_quote_qc, filter_quotes,
)
from src.qc.checks import (
    check_iv_convergence, generate_qc_report,
)
from src.storage.schemas import OptionQuoteRow, IvSolveResult


def _make_quote(bid=5.0, ask=5.20, strike=450.0, right="C", spread_pct=3.85):
    """Create a test OptionQuoteRow."""
    mid = (bid + ask) / 2.0
    if mid > 0:
        spread_pct = (ask - bid) / mid * 100.0
    strike_str = f"{strike:.0f}"
    return OptionQuoteRow(
        instrument_key=f"SPY_OPT_SMART_20260717_{strike_str}_{right}_100_USD",
        underlying_key="SPY_STK_SMART",
        expiry="20260717",
        strike=strike,
        right=right,
        maturity_years=0.25,
        bid=bid,
        ask=ask,
        last=(bid + ask) / 2,
        mid=mid,
        spread_pct=spread_pct,
        volume=100,
        open_interest=1000,
        quote_age_seconds=5.0,
    )


def test_spread_check():
    """Spread check should pass for tight and fail for wide."""
    config = {"max_spread_pct": 15.0, "threshold_version": "test_v1"}

    # Tight spread
    tight = _make_quote(bid=5.0, ask=5.10)
    result = check_spread_pct(tight, config)
    assert result.status == "pass", f"Tight spread should pass: {result.status}"

    # Wide spread
    wide = _make_quote(bid=1.0, ask=2.0)  # 66% spread
    result = check_spread_pct(wide, config)
    assert result.status == "fail", f"Wide spread should fail: {result.status}"

    print("✓ Spread check works correctly")


def test_bid_positivity():
    """Bid check should fail for zero bid."""
    config = {"min_bid": 0.01, "threshold_version": "test_v1"}

    pos = _make_quote(bid=1.0, ask=1.10)
    result = check_bid_positivity(pos, config)
    assert result.status == "pass"

    zero = _make_quote(bid=0.0, ask=0.05)
    result = check_bid_positivity(zero, config)
    assert result.status == "fail"
    assert result.reason_code == "ZERO_BID"

    print("✓ Bid positivity check works")


def test_intrinsic_value():
    """Should fail when mid is below intrinsic."""
    config = {"threshold_version": "test_v1"}
    spot = 455.0

    # Call with strike 450, intrinsic = 5, mid = 4 → below intrinsic
    below = _make_quote(bid=3.8, ask=4.2, strike=450.0, right="C")
    result = check_intrinsic_value(below, spot, config)
    assert result.status == "fail"
    assert result.reason_code == "BELOW_INTRINSIC"

    # Call with strike 450, intrinsic = 5, mid = 6 → above intrinsic
    above = _make_quote(bid=5.8, ask=6.2, strike=450.0, right="C")
    result = check_intrinsic_value(above, spot, config)
    assert result.status == "pass"

    print("✓ Intrinsic value check works")


def test_crossed_market():
    """Should fail when bid >= ask."""
    config = {"threshold_version": "test_v1"}

    normal = _make_quote(bid=5.0, ask=5.20)
    result = check_crossed_market(normal, config)
    assert result.status == "pass"

    crossed = _make_quote(bid=5.20, ask=5.0)
    result = check_crossed_market(crossed, config)
    assert result.status == "fail"
    assert result.reason_code == "CROSSED_MARKET"

    print("✓ Crossed market detection works")


def test_overall_qc():
    """Overall QC should classify as usable, caution, or reject."""
    spot = 450.0

    # Good quote
    good = _make_quote(bid=5.0, ask=5.10)
    status, results = run_quote_qc(good, spot)
    assert status in ("usable", "caution"), f"Good quote should be usable/caution: {status}"

    # Bad quote (zero bid)
    bad = _make_quote(bid=0.0, ask=0.05)
    status, results = run_quote_qc(bad, spot)
    assert status == "reject", f"Zero-bid should be rejected: {status}"

    print("✓ Overall QC classification works")


def test_filter_quotes():
    """Filter should separate accepted and rejected quotes."""
    spot = 450.0
    rows = [
        _make_quote(bid=5.0, ask=5.10),     # good
        _make_quote(bid=0.0, ask=0.05),      # bad: zero bid
        _make_quote(bid=3.0, ask=3.10),      # good
        _make_quote(bid=1.0, ask=10.0),      # bad: wide spread
    ]

    accepted, rejected, all_results = filter_quotes(rows, spot)

    assert len(accepted) + len(rejected) == len(rows)
    assert len(rejected) > 0, "Should have some rejections"
    assert len(all_results) > 0, "Should have QC results"

    print(f"✓ Filter: {len(accepted)} accepted, {len(rejected)} rejected")


def test_iv_convergence_check():
    """IV convergence check should reflect convergence ratio."""
    # 90% convergence (below 95% threshold)
    iv_results = []
    for i in range(100):
        iv_results.append(IvSolveResult(
            contract_key=f"test_{i}",
            snapshot_ts="2026-04-06",
            market_price=5.0,
            implied_vol=0.25 if i < 90 else None,
            converged=i < 90,
            iterations=5,
            residual=1e-8,
            lower_bound=0.001,
            upper_bound=5.0,
            failure_reason=None if i < 90 else "TEST_FAILURE",
        ))

    result = check_iv_convergence(iv_results)
    assert result.measured_value == 0.9
    assert result.status in ("warn", "fail"), \
        f"90% convergence should warn/fail: {result.status}"

    print("✓ IV convergence check works")


def test_qc_report_generation():
    """QC report should summarize all checks."""
    from src.storage.schemas import QcCheckResult

    checks = [
        QcCheckResult("test1", "key1", "pass", 0.5, 1.0, "v1", "OK"),
        QcCheckResult("test2", "key2", "warn", 0.8, 1.0, "v1", "CLOSE"),
        QcCheckResult("test3", "key3", "fail", 1.5, 1.0, "v1", "OVER"),
    ]

    report = generate_qc_report(checks)
    assert report["summary"]["total_checks"] == 3
    assert report["summary"]["pass"] == 1
    assert report["summary"]["warn"] == 1
    assert report["summary"]["fail"] == 1
    assert len(report["failures"]) == 1
    assert len(report["warnings"]) == 1

    print("✓ QC report generation works")


if __name__ == "__main__":
    print("\n=== QC Tests ===\n")
    test_spread_check()
    test_bid_positivity()
    test_intrinsic_value()
    test_crossed_market()
    test_overall_qc()
    test_filter_quotes()
    test_iv_convergence_check()
    test_qc_report_generation()
    print("\n✅ All QC tests passed!\n")

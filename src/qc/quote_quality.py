"""
Quote-level quality control for the volatility infrastructure platform.

Implements named, composable checks that classify option quotes as
usable, caution, or reject. Each check returns a structured result
with reason code and measured value.

From the PDF:
  "Do not implement QC as a monolithic if-statement. Break it into
   named checks and log each check separately."
"""

import math
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable, Tuple

from src.storage.schemas import OptionQuoteRow, QcCheckResult


# ─── Configuration ────────────────────────────────────────────────────

DEFAULT_QC_CONFIG = {
    "max_spread_pct": 15.0,
    "min_bid": 0.01,
    "max_quote_age_seconds": 30.0,
    "max_moneyness_for_intrinsic": 0.5,
    "min_open_interest": 0,
    "min_volume": 0,
    "threshold_version": "qc_v1",
}


# ─── Individual named checks ─────────────────────────────────────────

def check_spread_pct(
    row: OptionQuoteRow,
    config: Dict[str, Any],
) -> QcCheckResult:
    """Check that bid-ask spread percentage is within threshold."""
    threshold = config.get("max_spread_pct", 15.0)
    status = "pass" if row.spread_pct <= threshold else "fail"
    if row.spread_pct > threshold * 0.7 and status == "pass":
        status = "warn"

    return QcCheckResult(
        check_name="spread_pct",
        target_key=row.instrument_key,
        status=status,
        measured_value=row.spread_pct,
        threshold_value=threshold,
        threshold_version=config.get("threshold_version", "qc_v1"),
        reason_code="SPREAD_TOO_WIDE" if status == "fail" else "SPREAD_OK",
    )


def check_bid_positivity(
    row: OptionQuoteRow,
    config: Dict[str, Any],
) -> QcCheckResult:
    """Check that bid is positive (non-zero quote)."""
    min_bid = config.get("min_bid", 0.01)
    status = "pass" if row.bid >= min_bid else "fail"

    return QcCheckResult(
        check_name="bid_positivity",
        target_key=row.instrument_key,
        status=status,
        measured_value=row.bid,
        threshold_value=min_bid,
        threshold_version=config.get("threshold_version", "qc_v1"),
        reason_code="ZERO_BID" if status == "fail" else "BID_OK",
    )


def check_quote_age(
    row: OptionQuoteRow,
    config: Dict[str, Any],
) -> QcCheckResult:
    """Check that quote is not stale (within age threshold)."""
    threshold = config.get("max_quote_age_seconds", 30.0)
    status = "pass" if row.quote_age_seconds <= threshold else "fail"
    if row.quote_age_seconds > threshold * 0.8 and status == "pass":
        status = "warn"

    return QcCheckResult(
        check_name="quote_age",
        target_key=row.instrument_key,
        status=status,
        measured_value=row.quote_age_seconds,
        threshold_value=threshold,
        threshold_version=config.get("threshold_version", "qc_v1"),
        reason_code="STALE_QUOTE" if status == "fail" else "QUOTE_FRESH",
    )


def check_intrinsic_value(
    row: OptionQuoteRow,
    spot: float,
    config: Dict[str, Any],
) -> QcCheckResult:
    """
    Check that option mid is not below intrinsic value.

    A price below intrinsic is economically impossible for a tradable
    quote and indicates a data error or stale market.
    """
    if row.right == "C":
        intrinsic = max(spot - row.strike, 0.0)
    else:
        intrinsic = max(row.strike - spot, 0.0)

    # Allow small tolerance for bid-ask mechanics
    tolerance = 0.02
    is_below = row.mid < (intrinsic - tolerance) and row.mid > 0

    return QcCheckResult(
        check_name="intrinsic_value",
        target_key=row.instrument_key,
        status="fail" if is_below else "pass",
        measured_value=row.mid,
        threshold_value=intrinsic,
        threshold_version=config.get("threshold_version", "qc_v1"),
        reason_code="BELOW_INTRINSIC" if is_below else "INTRINSIC_OK",
        context={"intrinsic": intrinsic, "spot": spot},
    )


def check_crossed_market(
    row: OptionQuoteRow,
    config: Dict[str, Any],
) -> QcCheckResult:
    """Check for crossed or locked markets (bid >= ask)."""
    is_crossed = row.bid > 0 and row.ask > 0 and row.bid >= row.ask

    return QcCheckResult(
        check_name="crossed_market",
        target_key=row.instrument_key,
        status="fail" if is_crossed else "pass",
        measured_value=row.bid - row.ask if row.ask > 0 else 0.0,
        threshold_value=0.0,
        threshold_version=config.get("threshold_version", "qc_v1"),
        reason_code="CROSSED_MARKET" if is_crossed else "MARKET_OK",
    )


def check_monotonicity(
    row: OptionQuoteRow,
    spot: float,
    config: Dict[str, Any],
) -> QcCheckResult:
    """
    Basic monotonicity check: calls should decrease with strike,
    puts should increase with strike (relative to ATM).
    This is a hint check — not a hard rejection.
    """
    # Simple check: deep OTM options should have low prices
    if spot <= 0:
        return QcCheckResult(
            check_name="monotonicity",
            target_key=row.instrument_key,
            status="pass",
            measured_value=0.0,
            threshold_value=0.0,
            threshold_version=config.get("threshold_version", "qc_v1"),
            reason_code="NO_SPOT",
        )

    log_m = math.log(row.strike / spot) if row.strike > 0 else 0.0

    # Flag if deep OTM option has surprisingly high price
    if row.right == "C" and log_m > 0.3 and row.mid > spot * 0.1:
        status = "warn"
        reason = "HIGH_PRICE_DEEP_OTM_CALL"
    elif row.right == "P" and log_m < -0.3 and row.mid > spot * 0.1:
        status = "warn"
        reason = "HIGH_PRICE_DEEP_OTM_PUT"
    else:
        status = "pass"
        reason = "MONOTONICITY_OK"

    return QcCheckResult(
        check_name="monotonicity",
        target_key=row.instrument_key,
        status=status,
        measured_value=row.mid,
        threshold_value=spot * 0.1,
        threshold_version=config.get("threshold_version", "qc_v1"),
        reason_code=reason,
    )


# ─── QC pipeline ──────────────────────────────────────────────────────

ALL_CHECKS = [
    ("spread_pct", check_spread_pct),
    ("bid_positivity", check_bid_positivity),
    ("quote_age", check_quote_age),
    ("crossed_market", check_crossed_market),
]

# Checks requiring spot
SPOT_CHECKS = [
    ("intrinsic_value", check_intrinsic_value),
    ("monotonicity", check_monotonicity),
]


def run_quote_qc(
    row: OptionQuoteRow,
    spot: float,
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[str, List[QcCheckResult]]:
    """
    Run all QC checks on a single option quote.

    Returns:
        Tuple of (overall_status, list_of_check_results).
        overall_status is 'usable', 'caution', or 'reject'.
    """
    cfg = {**DEFAULT_QC_CONFIG, **(config or {})}
    results = []

    # Run basic checks
    for name, check_fn in ALL_CHECKS:
        results.append(check_fn(row, cfg))

    # Run spot-dependent checks
    for name, check_fn in SPOT_CHECKS:
        results.append(check_fn(row, spot, cfg))

    # Determine overall status
    has_fail = any(r.status == "fail" for r in results)
    has_warn = any(r.status == "warn" for r in results)

    if has_fail:
        overall = "reject"
    elif has_warn:
        overall = "caution"
    else:
        overall = "usable"

    return overall, results


def filter_quotes(
    rows: List[OptionQuoteRow],
    spot: float,
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[List[OptionQuoteRow], List[OptionQuoteRow], List[QcCheckResult]]:
    """
    Filter option quotes through the QC pipeline.

    Returns:
        Tuple of (accepted_rows, rejected_rows, all_qc_results).
        Accepted rows include both 'usable' and 'caution' quotes.
    """
    accepted = []
    rejected = []
    all_results = []

    for row in rows:
        status, results = run_quote_qc(row, spot, config)
        all_results.extend(results)

        if status == "reject":
            rejected.append(row)
        else:
            accepted.append(row)

    return accepted, rejected, all_results

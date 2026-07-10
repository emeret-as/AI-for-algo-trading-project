"""
Forward and implied carry engine for the volatility infrastructure.

Computes robust forward prices per maturity using put-call parity
reconstruction from liquid call-put pairs. This engine is foundational:
it determines the moneyness coordinate used by IV and surface modules.

Key equations from the PDF:
  F_i = K + exp(r*T) * (C_mid - P_mid)     (parity forward per strike)
  F = weighted_average(F_i, w_i)             (chosen forward)
  q = (1/T) * ln(S/F) + r                   (implied carry/dividend yield)
"""

import math
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict

from src.storage.schemas import (
    MarketStateSnapshot, OptionQuoteRow,
    ForwardCandidate, ForwardResult,
)
from src.utils.math_helpers import (
    median, median_absolute_deviation, robust_z_score,
    weighted_average as weighted_avg,
)


# ─── Configuration ────────────────────────────────────────────────────

DEFAULT_FORWARD_CONFIG = {
    "eligible_moneyness_band": 0.10,     # |ln(K/S)| < 10%
    "min_candidate_count": 2,
    "outlier_mad_threshold": 3.0,
    "min_bid_price": 0.05,
    "max_spread_pct": 20.0,
    "risk_free_rate": 0.05,
    "fallback_policy": "previous",       # 'previous', 'interpolate', 'reject'
}


# ─── Core logic ───────────────────────────────────────────────────────

def _find_eligible_pairs(
    snapshot: MarketStateSnapshot,
    expiry: str,
    config: Dict[str, Any],
) -> List[Tuple[OptionQuoteRow, OptionQuoteRow]]:
    """
    Find eligible call-put pairs near the money for a given expiry.

    Returns list of (call_row, put_row) tuples meeting quality criteria.
    """
    spot = snapshot.underlying_state.reference_spot
    if spot <= 0:
        return []

    band = config.get("eligible_moneyness_band", 0.10)
    min_bid = config.get("min_bid_price", 0.05)
    max_spread = config.get("max_spread_pct", 20.0)

    # Group options by (expiry, strike)
    calls: Dict[float, OptionQuoteRow] = {}
    puts: Dict[float, OptionQuoteRow] = {}

    for row in snapshot.option_rows:
        if row.expiry != expiry:
            continue
        if row.is_stale:
            continue

        # Moneyness check
        if spot > 0:
            log_m = abs(math.log(row.strike / spot))
            if log_m > band:
                continue

        # Quality checks
        if row.bid < min_bid:
            continue
        if row.spread_pct > max_spread:
            continue
        if row.mid <= 0:
            continue

        if row.right == "C":
            calls[row.strike] = row
        else:
            puts[row.strike] = row

    # Match pairs
    pairs = []
    for strike in sorted(calls.keys()):
        if strike in puts:
            pairs.append((calls[strike], puts[strike]))

    return pairs


def _compute_parity_forward(
    call_mid: float,
    put_mid: float,
    strike: float,
    T: float,
    r: float,
) -> float:
    """
    Compute parity forward: F = K + exp(r*T) * (C - P)

    This is Equation 2 from the PDF.
    """
    return strike + math.exp(r * T) * (call_mid - put_mid)


def _liquidity_weight(
    call: OptionQuoteRow,
    put: OptionQuoteRow,
    spot: float,
) -> float:
    """
    Compute liquidity-aware weight for a forward candidate.

    Higher weight for:
      - Near ATM (narrow moneyness)
      - Tight spreads
      - High volume and OI
    """
    if spot <= 0:
        return 0.0

    # Moneyness penalty
    log_m = abs(math.log(call.strike / spot))
    moneyness_w = math.exp(-10.0 * log_m * log_m)

    # Spread quality
    avg_spread = (call.spread_pct + put.spread_pct) / 2.0
    spread_w = max(0.0, 1.0 - avg_spread / 20.0)

    # Volume
    total_vol = call.volume + put.volume
    volume_w = min(total_vol / 200.0, 1.0)

    return moneyness_w * spread_w * (0.5 + 0.5 * volume_w)


def estimate_forward(
    snapshot: MarketStateSnapshot,
    expiry: str,
    maturity_years: float,
    config: Optional[Dict[str, Any]] = None,
) -> ForwardResult:
    """
    Estimate the forward price for one maturity.

    This is the primary entry point for forward computation.

    Steps:
      1. Find eligible call-put pairs near the money.
      2. Compute parity forward per strike.
      3. Weight by liquidity.
      4. Remove outliers using MAD.
      5. Compute weighted average as the chosen forward.
      6. Build full diagnostics bundle.

    Args:
        snapshot: Market state snapshot.
        expiry: Expiry string (YYYYMMDD).
        maturity_years: Time to expiry in years.
        config: Forward engine configuration.

    Returns:
        ForwardResult with chosen forward and full diagnostics.
    """
    cfg = {**DEFAULT_FORWARD_CONFIG, **(config or {})}
    r = cfg.get("risk_free_rate", 0.05)
    spot = snapshot.underlying_state.reference_spot
    underlying = snapshot.underlying_state.instrument_key

    # Step 1: Find eligible pairs
    pairs = _find_eligible_pairs(snapshot, expiry, cfg)

    # Build candidates
    candidates = []
    for call, put in pairs:
        f_i = _compute_parity_forward(
            call.mid, put.mid, call.strike, maturity_years, r
        )
        weight = _liquidity_weight(call, put, spot)
        residual = 0.0  # Will be computed after choosing forward

        candidates.append(ForwardCandidate(
            strike=call.strike,
            maturity_years=maturity_years,
            call_mid=call.mid,
            put_mid=put.mid,
            forward_estimate=f_i,
            weight=weight,
            quality_flag="raw",
            parity_residual=residual,
        ))

    # Edge case: not enough candidates
    min_count = cfg.get("min_candidate_count", 2)
    if len(candidates) < min_count:
        # Fallback: use spot * exp(r*T) as rough estimate
        fallback_fwd = spot * math.exp(r * maturity_years) if spot > 0 else 0.0
        return ForwardResult(
            underlying=underlying,
            snapshot_ts=snapshot.snapshot_ts,
            maturity_years=maturity_years,
            expiry=expiry,
            chosen_forward=fallback_fwd,
            confidence_score=0.0,
            candidate_count_raw=len(candidates),
            candidate_count_accepted=0,
            median_forward=fallback_fwd,
            weighted_mean_forward=fallback_fwd,
            mad_forward=0.0,
            candidates=candidates,
            fallback_used=True,
            fallback_reason=f"Only {len(candidates)} candidates (min={min_count})",
        )

    # Step 3-4: Outlier rejection using MAD
    forward_values = [c.forward_estimate for c in candidates]
    med_fwd = median(forward_values)
    mad_fwd = median_absolute_deviation(forward_values)
    threshold = cfg.get("outlier_mad_threshold", 3.0)

    accepted_candidates = []
    for c in candidates:
        z = abs(robust_z_score(c.forward_estimate, forward_values))
        residual = c.forward_estimate - med_fwd
        flag = "accepted" if z <= threshold else "outlier"

        accepted_candidates.append(ForwardCandidate(
            strike=c.strike,
            maturity_years=c.maturity_years,
            call_mid=c.call_mid,
            put_mid=c.put_mid,
            forward_estimate=c.forward_estimate,
            weight=c.weight if flag == "accepted" else 0.0,
            quality_flag=flag,
            parity_residual=residual,
        ))

    # Step 5: Weighted average of accepted candidates
    accepted_only = [c for c in accepted_candidates if c.quality_flag == "accepted"]

    if not accepted_only:
        # All rejected — use median as fallback
        chosen_forward = med_fwd
        confidence = 0.1
    else:
        values = [c.forward_estimate for c in accepted_only]
        weights = [c.weight for c in accepted_only]
        chosen_forward = weighted_avg(values, weights)

        # Confidence: based on candidate count and consistency
        count_factor = min(len(accepted_only) / 6.0, 1.0)
        consistency_factor = 1.0 - min(mad_fwd / max(abs(med_fwd), 1.0) * 10.0, 1.0)
        confidence = round(count_factor * consistency_factor, 4)

    return ForwardResult(
        underlying=underlying,
        snapshot_ts=snapshot.snapshot_ts,
        maturity_years=maturity_years,
        expiry=expiry,
        chosen_forward=round(chosen_forward, 6),
        confidence_score=max(confidence, 0.0),
        candidate_count_raw=len(candidates),
        candidate_count_accepted=len(accepted_only),
        median_forward=round(med_fwd, 6),
        weighted_mean_forward=round(chosen_forward, 6),
        mad_forward=round(mad_fwd, 6),
        candidates=accepted_candidates,
        fallback_used=len(accepted_only) == 0,
        fallback_reason="All candidates rejected" if not accepted_only else None,
    )


def build_forward_curve(
    snapshot: MarketStateSnapshot,
    config: Optional[Dict[str, Any]] = None,
) -> List[ForwardResult]:
    """
    Build the full forward curve for all maturities in the snapshot.

    Args:
        snapshot: Market state snapshot.
        config: Forward engine configuration.

    Returns:
        List of ForwardResult objects, one per maturity.
    """
    # Group option rows by expiry
    expiries: Dict[str, float] = {}
    for row in snapshot.option_rows:
        if row.expiry not in expiries:
            expiries[row.expiry] = row.maturity_years

    results = []
    for expiry, maturity in sorted(expiries.items()):
        if maturity <= 0:
            continue
        result = estimate_forward(snapshot, expiry, maturity, config)
        results.append(result)

    return results

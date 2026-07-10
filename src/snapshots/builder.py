"""
Market-state snapshot builder for the volatility infrastructure platform.

Transforms raw market events into coherent, deterministic market-state
snapshots. Snapshots are the canonical inputs to all downstream analytics.

Rules from the PDF:
  - Keep the snapshot builder pure: raw events in, snapshots out.
  - Do not call external services from inside it.
  - Spot fallbacks are labeled, not hidden.
  - Given the same raw events and snapshot parameters, repeated runs
    produce identical market-state rows.
"""

from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict

from src.storage.schemas import (
    RawMarketEvent, MarketStateSnapshot, UnderlyingState,
    OptionQuoteRow, OptionContract,
)
from src.utils.logging_helpers import build_logger


# ─── Configuration ────────────────────────────────────────────────────

DEFAULT_SNAPSHOT_CONFIG = {
    "max_quote_age_seconds": 30.0,
    "wide_spread_threshold_pct": 5.0,
    "fallback_spot_method": "last",  # 'last' or 'close'
    "stale_threshold_seconds": 60.0,
}


# ─── Pure functions ───────────────────────────────────────────────────

def latest_by_field(
    events: List[Dict[str, Any]],
    instrument_key: str,
) -> Dict[str, float]:
    """
    Extract the latest value for each field from raw events.

    Returns a dict of {field_name: field_value} using the most recent
    event for each field.
    """
    latest: Dict[str, Tuple[str, float]] = {}  # field -> (timestamp, value)

    for event in events:
        if event.get("instrument_key") != instrument_key:
            continue
        field_name = event.get("field_name", "")
        ts = event.get("receipt_ts", event.get("source_ts", ""))
        value = float(event.get("field_value", 0.0))

        if field_name not in latest or ts > latest[field_name][0]:
            latest[field_name] = (ts, value)

    return {k: v[1] for k, v in latest.items()}


def choose_reference_spot(
    bid: float,
    ask: float,
    last: float,
    config: Dict[str, Any],
) -> Tuple[float, str]:
    """
    Choose the reference spot price with labeled fallback.

    Priority: mid (if spread acceptable) → last → 0.0 with 'fallback' label.

    Returns:
        Tuple of (reference_spot, reference_type).
    """
    if bid > 0 and ask > 0 and ask >= bid:
        mid = (bid + ask) / 2.0
        spread_pct = (ask - bid) / mid * 100.0 if mid > 0 else 999.0

        if spread_pct <= config.get("wide_spread_threshold_pct", 5.0):
            return mid, "mid"

    # Fallback to last price
    if last > 0:
        return last, "last"

    return 0.0, "fallback"


def build_underlying_state(
    fields: Dict[str, float],
    instrument_key: str,
    config: Dict[str, Any],
) -> UnderlyingState:
    """Build an UnderlyingState from the latest field values."""
    bid = fields.get("bid", 0.0)
    ask = fields.get("ask", 0.0)
    last = fields.get("last", 0.0)
    volume = fields.get("volume", 0.0)

    mid = (bid + ask) / 2.0 if bid > 0 and ask > 0 else last
    spread_pct = ((ask - bid) / mid * 100.0) if mid > 0 and bid > 0 and ask > 0 else 0.0

    reference_spot, reference_type = choose_reference_spot(bid, ask, last, config)

    return UnderlyingState(
        instrument_key=instrument_key,
        bid=bid,
        ask=ask,
        last=last,
        mid=mid,
        spread_pct=round(spread_pct, 4),
        reference_spot=reference_spot,
        reference_type=reference_type,
        volume=volume,
        is_stale=reference_type == "fallback",
    )


def build_option_row(
    fields: Dict[str, float],
    contract: OptionContract,
    snapshot_ts: str,
    config: Dict[str, Any],
) -> OptionQuoteRow:
    """Build an OptionQuoteRow from the latest field values."""
    bid = fields.get("bid", 0.0)
    ask = fields.get("ask", 0.0)
    last = fields.get("last", 0.0)
    volume = fields.get("volume", 0.0)
    oi = fields.get("open_interest", 0.0)

    mid = (bid + ask) / 2.0 if bid > 0 and ask > 0 else 0.0
    spread_pct = ((ask - bid) / mid * 100.0) if mid > 0 else 0.0

    # Quote age calculation (simplified — uses 0 for synthetic data)
    quote_age = 0.0

    is_stale = quote_age > config.get("stale_threshold_seconds", 60.0) or (bid <= 0 and ask <= 0)

    return OptionQuoteRow(
        instrument_key=contract.instrument_key,
        underlying_key=contract.underlying_key,
        expiry=contract.expiry,
        strike=contract.strike,
        right=contract.right,
        maturity_years=contract.maturity_years,
        bid=bid,
        ask=ask,
        last=last,
        mid=mid,
        spread_pct=round(spread_pct, 4),
        volume=volume,
        open_interest=oi,
        quote_age_seconds=quote_age,
        is_stale=is_stale,
        multiplier=contract.multiplier,
    )


def derive_state_flags(
    underlying: UnderlyingState,
    option_rows: List[OptionQuoteRow],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Derive market-state flags for the snapshot."""
    total_options = len(option_rows)
    stale_options = sum(1 for o in option_rows if o.is_stale)
    active_options = total_options - stale_options

    # Group by maturity
    maturities = set(o.expiry for o in option_rows)

    return {
        "session_status": "open" if not underlying.is_stale else "degraded",
        "stale_underlying": underlying.is_stale,
        "total_option_count": total_options,
        "stale_option_count": stale_options,
        "active_option_count": active_options,
        "stale_option_ratio": stale_options / total_options if total_options > 0 else 0.0,
        "maturity_count": len(maturities),
        "fallback_spot_used": underlying.reference_type != "mid",
        "reference_type": underlying.reference_type,
    }


# ─── Main builder ─────────────────────────────────────────────────────

def build_snapshot(
    events: List[Dict[str, Any]],
    underlying_key: str,
    contracts: List[OptionContract],
    snapshot_ts: str,
    trade_date: str,
    config: Optional[Dict[str, Any]] = None,
) -> MarketStateSnapshot:
    """
    Build a complete market-state snapshot from raw events.

    This is the primary entry point for snapshot construction.
    It is a pure function: same inputs always produce the same output.

    Args:
        events: List of raw event dicts (from collector or replay).
        underlying_key: Instrument key for the underlying.
        contracts: List of OptionContract objects in the universe.
        snapshot_ts: Canonical snapshot timestamp (UTC ISO).
        trade_date: Trade date (YYYY-MM-DD).
        config: Snapshot configuration overrides.

    Returns:
        A deterministic MarketStateSnapshot.
    """
    cfg = {**DEFAULT_SNAPSHOT_CONFIG, **(config or {})}

    # Build underlying state
    underlying_fields = latest_by_field(events, underlying_key)
    underlying_state = build_underlying_state(
        underlying_fields, underlying_key, cfg
    )

    # Build option rows
    option_rows = []
    for contract in contracts:
        option_fields = latest_by_field(events, contract.instrument_key)
        if option_fields:  # Only include contracts with data
            row = build_option_row(option_fields, contract, snapshot_ts, cfg)
            option_rows.append(row)

    # Sort by expiry then strike for determinism
    option_rows.sort(key=lambda r: (r.expiry, r.strike, r.right))

    # Derive flags
    flags = derive_state_flags(underlying_state, option_rows, cfg)

    return MarketStateSnapshot(
        snapshot_ts=snapshot_ts,
        trade_date=trade_date,
        underlying_state=underlying_state,
        option_rows=option_rows,
        flags=flags,
    )

"""
Contract resolution and instrument master for the volatility infrastructure.

Provides canonical dataclasses for underlyings and option contracts,
instrument key construction/parsing, and synthetic universe generation
for testing.
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from datetime import date, timedelta
import math

from src.storage.schemas import (
    Underlying, OptionContract, build_instrument_key
)
from src.utils.dates import expiry_to_maturity_years, DayCountConvention


def generate_strike_ladder(
    spot: float,
    num_strikes: int = 21,
    strike_spacing_pct: float = 2.5,
) -> List[float]:
    """
    Generate a strike ladder centered around spot.

    Args:
        spot: Current underlying price.
        num_strikes: Total number of strikes (should be odd for symmetry).
        strike_spacing_pct: Spacing between strikes as % of spot.

    Returns:
        Sorted list of strike prices.
    """
    spacing = spot * strike_spacing_pct / 100.0
    center_idx = num_strikes // 2
    strikes = []
    for i in range(num_strikes):
        strike = round(spot + (i - center_idx) * spacing, 2)
        if strike > 0:
            strikes.append(strike)
    return sorted(strikes)


def generate_expiry_dates(
    valuation_date: date,
    num_monthly: int = 6,
    include_weeklies: bool = True,
) -> List[date]:
    """
    Generate option expiry dates.

    Monthly expiries are third Friday of each month.
    Weekly expiries are upcoming Fridays (if enabled).

    Args:
        valuation_date: Current date.
        num_monthly: Number of monthly expiries to generate.
        include_weeklies: Whether to include near-term weeklies.

    Returns:
        Sorted list of expiry dates.
    """
    expiries = set()

    # Monthly expiries: third Friday of each month
    current_month = valuation_date.month
    current_year = valuation_date.year

    for i in range(num_monthly):
        month = (current_month + i - 1) % 12 + 1
        year = current_year + (current_month + i - 1) // 12

        # Find third Friday
        first_day = date(year, month, 1)
        # Days until first Friday
        days_to_friday = (4 - first_day.weekday()) % 7
        first_friday = first_day + timedelta(days=days_to_friday)
        third_friday = first_friday + timedelta(weeks=2)

        if third_friday > valuation_date:
            expiries.add(third_friday)

    # Weekly expiries: next 4 Fridays
    if include_weeklies:
        days_to_friday = (4 - valuation_date.weekday()) % 7
        if days_to_friday == 0:
            days_to_friday = 7
        next_friday = valuation_date + timedelta(days=days_to_friday)
        for i in range(4):
            exp = next_friday + timedelta(weeks=i)
            if exp > valuation_date:
                expiries.add(exp)

    return sorted(expiries)


def build_option_universe(
    underlying: Underlying,
    valuation_date: date,
    spot: float,
    num_strikes: int = 21,
    strike_spacing_pct: float = 2.5,
    num_monthly: int = 6,
    include_weeklies: bool = True,
    multiplier: float = 100.0,
    day_count: DayCountConvention = DayCountConvention.ACT_365,
) -> List[OptionContract]:
    """
    Build the canonical option universe for an underlying.

    Args:
        underlying: Underlying instrument.
        valuation_date: Current valuation date.
        spot: Current spot price.
        num_strikes: Number of strikes per expiry.
        strike_spacing_pct: Strike spacing as % of spot.
        num_monthly: Number of monthly expiries.
        include_weeklies: Include near-term weekly expiries.
        multiplier: Contract multiplier.
        day_count: Day-count convention for maturity calculation.

    Returns:
        List of OptionContract objects forming the universe.
    """
    strikes = generate_strike_ladder(spot, num_strikes, strike_spacing_pct)
    expiries = generate_expiry_dates(valuation_date, num_monthly, include_weeklies)

    contracts = []
    con_id = 100000

    for expiry in expiries:
        expiry_str = expiry.strftime("%Y%m%d")
        maturity_years = expiry_to_maturity_years(
            valuation_date, expiry, day_count
        )

        for strike in strikes:
            for right in ["C", "P"]:
                key = build_instrument_key(
                    symbol=underlying.symbol,
                    security_type="OPT",
                    exchange=underlying.exchange,
                    expiry=expiry_str,
                    strike=strike,
                    right=right,
                    multiplier=multiplier,
                    currency=underlying.currency,
                )
                contracts.append(OptionContract(
                    instrument_key=key,
                    underlying_key=underlying.instrument_key,
                    expiry=expiry_str,
                    strike=strike,
                    right=right,
                    multiplier=multiplier,
                    exchange=underlying.exchange,
                    currency=underlying.currency,
                    contract_id_broker=str(con_id),
                    maturity_years=maturity_years,
                ))
                con_id += 1

    return contracts


def create_test_underlying(
    symbol: str = "SPY",
    exchange: str = "SMART",
    currency: str = "USD",
) -> Underlying:
    """Create a test underlying instrument."""
    key = build_instrument_key(symbol, "STK", exchange)
    return Underlying(
        symbol=symbol,
        security_type="STK",
        exchange=exchange,
        currency=currency,
        instrument_key=key,
        contract_id_broker="12345",
    )


def parse_instrument_key(key: str) -> Dict[str, str]:
    """
    Parse a canonical instrument key back into components.

    Format: SYMBOL_TYPE_EXCHANGE[_EXPIRY_STRIKE_RIGHT_MULT_CCY]
    """
    parts = key.split("_")
    result = {
        "symbol": parts[0],
        "security_type": parts[1],
        "exchange": parts[2],
    }
    if len(parts) >= 8:
        result.update({
            "expiry": parts[3],
            "strike": parts[4],
            "right": parts[5],
            "multiplier": parts[6],
            "currency": parts[7],
        })
    return result

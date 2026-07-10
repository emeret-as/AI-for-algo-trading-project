"""
Synthetic market data feed for development and testing.

Generates realistic option market data using Black-Scholes pricing
with configurable noise, spread patterns, and market conditions.
Supports calm, stressed, and sparse-liquidity scenarios.
"""

import math
import random
from datetime import date, datetime, timezone
from typing import List, Dict, Any, Optional

from src.storage.schemas import Underlying, OptionContract
from src.utils.math_helpers import norm_cdf, norm_pdf
from src.utils.dates import expiry_to_maturity_years


def _black_scholes_price(
    spot: float,
    strike: float,
    T: float,
    vol: float,
    r: float,
    is_call: bool,
) -> float:
    """Compute Black-Scholes option price for synthetic data generation."""
    if T <= 1e-10:
        if is_call:
            return max(spot - strike, 0.0)
        return max(strike - spot, 0.0)

    d1 = (math.log(spot / strike) + (r + 0.5 * vol * vol) * T) / (vol * math.sqrt(T))
    d2 = d1 - vol * math.sqrt(T)

    if is_call:
        return spot * norm_cdf(d1) - strike * math.exp(-r * T) * norm_cdf(d2)
    else:
        return strike * math.exp(-r * T) * norm_cdf(-d2) - spot * norm_cdf(-d1)


def _add_smile(
    base_vol: float,
    strike: float,
    spot: float,
    T: float,
    skew_strength: float = 0.15,
) -> float:
    """Add a realistic volatility smile/skew to the base vol."""
    log_m = math.log(strike / spot) if spot > 0 else 0.0
    # Skew: lower strikes have higher vol (equity-like)
    skew = -skew_strength * log_m
    # Smile: far-from-ATM options have higher vol
    smile = 0.03 * log_m * log_m / max(T, 0.01)
    return max(base_vol + skew + smile, 0.01)


class SyntheticFeed:
    """
    Generates synthetic market events mimicking IBKR data.

    Produces bid/ask/last/volume/OI events for a universe of
    options with realistic pricing based on Black-Scholes + skew.

    Market regimes:
      - calm: tight spreads, low vol, full liquidity
      - normal: moderate spreads and vol
      - stressed: wide spreads, high vol, some stale quotes
      - sparse: many missing quotes, wide spreads
    """

    def __init__(
        self,
        spot: float = 450.0,
        base_vol: float = 0.20,
        risk_free_rate: float = 0.05,
        regime: str = "normal",
        seed: Optional[int] = None,
    ):
        """
        Args:
            spot: Underlying spot price.
            base_vol: Base ATM volatility.
            risk_free_rate: Risk-free rate for pricing.
            regime: Market regime (calm, normal, stressed, sparse).
            seed: Random seed for reproducibility.
        """
        self.spot = spot
        self.base_vol = base_vol
        self.r = risk_free_rate
        self.regime = regime
        self._rng = random.Random(seed)

        # Regime-dependent parameters
        self._spread_factor = {
            "calm": 0.005,
            "normal": 0.015,
            "stressed": 0.05,
            "sparse": 0.08,
        }.get(regime, 0.015)

        self._stale_probability = {
            "calm": 0.0,
            "normal": 0.02,
            "stressed": 0.10,
            "sparse": 0.30,
        }.get(regime, 0.02)

        self._vol_noise = {
            "calm": 0.005,
            "normal": 0.01,
            "stressed": 0.03,
            "sparse": 0.02,
        }.get(regime, 0.01)

    def generate_underlying_events(
        self,
        underlying: Underlying,
        timestamp: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Generate bid/ask/last events for an underlying."""
        ts = timestamp or datetime.now(timezone.utc).isoformat()

        spread = self.spot * self._spread_factor * 0.2  # Underlying has tighter spread
        noise = self._rng.gauss(0, self.spot * 0.001)
        spot_adj = self.spot + noise

        bid = round(spot_adj - spread / 2, 2)
        ask = round(spot_adj + spread / 2, 2)
        last = round(spot_adj + self._rng.gauss(0, spread * 0.3), 2)

        events = []
        for field_name, value in [("bid", bid), ("ask", ask), ("last", last),
                                   ("volume", self._rng.randint(50000, 500000))]:
            events.append({
                "instrument_key": underlying.instrument_key,
                "field_name": field_name,
                "field_value": value,
                "source_ts": ts,
                "contract_id_broker": underlying.contract_id_broker,
            })
        return events

    def generate_option_events(
        self,
        contract: OptionContract,
        valuation_date: date,
        timestamp: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Generate bid/ask/last/volume/OI events for a single option.

        Uses Black-Scholes with smile to compute a fair price,
        then applies regime-dependent spreads and noise.
        """
        ts = timestamp or datetime.now(timezone.utc).isoformat()

        # Possibly mark as stale
        if self._rng.random() < self._stale_probability:
            return []  # Stale = no quote

        T = contract.maturity_years
        if T <= 0:
            return []

        # Compute fair price with smile
        vol = _add_smile(
            self.base_vol + self._rng.gauss(0, self._vol_noise),
            contract.strike, self.spot, T,
        )
        is_call = contract.right == "C"
        fair_price = _black_scholes_price(
            self.spot, contract.strike, T, vol, self.r, is_call
        )

        if fair_price < 0.01:
            fair_price = 0.01

        # Generate bid/ask with regime spread
        moneyness = abs(math.log(contract.strike / self.spot))
        spread_mult = 1.0 + 2.0 * moneyness  # Wider spread for far OTM
        half_spread = fair_price * self._spread_factor * spread_mult

        bid = round(max(fair_price - half_spread, 0.01), 2)
        ask = round(fair_price + half_spread, 2)
        last = round(fair_price + self._rng.gauss(0, half_spread * 0.5), 2)
        last = max(last, 0.01)

        # Volume and OI decrease away from ATM
        atm_factor = math.exp(-3.0 * moneyness * moneyness)
        volume = max(1, int(self._rng.gauss(500, 200) * atm_factor))
        oi = max(10, int(self._rng.gauss(5000, 2000) * atm_factor))

        events = []
        for field_name, value in [
            ("bid", bid), ("ask", ask), ("last", last),
            ("volume", volume), ("open_interest", oi),
        ]:
            events.append({
                "instrument_key": contract.instrument_key,
                "field_name": field_name,
                "field_value": value,
                "source_ts": ts,
                "contract_id_broker": contract.contract_id_broker,
            })
        return events

    def generate_full_snapshot_events(
        self,
        underlying: Underlying,
        contracts: List[OptionContract],
        valuation_date: date,
        timestamp: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Generate a full set of events for the underlying and all options."""
        ts = timestamp or datetime.now(timezone.utc).isoformat()
        events = self.generate_underlying_events(underlying, ts)
        for contract in contracts:
            events.extend(
                self.generate_option_events(contract, valuation_date, ts)
            )
        return events

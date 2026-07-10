"""
Canonical data schemas for the volatility infrastructure platform.

All table families are defined here as frozen dataclasses.
Every module in the platform must use these types as its input/output
contract. Field definitions follow Appendix E of the roadmap.

Schema design rules:
  - No decimals stored as formatted strings; use float with explicit scale.
  - Version fields on all derived analytics.
  - Timezone-aware UTC timestamps throughout.
  - Every derived record references source snapshot_ts.
  - QC results point to both the failing object and the run that produced it.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any


# ─── Instrument Master ────────────────────────────────────────────────

@dataclass(frozen=True)
class Underlying:
    """Canonical underlying instrument."""
    symbol: str
    security_type: str         # STK, IND, FUT
    exchange: str
    currency: str
    instrument_key: str        # canonical composite key
    contract_id_broker: str    # IBKR conId


@dataclass(frozen=True)
class OptionContract:
    """Canonical option contract."""
    instrument_key: str        # canonical composite key
    underlying_key: str        # reference to Underlying
    expiry: str                # YYYYMMDD
    strike: float
    right: str                 # C or P
    multiplier: float
    exchange: str
    currency: str
    contract_id_broker: str    # IBKR conId
    maturity_years: float = 0.0  # computed from expiry and valuation date


def build_instrument_key(
    symbol: str,
    security_type: str,
    exchange: str,
    expiry: str = "",
    strike: float = 0.0,
    right: str = "",
    multiplier: float = 100.0,
    currency: str = "USD",
) -> str:
    """
    Construct a canonical instrument key.

    Format: SYMBOL_TYPE_EXCHANGE[_EXPIRY_STRIKE_RIGHT_MULT_CCY]

    This key is the unique identifier used across all tables.
    """
    parts = [symbol, security_type, exchange]
    if expiry:
        strike_str = f"{strike:.2f}".rstrip('0').rstrip('.')
        parts.extend([expiry, strike_str, right, str(int(multiplier)), currency])
    return "_".join(parts)


# ─── Raw Market Events ────────────────────────────────────────────────

@dataclass(frozen=True)
class RawMarketEvent:
    """
    Immutable raw market observation from the collector.

    This is the highest-fidelity representation. Raw events are
    never modified after creation.
    """
    event_id: str              # unique within collector session
    collector_session_id: str
    instrument_key: str
    field_name: str            # bid, ask, last, volume, open_interest, etc.
    field_value: float
    receipt_ts: str            # UTC ISO timestamp when collector received
    source_ts: Optional[str] = None  # exchange timestamp if available
    contract_id_broker: str = ""


# ─── Market State Snapshots ───────────────────────────────────────────

@dataclass(frozen=True)
class UnderlyingState:
    """Underlying market state within a snapshot."""
    instrument_key: str
    bid: float
    ask: float
    last: float
    mid: float
    spread_pct: float          # (ask - bid) / mid * 100
    reference_spot: float      # chosen reference price
    reference_type: str        # 'mid', 'last', 'close', 'fallback'
    volume: float = 0.0
    is_stale: bool = False


@dataclass(frozen=True)
class OptionQuoteRow:
    """Single option quote within a market-state snapshot."""
    instrument_key: str
    underlying_key: str
    expiry: str
    strike: float
    right: str                 # C or P
    maturity_years: float
    bid: float
    ask: float
    last: float
    mid: float
    spread_pct: float
    volume: float
    open_interest: float
    quote_age_seconds: float   # age of quote at snapshot time
    is_stale: bool = False
    multiplier: float = 100.0


@dataclass(frozen=True)
class MarketStateSnapshot:
    """
    Complete market state at a point in time.

    This is the deterministic input to all downstream analytics.
    It must be reproducible from raw events.
    """
    snapshot_ts: str            # canonical UTC timestamp
    trade_date: str            # YYYY-MM-DD
    underlying_state: UnderlyingState
    option_rows: List[OptionQuoteRow]
    flags: Dict[str, Any] = field(default_factory=dict)
    # flags: session_status, stale_underlying, stale_option_count, etc.


# ─── Forward Curve ────────────────────────────────────────────────────

@dataclass(frozen=True)
class ForwardCandidate:
    """Individual parity-forward candidate from a single strike."""
    strike: float
    maturity_years: float
    call_mid: float
    put_mid: float
    forward_estimate: float    # K + exp(rT) * (C - P)
    weight: float
    quality_flag: str          # 'accepted', 'outlier', 'illiquid'
    parity_residual: float = 0.0


@dataclass(frozen=True)
class ForwardResult:
    """Forward estimate for one maturity of one underlying."""
    underlying: str
    snapshot_ts: str
    maturity_years: float
    expiry: str
    chosen_forward: float
    confidence_score: float    # 0.0 to 1.0
    candidate_count_raw: int
    candidate_count_accepted: int
    median_forward: float
    weighted_mean_forward: float
    mad_forward: float         # MAD of forward candidates
    candidates: List[ForwardCandidate]
    diagnostics_version: str = "v1"
    fallback_used: bool = False
    fallback_reason: Optional[str] = None


# ─── Implied Volatility Points ────────────────────────────────────────

@dataclass(frozen=True)
class IvSolveResult:
    """
    Result of implied-volatility inversion for a single option.

    Stores full diagnostics per the PDF specification: convergence
    status, iteration count, residual, bounds, and failure reason.
    """
    contract_key: str
    snapshot_ts: str
    market_price: float        # input mid price
    implied_vol: Optional[float]
    converged: bool
    iterations: int
    residual: float            # |model_price - market_price|
    lower_bound: float         # vol bracket lower
    upper_bound: float         # vol bracket upper
    failure_reason: Optional[str] = None
    model_name: str = "black76"
    model_version: str = "v1"
    log_moneyness: float = 0.0  # ln(K/F)
    total_variance: float = 0.0  # sigma^2 * T


# ─── Surface Parameters ───────────────────────────────────────────────

@dataclass(frozen=True)
class SviParameters:
    """SVI parameterization for one maturity slice."""
    a: float       # level
    b: float       # slope
    rho: float     # rotation
    m: float       # translation
    sigma: float   # curvature (ATM vol of vol)


@dataclass(frozen=True)
class SurfaceSlice:
    """Fitted surface for one maturity."""
    underlying: str
    snapshot_ts: str
    maturity_years: float
    expiry: str
    model_name: str            # 'svi', 'spline', 'flat'
    parameters: Optional[SviParameters]
    fit_rmse: float
    fit_max_error: float
    accepted_point_count: int
    is_fallback: bool = False
    fallback_reason: Optional[str] = None
    grid_moneyness: List[float] = field(default_factory=list)
    grid_total_variance: List[float] = field(default_factory=list)
    grid_implied_vol: List[float] = field(default_factory=list)


@dataclass(frozen=True)
class VolatilitySurface:
    """Complete volatility surface across all maturities."""
    underlying: str
    snapshot_ts: str
    slices: List[SurfaceSlice]
    model_version: str = "v1"
    config_hash: str = ""


# ─── Pricing Results ──────────────────────────────────────────────────

@dataclass(frozen=True)
class PricingResult:
    """Model price and Greeks for a single contract."""
    contract_key: str
    snapshot_ts: str
    model_price: float
    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float
    pricer_name: str           # 'black76', 'crr_tree', 'bjerksund_stensland'
    pricer_version: str = "v1"
    implied_vol_used: float = 0.0
    forward_used: float = 0.0
    spot_used: float = 0.0


# ─── Positions ─────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Position:
    """A portfolio position."""
    portfolio_id: str
    contract_key: str
    quantity: float            # positive = long, negative = short
    entry_price: float = 0.0
    valuation_ts: str = ""


# ─── Risk Aggregates ──────────────────────────────────────────────────

@dataclass(frozen=True)
class RiskLineItem:
    """Per-contract risk output."""
    contract_key: str
    portfolio_id: str
    quantity: float
    market_value: float
    delta: float               # position-adjusted
    gamma: float
    vega: float
    theta: float
    dollar_gamma: float        # gamma * S^2 * multiplier / 100
    dollar_vega: float         # vega * multiplier
    valuation_ts: str = ""
    analytics_version: str = ""


@dataclass(frozen=True)
class RiskAggregate:
    """Aggregated risk for a group of positions."""
    group_key: str             # underlying, maturity_bucket, or portfolio
    portfolio_id: str
    total_delta: float
    total_gamma: float
    total_vega: float
    total_theta: float
    total_dollar_gamma: float
    total_dollar_vega: float
    position_count: int
    valuation_ts: str = ""


# ─── Scenario Results ─────────────────────────────────────────────────

@dataclass(frozen=True)
class Scenario:
    """A named stress scenario."""
    scenario_id: str
    spot_shift_pct: float      # e.g., -10.0 means -10%
    vol_shift_abs: float       # e.g., 0.05 means +5 vol points
    time_roll_days: int        # e.g., 1 = roll forward 1 day
    description: str
    version: str = "v1"


@dataclass(frozen=True)
class ScenarioResult:
    """PnL result for one position under one scenario."""
    scenario_id: str
    contract_key: str
    portfolio_id: str
    base_price: float
    scenario_price: float
    pnl: float                 # (scenario_price - base_price) * quantity * multiplier
    quantity: float
    valuation_ts: str = ""


@dataclass(frozen=True)
class ScenarioSummary:
    """Aggregate scenario result."""
    scenario_id: str
    portfolio_id: str
    total_pnl: float
    worst_contributor_key: str
    worst_contributor_pnl: float
    position_count: int
    valuation_ts: str = ""


# ─── QC Results ────────────────────────────────────────────────────────

@dataclass(frozen=True)
class QcCheckResult:
    """Result of a single quality-control check."""
    check_name: str
    target_key: str            # instrument_key or maturity
    status: str                # 'pass', 'warn', 'fail'
    measured_value: float
    threshold_value: float
    threshold_version: str
    reason_code: str
    context: Dict[str, Any] = field(default_factory=dict)
    run_id: str = ""
    snapshot_ts: str = ""

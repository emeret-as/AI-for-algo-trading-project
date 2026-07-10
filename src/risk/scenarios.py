"""
Scenario engine for the volatility infrastructure platform.

Reprices portfolios under named stress scenarios with configurable
spot, vol, and time shocks. Supports both full repricing and local
Greek approximations, with worst-case and contributor analysis.

From the PDF:
  "Version the scenario grid. The scenario definition is part of
   the data lineage and must be queryable alongside the output."
"""

import math
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

from src.storage.schemas import (
    Scenario, ScenarioResult, ScenarioSummary,
    Position, PricingResult,
)
from src.pricing.european import price_european


# ─── Default scenario grid ────────────────────────────────────────────

DEFAULT_SCENARIOS = [
    Scenario("spot_down_10", -10.0, 0.0, 0, "Spot -10%", "v1"),
    Scenario("spot_down_5", -5.0, 0.0, 0, "Spot -5%", "v1"),
    Scenario("spot_down_2", -2.0, 0.0, 0, "Spot -2%", "v1"),
    Scenario("spot_up_2", 2.0, 0.0, 0, "Spot +2%", "v1"),
    Scenario("spot_up_5", 5.0, 0.0, 0, "Spot +5%", "v1"),
    Scenario("spot_up_10", 10.0, 0.0, 0, "Spot +10%", "v1"),
    Scenario("vol_up_5", 0.0, 0.05, 0, "Vol +5pts", "v1"),
    Scenario("vol_down_5", 0.0, -0.05, 0, "Vol -5pts", "v1"),
    Scenario("vol_up_10", 0.0, 0.10, 0, "Vol +10pts", "v1"),
    Scenario("time_1d", 0.0, 0.0, 1, "Time +1 day", "v1"),
    Scenario("time_5d", 0.0, 0.0, 5, "Time +5 days", "v1"),
    Scenario("crash_spot_down_10_vol_up_10", -10.0, 0.10, 0, "Crash: Spot -10% Vol +10pts", "v1"),
    Scenario("rally_spot_up_5_vol_down_3", 5.0, -0.03, 0, "Rally: Spot +5% Vol -3pts", "v1"),
    Scenario("weekend_decay", 0.0, 0.0, 3, "Weekend theta decay", "v1"),
]


def reprice_under_scenario(
    scenario: Scenario,
    position: Position,
    base_pricing: PricingResult,
    multiplier: float = 100.0,
) -> ScenarioResult:
    """
    Reprice a single position under a scenario using full repricing.

    Applies spot shift, vol shift, and time roll to the pricing inputs,
    then recomputes the option price.

    Args:
        scenario: Stress scenario to apply.
        position: Portfolio position.
        base_pricing: Base-case pricing result.
        multiplier: Contract multiplier.

    Returns:
        ScenarioResult with base and scenario prices and PnL.
    """
    # Apply scenario shocks
    spot_shift = 1.0 + scenario.spot_shift_pct / 100.0
    scenario_spot = base_pricing.spot_used * spot_shift
    scenario_forward = base_pricing.forward_used * spot_shift

    scenario_vol = max(base_pricing.implied_vol_used + scenario.vol_shift_abs, 0.001)

    # Extract contract details from key
    parts = base_pricing.contract_key.split("_")
    strike = float(parts[4]) if len(parts) > 4 else 0.0
    right = parts[5] if len(parts) > 5 else "C"
    is_call = right == "C"

    # Time roll
    # Estimate T from base pricing (approximate)
    if base_pricing.implied_vol_used > 0 and base_pricing.vega != 0:
        # Rough T estimate from vega relationship
        T_approx = 0.25  # default
    else:
        T_approx = 0.25

    # Better: use log_moneyness if available
    if scenario_forward > 0 and strike > 0 and base_pricing.implied_vol_used > 0:
        # Back out T from base model price using known inputs
        T_approx = max(0.001, T_approx - scenario.time_roll_days / 365.0)

    T_scenario = max(T_approx - scenario.time_roll_days / 365.0, 0.001)

    # Full repricing
    result = price_european(
        forward=scenario_forward,
        strike=strike,
        T=T_scenario,
        vol=scenario_vol,
        r=0.05,
        is_call=is_call,
        spot=scenario_spot,
    )

    scenario_price = result.price
    pnl = (scenario_price - base_pricing.model_price) * position.quantity * multiplier

    return ScenarioResult(
        scenario_id=scenario.scenario_id,
        contract_key=position.contract_key,
        portfolio_id=position.portfolio_id,
        base_price=base_pricing.model_price,
        scenario_price=round(scenario_price, 6),
        pnl=round(pnl, 2),
        quantity=position.quantity,
        valuation_ts=base_pricing.snapshot_ts,
    )


def greek_approximation_pnl(
    scenario: Scenario,
    pricing: PricingResult,
    position: Position,
    multiplier: float = 100.0,
) -> float:
    """
    Approximate scenario PnL using local Greek expansion.

    PnL ≈ Delta * ΔS + 0.5 * Gamma * ΔS² + Vega * Δσ + Theta * Δt

    This is faster but less accurate for large shocks.

    Returns:
        Approximate PnL in dollars.
    """
    S = pricing.spot_used
    dS = S * scenario.spot_shift_pct / 100.0
    dvol = scenario.vol_shift_abs  # absolute vol shift
    dt = scenario.time_roll_days  # in days

    qty = position.quantity

    pnl = (
        pricing.delta * dS
        + 0.5 * pricing.gamma * dS * dS
        + pricing.vega * (dvol / 0.01)  # vega is per 0.01 vol
        + pricing.theta * dt            # theta is per day
    ) * qty * multiplier

    return round(pnl, 2)


def run_scenario_engine(
    scenarios: List[Scenario],
    positions: List[Position],
    pricing_results: Dict[str, PricingResult],
    multiplier: float = 100.0,
    use_full_repricing: bool = True,
) -> Tuple[List[ScenarioResult], List[ScenarioSummary]]:
    """
    Run the full scenario engine across all scenarios and positions.

    Args:
        scenarios: List of Scenario definitions.
        positions: Portfolio positions.
        pricing_results: Dict of contract_key -> PricingResult.
        multiplier: Contract multiplier.
        use_full_repricing: If True, use full repricing; else Greek approx.

    Returns:
        Tuple of (all_results, summaries_per_scenario).
    """
    if not scenarios:
        scenarios = DEFAULT_SCENARIOS

    all_results = []
    summaries = []

    for scenario in scenarios:
        scenario_results = []

        for pos in positions:
            pr = pricing_results.get(pos.contract_key)
            if pr is None:
                continue

            if use_full_repricing:
                result = reprice_under_scenario(scenario, pos, pr, multiplier)
            else:
                approx_pnl = greek_approximation_pnl(scenario, pr, pos, multiplier)
                result = ScenarioResult(
                    scenario_id=scenario.scenario_id,
                    contract_key=pos.contract_key,
                    portfolio_id=pos.portfolio_id,
                    base_price=pr.model_price,
                    scenario_price=0.0,  # not computed in approx mode
                    pnl=approx_pnl,
                    quantity=pos.quantity,
                    valuation_ts=pr.snapshot_ts,
                )
            scenario_results.append(result)

        all_results.extend(scenario_results)

        # Build summary
        if scenario_results:
            total_pnl = sum(r.pnl for r in scenario_results)
            worst = min(scenario_results, key=lambda r: r.pnl)
            portfolio_id = scenario_results[0].portfolio_id

            summaries.append(ScenarioSummary(
                scenario_id=scenario.scenario_id,
                portfolio_id=portfolio_id,
                total_pnl=round(total_pnl, 2),
                worst_contributor_key=worst.contract_key,
                worst_contributor_pnl=worst.pnl,
                position_count=len(scenario_results),
                valuation_ts=scenario_results[0].valuation_ts,
            ))

    return all_results, summaries


def top_contributors(
    results: List[ScenarioResult],
    scenario_id: str,
    n: int = 5,
) -> List[ScenarioResult]:
    """
    Extract top N worst PnL contributors for a scenario.

    Useful for risk attribution and concentration analysis.
    """
    scenario_results = [r for r in results if r.scenario_id == scenario_id]
    return sorted(scenario_results, key=lambda r: r.pnl)[:n]

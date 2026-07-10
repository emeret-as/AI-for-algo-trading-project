"""
Risk aggregation for the volatility infrastructure platform.

Merges positions with analytics results to produce line-level
and aggregate sensitivities. Supports grouping by underlying,
maturity bucket, or portfolio.

From the PDF:
  "Preserve line-level outputs for audit. Keep aggregate formulas
   transparent and queryable."
"""

from typing import List, Dict, Optional, Tuple
from collections import defaultdict

from src.storage.schemas import (
    Position, PricingResult, RiskLineItem, RiskAggregate,
)
from src.risk.greeks import compute_dollar_gamma, compute_dollar_vega


def compute_line_items(
    positions: List[Position],
    pricing_results: Dict[str, PricingResult],
    multiplier: float = 100.0,
) -> List[RiskLineItem]:
    """
    Compute line-level risk for each position.

    Merges position quantities with pricing Greeks to produce
    position-adjusted sensitivities.

    Args:
        positions: List of portfolio positions.
        pricing_results: Dict of contract_key -> PricingResult.
        multiplier: Contract multiplier.

    Returns:
        List of RiskLineItem, one per position.
    """
    line_items = []

    for pos in positions:
        pr = pricing_results.get(pos.contract_key)
        if pr is None:
            continue

        qty = pos.quantity
        market_value = pr.model_price * qty * multiplier

        dollar_gamma = compute_dollar_gamma(
            pr.gamma, pr.spot_used, multiplier, qty
        )
        dollar_vega = compute_dollar_vega(pr.vega, multiplier, qty)

        line_items.append(RiskLineItem(
            contract_key=pos.contract_key,
            portfolio_id=pos.portfolio_id,
            quantity=qty,
            market_value=round(market_value, 2),
            delta=round(pr.delta * qty, 6),
            gamma=round(pr.gamma * qty, 8),
            vega=round(pr.vega * qty, 6),
            theta=round(pr.theta * qty, 6),
            dollar_gamma=round(dollar_gamma, 2),
            dollar_vega=round(dollar_vega, 2),
            valuation_ts=pr.snapshot_ts,
        ))

    return line_items


def aggregate_risk(
    line_items: List[RiskLineItem],
    group_by: str = "portfolio_id",
) -> List[RiskAggregate]:
    """
    Aggregate risk by a configured grouping key.

    Supports grouping by portfolio_id, underlying, or maturity bucket.

    Args:
        line_items: List of RiskLineItem.
        group_by: Field to group by.

    Returns:
        List of RiskAggregate, one per group.
    """
    groups: Dict[str, List[RiskLineItem]] = defaultdict(list)

    for item in line_items:
        key = getattr(item, group_by, item.portfolio_id)
        groups[key].append(item)

    aggregates = []
    for group_key, items in sorted(groups.items()):
        portfolio_id = items[0].portfolio_id if items else ""
        valuation_ts = items[0].valuation_ts if items else ""

        aggregates.append(RiskAggregate(
            group_key=group_key,
            portfolio_id=portfolio_id,
            total_delta=round(sum(i.delta for i in items), 6),
            total_gamma=round(sum(i.gamma for i in items), 8),
            total_vega=round(sum(i.vega for i in items), 6),
            total_theta=round(sum(i.theta for i in items), 6),
            total_dollar_gamma=round(sum(i.dollar_gamma for i in items), 2),
            total_dollar_vega=round(sum(i.dollar_vega for i in items), 2),
            position_count=len(items),
            valuation_ts=valuation_ts,
        ))

    return aggregates


def risk_summary_table(
    aggregates: List[RiskAggregate],
) -> List[Dict[str, any]]:
    """
    Format risk aggregates as a summary table for reporting.

    Returns a list of dicts suitable for tabular display.
    """
    return [
        {
            "group": agg.group_key,
            "portfolio": agg.portfolio_id,
            "positions": agg.position_count,
            "delta": f"{agg.total_delta:+.4f}",
            "gamma": f"{agg.total_gamma:+.6f}",
            "vega": f"{agg.total_vega:+.4f}",
            "theta": f"{agg.total_theta:+.4f}",
            "$ gamma": f"{agg.total_dollar_gamma:+,.0f}",
            "$ vega": f"{agg.total_dollar_vega:+,.0f}",
        }
        for agg in aggregates
    ]

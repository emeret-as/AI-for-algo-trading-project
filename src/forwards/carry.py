"""
Implied carry and dividend yield diagnostics.

Derives the implied carry/dividend yield from spot and forward,
providing diagnostic comparison against rate expectations.

Equation from the PDF:
  q = (1/T) * ln(S/F) + r
"""

import math
from typing import List, Optional
from dataclasses import dataclass

from src.storage.schemas import ForwardResult


@dataclass(frozen=True)
class CarryDiagnostic:
    """Implied carry/dividend diagnostic for one maturity."""
    underlying: str
    expiry: str
    maturity_years: float
    spot: float
    forward: float
    risk_free_rate: float
    implied_carry_yield: float   # q = (1/T) * ln(S/F) + r
    implied_dividend_yield: float  # same as carry for equities
    forward_confidence: float
    is_reliable: bool


def compute_implied_carry(
    spot: float,
    forward: float,
    T: float,
    r: float,
) -> float:
    """
    Compute implied carry/dividend yield from spot and forward.

    q = (1/T) * ln(S/F) + r

    Args:
        spot: Reference spot price.
        forward: Estimated forward price.
        T: Time to maturity in years.
        r: Risk-free rate.

    Returns:
        Implied carry yield.
    """
    if T <= 1e-10 or spot <= 0 or forward <= 0:
        return 0.0
    return (1.0 / T) * math.log(spot / forward) + r


def build_carry_curve(
    spot: float,
    forward_results: List[ForwardResult],
    risk_free_rate: float = 0.05,
    min_confidence: float = 0.3,
) -> List[CarryDiagnostic]:
    """
    Build implied carry/dividend curve from forward results.

    Args:
        spot: Reference spot price.
        forward_results: List of ForwardResult from the forward engine.
        risk_free_rate: Risk-free rate assumption.
        min_confidence: Minimum forward confidence for reliable carry.

    Returns:
        List of CarryDiagnostic, one per maturity.
    """
    diagnostics = []

    for fwd in forward_results:
        if fwd.maturity_years <= 0:
            continue

        carry = compute_implied_carry(
            spot, fwd.chosen_forward, fwd.maturity_years, risk_free_rate
        )

        is_reliable = (
            fwd.confidence_score >= min_confidence
            and not fwd.fallback_used
            and fwd.chosen_forward > 0
        )

        diagnostics.append(CarryDiagnostic(
            underlying=fwd.underlying,
            expiry=fwd.expiry,
            maturity_years=fwd.maturity_years,
            spot=spot,
            forward=fwd.chosen_forward,
            risk_free_rate=risk_free_rate,
            implied_carry_yield=round(carry, 6),
            implied_dividend_yield=round(carry, 6),
            forward_confidence=fwd.confidence_score,
            is_reliable=is_reliable,
        ))

    return diagnostics

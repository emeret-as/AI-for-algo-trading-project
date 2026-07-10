"""
Greeks computation for the volatility infrastructure platform.

Computes canonical per-contract and portfolio-level sensitivities.
Delta and gamma are w.r.t. underlying reference price; vega is per
absolute vol point. Dollar Greeks use the contract multiplier.

From the PDF:
  "All first-order and second-order sensitivities should be computed
   in a unified unit system."
"""

import math
from typing import List, Optional, Dict

from src.storage.schemas import (
    PricingResult, OptionQuoteRow, IvSolveResult, ForwardResult,
)
from src.pricing.european import price_european


def compute_contract_greeks(
    row: OptionQuoteRow,
    iv_result: IvSolveResult,
    forward: float,
    spot: float,
    r: float = 0.05,
) -> Optional[PricingResult]:
    """
    Compute full Greeks for a single option contract.

    Uses the fitted implied volatility and forward price to compute
    model price and analytic Greeks.

    Args:
        row: Option quote row from snapshot.
        iv_result: Converged IV solve result.
        forward: Forward price for this maturity.
        spot: Underlying reference spot.
        r: Risk-free rate.

    Returns:
        PricingResult with model price and all Greeks, or None if IV missing.
    """
    if not iv_result.converged or iv_result.implied_vol is None:
        return None

    is_call = row.right == "C"
    result = price_european(
        forward=forward,
        strike=row.strike,
        T=row.maturity_years,
        vol=iv_result.implied_vol,
        r=r,
        is_call=is_call,
        spot=spot,
    )

    return PricingResult(
        contract_key=row.instrument_key,
        snapshot_ts=iv_result.snapshot_ts,
        model_price=round(result.price, 6),
        delta=round(result.delta, 6),
        gamma=round(result.gamma, 8),
        vega=round(result.vega, 6),
        theta=round(result.theta, 6),
        rho=round(result.rho, 6),
        pricer_name="black76",
        implied_vol_used=iv_result.implied_vol,
        forward_used=forward,
        spot_used=spot,
    )


def compute_dollar_gamma(
    gamma: float,
    spot: float,
    multiplier: float = 100.0,
    quantity: float = 1.0,
) -> float:
    """
    Compute dollar gamma.

    Dollar Gamma = Gamma * S² * Multiplier * Quantity / 100

    This represents the change in dollar delta for a 1% move.
    """
    return gamma * spot * spot * multiplier * quantity / 100.0


def compute_dollar_vega(
    vega: float,
    multiplier: float = 100.0,
    quantity: float = 1.0,
) -> float:
    """
    Compute dollar vega.

    Dollar Vega = Vega * Multiplier * Quantity

    Vega is already per 0.01 vol change.
    """
    return vega * multiplier * quantity


def finite_difference_greeks(
    forward: float,
    strike: float,
    T: float,
    vol: float,
    r: float,
    is_call: bool,
    spot: Optional[float] = None,
    spot_bump: float = 0.01,
    vol_bump: float = 0.01,
) -> Dict[str, float]:
    """
    Compute Greeks using finite differences for validation.

    Used to validate analytic Greeks. The bump sizes are versioned
    and documented per the PDF requirement.

    Args:
        forward: Forward price.
        strike: Strike price.
        T: Time to maturity.
        vol: Implied volatility.
        r: Risk-free rate.
        is_call: True for call.
        spot: Underlying spot (for delta/gamma bumps).
        spot_bump: Relative spot bump size.
        vol_bump: Absolute vol bump size.

    Returns:
        Dict with fd_delta, fd_gamma, fd_vega, fd_theta.
    """
    S = spot if spot is not None else forward * math.exp(-r * T)
    bump_abs = S * spot_bump

    # Delta: dP/dS
    f_up = forward * (1 + spot_bump)
    f_dn = forward * (1 - spot_bump)
    p_up = price_european(f_up, strike, T, vol, r, is_call, S + bump_abs).price
    p_dn = price_european(f_dn, strike, T, vol, r, is_call, S - bump_abs).price
    p_mid = price_european(forward, strike, T, vol, r, is_call, S).price
    fd_delta = (p_up - p_dn) / (2 * bump_abs)

    # Gamma: d²P/dS²
    fd_gamma = (p_up - 2 * p_mid + p_dn) / (bump_abs ** 2)

    # Vega: dP/dσ (per 0.01)
    p_vol_up = price_european(forward, strike, T, vol + vol_bump, r, is_call, S).price
    fd_vega = (p_vol_up - p_mid)  # per vol_bump

    # Theta: dP/dt (per day)
    dt = 1.0 / 365.0
    if T > dt:
        p_later = price_european(forward, strike, T - dt, vol, r, is_call, S).price
        fd_theta = p_later - p_mid
    else:
        fd_theta = 0.0

    return {
        "fd_delta": fd_delta,
        "fd_gamma": fd_gamma,
        "fd_vega": fd_vega,
        "fd_theta": fd_theta,
        "spot_bump_pct": spot_bump * 100,
        "vol_bump_abs": vol_bump,
    }

"""
American option pricing for the volatility infrastructure platform.

Implements a CRR binomial tree pricer with early-exercise detection
and optional Bjerksund-Stensland closed-form approximation.

From the PDF:
  "The first production version should expose a lattice-based pricer
   and may optionally add a closed-form approximation for speed.
   The implementation target is not novelty but stable, testable behavior."
"""

import math
from dataclasses import dataclass
from typing import Optional

from src.utils.math_helpers import norm_cdf


@dataclass(frozen=True)
class AmericanPriceResult:
    """Result of American option pricing."""
    price: float
    european_price: float       # for comparison
    early_exercise_premium: float
    delta: float
    gamma: float
    vega: float
    pricer_name: str
    steps: int
    is_call: bool


def crr_tree_price(
    spot: float,
    strike: float,
    T: float,
    vol: float,
    r: float,
    q: float,
    is_call: bool,
    steps: int = 200,
) -> AmericanPriceResult:
    """
    Cox-Ross-Rubinstein binomial tree for American options.

    Implements backward induction with early-exercise check at each node.
    Uses steps/2 for delta and gamma via finite differences.

    Args:
        spot: Current spot price S.
        strike: Strike price K.
        T: Time to maturity in years.
        vol: Implied volatility.
        r: Risk-free rate.
        q: Continuous dividend yield.
        is_call: True for call, False for put.
        steps: Number of tree steps (higher = more accurate).

    Returns:
        AmericanPriceResult with price and Greeks.
    """
    if T <= 1e-10:
        if is_call:
            intrinsic = max(spot - strike, 0.0)
        else:
            intrinsic = max(strike - spot, 0.0)
        return AmericanPriceResult(
            price=intrinsic,
            european_price=intrinsic,
            early_exercise_premium=0.0,
            delta=1.0 if (is_call and spot > strike) else (-1.0 if (not is_call and spot < strike) else 0.0),
            gamma=0.0,
            vega=0.0,
            pricer_name="crr_tree",
            steps=steps,
            is_call=is_call,
        )

    dt = T / steps
    u = math.exp(vol * math.sqrt(dt))
    d = 1.0 / u
    disc = math.exp(-r * dt)
    p_up = (math.exp((r - q) * dt) - d) / (u - d)
    p_dn = 1.0 - p_up

    # Validate probability
    if p_up < 0 or p_up > 1:
        # Fallback to more steps
        if steps < 50:
            return crr_tree_price(spot, strike, T, vol, r, q, is_call, 200)

    # Build terminal payoffs
    n = steps
    prices = [spot * (u ** (n - 2 * j)) for j in range(n + 1)]

    if is_call:
        values = [max(p - strike, 0.0) for p in prices]
    else:
        values = [max(strike - p, 0.0) for p in prices]

    # Store European values for comparison
    euro_values = values[:]

    # Backward induction with early exercise
    for i in range(n - 1, -1, -1):
        for j in range(i + 1):
            s_node = spot * (u ** (i - 2 * j))
            hold = disc * (p_up * values[j] + p_dn * values[j + 1])

            if is_call:
                exercise = max(s_node - strike, 0.0)
            else:
                exercise = max(strike - s_node, 0.0)

            values[j] = max(hold, exercise)

            # European: no early exercise
            euro_values[j] = disc * (p_up * euro_values[j] + p_dn * euro_values[j + 1])

    american_price = values[0]
    european_price = euro_values[0]

    # Greeks via finite differences
    # Delta: bump spot by small amount
    bump = spot * 0.01
    price_up = _crr_scalar(spot + bump, strike, T, vol, r, q, is_call, steps)
    price_dn = _crr_scalar(spot - bump, strike, T, vol, r, q, is_call, steps)
    delta = (price_up - price_dn) / (2.0 * bump)

    # Gamma
    gamma = (price_up - 2.0 * american_price + price_dn) / (bump * bump)

    # Vega: bump vol by 0.01
    vol_bump = 0.01
    price_vol_up = _crr_scalar(spot, strike, T, vol + vol_bump, r, q, is_call, steps)
    vega = (price_vol_up - american_price)  # per 0.01 vol

    return AmericanPriceResult(
        price=american_price,
        european_price=european_price,
        early_exercise_premium=american_price - european_price,
        delta=delta,
        gamma=gamma,
        vega=vega,
        pricer_name="crr_tree",
        steps=steps,
        is_call=is_call,
    )


def _crr_scalar(
    spot: float,
    strike: float,
    T: float,
    vol: float,
    r: float,
    q: float,
    is_call: bool,
    steps: int,
) -> float:
    """Scalar CRR tree price (no Greeks) for finite-difference bumps."""
    if T <= 1e-10:
        if is_call:
            return max(spot - strike, 0.0)
        return max(strike - spot, 0.0)

    dt = T / steps
    u = math.exp(vol * math.sqrt(dt))
    d = 1.0 / u
    disc = math.exp(-r * dt)
    p_up = (math.exp((r - q) * dt) - d) / (u - d)

    n = steps
    prices = [spot * (u ** (n - 2 * j)) for j in range(n + 1)]

    if is_call:
        values = [max(p - strike, 0.0) for p in prices]
    else:
        values = [max(strike - p, 0.0) for p in prices]

    for i in range(n - 1, -1, -1):
        for j in range(i + 1):
            s_node = spot * (u ** (i - 2 * j))
            hold = disc * (p_up * values[j] + (1 - p_up) * values[j + 1])
            if is_call:
                exercise = max(s_node - strike, 0.0)
            else:
                exercise = max(strike - s_node, 0.0)
            values[j] = max(hold, exercise)

    return values[0]


def bjerksund_stensland_price(
    spot: float,
    strike: float,
    T: float,
    vol: float,
    r: float,
    q: float,
    is_call: bool,
) -> float:
    """
    Bjerksund-Stensland (1993) approximation for American options.

    Faster than tree but less accurate. Used as a speed optimization
    when tree computation is too slow.
    """
    if T <= 1e-10:
        if is_call:
            return max(spot - strike, 0.0)
        return max(strike - spot, 0.0)

    # For puts, use put-call symmetry
    if not is_call:
        return bjerksund_stensland_price(strike, spot, T, vol, r - q + r, q, True)

    # BS93 call approximation
    sigma2 = vol * vol
    b = r - q

    beta = (0.5 - b / sigma2) + math.sqrt(
        ((b / sigma2 - 0.5) ** 2) + 2.0 * r / sigma2
    )

    B_inf = beta / (beta - 1.0) * strike
    B_0 = max(strike, r / q * strike) if q > 0 else strike

    h = -(b * T + 2.0 * vol * math.sqrt(T)) * (B_0 / (B_inf - B_0))
    S_star = B_inf + (B_0 - B_inf) * math.exp(h)

    if spot >= S_star:
        return spot - strike

    alpha = (S_star - strike) * (S_star ** (-beta))

    # European call price
    d1 = (math.log(spot / strike) + (b + 0.5 * sigma2) * T) / (vol * math.sqrt(T))
    d2 = d1 - vol * math.sqrt(T)
    euro_call = spot * math.exp((b - r) * T) * norm_cdf(d1) - strike * math.exp(-r * T) * norm_cdf(d2)

    # Approximation terms
    d1_star = (math.log(spot / S_star) + (b + 0.5 * sigma2) * T) / (vol * math.sqrt(T))

    american_call = (
        alpha * (spot ** beta)
        - alpha * _phi(spot, T, beta, S_star, S_star, r, b, vol)
        + _phi(spot, T, 1, S_star, S_star, r, b, vol)
        - _phi(spot, T, 1, strike, S_star, r, b, vol)
        - strike * _phi(spot, T, 0, S_star, S_star, r, b, vol)
        + strike * _phi(spot, T, 0, strike, S_star, r, b, vol)
    )

    return max(american_call, euro_call)


def _phi(S, T, gamma_val, H, X, r, b, vol):
    """Helper function for Bjerksund-Stensland formula."""
    sigma2 = vol * vol
    lambda_val = (-r + gamma_val * b + 0.5 * gamma_val * (gamma_val - 1) * sigma2) * T
    kappa = (2 * b / sigma2) + (2 * gamma_val - 1)

    if X <= 0 or H <= 0 or S <= 0:
        return 0.0

    d1 = -(math.log(S / H) + (b + (gamma_val - 0.5) * sigma2) * T) / (vol * math.sqrt(T))
    d2 = -(math.log(X * X / (S * H)) + (b + (gamma_val - 0.5) * sigma2) * T) / (vol * math.sqrt(T))

    return (
        math.exp(lambda_val)
        * (S ** gamma_val)
        * (norm_cdf(d1) - (X / S) ** kappa * norm_cdf(d2))
    )

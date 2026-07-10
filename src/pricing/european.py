"""
European option pricing for the volatility infrastructure platform.

Implements forward-consistent Black-76/Black-Scholes pricing with
full analytic Greeks. This is the trusted pricing library — all
downstream analytics must call these functions rather than
reimplementing formulas.

Equations from the PDF:
  d1 = [ln(F/K) + 0.5*σ²*T] / (σ*√T)
  d2 = d1 - σ*√T
  Call = exp(-rT) * [F*N(d1) - K*N(d2)]
  Put  = exp(-rT) * [K*N(-d2) - F*N(-d1)]

Unit conventions:
  - Delta: per unit change in underlying price
  - Gamma: per unit change in underlying price
  - Vega: per 1 absolute vol point (0.01)
  - Theta: per calendar day (negative for long positions)
"""

import math
from dataclasses import dataclass
from typing import Optional

from src.utils.math_helpers import norm_cdf, norm_pdf


@dataclass(frozen=True)
class EuropeanPriceResult:
    """Complete pricing result with all Greeks."""
    price: float
    delta: float
    gamma: float
    vega: float           # per 1 vol point (0.01)
    theta: float          # per calendar day
    rho: float            # per 1% rate change
    d1: float
    d2: float
    forward: float
    intrinsic: float


def black76_price(
    forward: float,
    strike: float,
    T: float,
    vol: float,
    r: float,
    is_call: bool,
) -> float:
    """
    Black-76 European option price.

    Uses the forward-consistent form where F is the forward price
    for maturity T.

    Args:
        forward: Forward price F(T).
        strike: Strike price K.
        T: Time to maturity in years.
        vol: Implied volatility (annualized, as decimal e.g. 0.20).
        r: Risk-free rate (annualized, as decimal).
        is_call: True for call, False for put.

    Returns:
        Option price.
    """
    if T <= 1e-10:
        if is_call:
            return max(forward - strike, 0.0)
        return max(strike - forward, 0.0)

    if vol <= 1e-10:
        df = math.exp(-r * T)
        if is_call:
            return df * max(forward - strike, 0.0)
        return df * max(strike - forward, 0.0)

    sqrt_T = math.sqrt(T)
    d1 = (math.log(forward / strike) + 0.5 * vol * vol * T) / (vol * sqrt_T)
    d2 = d1 - vol * sqrt_T
    df = math.exp(-r * T)

    if is_call:
        return df * (forward * norm_cdf(d1) - strike * norm_cdf(d2))
    else:
        return df * (strike * norm_cdf(-d2) - forward * norm_cdf(-d1))


def black_scholes_price(
    spot: float,
    strike: float,
    T: float,
    vol: float,
    r: float,
    q: float,
    is_call: bool,
) -> float:
    """
    Black-Scholes option price with continuous dividend yield.

    Args:
        spot: Current spot price S.
        strike: Strike price K.
        T: Time to maturity in years.
        vol: Implied volatility.
        r: Risk-free rate.
        q: Continuous dividend yield.
        is_call: True for call, False for put.

    Returns:
        Option price.
    """
    forward = spot * math.exp((r - q) * T)
    return black76_price(forward, strike, T, vol, r, is_call)


def price_european(
    forward: float,
    strike: float,
    T: float,
    vol: float,
    r: float,
    is_call: bool,
    spot: Optional[float] = None,
) -> EuropeanPriceResult:
    """
    Full European option pricing with analytic Greeks.

    This is the canonical pricing entry point for European options.

    Args:
        forward: Forward price F(T).
        strike: Strike price K.
        T: Time to maturity in years.
        vol: Implied volatility.
        r: Risk-free rate.
        is_call: True for call, False for put.
        spot: Optional spot price for delta/gamma (defaults to F*exp(-rT)).

    Returns:
        EuropeanPriceResult with price and all Greeks.
    """
    S = spot if spot is not None else forward * math.exp(-r * T)

    if T <= 1e-10:
        intrinsic_call = max(forward - strike, 0.0)
        intrinsic_put = max(strike - forward, 0.0)
        price = intrinsic_call if is_call else intrinsic_put
        delta = (1.0 if forward > strike else 0.0) if is_call else (-1.0 if forward < strike else 0.0)
        return EuropeanPriceResult(
            price=price, delta=delta, gamma=0.0, vega=0.0,
            theta=0.0, rho=0.0, d1=0.0, d2=0.0,
            forward=forward, intrinsic=price,
        )

    sqrt_T = math.sqrt(T)
    vol_sqrt_T = vol * sqrt_T
    d1 = (math.log(forward / strike) + 0.5 * vol * vol * T) / vol_sqrt_T
    d2 = d1 - vol_sqrt_T
    df = math.exp(-r * T)

    # Price
    if is_call:
        price = df * (forward * norm_cdf(d1) - strike * norm_cdf(d2))
        intrinsic = max(forward - strike, 0.0) * df
    else:
        price = df * (strike * norm_cdf(-d2) - forward * norm_cdf(-d1))
        intrinsic = max(strike - forward, 0.0) * df

    # Greeks (with respect to spot)
    # Delta
    if is_call:
        delta = df * norm_cdf(d1) * (forward / S) if S > 0 else 0.0
    else:
        delta = df * (norm_cdf(d1) - 1.0) * (forward / S) if S > 0 else 0.0

    # Gamma
    if S > 0 and vol_sqrt_T > 0:
        gamma = df * norm_pdf(d1) * (forward / (S * S * vol_sqrt_T))
    else:
        gamma = 0.0

    # Vega (per 0.01 vol change)
    vega = df * forward * norm_pdf(d1) * sqrt_T * 0.01

    # Theta (per calendar day)
    theta_term1 = -df * forward * norm_pdf(d1) * vol / (2.0 * sqrt_T)
    if is_call:
        theta_term2 = -r * df * (forward * norm_cdf(d1) - strike * norm_cdf(d2))
    else:
        theta_term2 = -r * df * (strike * norm_cdf(-d2) - forward * norm_cdf(-d1))
    theta = (theta_term1 + theta_term2) / 365.0

    # Rho (per 1% rate change)
    if is_call:
        rho = T * df * strike * norm_cdf(d2) * 0.01
    else:
        rho = -T * df * strike * norm_cdf(-d2) * 0.01

    return EuropeanPriceResult(
        price=price,
        delta=delta,
        gamma=gamma,
        vega=vega,
        theta=theta,
        rho=rho,
        d1=d1,
        d2=d2,
        forward=forward,
        intrinsic=intrinsic,
    )


def bs_vega(
    forward: float,
    strike: float,
    T: float,
    vol: float,
    r: float,
) -> float:
    """
    Black-76 vega (derivative of price w.r.t. volatility).

    Used by the IV solver for Newton acceleration.
    Returns vega per unit vol (not per 0.01).
    """
    if T <= 1e-10 or vol <= 1e-10:
        return 0.0

    sqrt_T = math.sqrt(T)
    d1 = (math.log(forward / strike) + 0.5 * vol * vol * T) / (vol * sqrt_T)
    df = math.exp(-r * T)

    return df * forward * norm_pdf(d1) * sqrt_T

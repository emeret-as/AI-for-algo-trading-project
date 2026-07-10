"""
Implied volatility solver for the volatility infrastructure platform.

Separates economics from numerics. The solver uses a bracketed Brent
method as the primary safe path, with optional Newton acceleration
inside the bracket. Returns full diagnostics even on success.

From the PDF:
  "Use bracketed methods wherever possible for reliability.
   Failed solves should return structured diagnostics, not NaN without context."
"""

import math
from typing import Optional, List, Callable

from src.storage.schemas import IvSolveResult
from src.pricing.european import black76_price, bs_vega


# ─── Configuration ────────────────────────────────────────────────────

DEFAULT_SOLVER_CONFIG = {
    "vol_lower_bound": 0.001,      # near zero but not zero
    "vol_upper_bound": 5.0,        # 500% vol for extreme cases
    "max_iterations": 100,
    "price_tolerance": 1e-8,
    "vol_tolerance": 1e-8,
    "use_newton_acceleration": True,
    "model_name": "black76",
    "model_version": "v1",
}


# ─── Pre-checks ───────────────────────────────────────────────────────

def _intrinsic_value(
    forward: float,
    strike: float,
    r: float,
    T: float,
    is_call: bool,
) -> float:
    """Compute discounted intrinsic value."""
    df = math.exp(-r * T)
    if is_call:
        return df * max(forward - strike, 0.0)
    return df * max(strike - forward, 0.0)


def _validate_inputs(
    market_price: float,
    forward: float,
    strike: float,
    T: float,
    r: float,
    is_call: bool,
    config: dict,
) -> Optional[str]:
    """
    Pre-validate inputs before entering the root finder.

    Returns a failure reason string if invalid, None if OK.
    """
    if market_price <= 0:
        return "NON_POSITIVE_PRICE"

    if forward <= 0:
        return "NON_POSITIVE_FORWARD"

    if strike <= 0:
        return "NON_POSITIVE_STRIKE"

    if T <= 0:
        return "NON_POSITIVE_MATURITY"

    # Check below intrinsic
    intrinsic = _intrinsic_value(forward, strike, r, T, is_call)
    if market_price < intrinsic - 1e-6:
        return "BELOW_INTRINSIC"

    # Check above upper bound (max price = forward for call, strike for put)
    df = math.exp(-r * T)
    if is_call:
        max_price = df * forward
    else:
        max_price = df * strike
    if market_price > max_price + 1e-6:
        return "ABOVE_MAX_PRICE"

    return None


# ─── Brent solver ─────────────────────────────────────────────────────

def _brent_solve(
    f: Callable[[float], float],
    a: float,
    b: float,
    tol: float,
    max_iter: int,
) -> tuple:
    """
    Brent's method for root finding in [a, b].

    Returns (root, converged, iterations, residual).
    """
    fa = f(a)
    fb = f(b)

    if fa * fb > 0:
        # No sign change — try to find bracket
        return (a + b) / 2, False, 0, min(abs(fa), abs(fb))

    if abs(fa) < abs(fb):
        a, b = b, a
        fa, fb = fb, fa

    c = a
    fc = fa
    mflag = True
    s = 0.0
    d = 0.0

    for iteration in range(max_iter):
        if abs(fb) < tol:
            return b, True, iteration + 1, abs(fb)

        if abs(b - a) < tol:
            return b, True, iteration + 1, abs(fb)

        if fa != fc and fb != fc:
            # Inverse quadratic interpolation
            s = (a * fb * fc / ((fa - fb) * (fa - fc))
                 + b * fa * fc / ((fb - fa) * (fb - fc))
                 + c * fa * fb / ((fc - fa) * (fc - fb)))
        else:
            # Secant
            if abs(fb - fa) < 1e-15:
                s = (a + b) / 2
            else:
                s = b - fb * (b - a) / (fb - fa)

        # Conditions for bisection
        cond1 = not ((3 * a + b) / 4 < s < b or b < s < (3 * a + b) / 4)
        cond2 = mflag and abs(s - b) >= abs(b - c) / 2
        cond3 = (not mflag) and abs(s - b) >= abs(c - d) / 2
        cond4 = mflag and abs(b - c) < tol
        cond5 = (not mflag) and abs(c - d) < tol

        if cond1 or cond2 or cond3 or cond4 or cond5:
            s = (a + b) / 2
            mflag = True
        else:
            mflag = False

        fs = f(s)
        d = c
        c = b
        fc = fb

        if fa * fs < 0:
            b = s
            fb = fs
        else:
            a = s
            fa = fs

        if abs(fa) < abs(fb):
            a, b = b, a
            fa, fb = fb, fa

    return b, False, max_iter, abs(fb)


# ─── Main solver ──────────────────────────────────────────────────────

def solve_implied_vol(
    market_price: float,
    forward: float,
    strike: float,
    T: float,
    r: float,
    is_call: bool,
    contract_key: str = "",
    snapshot_ts: str = "",
    config: Optional[dict] = None,
) -> IvSolveResult:
    """
    Solve for implied volatility from a market price.

    Uses bracketed Brent method with optional Newton acceleration.
    Returns full diagnostics in all cases (success and failure).

    Args:
        market_price: Observed option mid price.
        forward: Forward price F(T).
        strike: Strike price K.
        T: Time to maturity in years.
        r: Risk-free rate.
        is_call: True for call, False for put.
        contract_key: Instrument key for diagnostics.
        snapshot_ts: Snapshot timestamp for diagnostics.
        config: Solver configuration overrides.

    Returns:
        IvSolveResult with implied vol and full diagnostics.
    """
    cfg = {**DEFAULT_SOLVER_CONFIG, **(config or {})}

    vol_lo = cfg["vol_lower_bound"]
    vol_hi = cfg["vol_upper_bound"]
    max_iter = cfg["max_iterations"]
    price_tol = cfg["price_tolerance"]

    # Pre-validation
    failure = _validate_inputs(market_price, forward, strike, T, r, is_call, cfg)
    if failure:
        return IvSolveResult(
            contract_key=contract_key,
            snapshot_ts=snapshot_ts,
            market_price=market_price,
            implied_vol=None,
            converged=False,
            iterations=0,
            residual=abs(market_price),
            lower_bound=vol_lo,
            upper_bound=vol_hi,
            failure_reason=failure,
            model_name=cfg["model_name"],
            model_version=cfg["model_version"],
            log_moneyness=math.log(strike / forward) if forward > 0 else 0.0,
            total_variance=0.0,
        )

    # Define the root function: f(σ) = BS(σ) - market_price
    def root_fn(vol):
        return black76_price(forward, strike, T, vol, r, is_call) - market_price

    # Try Newton first if enabled (faster convergence)
    if cfg.get("use_newton_acceleration", True):
        iv_newton = _newton_solve(
            market_price, forward, strike, T, r, is_call,
            vol_lo, vol_hi, max_iter, price_tol,
        )
        if iv_newton is not None:
            log_m = math.log(strike / forward) if forward > 0 else 0.0
            return IvSolveResult(
                contract_key=contract_key,
                snapshot_ts=snapshot_ts,
                market_price=market_price,
                implied_vol=iv_newton,
                converged=True,
                iterations=-1,  # Newton
                residual=abs(root_fn(iv_newton)),
                lower_bound=vol_lo,
                upper_bound=vol_hi,
                failure_reason=None,
                model_name=cfg["model_name"],
                model_version=cfg["model_version"],
                log_moneyness=log_m,
                total_variance=iv_newton * iv_newton * T,
            )

    # Brent solver
    vol, converged, iterations, residual = _brent_solve(
        root_fn, vol_lo, vol_hi, price_tol, max_iter
    )

    log_m = math.log(strike / forward) if forward > 0 else 0.0

    if converged:
        return IvSolveResult(
            contract_key=contract_key,
            snapshot_ts=snapshot_ts,
            market_price=market_price,
            implied_vol=vol,
            converged=True,
            iterations=iterations,
            residual=residual,
            lower_bound=vol_lo,
            upper_bound=vol_hi,
            failure_reason=None,
            model_name=cfg["model_name"],
            model_version=cfg["model_version"],
            log_moneyness=log_m,
            total_variance=vol * vol * T,
        )
    else:
        return IvSolveResult(
            contract_key=contract_key,
            snapshot_ts=snapshot_ts,
            market_price=market_price,
            implied_vol=None,
            converged=False,
            iterations=iterations,
            residual=residual,
            lower_bound=vol_lo,
            upper_bound=vol_hi,
            failure_reason="BRENT_NO_CONVERGENCE",
            model_name=cfg["model_name"],
            model_version=cfg["model_version"],
            log_moneyness=log_m,
            total_variance=0.0,
        )


def _newton_solve(
    market_price: float,
    forward: float,
    strike: float,
    T: float,
    r: float,
    is_call: bool,
    vol_lo: float,
    vol_hi: float,
    max_iter: int,
    tol: float,
) -> Optional[float]:
    """
    Newton-Raphson solver with bracket safety.

    Returns implied vol if converged, None otherwise.
    Falls back to None if iterate leaves the bracket.
    """
    # Initial guess: Brenner-Subrahmanyam approximation
    vol = market_price / (forward * math.sqrt(T / (2 * math.pi))) if T > 0 else 0.2
    vol = max(vol_lo, min(vol, vol_hi))

    for _ in range(min(max_iter, 20)):
        price = black76_price(forward, strike, T, vol, r, is_call)
        vega = bs_vega(forward, strike, T, vol, r)

        if vega < 1e-15:
            return None  # Vega collapsed, Newton unsafe

        diff = price - market_price
        if abs(diff) < tol:
            if vol_lo <= vol <= vol_hi:
                return vol
            return None

        vol_new = vol - diff / vega

        # Stay in bracket
        if vol_new < vol_lo or vol_new > vol_hi:
            return None

        vol = vol_new

    return None


def solve_chain_iv(
    snapshot,
    forward_results: List,
    config: Optional[dict] = None,
) -> List[IvSolveResult]:
    """
    Solve implied volatilities for all options in a snapshot.

    Uses the forward curve to determine forward prices per maturity.

    Args:
        snapshot: MarketStateSnapshot.
        forward_results: List of ForwardResult from forward engine.
        config: Solver configuration.

    Returns:
        List of IvSolveResult for all eligible options.
    """
    from src.storage.schemas import ForwardResult as FR

    # Build forward lookup by expiry
    fwd_lookup = {}
    for fwd in forward_results:
        fwd_lookup[fwd.expiry] = fwd.chosen_forward

    r = (config or {}).get("risk_free_rate", 0.05)
    results = []

    for row in snapshot.option_rows:
        if row.is_stale or row.mid <= 0:
            continue

        forward = fwd_lookup.get(row.expiry)
        if forward is None or forward <= 0:
            continue

        result = solve_implied_vol(
            market_price=row.mid,
            forward=forward,
            strike=row.strike,
            T=row.maturity_years,
            r=r,
            is_call=(row.right == "C"),
            contract_key=row.instrument_key,
            snapshot_ts=snapshot.snapshot_ts,
            config=config,
        )
        results.append(result)

    return results

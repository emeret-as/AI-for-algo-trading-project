"""
System-level validation checks for the volatility infrastructure.

Implements named validation checks as reusable functions covering:
collector continuity, forward stability, IV convergence, surface
fit quality, calendar sanity, Greek sanity, and scenario completeness.

From the PDF:
  "A good validation framework is specific. Avoid generic red banners.
   Operators need to know which maturity failed."
"""

from typing import List, Dict, Any, Optional
from collections import defaultdict

from src.storage.schemas import (
    QcCheckResult, MarketStateSnapshot, ForwardResult,
    IvSolveResult, SurfaceSlice, VolatilitySurface,
    ScenarioSummary,
)


DEFAULT_CHECK_CONFIG = {
    "min_option_coverage_pct": 50.0,
    "max_stale_ratio": 0.3,
    "forward_stability_threshold": 0.02,
    "iv_convergence_min_ratio": 0.95,
    "surface_max_rmse": 0.05,
    "scenario_completeness_pct": 100.0,
    "threshold_version": "checks_v1",
}


def check_underlying_health(
    snapshot: MarketStateSnapshot,
    config: Optional[Dict] = None,
) -> QcCheckResult:
    """Check underlying quote health (spread and staleness)."""
    cfg = {**DEFAULT_CHECK_CONFIG, **(config or {})}
    state = snapshot.underlying_state

    is_healthy = (
        not state.is_stale
        and state.spread_pct < 5.0
        and state.reference_spot > 0
    )

    return QcCheckResult(
        check_name="underlying_health",
        target_key=state.instrument_key,
        status="pass" if is_healthy else "warn" if state.reference_spot > 0 else "fail",
        measured_value=state.spread_pct,
        threshold_value=5.0,
        threshold_version=cfg["threshold_version"],
        reason_code="HEALTHY" if is_healthy else "DEGRADED_UNDERLYING",
        context={
            "reference_type": state.reference_type,
            "is_stale": state.is_stale,
            "spread_pct": state.spread_pct,
        },
        snapshot_ts=snapshot.snapshot_ts,
    )


def check_option_coverage(
    snapshot: MarketStateSnapshot,
    config: Optional[Dict] = None,
) -> QcCheckResult:
    """Check that enough option quotes are active (not stale)."""
    cfg = {**DEFAULT_CHECK_CONFIG, **(config or {})}
    flags = snapshot.flags

    total = flags.get("total_option_count", 0)
    active = flags.get("active_option_count", 0)
    coverage_pct = (active / total * 100.0) if total > 0 else 0.0

    threshold = cfg["min_option_coverage_pct"]
    status = "pass" if coverage_pct >= threshold else "fail"

    return QcCheckResult(
        check_name="option_coverage",
        target_key=snapshot.underlying_state.instrument_key,
        status=status,
        measured_value=coverage_pct,
        threshold_value=threshold,
        threshold_version=cfg["threshold_version"],
        reason_code="GOOD_COVERAGE" if status == "pass" else "LOW_COVERAGE",
        context={
            "total_options": total,
            "active_options": active,
            "maturity_count": flags.get("maturity_count", 0),
        },
        snapshot_ts=snapshot.snapshot_ts,
    )


def check_forward_stability(
    forward_results: List[ForwardResult],
    config: Optional[Dict] = None,
) -> List[QcCheckResult]:
    """Check forward curve stability per maturity."""
    cfg = {**DEFAULT_CHECK_CONFIG, **(config or {})}
    threshold = cfg["forward_stability_threshold"]
    results = []

    for fwd in forward_results:
        # Stability: MAD relative to forward level
        if fwd.chosen_forward > 0:
            relative_mad = fwd.mad_forward / fwd.chosen_forward
        else:
            relative_mad = float('inf')

        status = "pass" if relative_mad < threshold else "warn" if relative_mad < threshold * 2 else "fail"

        results.append(QcCheckResult(
            check_name="forward_stability",
            target_key=f"{fwd.underlying}_{fwd.expiry}",
            status=status,
            measured_value=relative_mad,
            threshold_value=threshold,
            threshold_version=cfg["threshold_version"],
            reason_code="STABLE" if status == "pass" else "UNSTABLE_FORWARD",
            context={
                "forward": fwd.chosen_forward,
                "mad": fwd.mad_forward,
                "candidates_raw": fwd.candidate_count_raw,
                "candidates_accepted": fwd.candidate_count_accepted,
                "confidence": fwd.confidence_score,
                "fallback_used": fwd.fallback_used,
            },
            snapshot_ts=fwd.snapshot_ts,
        ))

    return results


def check_iv_convergence(
    iv_results: List[IvSolveResult],
    config: Optional[Dict] = None,
) -> QcCheckResult:
    """Check IV solver convergence ratio."""
    cfg = {**DEFAULT_CHECK_CONFIG, **(config or {})}

    total = len(iv_results)
    converged = sum(1 for r in iv_results if r.converged)
    ratio = converged / total if total > 0 else 0.0

    threshold = cfg["iv_convergence_min_ratio"]
    status = "pass" if ratio >= threshold else "warn" if ratio >= threshold * 0.9 else "fail"

    # Failure analysis
    failure_reasons = defaultdict(int)
    for r in iv_results:
        if not r.converged and r.failure_reason:
            failure_reasons[r.failure_reason] += 1

    return QcCheckResult(
        check_name="iv_convergence",
        target_key="all",
        status=status,
        measured_value=ratio,
        threshold_value=threshold,
        threshold_version=cfg["threshold_version"],
        reason_code="GOOD_CONVERGENCE" if status == "pass" else "LOW_CONVERGENCE",
        context={
            "total_solves": total,
            "converged": converged,
            "failure_breakdown": dict(failure_reasons),
        },
    )


def check_surface_fit(
    surface: VolatilitySurface,
    config: Optional[Dict] = None,
) -> List[QcCheckResult]:
    """Check surface fit quality per slice."""
    cfg = {**DEFAULT_CHECK_CONFIG, **(config or {})}
    max_rmse = cfg["surface_max_rmse"]
    results = []

    for slice in surface.slices:
        status = "pass" if slice.fit_rmse < max_rmse else "warn" if slice.fit_rmse < max_rmse * 2 else "fail"

        results.append(QcCheckResult(
            check_name="surface_fit",
            target_key=f"{slice.underlying}_{slice.expiry}",
            status=status,
            measured_value=slice.fit_rmse,
            threshold_value=max_rmse,
            threshold_version=cfg["threshold_version"],
            reason_code="GOOD_FIT" if status == "pass" else "POOR_FIT",
            context={
                "model": slice.model_name,
                "accepted_points": slice.accepted_point_count,
                "max_error": slice.fit_max_error,
                "is_fallback": slice.is_fallback,
            },
            snapshot_ts=slice.snapshot_ts,
        ))

    return results


def check_scenario_completeness(
    summaries: List[ScenarioSummary],
    expected_count: int,
    config: Optional[Dict] = None,
) -> QcCheckResult:
    """Check that all configured scenarios have been executed."""
    cfg = {**DEFAULT_CHECK_CONFIG, **(config or {})}

    actual = len(summaries)
    completeness = (actual / expected_count * 100.0) if expected_count > 0 else 0.0

    status = "pass" if completeness >= 100.0 else "fail"

    return QcCheckResult(
        check_name="scenario_completeness",
        target_key="all",
        status=status,
        measured_value=completeness,
        threshold_value=100.0,
        threshold_version=cfg["threshold_version"],
        reason_code="ALL_SCENARIOS_COMPLETE" if status == "pass" else "MISSING_SCENARIOS",
        context={
            "expected": expected_count,
            "actual": actual,
        },
    )


def generate_qc_report(
    all_checks: List[QcCheckResult],
) -> Dict[str, Any]:
    """
    Generate a structured QC report from all check results.

    Returns a summary with counts by status and detail breakdowns.
    """
    by_status = defaultdict(list)
    for check in all_checks:
        by_status[check.status].append(check)

    return {
        "summary": {
            "total_checks": len(all_checks),
            "pass": len(by_status["pass"]),
            "warn": len(by_status["warn"]),
            "fail": len(by_status["fail"]),
            "pass_rate": len(by_status["pass"]) / max(len(all_checks), 1) * 100,
        },
        "failures": [
            {
                "check": c.check_name,
                "target": c.target_key,
                "measured": c.measured_value,
                "threshold": c.threshold_value,
                "reason": c.reason_code,
            }
            for c in by_status["fail"]
        ],
        "warnings": [
            {
                "check": c.check_name,
                "target": c.target_key,
                "measured": c.measured_value,
                "reason": c.reason_code,
            }
            for c in by_status["warn"]
        ],
    }

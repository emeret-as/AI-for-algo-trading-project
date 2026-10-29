"""
Job orchestration for the volatility infrastructure platform.

Provides entry points for scheduled jobs. Jobs are thin wrappers
that orchestrate library calls, write manifests, and emit metrics.

From the PDF:
  "Use dependency ordering so downstream jobs do not run on
   incomplete upstream data."
"""

import json
from datetime import datetime, timezone, date
from typing import Dict, Any, Optional, List

from src.utils.logging_helpers import build_logger, JobRunContext
from src.storage.writers import PartitionedWriter
from src.storage.schemas import Position
from src.risk.aggregation import compute_line_items, aggregate_risk, risk_summary_table


def run_full_pipeline(
    trade_date_str: str,
    underlying_symbol: str = "SPY",
    spot: float = 450.0,
    data_dir: str = "data",
    config: Optional[Dict[str, Any]] = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """
    Run the full analytics pipeline for one business day.

    Sequence (from the PDF Part IV-F):
      1. Build instrument universe
      2. Generate/load raw market events
      3. Build market-state snapshot
      4. Build forward curves
      5. Solve implied volatilities
      6. Fit volatility surface
      7. Compute Greeks and risk
      8. Run scenario engine
      9. Run QC suite
      10. Write outputs and manifest

    Args:
        trade_date_str: Trade date (YYYY-MM-DD).
        underlying_symbol: Underlying ticker.
        spot: Underlying spot price.
        data_dir: Root data directory.
        config: Pipeline configuration overrides.
        dry_run: If True, do not write outputs.

    Returns:
        Pipeline results summary.
    """
    from src.universe.contracts import (
        create_test_underlying, build_option_universe,
    )
    from src.collectors.synthetic_feed import SyntheticFeed
    from src.snapshots.builder import build_snapshot
    from src.forwards.engine import build_forward_curve
    from src.forwards.carry import build_carry_curve
    from src.qc.quote_quality import filter_quotes
    from src.iv.solver import solve_chain_iv
    from src.surfaces.calibration import fit_surface
    from src.risk.greeks import compute_contract_greeks
    from src.risk.scenarios import run_scenario_engine, DEFAULT_SCENARIOS
    from src.qc.checks import (
        check_underlying_health, check_option_coverage,
        check_forward_stability, check_iv_convergence,
        check_surface_fit, check_scenario_completeness,
        generate_qc_report,
    )

    cfg = config or {}
    logger = build_logger("orchestration.jobs", structured=False)
    run_ctx = JobRunContext.create(trade_date_str)

    logger.info(f"=== Pipeline start: {trade_date_str} | {underlying_symbol} ===")
    results = {"trade_date": trade_date_str, "underlying": underlying_symbol}
    valuation_date = date.fromisoformat(trade_date_str)

    # ─── Step 1: Universe ─────────────────────────────────────────
    logger.info("Step 1: Building instrument universe")
    underlying = create_test_underlying(underlying_symbol)
    contracts = build_option_universe(
        underlying, valuation_date, spot,
        num_strikes=cfg.get("num_strikes", 21),
        num_monthly=cfg.get("num_monthly", 4),
        include_weeklies=cfg.get("include_weeklies", True),
    )
    results["universe_size"] = len(contracts)
    logger.info(f"  Universe: {len(contracts)} contracts")

    # ─── Step 2: Raw events ───────────────────────────────────────
    logger.info("Step 2: Generating synthetic market data")
    feed = SyntheticFeed(
        spot=spot,
        base_vol=cfg.get("base_vol", 0.20),
        regime=cfg.get("regime", "normal"),
        seed=cfg.get("seed", 42),
    )
    events = feed.generate_full_snapshot_events(
        underlying, contracts, valuation_date,
    )
    results["raw_events"] = len(events)
    logger.info(f"  Generated {len(events)} raw events")

    # ─── Step 3: Snapshot ─────────────────────────────────────────
    logger.info("Step 3: Building market-state snapshot")
    snapshot_ts = datetime.now(timezone.utc).isoformat()
    snapshot = build_snapshot(
        events, underlying.instrument_key, contracts,
        snapshot_ts, trade_date_str,
    )
    results["snapshot_options"] = len(snapshot.option_rows)
    results["snapshot_flags"] = snapshot.flags
    logger.info(f"  Snapshot: {len(snapshot.option_rows)} option rows")

    # ─── Step 4: QC filter ────────────────────────────────────────
    logger.info("Step 4: Running quote QC")
    accepted, rejected, qc_results = filter_quotes(
        snapshot.option_rows, snapshot.underlying_state.reference_spot,
    )
    results["qc_accepted"] = len(accepted)
    results["qc_rejected"] = len(rejected)
    logger.info(f"  QC: {len(accepted)} accepted, {len(rejected)} rejected")

    # ─── Step 5: Forwards ─────────────────────────────────────────
    logger.info("Step 5: Building forward curves")
    forward_results = build_forward_curve(snapshot)
    results["forward_count"] = len(forward_results)
    high_conf = sum(1 for f in forward_results if f.confidence_score > 0.5)
    logger.info(f"  Forwards: {len(forward_results)} maturities, {high_conf} high confidence")

    # ─── Step 5b: Carry ───────────────────────────────────────────
    carry_diags = build_carry_curve(
        snapshot.underlying_state.reference_spot, forward_results,
    )

    # ─── Step 6: IV solve ─────────────────────────────────────────
    logger.info("Step 6: Solving implied volatilities")
    iv_results = solve_chain_iv(snapshot, forward_results, cfg)
    converged = sum(1 for r in iv_results if r.converged)
    results["iv_total"] = len(iv_results)
    results["iv_converged"] = converged
    conv_ratio = converged / len(iv_results) if iv_results else 0.0
    logger.info(f"  IV: {converged}/{len(iv_results)} converged ({conv_ratio:.1%})")

    # ─── Step 7: Surface ──────────────────────────────────────────
    logger.info("Step 7: Fitting volatility surface")
    surface = fit_surface(iv_results, underlying.instrument_key, snapshot_ts)
    svi_count = sum(1 for s in surface.slices if s.model_name == "svi")
    results["surface_slices"] = len(surface.slices)
    results["svi_fits"] = svi_count
    logger.info(f"  Surface: {len(surface.slices)} slices, {svi_count} SVI fits")

    # ─── Step 8: Greeks ───────────────────────────────────────────
    logger.info("Step 8: Computing Greeks")
    fwd_lookup = {f.expiry: f.chosen_forward for f in forward_results}
    iv_lookup = {r.contract_key: r for r in iv_results if r.converged}

    pricing_results = {}
    for row in snapshot.option_rows:
        iv = iv_lookup.get(row.instrument_key)
        fwd = fwd_lookup.get(row.expiry)
        if iv and fwd:
            pr = compute_contract_greeks(
                row, iv, fwd, snapshot.underlying_state.reference_spot,
            )
            if pr:
                pricing_results[row.instrument_key] = pr

    results["priced_contracts"] = len(pricing_results)
    logger.info(f"  Priced: {len(pricing_results)} contracts")

    # ─── Step 9: Risk ─────────────────────────────────────────────
    logger.info("Step 9: Computing risk aggregates")
    # Create synthetic positions for demo
    positions = [
        Position(
            portfolio_id="demo_portfolio",
            contract_key=key,
            quantity=10.0 if i % 3 == 0 else -5.0,
            valuation_ts=snapshot_ts,
        )
        for i, key in enumerate(list(pricing_results.keys())[:20])
    ]

    line_items = compute_line_items(positions, pricing_results)
    aggregates = aggregate_risk(line_items)
    results["risk_positions"] = len(positions)
    results["risk_aggregates"] = len(aggregates)
    results["risk_table"] = risk_summary_table(aggregates)
    logger.info(f"  Risk: {len(positions)} positions, {len(aggregates)} aggregates")

    # ─── Step 10: Scenarios ───────────────────────────────────────
    logger.info("Step 10: Running scenario engine")
    scenario_results, summaries = run_scenario_engine(
        DEFAULT_SCENARIOS, positions, pricing_results,
    )
    scenario_rows = [
        {
            'scenario': summary.scenario_id,
            'portfolio': summary.portfolio_id,
            'total_pnl': round(summary.total_pnl, 2),
            'worst_contributor': summary.worst_contributor_key,
            'worst_pnl': round(summary.worst_contributor_pnl, 2),
            'positions': summary.position_count,
        }
        for summary in summaries
    ]
    results['scenario_table'] = scenario_rows
    results["scenario_count"] = len(summaries)
    if summaries:
        worst = min(summaries, key=lambda s: s.total_pnl)
        results["worst_scenario"] = worst.scenario_id
        results["worst_pnl"] = worst.total_pnl
    logger.info(f"  Scenarios: {len(summaries)} completed")

    # ─── Step 11: QC suite ────────────────────────────────────────
    logger.info("Step 11: Running QC suite")
    all_qc_checks = []
    all_qc_checks.append(check_underlying_health(snapshot))
    all_qc_checks.append(check_option_coverage(snapshot))
    all_qc_checks.extend(check_forward_stability(forward_results))
    all_qc_checks.append(check_iv_convergence(iv_results))
    all_qc_checks.extend(check_surface_fit(surface))
    all_qc_checks.append(check_scenario_completeness(summaries, len(DEFAULT_SCENARIOS)))

    qc_report = generate_qc_report(all_qc_checks)
    results["qc_report"] = qc_report
    results["qc_table"] = [
        {
            "check": check.check_name,
            "target": check.target_key,
            "status": check.status,
            "measured": check.measured_value,
            "threshold": check.threshold_value,
            "reason": check.reason_code,
        }
        for check in all_qc_checks
    ]
    logger.info(
        f"  QC: {qc_report['summary']['pass']} pass, "
        f"{qc_report['summary']['warn']} warn, "
        f"{qc_report['summary']['fail']} fail"
    )

    # ─── Step 12: Write outputs ───────────────────────────────────
    if not dry_run:
        logger.info("Step 12: Writing outputs")
        writer = PartitionedWriter(data_dir, "pipeline_results", run_ctx.run_id)
        writer.write_manifest(
            {**run_ctx.to_manifest(), "results_summary": results},
            trade_date_str,
        )
        logger.info("  Manifest written")

    logger.info(f"=== Pipeline complete: {trade_date_str} ===")
    return results

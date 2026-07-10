"""
Historical replay for the volatility infrastructure platform.

Enables reconstruction of historical analytics from stored raw data
using the same code path as live analytics.

From the PDF:
  "The replay code path should call the same libraries as live processing.
   Resist the temptation to fork a separate 'historical only' implementation."
"""

from datetime import date, timedelta
from typing import List, Dict, Any, Optional

from src.orchestration.jobs import run_full_pipeline
from src.utils.logging_helpers import build_logger


def replay_date(
    trade_date_str: str,
    underlying_symbol: str = "SPY",
    spot: float = 450.0,
    data_dir: str = "data",
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Replay analytics for a single historical date.

    Uses the same code path as live processing (run_full_pipeline).
    The only difference is the data source: stored raw events instead
    of live broker feed.

    Args:
        trade_date_str: Historical trade date (YYYY-MM-DD).
        underlying_symbol: Underlying ticker.
        spot: Historical spot price for that date.
        data_dir: Data directory.
        config: Pipeline configuration.

    Returns:
        Pipeline results for the replayed date.
    """
    logger = build_logger("orchestration.replay", structured=False)
    logger.info(f"Replaying {trade_date_str} for {underlying_symbol}")

    # Use the same pipeline — same code path as live
    cfg = config or {}
    cfg["replay_mode"] = True

    result = run_full_pipeline(
        trade_date_str=trade_date_str,
        underlying_symbol=underlying_symbol,
        spot=spot,
        data_dir=data_dir,
        config=cfg,
    )

    result["replay"] = True
    result["replay_date"] = trade_date_str
    return result


def replay_date_range(
    start_date_str: str,
    end_date_str: str,
    underlying_symbol: str = "SPY",
    spot_series: Optional[Dict[str, float]] = None,
    data_dir: str = "data",
    config: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Replay analytics across a date range.

    Args:
        start_date_str: Start date (YYYY-MM-DD).
        end_date_str: End date (YYYY-MM-DD).
        underlying_symbol: Underlying ticker.
        spot_series: Optional dict of date -> spot price.
        data_dir: Data directory.
        config: Pipeline configuration.

    Returns:
        List of pipeline results, one per date.
    """
    logger = build_logger("orchestration.replay", structured=False)
    start = date.fromisoformat(start_date_str)
    end = date.fromisoformat(end_date_str)

    default_spot = 450.0
    results = []
    current = start

    while current <= end:
        # Skip weekends
        if current.weekday() >= 5:
            current += timedelta(days=1)
            continue

        date_str = current.isoformat()
        spot = (spot_series or {}).get(date_str, default_spot)

        try:
            result = replay_date(
                date_str, underlying_symbol, spot, data_dir, config,
            )
            results.append(result)
            logger.info(f"Replay {date_str}: OK")
        except Exception as e:
            logger.error(f"Replay {date_str}: FAILED - {e}")
            results.append({
                "trade_date": date_str,
                "replay": True,
                "error": str(e),
            })

        current += timedelta(days=1)

    logger.info(
        f"Replay complete: {len(results)} dates, "
        f"{sum(1 for r in results if 'error' not in r)} successful"
    )
    return results

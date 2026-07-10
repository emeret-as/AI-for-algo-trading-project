"""
Raw event capture and persistence for the volatility infrastructure.

Implements the collector service that normalizes broker events,
stamps them with timestamps, and persists them to the raw layer.

Rules from the PDF:
  - Callbacks should only normalize, stamp, and persist.
  - Never compute analytics inside the collector callback.
  - Append-only semantics.
  - Malformed events quarantined, never silently dropped.
"""

import uuid
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

from src.storage.schemas import RawMarketEvent
from src.storage.writers import PartitionedWriter
from src.utils.logging_helpers import build_logger


class RawCollector:
    """
    Raw market event collector service.

    Normalizes incoming broker events into RawMarketEvent objects
    and persists them to the raw data layer. The collector callback
    is intentionally lightweight — no analytics computation.
    """

    def __init__(
        self,
        writer: PartitionedWriter,
        session_id: Optional[str] = None,
        batch_size: int = 100,
    ):
        """
        Args:
            writer: Partitioned writer for raw events.
            session_id: Unique session identifier.
            batch_size: Number of events to batch before flushing.
        """
        self.session_id = session_id or f"sess_{uuid.uuid4().hex[:8]}"
        self._writer = writer
        self._batch_size = batch_size
        self._buffer: List[RawMarketEvent] = []
        self._event_counter = 0
        self._logger = build_logger("collectors.raw_writer", structured=False)
        self._trade_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def on_event(self, broker_event: Dict[str, Any]) -> None:
        """
        Process a single broker event.

        This is the callback method. It normalizes, stamps, and buffers.
        No heavy computation should happen here.
        """
        try:
            normalized = self._normalize_event(broker_event)
            self._buffer.append(normalized)
            self._event_counter += 1

            if len(self._buffer) >= self._batch_size:
                self.flush()

        except Exception as e:
            self._logger.warning(
                f"Failed to normalize event: {e}",
                extra={"extra_fields": {"raw_event": str(broker_event)[:200]}}
            )

    def _normalize_event(self, broker_event: Dict[str, Any]) -> RawMarketEvent:
        """
        Normalize a broker event into a RawMarketEvent.

        Extracts instrument_key, field_name, field_value, and timestamps.
        """
        receipt_ts = datetime.now(timezone.utc).isoformat()
        event_id = f"{self.session_id}_{self._event_counter:08d}"

        return RawMarketEvent(
            event_id=event_id,
            collector_session_id=self.session_id,
            instrument_key=broker_event.get("instrument_key", "UNKNOWN"),
            field_name=broker_event.get("field_name", "unknown"),
            field_value=float(broker_event.get("field_value", 0.0)),
            receipt_ts=receipt_ts,
            source_ts=broker_event.get("source_ts"),
            contract_id_broker=broker_event.get("contract_id_broker", ""),
        )

    def flush(self) -> Dict[str, Any]:
        """Flush the event buffer to storage."""
        if not self._buffer:
            return {"records_written": 0}

        result = self._writer.write_records(
            records=self._buffer,
            trade_date=self._trade_date,
            filename_prefix="raw_events",
        )

        self._logger.info(
            f"Flushed {len(self._buffer)} raw events to {result.get('path', '')}"
        )
        self._buffer.clear()
        return result

    def set_trade_date(self, trade_date: str) -> None:
        """Set the trade date for partitioning."""
        self._trade_date = trade_date

    @property
    def stats(self) -> Dict[str, Any]:
        """Return collector statistics."""
        return {
            "session_id": self.session_id,
            "total_events": self._event_counter,
            "buffer_size": len(self._buffer),
            "trade_date": self._trade_date,
            "writer_stats": self._writer.stats,
        }

    def close(self) -> None:
        """Flush remaining events and close."""
        self.flush()
        self._logger.info(
            f"Collector session {self.session_id} closed. "
            f"Total events: {self._event_counter}"
        )

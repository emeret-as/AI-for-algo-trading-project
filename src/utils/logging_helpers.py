"""
Structured logging helpers for the volatility infrastructure platform.

Provides correlation-ID-aware logging, structured log formatting,
and a logger builder used by all services.
"""

import logging
import json
import sys
import uuid
from datetime import datetime, timezone
from typing import Optional, Dict, Any


class StructuredFormatter(logging.Formatter):
    """
    JSON-structured log formatter for production observability.

    Each log line is a JSON object containing timestamp, level,
    message, module, and any additional context fields.
    """

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "module": record.name,
            "message": record.getMessage(),
            "lineno": record.lineno,
        }

        # Add correlation ID if present
        if hasattr(record, "correlation_id"):
            log_entry["correlation_id"] = record.correlation_id

        # Add extra fields
        if hasattr(record, "extra_fields"):
            log_entry.update(record.extra_fields)

        # Add exception info if present
        if record.exc_info and record.exc_info[0] is not None:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry)


class CorrelationAdapter(logging.LoggerAdapter):
    """
    Logger adapter that injects correlation_id into every log record.

    Correlation IDs link collector sessions to analytics jobs,
    enabling end-to-end tracing through the pipeline.
    """

    def process(self, msg, kwargs):
        extra = kwargs.get("extra", {})
        extra["correlation_id"] = self.extra.get("correlation_id", "unknown")
        kwargs["extra"] = extra
        return msg, kwargs


def build_logger(
    name: str,
    correlation_id: Optional[str] = None,
    level: int = logging.INFO,
    structured: bool = True,
) -> logging.LoggerAdapter:
    """
    Build a structured logger with correlation ID support.

    This is the canonical logger builder used by all services.
    Every module should obtain its logger through this function.

    Args:
        name: Logger name (typically the module name).
        correlation_id: Unique ID linking related log entries.
        level: Logging level.
        structured: If True, use JSON formatting.

    Returns:
        A LoggerAdapter with correlation ID injection.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers on repeated calls
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)

        if structured:
            handler.setFormatter(StructuredFormatter())
        else:
            handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
                )
            )

        logger.addHandler(handler)

    cid = correlation_id or str(uuid.uuid4())[:8]
    return CorrelationAdapter(logger, {"correlation_id": cid})


class JobRunContext:
    """
    Execution context for a pipeline job.

    Captures the run_id, environment, code version, config hashes,
    and correlation ID for provenance tracking.
    """

    def __init__(
        self,
        run_id: str,
        environment: str,
        code_version: str,
        config_hashes: Dict[str, str],
        trade_date: str,
        correlation_id: Optional[str] = None,
    ):
        self.run_id = run_id
        self.environment = environment
        self.code_version = code_version
        self.config_hashes = config_hashes
        self.trade_date = trade_date
        self.correlation_id = correlation_id or str(uuid.uuid4())[:12]
        self.start_ts = datetime.now(timezone.utc).isoformat()

    @classmethod
    def create(
        cls,
        trade_date: str,
        environment: str = "development",
        code_version: str = "vol-infra-4.0.0",
    ) -> "JobRunContext":
        """Create a new job run context with auto-generated IDs."""
        run_id = f"{trade_date}_{environment}_{uuid.uuid4().hex[:6]}"
        return cls(
            run_id=run_id,
            environment=environment,
            code_version=code_version,
            config_hashes={},
            trade_date=trade_date,
        )

    def to_manifest(self) -> Dict[str, Any]:
        """Export as a manifest dictionary for persistence."""
        return {
            "run_id": self.run_id,
            "environment": self.environment,
            "code_version": self.code_version,
            "config_hashes": self.config_hashes,
            "trade_date": self.trade_date,
            "correlation_id": self.correlation_id,
            "start_ts": self.start_ts,
        }

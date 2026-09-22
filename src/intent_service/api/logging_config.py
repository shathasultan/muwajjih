"""Structured JSON logging, correlated by trace id.

Two problems this solves.

First, without it `INTENT_LOG_LEVEL` is dead configuration: under uvicorn's
default setup the root logger sits at WARNING with no handler attached to
this package, so every `logger.info(...)` line is discarded. That would make
the `trace_id` returned to callers useless -- there would be nothing to grep.

Second, and the reason the output is JSON rather than a formatted line: a log
aggregator can filter on `trace_id` as a field, but only if it is a field.
A human-readable string like `predict_ok trace_id=abc` requires every consumer
to re-parse it with a regex that breaks the moment the message is reworded.

PRIVACY: `format` copies only whitelisted keys out of the record's `extra`.
Message text is customer data and is never logged -- not at DEBUG, not on the
error path. What IS logged is a trace id, which correlates a log line to a
request without revealing anything about its content. That is the whole point
of handing the id back to the caller: a user can quote it in a support ticket
instead of pasting their message into one.
"""

import datetime as dt
import json
import logging
import sys
from typing import Any

# Everything else on a LogRecord is stdlib noise. Anything a call site wants
# in the output goes through `extra=` and must be named here -- an allowlist,
# so a future `logger.info(..., extra={"text": message.text})` silently drops
# the customer data instead of shipping it to the log aggregator.
CONTEXT_FIELDS = frozenset(
    {"trace_id", "event", "action", "intent", "priority", "confidence", "cached", "path", "status"}
)


class JsonFormatter(logging.Formatter):
    """Render a LogRecord as a single line of JSON."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": dt.datetime.fromtimestamp(record.created, tz=dt.UTC).isoformat(
                timespec="milliseconds"
            ),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        for key, value in record.__dict__.items():
            if key in CONTEXT_FIELDS:
                payload[key] = value

        if record.exc_info:
            # The traceback goes in a field of its own rather than being
            # concatenated onto the message, so multi-line tracebacks do not
            # break one-JSON-object-per-line parsing.
            payload["exception"] = self.formatException(record.exc_info)

        # ensure_ascii=False keeps Arabic readable in the log stream instead
        # of turning every character into a \uXXXX escape.
        return json.dumps(payload, ensure_ascii=False, default=str)


def configure_logging(level: str) -> None:
    """Attach a JSON stdout handler to this package's logger.

    Writes to stdout (not stderr) because a container's stdout is the
    conventional log stream that orchestrators collect. `propagate = False`
    keeps these lines from being duplicated by uvicorn's own root handler.
    """
    resolved = logging.getLevelNamesMapping().get(level.upper())
    if resolved is None:
        raise ValueError(
            f"Invalid INTENT_LOG_LEVEL: {level!r}. "
            "Use one of DEBUG, INFO, WARNING, ERROR, CRITICAL."
        )

    package_logger = logging.getLogger("intent_service")
    package_logger.setLevel(resolved)
    package_logger.propagate = False

    # Idempotent: create_app may be called more than once in a test session,
    # and stacking handlers would duplicate every line.
    if not any(getattr(h, "_intent_service_handler", False) for h in package_logger.handlers):
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JsonFormatter())
        handler._intent_service_handler = True  # type: ignore[attr-defined]
        package_logger.addHandler(handler)

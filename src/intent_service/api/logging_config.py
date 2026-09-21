"""Logging setup for the service.

Without this, `INTENT_LOG_LEVEL` is dead configuration: under uvicorn's
default setup the root logger sits at WARNING with no handler attached to
this package, so every `logger.info("predict_ok trace_id=...")` line is
discarded. That would make the `trace_id` returned to callers useless --
there would be nothing to grep for.
"""

import logging
import sys

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"


def configure_logging(level: str) -> None:
    """Attach a stdout handler to this package's logger.

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
        handler.setFormatter(logging.Formatter(LOG_FORMAT))
        handler._intent_service_handler = True  # type: ignore[attr-defined]
        package_logger.addHandler(handler)

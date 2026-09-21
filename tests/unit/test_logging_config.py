"""Tests for logging setup.

`INTENT_LOG_LEVEL` was dead configuration once: it was declared in Settings
but never read, so every `logger.info(...)` line in the service -- including
the `trace_id` the API returns to callers -- was discarded. These tests keep
that wiring honest.
"""

import logging

import pytest

from intent_service.api.logging_config import configure_logging


@pytest.fixture(autouse=True)
def _reset_package_logger():
    logger = logging.getLogger("intent_service")
    original_handlers = list(logger.handlers)
    original_level = logger.level
    logger.handlers.clear()
    yield
    logger.handlers[:] = original_handlers
    logger.setLevel(original_level)


def test_configure_attaches_a_handler_at_the_requested_level() -> None:
    configure_logging("DEBUG")

    logger = logging.getLogger("intent_service")
    assert logger.level == logging.DEBUG
    assert logger.handlers


def test_info_records_are_actually_emitted(capsys) -> None:
    """The regression that matters: an INFO line must reach stdout."""
    configure_logging("INFO")

    logging.getLogger("intent_service.test").info("predict_ok trace_id=abc123")

    assert "predict_ok trace_id=abc123" in capsys.readouterr().out


def test_configure_is_idempotent() -> None:
    """create_app may run more than once per process; stacking handlers
    would duplicate every log line.

    Counts only handlers this module installed -- pytest injects its own
    LogCaptureHandler, which is not ours to assert about.
    """
    configure_logging("INFO")
    configure_logging("INFO")

    ours = [
        h
        for h in logging.getLogger("intent_service").handlers
        if getattr(h, "_intent_service_handler", False)
    ]
    assert len(ours) == 1


def test_invalid_level_fails_loudly() -> None:
    with pytest.raises(ValueError, match="Invalid INTENT_LOG_LEVEL"):
        configure_logging("VERBOSE")

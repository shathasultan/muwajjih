"""Shared fixtures.

The fakes here are the payoff for the Protocol-based ports: a whole test
suite for the service and API layers runs without sklearn, without an
artifact on disk, and without Redis.
"""

from typing import ClassVar

import pytest

from intent_service.config import Settings
from intent_service.domain.policy import PolicyThresholds

TEST_KEY = "test-key-0123456789"


class FakeIntentModel:
    """Satisfies the `IntentModel` port. Returns whatever it was told to.

    Six lines is the entire cost of testing the service layer -- no mocking
    framework, no patching, no artifact. That is the point of the port.
    """

    model_version = "test-v1"
    labels: ClassVar[list[str]] = [
        "complaint",
        "order_status",
        "praise",
        "price_inquiry",
        "return_refund",
        "support_request",
    ]

    def __init__(self, label: str = "complaint", confidence: float = 0.9) -> None:
        self.label = label
        self.confidence = confidence
        self.calls = 0

    def predict(self, text: str) -> tuple[str, float]:
        self.calls += 1
        return self.label, self.confidence


@pytest.fixture
def thresholds() -> PolicyThresholds:
    """Explicit values, never production's. A test that silently inherits a
    tuned threshold stops testing the band it claims to test the moment
    operations retunes it."""
    return PolicyThresholds(auto_route_floor=0.60, reject_floor=0.25)


@pytest.fixture
def settings(tmp_path: object) -> Settings:
    return Settings(
        api_keys=TEST_KEY,
        require_api_key=True,
        redis_url="",
        rate_limit_requests=1000,
        log_level="WARNING",
    )

"""Unit tests for the security primitives, in isolation from HTTP."""

import time

from intent_service.api.security import SlidingWindowRateLimiter, is_authorised


def test_valid_key_is_accepted() -> None:
    assert is_authorised("secret-key", ["secret-key", "other"]) is True


def test_wrong_key_is_rejected() -> None:
    assert is_authorised("wrong", ["secret-key"]) is False


def test_missing_key_is_rejected() -> None:
    assert is_authorised(None, ["secret-key"]) is False
    assert is_authorised("", ["secret-key"]) is False


def test_no_configured_keys_rejects_everything() -> None:
    assert is_authorised("anything", []) is False


def test_rate_limiter_allows_up_to_the_budget() -> None:
    limiter = SlidingWindowRateLimiter(max_requests=3, window_seconds=60)

    assert [limiter.check("caller")[0] for _ in range(3)] == [True, True, True]


def test_rate_limiter_blocks_past_the_budget() -> None:
    limiter = SlidingWindowRateLimiter(max_requests=2, window_seconds=60)
    limiter.check("caller")
    limiter.check("caller")

    allowed, retry_after = limiter.check("caller")

    assert allowed is False
    assert retry_after >= 1


def test_rate_limit_budgets_are_per_caller() -> None:
    """One noisy caller must not consume another caller's budget."""
    limiter = SlidingWindowRateLimiter(max_requests=1, window_seconds=60)
    limiter.check("caller-a")

    assert limiter.check("caller-a")[0] is False
    assert limiter.check("caller-b")[0] is True


def test_window_slides_so_budget_recovers() -> None:
    limiter = SlidingWindowRateLimiter(max_requests=1, window_seconds=0.05)
    limiter.check("caller")
    assert limiter.check("caller")[0] is False

    time.sleep(0.06)

    assert limiter.check("caller")[0] is True


def test_expired_callers_are_swept_from_memory() -> None:
    """Without sweeping, the dict grows for the process lifetime: every new
    IP adds an entry that is never removed."""
    limiter = SlidingWindowRateLimiter(max_requests=5, window_seconds=0.05)
    for i in range(50):
        limiter.check(f"caller-{i}")

    assert limiter.tracked_callers == 50

    time.sleep(0.06)
    limiter.check("someone-new")  # triggers the amortised sweep

    assert limiter.tracked_callers == 1


def test_sweep_keeps_callers_still_inside_their_window() -> None:
    limiter = SlidingWindowRateLimiter(max_requests=5, window_seconds=0.05)
    limiter.check("old")
    time.sleep(0.06)
    limiter.check("recent")
    limiter.check("recent-again")

    assert "ip:old" not in limiter._hits
    assert limiter.tracked_callers == 2

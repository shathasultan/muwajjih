"""Authentication, rate limiting, and hardening for the HTTP layer.

Security lives in the API layer only: the domain and service layers stay
unaware that authentication exists, exactly as they stay unaware that HTTP
exists. Swapping API keys for OAuth later touches this file and nothing else.

Deliberately dependency-free (stdlib + FastAPI). Every control here has a
documented limitation in README.md -- an in-process rate limiter is honest
about not being shared across replicas rather than pretending otherwise.
"""

import secrets
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field

from fastapi import HTTPException, Request, status

API_KEY_HEADER = "X-API-Key"


def is_authorised(presented_key: str | None, valid_keys: list[str]) -> bool:
    """Constant-time key comparison.

    A plain `presented in valid_keys` leaks key content through timing: the
    comparison returns as soon as bytes differ, so an attacker can recover a
    key character by character. `compare_digest` always takes the same time.
    """
    if not presented_key:
        return False
    return any(secrets.compare_digest(presented_key, valid) for valid in valid_keys)


def require_api_key(request: Request) -> None:
    """FastAPI dependency: reject the request unless it carries a valid key.

    `/health` deliberately does NOT use this -- an orchestrator's liveness
    probe must work without credentials, and it exposes no business data.
    """
    settings = request.app.state.app_state.settings
    if not settings.require_api_key:
        return

    presented = request.headers.get(API_KEY_HEADER)
    if not is_authorised(presented, settings.api_key_list):
        # One generic message for both "no key" and "wrong key": telling a
        # caller which of the two failed hands an attacker free information.
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            "missing or invalid API key",
            headers={"WWW-Authenticate": API_KEY_HEADER},
        )


@dataclass
class SlidingWindowRateLimiter:
    """Fixed request budget per caller over a sliding time window.

    Keyed by API key when present, falling back to client IP. In-process and
    per-replica by design -- a multi-replica deployment needs Redis; see the
    limitations section of README.md.
    """

    max_requests: int
    window_seconds: float
    _hits: dict[str, deque[float]] = field(default_factory=lambda: defaultdict(deque))

    def check(self, caller: str) -> tuple[bool, int]:
        """Return (allowed, retry_after_seconds)."""
        now = time.monotonic()
        window = self._hits[caller]

        # Drop timestamps that fell out of the window.
        cutoff = now - self.window_seconds
        while window and window[0] <= cutoff:
            window.popleft()

        if len(window) >= self.max_requests:
            retry_after = max(1, int(window[0] + self.window_seconds - now) + 1)
            return False, retry_after

        window.append(now)
        return True, 0


def caller_identity(request: Request) -> str:
    """Identify the caller for rate-limiting purposes."""
    key = request.headers.get(API_KEY_HEADER)
    if key:
        # Never use the raw key as a dict key that might be logged -- a short
        # prefix is enough to separate callers without storing the secret.
        return f"key:{key[:8]}"
    client = request.client
    return f"ip:{client.host}" if client else "ip:unknown"


# Applied to every response. Individually small, collectively they close the
# common browser-side attack surface for an API that may be called from a page.
SECURITY_HEADERS = {
    # Stop browsers guessing a different content type than we declared.
    "X-Content-Type-Options": "nosniff",
    # This API is never meant to be framed.
    "X-Frame-Options": "DENY",
    # Don't leak the calling URL to third parties.
    "Referrer-Policy": "no-referrer",
    # Prediction responses are per-request; never let a proxy cache them.
    "Cache-Control": "no-store",
    # An API returns JSON, not scripts or frames.
    "Content-Security-Policy": "default-src 'none'; frame-ancestors 'none'",
    # Browsers should not grant this origin any powerful features.
    "Permissions-Policy": "geolocation=(), microphone=(), camera=()",
}

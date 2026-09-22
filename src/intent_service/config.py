"""Typed, fail-fast configuration. One place to read every environment
variable -- never scatter `os.environ["X"]` across the codebase.

Security posture: the defaults are the SAFE ones. `require_api_key` is True
and `cors_origins` is empty out of the box, so a careless deployment is
locked down rather than wide open. Opening either is a deliberate,
visible act.
"""

from pathlib import Path

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="INTENT_",
        env_file=".env",
        extra="ignore",
        protected_namespaces=(),
    )

    # --- model -------------------------------------------------------------
    model_path: Path = Path("models/intent_model.joblib")

    # --- decision policy ----------------------------------------------------
    # The automation boundary. Operations tunes these without a code change;
    # the domain validates their relative order (see PolicyThresholds).
    auto_route_floor: float = Field(default=0.60, ge=0.0, le=1.0)
    reject_floor: float = Field(default=0.25, ge=0.0, le=1.0)

    # --- decision cache (extension) -----------------------------------------
    # Empty means "no Redis": the service falls back to a bounded in-process
    # cache. Configuring a URL is opt-in, so the default deployment has no
    # external dependency at all.
    redis_url: str = ""
    cache_ttl_seconds: int = Field(default=300, ge=1)
    # Short by design: a hung cache must never stall a prediction. See
    # RedisDecisionCache.connect.
    cache_timeout_seconds: float = Field(default=0.25, gt=0)

    # --- service -----------------------------------------------------------
    log_level: str = "INFO"
    api_title: str = "Intent Classification Service"
    # Secure by default: interactive docs enumerate every endpoint and
    # schema, so they are opt-in rather than opt-out.
    enable_docs: bool = False

    # --- authentication ----------------------------------------------------
    # Comma-separated list, e.g. INTENT_API_KEYS="key-one,key-two".
    api_keys: str = ""
    require_api_key: bool = True

    # --- rate limiting -----------------------------------------------------
    rate_limit_requests: int = Field(default=60, ge=1)
    rate_limit_window_seconds: float = Field(default=60.0, gt=0)

    # --- transport hardening ------------------------------------------------
    # Empty means "deny all cross-origin requests" -- the safe default for an
    # API with no known browser client.
    cors_origins: str = ""
    max_request_bytes: int = Field(default=16_384, ge=1)
    # Enable ONLY when a trusted reverse proxy overwrites
    # X-Forwarded-For. The header is caller-controlled otherwise.
    trust_proxy_headers: bool = False

    @property
    def api_key_list(self) -> list[str]:
        return [key.strip() for key in self.api_keys.split(",") if key.strip()]

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    @model_validator(mode="after")
    def _reject_inverted_thresholds(self) -> "Settings":
        """Catch an impossible policy at startup rather than at the first
        request. With reject_floor above auto_route_floor, the human_review
        band is empty and the service would silently never ask for a human
        -- a safety regression that no status code would reveal."""
        if self.reject_floor > self.auto_route_floor:
            raise ValueError(
                f"INTENT_REJECT_FLOOR ({self.reject_floor}) must not exceed "
                f"INTENT_AUTO_ROUTE_FLOOR ({self.auto_route_floor}); otherwise "
                "no message could ever be sent to human review."
            )
        return self

    @model_validator(mode="after")
    def _reject_auth_without_keys(self) -> "Settings":
        """Fail at startup, not at the first request.

        Booting with authentication demanded but no keys configured would
        make every call 401 -- an outage that looks like a code bug. Refusing
        to start names the real cause immediately.
        """
        if self.require_api_key and not self.api_key_list:
            raise ValueError(
                "INTENT_REQUIRE_API_KEY is true but INTENT_API_KEYS is empty. "
                "Set INTENT_API_KEYS, or set INTENT_REQUIRE_API_KEY=false for "
                "local development only."
            )
        return self

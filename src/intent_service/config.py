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
    confidence_floor: float = Field(default=0.15, ge=0.0, le=1.0)

    # --- service -----------------------------------------------------------
    log_level: str = "INFO"
    api_title: str = "Intent Classification Service"
    enable_docs: bool = True

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

    @property
    def api_key_list(self) -> list[str]:
        return [key.strip() for key in self.api_keys.split(",") if key.strip()]

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

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

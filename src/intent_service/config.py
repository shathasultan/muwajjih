"""Typed, fail-fast configuration. One place to read every environment
variable -- never scatter `os.environ["X"]` across the codebase.

Every value has a safe default, so `docker run` works out of the box, and
any of them can be overridden from the environment or a `.env` file
without editing a line of Python.
"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="INTENT_",
        env_file=".env",
        extra="ignore",
        protected_namespaces=(),
    )

    model_path: Path = Path("models/intent_model.joblib")
    log_level: str = "INFO"
    api_title: str = "Intent Classification Service"
    confidence_floor: float = 0.15  # below this, the API flags low-confidence

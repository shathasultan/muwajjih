"""Unit tests for configuration parsing and its fail-fast guardrail."""

import pytest
from pydantic import ValidationError

from intent_service.config import Settings


def test_api_keys_are_parsed_and_trimmed() -> None:
    settings = Settings(api_keys=" key-one , key-two ,, ")

    assert settings.api_key_list == ["key-one", "key-two"]


def test_cors_origins_are_parsed() -> None:
    settings = Settings(api_keys="k", cors_origins="https://a.com, https://b.com")

    assert settings.cors_origin_list == ["https://a.com", "https://b.com"]


def test_cors_denies_all_origins_by_default() -> None:
    assert Settings(api_keys="k").cors_origin_list == []


def test_auth_is_required_by_default() -> None:
    assert Settings(api_keys="k").require_api_key is True


def test_requiring_auth_without_keys_fails_at_startup() -> None:
    """Secure-by-default must not mean 'boots and 401s everything'."""
    with pytest.raises(ValidationError, match="INTENT_API_KEYS is empty"):
        Settings(require_api_key=True, api_keys="")


def test_auth_can_be_disabled_explicitly_for_local_dev() -> None:
    assert Settings(require_api_key=False, api_keys="").require_api_key is False

"""Tests for dtst.user_agent."""

from __future__ import annotations

import pytest

from dtst.user_agent import DEFAULT_USER_AGENT, get_user_agent


def test_default_when_env_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DTST_USER_AGENT", raising=False)
    assert get_user_agent() == DEFAULT_USER_AGENT


def test_custom_env_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DTST_USER_AGENT", "custom/1.0")
    assert get_user_agent() == "custom/1.0"


def test_empty_string_env_returns_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DTST_USER_AGENT", "")
    assert get_user_agent() == ""


def test_default_user_agent_shape() -> None:
    assert isinstance(DEFAULT_USER_AGENT, str)
    assert DEFAULT_USER_AGENT.startswith("dtst/")


def test_uses_clean_env_fixture(clean_dtst_env) -> None:
    # With the fixture applied, the default is returned.
    assert get_user_agent() == DEFAULT_USER_AGENT

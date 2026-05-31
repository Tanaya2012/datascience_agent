"""
Tests for configs/model_config.py — model-agnostic model resolution.

These target the pure decision function `_resolve_spec`, which needs no
google-adk install. The LiteLLM construction path is covered separately and
tolerates ADK being absent.
"""

from __future__ import annotations

import pytest

from datascience_agent.configs.model_config import (
    DEFAULT_MODEL,
    _resolve_spec,
    resolve_model,
)


@pytest.fixture(autouse=True)
def _clear_model_env(monkeypatch):
    """Ensure env vars don't leak between tests."""
    monkeypatch.delenv("AGENT_MODEL", raising=False)
    monkeypatch.delenv("LLM_PROVIDER", raising=False)


class TestResolveSpec:
    def test_default_when_nothing_set(self):
        assert _resolve_spec() == ("gemini", DEFAULT_MODEL)

    def test_explicit_gemini_arg(self):
        assert _resolve_spec(model="gemini-2.0-flash") == ("gemini", "gemini-2.0-flash")

    def test_agent_model_env(self, monkeypatch):
        monkeypatch.setenv("AGENT_MODEL", "gemini-1.5-pro")
        assert _resolve_spec() == ("gemini", "gemini-1.5-pro")

    def test_explicit_arg_overrides_env(self, monkeypatch):
        monkeypatch.setenv("AGENT_MODEL", "gemini-1.5-pro")
        assert _resolve_spec(model="gemini-2.0-flash") == ("gemini", "gemini-2.0-flash")

    def test_anthropic_provider_prefixes_model(self):
        assert _resolve_spec(model="claude-sonnet-4-6", provider="anthropic") == (
            "litellm",
            "anthropic/claude-sonnet-4-6",
        )

    def test_provider_via_env(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        monkeypatch.setenv("AGENT_MODEL", "gpt-4o")
        assert _resolve_spec() == ("litellm", "openai/gpt-4o")

    def test_already_prefixed_model_not_double_prefixed(self):
        assert _resolve_spec(model="openai/gpt-4o", provider="openai") == (
            "litellm",
            "openai/gpt-4o",
        )

    def test_prefixed_nongemini_model_without_provider_is_litellm(self):
        # A "provider/model" string implies LiteLLM even if provider hint is blank.
        assert _resolve_spec(model="anthropic/claude-sonnet-4-6") == (
            "litellm",
            "anthropic/claude-sonnet-4-6",
        )

    def test_gemini_prefixed_model_stays_gemini(self):
        assert _resolve_spec(model="gemini-2.5-pro") == ("gemini", "gemini-2.5-pro")

    @pytest.mark.parametrize("prov", ["", "google", "gemini", "vertex", "GOOGLE"])
    def test_gemini_provider_aliases(self, prov):
        kind, _ = _resolve_spec(model="gemini-2.0-flash", provider=prov)
        assert kind == "gemini"


class TestResolveModel:
    def test_gemini_returns_plain_string(self):
        assert resolve_model(model="gemini-2.0-flash") == "gemini-2.0-flash"

    def test_litellm_path_builds_or_raises_cleanly(self):
        # Without provider extras installed this raises a clear RuntimeError rather
        # than an opaque ImportError; with them, it returns a LiteLlm instance.
        try:
            result = resolve_model(model="claude-sonnet-4-6", provider="anthropic")
        except RuntimeError as exc:
            assert "LiteLLM" in str(exc)
        else:
            assert result.__class__.__name__ == "LiteLlm"

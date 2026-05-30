"""
Model-agnostic resolution of the LLM used by the agent (and, later, specialists).

The orchestrator must not be hard-coupled to Gemini. ADK accepts either a bare
Gemini model-name string (e.g. ``"gemini-2.5-pro"``) or a ``LiteLlm`` instance
wrapping any LiteLLM-supported provider (``"anthropic/claude-..."``,
``"openai/gpt-..."``, ``"ollama/..."``). This module decides which to build from
explicit arguments or environment variables.

Resolution order (highest priority first): explicit argument → environment → default.

Environment variables:
  - ``AGENT_MODEL``   model name (may carry a ``provider/model`` prefix)
  - ``LLM_PROVIDER``  provider hint: google/gemini → bare string; anything else → LiteLLM

Design note: the *decision* (`_resolve_spec`) is pure and import-free so it is
testable without google-adk installed; only `resolve_model` imports ADK, and only
on the LiteLLM path.
"""

from __future__ import annotations

import os
from typing import Literal

# A strong, capable default. Override via AGENT_MODEL / LLM_PROVIDER.
DEFAULT_MODEL = "gemini-2.5-pro"

# Provider hints that mean "use a plain Gemini model-name string" (ADK native path).
_GEMINI_PROVIDERS = {"", "google", "gemini", "google-genai", "googleai", "vertex", "vertexai"}

ModelKind = Literal["gemini", "litellm"]


def _resolve_spec(
    model: str | None = None,
    provider: str | None = None,
) -> tuple[ModelKind, str]:
    """
    Decide the model kind and concrete model string (pure, no ADK import).

    Returns ``("gemini", "gemini-2.5-pro")`` or ``("litellm", "anthropic/claude-...")``.

    Rules:
      - provider in _GEMINI_PROVIDERS → Gemini, UNLESS the model string carries a
        non-gemini ``provider/model`` prefix (then it's clearly a LiteLLM spec).
      - any other provider → LiteLLM; prefix the model with ``provider/`` if it
        isn't already prefixed.
    """
    provider = (provider if provider is not None else os.environ.get("LLM_PROVIDER", "")).strip().lower()
    model = (model or os.environ.get("AGENT_MODEL") or DEFAULT_MODEL).strip()

    if provider in _GEMINI_PROVIDERS:
        # A prefixed, non-gemini model string overrides a missing/gemini provider.
        if "/" in model and not model.startswith("gemini"):
            return "litellm", model
        return "gemini", model

    # Explicit non-gemini provider → route through LiteLLM.
    litellm_model = model if "/" in model else f"{provider}/{model}"
    return "litellm", litellm_model


def resolve_model(model: str | None = None, provider: str | None = None):
    """
    Return a model spec ADK accepts: a Gemini model-name ``str`` or a ``LiteLlm``.

    Args:
        model: explicit model name (overrides ``AGENT_MODEL``).
        provider: explicit provider hint (overrides ``LLM_PROVIDER``).

    The LiteLLM path imports ``google.adk`` lazily so the Gemini path (and tests)
    work without provider extras installed.
    """
    kind, model_str = _resolve_spec(model, provider)
    if kind == "gemini":
        return model_str
    return _build_litellm(model_str)


def _build_litellm(model_str: str):
    try:
        from google.adk.models.lite_llm import LiteLlm  # type: ignore[import]
    except Exception as exc:  # pragma: no cover - depends on optional install
        raise RuntimeError(
            f"Model '{model_str}' requires LiteLLM, but google.adk's LiteLlm could not be "
            f"imported ({exc}). Install litellm and the provider SDK, and set the provider's "
            "API key (e.g. ANTHROPIC_API_KEY / OPENAI_API_KEY)."
        ) from exc
    return LiteLlm(model=model_str)

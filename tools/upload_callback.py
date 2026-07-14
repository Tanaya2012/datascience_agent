"""
Upload auto-ingest callback (D20).

An ``adk web`` upload arrives as an inline data ``Part`` on the *user* message.
``ingest_uploaded_file`` lives on the ``data_steward`` specialist, but the inline
Part does **not** survive the orchestrator→specialist ``AgentTool`` delegation — the
specialist's sub-runner sees the orchestrator's delegation text, not the original
human message — so the tool never sees a web-UI upload in the multi-agent topology
(confirmed 0/3 through the orchestrator vs. 3/3 single-agent; see DECISIONS.md D20).

This ``before_agent_callback`` runs on the **orchestrator**, which *does* receive the
original user message, and materializes any inline upload into a versioned dataset
artifact + ``current_dataset_key`` **before routing**. Downstream specialists then
pick it up via shared session state (which crosses the boundary). Deterministic — no
reliance on the LLM choosing to call a tool.
"""

from __future__ import annotations

from typing import Optional

from .ingestion import _ingest_and_register


def _has_inline_upload(callback_context) -> bool:
    """True if this turn's user message carries an inline (uploaded) data Part."""
    user_content = getattr(callback_context, "user_content", None)
    parts = getattr(user_content, "parts", None) or []
    return any(getattr(p, "inline_data", None) and p.inline_data.data for p in parts)


async def ingest_upload_callback(callback_context) -> Optional[object]:
    """Auto-ingest an inline-uploaded file on the orchestrator's turn, before routing.

    No-op when the turn carries no upload (cheap check; runs every orchestrator turn).
    On a parse/format failure, leaves the dataset unloaded rather than crashing the
    turn — the agent then asks the user for a valid file. Always returns ``None`` so
    the agent proceeds normally, now with the upload loaded as the current dataset.
    """
    if not _has_inline_upload(callback_context):
        return None
    try:
        await _ingest_and_register(
            callback_context, filename=None, mime_type=None,
            is_secondary=False, secondary_name=None,
        )
    except Exception:
        # Malformed/unsupported upload — don't crash the turn; the agent surfaces it.
        pass
    return None

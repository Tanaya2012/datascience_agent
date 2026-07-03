"""
Gated MCP stdio-connector factory (generalized in D16).

The reusable pattern: register a stdio ``McpToolset`` **only when its launcher is on
PATH and an optional credential/availability gate passes** — so a dead server never
spams per-turn ADK errors. Originally Kaggle-specific (`kaggle-mcp` via `uvx`); Kaggle
moved off MCP to the `kaggle` library (D16, `tools/kaggle_tool.py`), so this is now a
generic factory for *future* stdio MCP servers.

Fit reminder (ARCHITECTURE "MCP connectors — selection criterion"): MCP suits the
**control plane** (metadata/search/query results, remote side effects) — SQL servers,
web-fetch. Bulk local-file delivery (data plane) stays in-process tools + artifacts.
"""

from __future__ import annotations

import logging
import shutil
from typing import Callable, Optional, Sequence

logger = logging.getLogger(__name__)


def launcher_available(command: str) -> bool:
    """True if ``command`` (e.g. ``uvx``) is on PATH."""
    return shutil.which(command) is not None


def uvx_available() -> bool:
    """True if the ``uvx`` launcher (from ``uv``) is on PATH."""
    return launcher_available("uvx")


def maybe_stdio_toolset(
    command: str,
    args: Sequence[str],
    tool_filter: Sequence[str],
    gate: Optional[Callable[[], bool]] = None,
    name: Optional[str] = None,
) -> Optional[object]:
    """
    Build a stdio ``McpToolset`` if the launcher is present and ``gate()`` passes,
    else ``None``.

    Returning ``None`` (instead of raising) lets callers append conditionally:
    ``ts = maybe_stdio_toolset(...);  tools += [ts] if ts else []``.

    Args:
        command: launcher on PATH (e.g. "uvx", "npx").
        args: launcher args (e.g. ["some-mcp-server"]).
        tool_filter: MCP tool names to expose.
        gate: optional predicate (e.g. credentials-present); skipped if it returns False.
        name: label for log messages.
    """
    label = name or " ".join([command, *args])
    if not launcher_available(command):
        logger.info("Launcher '%s' not on PATH — skipping MCP toolset '%s'.", command, label)
        return None
    if gate is not None and not gate():
        logger.info("Gate failed — skipping MCP toolset '%s'.", label)
        return None

    # Imported lazily so the package imports cleanly even if ADK's mcp extra shifts.
    from google.adk.tools.mcp_tool import McpToolset, StdioConnectionParams  # type: ignore[import]
    from mcp import StdioServerParameters  # type: ignore[import]

    return McpToolset(
        connection_params=StdioConnectionParams(
            server_params=StdioServerParameters(command=command, args=list(args)),
        ),
        tool_filter=list(tool_filter),
    )

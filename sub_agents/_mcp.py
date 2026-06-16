"""
MCP connector helpers for specialists (M2b).

The Kaggle MCP server runs via ``uvx kaggle-mcp``. ``uvx`` is not always present
(it ships with ``uv``), and registering an MCP toolset whose server can't start
makes ADK log an error on every turn. So we register the toolset **only when
``uvx`` is on PATH** — otherwise the agent stays clean and fully functional
without Kaggle. This is the reusable pattern for future MCP connectors.

Auth: the Kaggle MCP reads ``~/.kaggle/kaggle.json`` (or KAGGLE_USERNAME /
KAGGLE_KEY from the inherited environment); we set no explicit env override so the
subprocess inherits the parent environment, including PATH.
"""

from __future__ import annotations

import logging
import shutil
from typing import Optional

logger = logging.getLogger(__name__)

KAGGLE_TOOLS = ["search_kaggle_datasets", "download_kaggle_dataset"]


def uvx_available() -> bool:
    """True if the ``uvx`` launcher (from ``uv``) is on PATH."""
    return shutil.which("uvx") is not None


def maybe_kaggle_toolset() -> Optional[object]:
    """
    Build the Kaggle ``McpToolset`` if ``uvx`` is available, else ``None``.

    Returning ``None`` (instead of raising) lets callers append the toolset
    conditionally: ``tools = [...]; ts = maybe_kaggle_toolset();  tools += [ts] if ts``.
    """
    if not uvx_available():
        logger.info(
            "uvx not found on PATH — skipping Kaggle MCP toolset. Install `uv` "
            "(https://astral.sh/uv) and set Kaggle credentials to enable it."
        )
        return None

    # Imported lazily so the package imports cleanly even if ADK's mcp extra shifts.
    from google.adk.tools.mcp_tool import McpToolset, StdioConnectionParams  # type: ignore[import]
    from mcp import StdioServerParameters  # type: ignore[import]

    return McpToolset(
        connection_params=StdioConnectionParams(
            server_params=StdioServerParameters(
                command="uvx",
                args=["kaggle-mcp"],
            ),
        ),
        tool_filter=list(KAGGLE_TOOLS),
    )

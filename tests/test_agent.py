"""
Tests for the root agent configuration, focusing on the Kaggle MCP toolset.

These tests verify that the MCPToolset is correctly wired up with the right
tool filter and subprocess command — without starting an actual MCP server.
"""

from __future__ import annotations

import pytest
from google.adk.tools.mcp_tool import MCPToolset


class TestKaggleMcpToolset:
    """Verify _kaggle_mcp is configured correctly."""

    def test_is_mcp_toolset_instance(self):
        from datascience_agent.agent import _kaggle_mcp

        assert isinstance(_kaggle_mcp, MCPToolset)

    def test_tool_filter_contains_search_and_download(self):
        from datascience_agent.agent import _kaggle_mcp

        assert "search_kaggle_datasets" in _kaggle_mcp.tool_filter
        assert "download_kaggle_dataset" in _kaggle_mcp.tool_filter

    def test_tool_filter_no_extra_tools(self):
        from datascience_agent.agent import _kaggle_mcp

        assert len(_kaggle_mcp.tool_filter) == 2

    def test_mcp_command_is_uvx(self):
        from datascience_agent.agent import _kaggle_mcp

        server_params = _kaggle_mcp._connection_params.server_params
        assert server_params.command == "uvx"

    def test_mcp_args_include_kaggle_mcp(self):
        from datascience_agent.agent import _kaggle_mcp

        server_params = _kaggle_mcp._connection_params.server_params
        assert "kaggle-mcp" in server_params.args

    def test_no_explicit_env_set(self):
        # Auth is handled via ~/.kaggle/kaggle.json; no env override should be set
        # so the subprocess inherits the full parent environment (including PATH).
        from datascience_agent.agent import _kaggle_mcp

        assert _kaggle_mcp._connection_params.server_params.env is None


class TestAgentToolRegistration:
    """Verify root_agent has the Kaggle MCPToolset in its tools list."""

    def test_kaggle_mcp_in_agent_tools(self):
        from datascience_agent.agent import root_agent, _kaggle_mcp

        assert _kaggle_mcp in root_agent.tools

    def test_agent_has_eight_python_tools(self):
        from datascience_agent.agent import root_agent
        from google.adk.tools.mcp_tool import MCPToolset

        python_tools = [t for t in root_agent.tools if not isinstance(t, MCPToolset)]
        assert len(python_tools) == 8

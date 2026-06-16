"""
Tests for the conditional Kaggle MCP toolset (M2b).

The toolset must register only when ``uvx`` is on PATH, and the Data Steward must
build cleanly either way. We monkeypatch ``shutil.which`` so the tests are
deterministic regardless of whether ``uv`` is installed on the dev box.
"""

from __future__ import annotations

import pytest

from datascience_agent.sub_agents import _mcp
from datascience_agent.sub_agents.data_steward import build_data_steward


def _force_uvx(monkeypatch, present: bool):
    monkeypatch.setattr(
        _mcp.shutil, "which", lambda cmd: "/usr/bin/uvx" if present else None
    )


class TestMaybeKaggleToolset:
    def test_returns_none_when_uvx_absent(self, monkeypatch):
        _force_uvx(monkeypatch, present=False)
        assert _mcp.uvx_available() is False
        assert _mcp.maybe_kaggle_toolset() is None

    def test_builds_toolset_when_uvx_present(self, monkeypatch):
        _force_uvx(monkeypatch, present=True)
        ts = _mcp.maybe_kaggle_toolset()
        from google.adk.tools.mcp_tool import McpToolset

        assert isinstance(ts, McpToolset)

    def test_toolset_filter_and_command(self, monkeypatch):
        _force_uvx(monkeypatch, present=True)
        ts = _mcp.maybe_kaggle_toolset()
        assert set(ts.tool_filter) == {"search_kaggle_datasets", "download_kaggle_dataset"}
        server_params = ts._connection_params.server_params
        assert server_params.command == "uvx"
        assert "kaggle-mcp" in server_params.args
        # No explicit env override → subprocess inherits parent env (PATH, creds).
        assert server_params.env is None


class TestDataStewardConditionalWiring:
    def test_steward_has_no_mcp_toolset_when_uvx_absent(self, monkeypatch):
        _force_uvx(monkeypatch, present=False)
        steward = build_data_steward()
        from google.adk.tools.mcp_tool import McpToolset

        assert not any(isinstance(t, McpToolset) for t in steward.tools)
        # Function tools always present.
        fn_tools = {t.__name__ for t in steward.tools if hasattr(t, "__name__")}
        assert fn_tools == {"dataset_loader", "ingest_uploaded_file", "profile_dataset"}

    def test_steward_includes_mcp_toolset_when_uvx_present(self, monkeypatch):
        _force_uvx(monkeypatch, present=True)
        steward = build_data_steward()
        from google.adk.tools.mcp_tool import McpToolset

        assert any(isinstance(t, McpToolset) for t in steward.tools)

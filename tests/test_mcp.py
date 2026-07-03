"""
Tests for the generic gated MCP stdio-connector factory (`sub_agents/_mcp.py`).

Generalized in D16 (Kaggle moved off MCP). We monkeypatch `shutil.which` so tests
are deterministic regardless of what launchers are installed on the dev box.
"""

from __future__ import annotations

from datascience_agent.sub_agents import _mcp


def _force_launcher(monkeypatch, present: bool):
    monkeypatch.setattr(_mcp.shutil, "which", lambda cmd: f"/usr/bin/{cmd}" if present else None)


class TestMaybeStdioToolset:
    def test_returns_none_when_launcher_absent(self, monkeypatch):
        _force_launcher(monkeypatch, present=False)
        assert _mcp.uvx_available() is False
        ts = _mcp.maybe_stdio_toolset("uvx", ["some-mcp"], ["a", "b"])
        assert ts is None

    def test_returns_none_when_gate_fails(self, monkeypatch):
        _force_launcher(monkeypatch, present=True)
        ts = _mcp.maybe_stdio_toolset("uvx", ["some-mcp"], ["a"], gate=lambda: False)
        assert ts is None

    def test_builds_toolset_when_launcher_and_gate_ok(self, monkeypatch):
        _force_launcher(monkeypatch, present=True)
        ts = _mcp.maybe_stdio_toolset("uvx", ["some-mcp"], ["tool_a", "tool_b"], gate=lambda: True)
        from google.adk.tools.mcp_tool import McpToolset

        assert isinstance(ts, McpToolset)
        assert set(ts.tool_filter) == {"tool_a", "tool_b"}
        server_params = ts._connection_params.server_params
        assert server_params.command == "uvx"
        assert server_params.args == ["some-mcp"]

    def test_no_gate_means_launcher_only(self, monkeypatch):
        _force_launcher(monkeypatch, present=True)
        ts = _mcp.maybe_stdio_toolset("npx", ["x"], ["t"])  # gate=None
        assert ts is not None

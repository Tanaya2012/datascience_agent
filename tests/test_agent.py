"""
Tests for the orchestrator agent topology (M2).

The root agent is now a coordinator that delegates to four specialist sub-agents
via ``AgentTool`` — it holds no data tools directly. These tests verify the
topology without starting any LLM or MCP server.
"""

from __future__ import annotations

from google.adk.agents import LlmAgent
from google.adk.tools.agent_tool import AgentTool


def _function_tool_names(agent) -> set:
    # Only plain function tools; toolsets (optional Kaggle McpToolset) are excluded.
    return {t.__name__ for t in agent.tools if hasattr(t, "__name__")}


class TestOrchestratorTopology:
    """root_agent wraps exactly the four specialists as AgentTools."""

    def test_root_agent_name_unchanged(self):
        from datascience_agent.agent import root_agent

        # Stable name keeps `adk run/web datascience_agent` discovery working.
        assert root_agent.name == "data_science_agent"

    def test_root_tools_are_all_agent_tools(self):
        from datascience_agent.agent import root_agent

        assert len(root_agent.tools) == 5
        assert all(isinstance(t, AgentTool) for t in root_agent.tools)

    def test_root_delegates_to_the_specialists(self):
        from datascience_agent.agent import root_agent

        names = {t.agent.name for t in root_agent.tools}
        assert names == {
            "data_steward",
            "cleaning_specialist",
            "analysis_specialist",
            "feature_engineering_specialist",
            "reporting_specialist",
        }

    def test_root_holds_no_raw_data_tools(self):
        from datascience_agent.agent import root_agent

        # The orchestrator routes; it must not carry data tools itself.
        assert not any(callable(t) and not isinstance(t, AgentTool) for t in root_agent.tools)

    def test_root_has_nonempty_instruction(self):
        from datascience_agent.agent import root_agent

        assert isinstance(root_agent.instruction, str) and root_agent.instruction.strip()


class TestSpecialistTooling:
    """Each specialist owns its expected toolset (verified via the orchestrator)."""

    def _specialist(self, name) -> LlmAgent:
        from datascience_agent.agent import root_agent

        return next(t.agent for t in root_agent.tools if t.agent.name == name)

    def test_data_steward_tools(self):
        # Function tools are fixed; the Kaggle McpToolset is optional (uvx-gated).
        tools = _function_tool_names(self._specialist("data_steward"))
        assert tools == {"dataset_loader", "ingest_uploaded_file", "profile_dataset"}

    def test_cleaning_specialist_tools(self):
        tools = _function_tool_names(self._specialist("cleaning_specialist"))
        assert tools == {
            "handle_missing_values",
            "standardize_formats",
            "deduplicate_dataset",
            "merge_datasets",
            "validate_dataset",
        }

    def test_analysis_specialist_tools(self):
        tools = _function_tool_names(self._specialist("analysis_specialist"))
        assert tools == {"profile_dataset", "explore_dataset", "plot_dataset", "run_python"}

    def test_feature_engineering_specialist_tools(self):
        tools = _function_tool_names(self._specialist("feature_engineering_specialist"))
        assert tools == {
            "encode_features",
            "scale_features",
            "bin_columns",
            "engineer_datetime_features",
            "run_python",
        }

    def test_reporting_specialist_tools(self):
        tools = _function_tool_names(self._specialist("reporting_specialist"))
        assert tools == {"generate_output"}

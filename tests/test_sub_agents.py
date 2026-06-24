"""
Tests for the specialist builders in ``sub_agents/`` (M2).

Verify each builder returns a configured ``LlmAgent`` with the right name,
toolset, a non-empty focused instruction, and a resolved model — without
contacting any LLM.
"""

from __future__ import annotations

import pytest
from google.adk.agents import LlmAgent

from datascience_agent.sub_agents import (
    build_analysis_specialist,
    build_cleaning_specialist,
    build_data_steward,
    build_reporting_specialist,
)


def _tool_names(agent) -> set[str]:
    # Only the plain function tools — toolsets (e.g. the optional Kaggle McpToolset,
    # present only when uvx is installed) have no __name__ and are ignored here.
    return {t.__name__ for t in agent.tools if hasattr(t, "__name__")}


BUILDERS = {
    "data_steward": (
        build_data_steward,
        {"dataset_loader", "ingest_uploaded_file", "profile_dataset"},
    ),
    "cleaning_specialist": (
        build_cleaning_specialist,
        {
            "handle_missing_values",
            "standardize_formats",
            "deduplicate_dataset",
            "merge_datasets",
            "validate_dataset",
        },
    ),
    "analysis_specialist": (
        build_analysis_specialist,
        {"profile_dataset", "explore_dataset", "plot_dataset", "run_python"},
    ),
    "reporting_specialist": (build_reporting_specialist, {"generate_output"}),
}


@pytest.mark.parametrize("name,spec", BUILDERS.items())
def test_builder_returns_llm_agent_with_expected_name(name, spec):
    builder, _ = spec
    agent = builder()
    assert isinstance(agent, LlmAgent)
    assert agent.name == name


@pytest.mark.parametrize("name,spec", BUILDERS.items())
def test_builder_has_expected_tools(name, spec):
    builder, expected_tools = spec
    agent = builder()
    assert _tool_names(agent) == expected_tools


@pytest.mark.parametrize("name,spec", BUILDERS.items())
def test_builder_has_focused_instruction_and_description(name, spec):
    builder, _ = spec
    agent = builder()
    assert isinstance(agent.instruction, str) and agent.instruction.strip()
    assert isinstance(agent.description, str) and agent.description.strip()


def test_model_override_is_respected():
    # An explicit model string flows straight through to the agent.
    agent = build_data_steward(model="gemini-2.5-flash")
    assert agent.model == "gemini-2.5-flash"

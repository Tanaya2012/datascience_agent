"""
Guards on the orchestrator/specialist instructions.

1. ADK instruction-template collisions (M4.5, D19).

ADK's LlmAgent renders a string `instruction` through session-state templating: any
`{identifier}` is treated as a state variable and raises
`KeyError: Context variable not found: <name>` when that variable is absent — crashing
the turn whenever the agent is invoked. The live bug bash hit this via the
Feature-Engineering instruction's literal `{col}_binned` / `{col}_{feature}` examples.

This asserts no orchestrator/specialist instruction contains a single-brace `{token}`
(doubled `{{...}}` escapes are fine). Deterministic — no LLM needed.

2. Routing-blurb drift (D37). The orchestrator picks a specialist from its one-line
blurb in the instruction, never from the specialist's tool list. When the blurbs fell
two milestones behind the rosters (audit F9), the LLM was steered to `run_python` for
work a deterministic, audited tool already covered. Every deterministic tool must
therefore be named — by capability, not function name — in its specialist's blurb.
"""

from __future__ import annotations

import re

import pytest

from datascience_agent.agent import root_agent
from datascience_agent.sub_agents import (
    analysis_specialist,
    cleaning_specialist,
    data_steward,
    feature_engineering_specialist,
    modeling_specialist,
    reporting_specialist,
)

# A single-brace {token}/{token?} that ADK would try to resolve from state, not a
# doubled {{...}} escape.
_TEMPLATE_RE = re.compile(r"(?<!\{)\{([A-Za-z_][A-Za-z0-9_]*\??)\}(?!\})")


@pytest.mark.parametrize(
    "agent",
    [root_agent, data_steward, cleaning_specialist, analysis_specialist,
     feature_engineering_specialist, modeling_specialist, reporting_specialist],
    ids=lambda a: a.name,
)
def test_instruction_has_no_template_collisions(agent):
    instr = agent.instruction
    assert isinstance(instr, str), f"{agent.name}: instruction is not a plain string"
    hits = _TEMPLATE_RE.findall(instr)
    assert not hits, (
        f"{agent.name}: instruction contains {{...}} that ADK will interpolate as "
        f"session-state variables (crashes the turn): {hits}. Rephrase without braces "
        f"or double them to escape."
    )


# Deterministic tool -> phrase(s) its specialist's orchestrator blurb must contain
# (case-insensitive). `run_python` is the escape hatch, not a routing signal.
_BLURB_MUST_MENTION = {
    "data_steward": {
        "dataset_loader": ["local"],
        "ingest_uploaded_file": ["upload"],
        "search_kaggle": ["kaggle"],
        "download_kaggle": ["kaggle"],
        "profile_dataset": ["profile"],
    },
    "cleaning_specialist": {
        "handle_missing_values": ["missing"],
        "standardize_formats": ["standardiz"],
        "deduplicate_dataset": ["dedup"],
        "merge_datasets": ["merge"],
        "validate_dataset": ["quality score"],
    },
    "analysis_specialist": {
        "profile_dataset": ["eda"],
        "explore_dataset": ["correlation", "distribution", "target"],
        "plot_dataset": ["chart"],
        "statistical_test": ["statistical test"],
    },
    "feature_engineering_specialist": {
        "encode_features": ["encod"],
        "scale_features": ["scal"],
        "bin_columns": ["bin"],
        "engineer_datetime_features": ["date"],
    },
    "modeling_specialist": {
        "train_model": ["train"],
        "evaluate_model": ["evaluat"],
    },
    "reporting_specialist": {
        "generate_output": ["export"],
    },
}


def _blurbs(instruction: str) -> dict[str, str]:
    """Split the '## Your specialists' section into {agent_name: blurb text}."""
    section = instruction.split("## Your specialists", 1)[1].split("\n## ", 1)[0]
    parts = re.split(r"^- \*\*(\w+)\*\*", section, flags=re.M)
    return {name: text.lower() for name, text in zip(parts[1::2], parts[2::2])}


def test_every_specialist_has_a_routing_blurb():
    specialists = {t.agent.name for t in root_agent.tools}
    assert set(_blurbs(root_agent.instruction)) == specialists


@pytest.mark.parametrize("specialist", sorted(_BLURB_MUST_MENTION))
def test_routing_blurb_covers_every_deterministic_tool(specialist):
    agent = next(t.agent for t in root_agent.tools if t.agent.name == specialist)
    roster = {t.__name__ for t in agent.tools if hasattr(t, "__name__")} - {"run_python"}
    expected = _BLURB_MUST_MENTION[specialist]
    # A tool added to the roster without a phrase here fails, forcing a blurb update.
    assert roster == set(expected), (
        f"{specialist}: roster {sorted(roster)} != described tools {sorted(expected)}; "
        f"update agent.py's blurb and _BLURB_MUST_MENTION together."
    )
    blurb = _blurbs(root_agent.instruction)[specialist]
    missing = {tool: p for tool, phrases in expected.items() for p in phrases if p not in blurb}
    assert not missing, f"{specialist}'s orchestrator blurb never mentions: {missing}"

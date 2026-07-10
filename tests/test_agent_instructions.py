"""
Guard against ADK instruction-template collisions (M4.5, D19).

ADK's LlmAgent renders a string `instruction` through session-state templating: any
`{identifier}` is treated as a state variable and raises
`KeyError: Context variable not found: <name>` when that variable is absent — crashing
the turn whenever the agent is invoked. The live bug bash hit this via the
Feature-Engineering instruction's literal `{col}_binned` / `{col}_{feature}` examples.

This asserts no orchestrator/specialist instruction contains a single-brace `{token}`
(doubled `{{...}}` escapes are fine). Deterministic — no LLM needed.
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
    reporting_specialist,
)

# A single-brace {token}/{token?} that ADK would try to resolve from state, not a
# doubled {{...}} escape.
_TEMPLATE_RE = re.compile(r"(?<!\{)\{([A-Za-z_][A-Za-z0-9_]*\??)\}(?!\})")


@pytest.mark.parametrize(
    "agent",
    [root_agent, data_steward, cleaning_specialist, analysis_specialist,
     feature_engineering_specialist, reporting_specialist],
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

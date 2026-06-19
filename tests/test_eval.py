"""
ADK eval-framework tests (M2c).

Two layers:
  * ``test_routing_evalset_parses`` — always runs; validates the committed
    evalset deserializes into ADK's ``EvalSet`` schema (cheap structural guard).
  * ``test_routing_eval`` — gated behind ``RUN_LLM_EVALS=1`` + an API key; runs
    the orchestrator end-to-end via ``AgentEvaluator`` and checks the response
    matches the reference (ROUGE ≥ 0.3, see evals/test_config.json).

The committed evalset uses a ``{{FIXTURE_CSV}}`` token (portable across
machines); both tests substitute the absolute path to evals/fixtures/tiny.csv.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pytest

_PROJECT = Path(__file__).resolve().parent.parent
_EVALS = _PROJECT / "evals"
_EVALSET = _EVALS / "routing.evalset.json"
_FIXTURE = _EVALS / "fixtures" / "tiny.csv"


def _materialize_evalset(dest_dir: Path) -> Path:
    """Write a concrete evalset (token replaced) + its test_config into dest_dir."""
    raw = _EVALSET.read_text().replace("{{FIXTURE_CSV}}", str(_FIXTURE))
    out = dest_dir / "routing.evalset.json"
    out.write_text(raw)
    shutil.copy(_EVALS / "test_config.json", dest_dir / "test_config.json")
    return out


def test_routing_evalset_parses(tmp_path):
    from google.adk.evaluation.eval_set import EvalSet

    data = json.loads(_materialize_evalset(tmp_path).read_text())
    eval_set = EvalSet.model_validate(data)
    assert eval_set.eval_set_id == "routing"
    assert eval_set.eval_cases and eval_set.eval_cases[0].conversation
    assert _FIXTURE.exists()
    # Token was substituted with a concrete, existing path.
    user_text = eval_set.eval_cases[0].conversation[0].user_content.parts[0].text
    assert str(_FIXTURE) in user_text


def _llm_evals_enabled() -> bool:
    return os.environ.get("RUN_LLM_EVALS") == "1" and bool(
        os.environ.get("GOOGLE_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
    )


@pytest.mark.llm
@pytest.mark.skipif(not _llm_evals_enabled(), reason="set RUN_LLM_EVALS=1 + API key to run LLM evals")
@pytest.mark.asyncio
async def test_routing_eval(tmp_path):
    from google.adk.evaluation.agent_evaluator import AgentEvaluator

    evalset = _materialize_evalset(tmp_path)
    # agent_module is the importable package exposing root_agent.
    await AgentEvaluator.evaluate(
        agent_module="datascience_agent",
        eval_dataset_file_path_or_dir=str(evalset),
        num_runs=1,
    )

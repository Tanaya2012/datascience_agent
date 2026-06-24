"""
Cross-specialist regression test (guards against the D13 class of bug).

In the multi-agent topology each specialist is a separate LLM context, so a tool
invoked by one specialist only knows the natural-language request — NOT the
physical artifact key produced by a tool an earlier specialist ran. The contract
that makes this work: every dataset-consuming tool defaults
`dataset_artifact_key` to `state.current_dataset_key`.

This test exercises a realistic pipeline where each step is owned by a different
specialist and **no artifact key is ever threaded through** — each tool must pick
up the current dataset from shared session state. A single `mock_ctx` (shared
state dict + filesystem artifact fallback) faithfully models cross-`AgentTool`
state sharing, whose propagation is itself verified separately. Before D13 the
key-requiring tools would have failed here with "no dataset loaded".
"""

from __future__ import annotations

import pytest

from tools.artifact_utils import get_session_state
from tools.dataset_loader import dataset_loader
from tools.data_profiler import profile_dataset
from tools.eda import explore_dataset
from tools.visualization import plot_dataset
from tools.cleaning.missing_handler import handle_missing_values
from tools.validator import validate_dataset
from tools.output_generator import generate_output


def _current_key(mock_ctx) -> str | None:
    return get_session_state(mock_ctx).current_dataset_key


@pytest.mark.asyncio
async def test_keyless_pipeline_across_specialists(mock_ctx, csv_file, tmp_path):
    # 1. Data Steward — load (the ONLY call that supplies an identifier).
    load = await dataset_loader("local", str(csv_file), tool_context=mock_ctx)
    assert load["success"] is True
    loaded_key = _current_key(mock_ctx)
    assert loaded_key, "load must set current_dataset_key"

    # 2. Data Steward — profile, no key passed.
    prof = await profile_dataset(tool_context=mock_ctx)
    assert prof["success"] is True

    # 3. Analysis — EDA, no key passed.
    eda = await explore_dataset(tool_context=mock_ctx)
    assert eda["success"] is True
    assert eda["output_artifact_key"] == loaded_key  # read-only: still on the loaded version

    # 4. Cleaning — impute missing values, no key passed; this MUTATES → new version.
    fill = await handle_missing_values(
        strategy_config={"Age": "median", "Salary": "mean"}, tool_context=mock_ctx
    )
    assert fill["success"] is True
    cleaned_key = _current_key(mock_ctx)
    assert cleaned_key and cleaned_key != loaded_key, "cleaning must advance current_dataset_key"

    # 5. Analysis — plot the cleaned data, no key passed (uses the NEW current version).
    plot = await plot_dataset(chart_kind="histogram", x="Age", tool_context=mock_ctx)
    assert plot["success"] is True
    assert plot["plot_artifact_key"].endswith(".png")

    # 6. Cleaning — validate, no key passed.
    val = await validate_dataset(tool_context=mock_ctx)
    assert val["success"] is True
    assert val["quality_score"] is not None

    # 7. Reporting — export, no key passed.
    out = await generate_output(output_dir=str(tmp_path), tool_context=mock_ctx)
    assert out["success"] is True
    assert out["csv_path"]


@pytest.mark.asyncio
async def test_keyless_tool_errors_cleanly_when_nothing_loaded(mock_ctx):
    # The flip side of the contract: with no dataset loaded and no key, the tool
    # must fail gracefully (not crash) — what a specialist sees if it runs early.
    for result in (
        await explore_dataset(tool_context=mock_ctx),
        await validate_dataset(tool_context=mock_ctx),
        await plot_dataset(chart_kind="histogram", x="x", tool_context=mock_ctx),
    ):
        assert result["success"] is False
        assert "No dataset loaded" in result["error_message"]

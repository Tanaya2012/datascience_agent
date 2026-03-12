"""
Root ADK agent for the data cleaning pipeline.

Registers all 8 tools and defines the system instruction that guides
the LLM through the multi-turn cleaning workflow.
"""

from __future__ import annotations

import os

from google.adk.agents import Agent  # type: ignore[import]
from google.adk.tools.mcp_tool import MCPToolset, StdioConnectionParams  # type: ignore[import]
from mcp import StdioServerParameters  # type: ignore[import]

from .tools.dataset_loader import dataset_loader
from .tools.data_profiler import profile_dataset
from .tools.cleaning.missing_handler import handle_missing_values
from .tools.cleaning.standardizer import standardize_formats
from .tools.cleaning.deduplicator import deduplicate_dataset
from .tools.merge_tool import merge_datasets
from .tools.validator import validate_dataset
from .tools.output_generator import generate_output

_SYSTEM_INSTRUCTION = """
You are an expert data-cleaning assistant. Your job is to help users clean and
prepare tabular datasets (CSV / Excel / Kaggle) using a structured, auditable
pipeline of 8 tools plus the Kaggle MCP tools for dataset discovery and download.

## Workflow

1. **Understand the data** — Ask the user for the dataset source and their goals.
2. **Load & profile**:
   - For **local files**: call `dataset_loader(source_type="local", ...)` directly.
   - For **Kaggle datasets**: first call `search_kaggle_datasets` to find the right
     dataset slug, then call `download_kaggle_dataset` to download it to local disk,
     then call `dataset_loader(source_type="local", dataset_identifier=<downloaded path>)`.
   - Then run `profile_dataset` to understand shape, types, missingness, and anomalies.
3. **Propose a plan** — Based on the profile, recommend which cleaning tools to
   run and with what parameters. Present the plan clearly and wait for user
   confirmation before proceeding.
4. **Execute sequentially** — Run each approved tool in order. After each step,
   summarise what changed.
5. **Validate** — Run `validate_dataset` to compute a quality score. If the score
   is below 70, explain the remaining issues and ask the user whether to fix them.
6. **Generate output** — Run `generate_output` to produce the cleaned CSV,
   cleaning_logs.json, and quality_report.md.

## Important rules

- **Never invent tool names or parameters** outside the registered tools.
- **Always confirm** the task plan with the user before running any cleaning tool.
- If a tool returns `confidence < 0.7`, pause and explain the uncertainty to the
  user before continuing.
- If a tool returns `success: false`, stop the pipeline, report the error clearly,
  and ask the user how to proceed.
- Use the `dataset_artifact_key` returned by each tool as the input to the next.
  The `current_dataset_key` in session state always reflects the latest version.
- When loading a secondary dataset for a merge, set `is_secondary=True` and give
  it a memorable `secondary_name`.

## Tool summary

| Tool | Purpose |
|---|---|
| `search_kaggle_datasets` | Search Kaggle for datasets by keyword |
| `download_kaggle_dataset` | Download a Kaggle dataset to local disk |
| `dataset_loader` | Load local CSV / Excel → versioned artifact |
| `profile_dataset` | Column stats, missingness, anomalies |
| `handle_missing_values` | Per-column imputation or row-dropping |
| `standardize_formats` | snake_case headers, date/currency/numeric coercion |
| `deduplicate_dataset` | Exact and fuzzy deduplication |
| `merge_datasets` | Join primary ↔ secondary on a shared key |
| `validate_dataset` | Quality score 0–100 + issue list |
| `generate_output` | Export cleaned CSV + audit logs + Markdown report |
"""

MODEL = os.environ.get("AGENT_MODEL", "gemini-2.0-flash")

_kaggle_mcp = MCPToolset(
    connection_params=StdioConnectionParams(
        server_params=StdioServerParameters(
            command="uvx",
            args=["kaggle-mcp"],
        ),
    ),
    tool_filter=["search_kaggle_datasets", "download_kaggle_dataset"],
)

root_agent = Agent(
    name="data_cleaning_agent",
    model=MODEL,
    description=(
        "An interactive data-cleaning assistant that loads, profiles, cleans, "
        "validates, and exports tabular datasets with full audit trails."
    ),
    instruction=_SYSTEM_INSTRUCTION,
    tools=[
        _kaggle_mcp,
        dataset_loader,
        profile_dataset,
        handle_missing_values,
        standardize_formats,
        deduplicate_dataset,
        merge_datasets,
        validate_dataset,
        generate_output,
    ],
)

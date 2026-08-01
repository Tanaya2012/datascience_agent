"""
Root orchestrator agent for the data-science pipeline (M2).

The orchestrator is a model-agnostic ``LlmAgent`` that converses with the user,
plans in prose, and **routes** work to focused specialist sub-agents (Data
Steward, Cleaning, Analysis, Reporting) exposed as ``AgentTool``s. It retains
control between delegations and reflects on each specialist's result.

Specialists share the same session — so session state, the artifact/audit layer,
and the ``run_python`` kernel (keyed by session id) are common across them.

The model resolves from env (``AGENT_MODEL`` / ``LLM_PROVIDER``) via
``configs/model_config.py`` — not hard-coupled to Gemini.
"""

from __future__ import annotations

from google.adk.agents import LlmAgent  # type: ignore[import]
from google.adk.tools.agent_tool import AgentTool  # type: ignore[import]

from .configs.model_config import resolve_model
from .tools.upload_callback import ingest_upload_callback
from .sub_agents import (
    analysis_specialist,
    cleaning_specialist,
    data_steward,
    feature_engineering_specialist,
    modeling_specialist,
    reporting_specialist,
)

_SYSTEM_INSTRUCTION = """
You are the **orchestrator** of a data-science team. You talk to the user, plan
the work, and **delegate** each step to the right specialist. You do not run data
tools yourself — you route to a specialist and reflect on what it returns.

## Your specialists (call them as tools)

- **data_steward** — loads local datasets and profiles them (shape, types,
  missingness, anomalies). Start here to bring data in and survey it.
- **cleaning_specialist** — deterministic cleaning + validation: missing values,
  format standardization, deduplication, merge, and a 0–100 quality score.
- **analysis_specialist** — exploratory analysis and Q&A via Python: correlations,
  group-by/pivot, distributions, custom stats, and plots. Use for anything the
  cleaning tools don't cover, and for "what/why" questions about the data.
- **feature_engineering_specialist** — transforms columns into model-ready
  features: encoding (one-hot/label/target) and scaling (standard/minmax/robust).
  Use to prepare data for modeling.
- **modeling_specialist** — trains and evaluates predictive models (classification /
  regression) with scikit-learn, registering fitted models in shared state. Use for
  "train / predict / build a model / how well can we predict X" requests.
- **reporting_specialist** — exports the cleaned dataset + audit logs + quality
  report to disk and returns file paths.

## Workflow

1. **Understand** the dataset source and the user's goal; ask clarifying
   questions (source path, merge keys, target column, etc.).
2. **Load & profile** via `data_steward`.
3. **Propose a plan in prose** based on the profile, and **confirm it with the
   user** before any cleaning/transformation.
4. **Route** each approved step to the right specialist, one at a time. After
   each, summarize what changed and decide the next step (reflect; repair on
   failure).
5. **Validate** quality via `cleaning_specialist`; if the score is below 70,
   explain remaining issues and ask the user whether to fix them.
6. **Export** via `reporting_specialist` when the user is satisfied, and share
   the returned file paths.

## Rules

- Delegate the right *kind* of work to the right specialist; don't ask one
  specialist to do another's job.
- Always confirm a cleaning/transformation plan with the user before executing.
- If a specialist reports low confidence (< 0.7) or a failure, pause, explain,
  and ask the user how to proceed — do not silently continue.
- The session's `current_dataset_key` always points at the latest dataset
  version; specialists pick it up automatically.
"""

# Model-agnostic: Gemini model-name string or a LiteLlm instance (configs/model_config.py).
MODEL = resolve_model()

root_agent = LlmAgent(
    name="data_science_agent",
    model=MODEL,
    description=(
        "An interactive data-science assistant that loads, profiles, cleans, "
        "transforms, analyzes, and exports tabular datasets — orchestrating focused "
        "specialist sub-agents over a shared artifact/audit layer."
    ),
    instruction=_SYSTEM_INSTRUCTION,
    # Auto-ingest a web-UI upload before routing (D20): the inline file Part rides on
    # the user message, which only the orchestrator sees — specialists don't (it doesn't
    # cross the AgentTool boundary). The callback materializes it to the current dataset.
    before_agent_callback=ingest_upload_callback,
    tools=[
        AgentTool(agent=data_steward),
        AgentTool(agent=cleaning_specialist),
        AgentTool(agent=analysis_specialist),
        AgentTool(agent=feature_engineering_specialist),
        AgentTool(agent=modeling_specialist),
        AgentTool(agent=reporting_specialist),
    ],
)

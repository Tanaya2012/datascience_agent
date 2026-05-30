# datascience_agent

An interactive, LLM-guided **data-science agent** built on Google's Agent
Development Kit (ADK). You describe a tabular dataset (local CSV/Excel/Parquet or
Kaggle) and your goals in natural language; the agent loads, profiles, cleans,
explores, transforms, analyzes, models, and reports — keeping a full audit trail.

## Architecture (in brief)

- **Hybrid:** deterministic, auditable tools form the safe core; a sandboxed
  **code-execution escape hatch** handles anything not covered by a tool. Every
  data mutation — tool or generated code — is checkpointed as a versioned Parquet
  artifact plus a transformation log.
- **Multi-agent:** a model-agnostic orchestrator routes to focused specialists
  (data steward, cleaning, analysis/EDA, feature engineering, modeling, reporting).
- **Model-agnostic:** orchestrator model resolved from env via ADK + LiteLLM
  (Gemini / Claude / OpenAI / local). See `configs/model_config.py`.

This is an active rebuild from an earlier cleaning-only pipeline. **Start with
`.context/`** — `STATUS.md`, `ROADMAP.md`, `DECISIONS.md`, `ARCHITECTURE.md` —
for current state, milestones, rationale, and target design.

## Quick start

```bash
pip install -r requirements.txt        # agent runtime deps
# (later) bootstrap the code-execution worker venv — see scripts/

# configure: copy/edit .env  →  GOOGLE_API_KEY (or other provider key),
#            AGENT_MODEL, LLM_PROVIDER, KAGGLE_USERNAME/KAGGLE_KEY

adk run        # CLI    |    adk web    # browser UI
pytest         # tests
```

## Status

Legacy cleaning core (8 tools + 167 tests) is complete and reused as the
deterministic layer. Capability expansion is tracked in `.context/ROADMAP.md`.

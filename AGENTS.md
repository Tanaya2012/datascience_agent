# AGENTS.md

Model-agnostic data-science agent on Google ADK (hybrid: deterministic tools +
a code-execution escape hatch; full audit trail).

**Before doing anything, read `.context/` — it is the source of truth:**
`STATUS.md` (where we left off) → `ROADMAP.md` (milestones) →
`DECISIONS.md` (decisions + rationale) → `ARCHITECTURE.md` (design).

**Every session:** update `STATUS.md` + tick `ROADMAP.md` as work lands; append
to `DECISIONS.md` on any decision. Keep volatile state in `.context/`, not here.

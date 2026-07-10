"""
Live-agent bug bash (M4.5) — drives the real orchestrator through adversarial,
compounding, and deliberately-broken scenarios, several times each (the LLM is
non-deterministic, so one run only samples one trajectory), asserting *state/artifact
invariants* after each run rather than exact tool trajectories (LLM-noisy — D11).

What it catches that the deterministic fuzzer can't: orchestrator↔specialist routing,
multi-turn plan/approval gating, and error-recovery loops (the transcript-#12 failure
mode). Inner tool calls run inside a specialist's sub-runner, so the parent event
stream shows only delegations — but the shared `pipeline_state` (transformation_logs,
column_lineage, warnings, confidence) records what the inner tools actually did.

Run (needs GOOGLE_API_KEY, uses real LLM):
  conda run -n dsagent python scripts/live_bug_bash.py [repeats]
Findings + transcripts → scratchpad/live_bug_bash_findings.json.
"""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
from pathlib import Path

from dotenv import load_dotenv

MAX_EVENTS = 120          # per turn; exceeding it is itself a finding (runaway/loop)
LOOP_REPEAT = 4           # same (delegation, args) more than this = loop
SCRATCH = Path("/private/tmp/claude-501/-Users-tushar-interests-datascience-agent/"
               "735ec4f5-1cde-4bc3-88b2-07d8a842ffc4/scratchpad")

# An "honest" recovery response either names the failure OR asks for what's missing /
# lists what's available — anything but silently fabricating a result.
_ERR_WORDS = ("could not", "couldn't", "cannot", "can't", "unable", "not found",
              "doesn't exist", "does not exist", "no dataset", "no column", "isn't",
              "not available", "failed", "error", "invalid", "missing", "no such",
              "not in", "aren't", "don't have", "do not have", "provide", "need to",
              "need a", "which dataset", "available column", "path to", "please provide",
              "could you")


# ---------------------------------------------------------------------------
# messy fixtures
# ---------------------------------------------------------------------------

MESSY_SALES = (
    "Customer Name,Sale Amount,Order Date,Region,Email,Units Sold\n"
    "Alice Johnson,\"$1,200.50\",2023-01-15,West,alice@x.com,5\n"
    "Bob Smith,$850.00,01/22/2023,East,,3\n"
    "Alice Johnson,\"$1,200.50\",2023-01-15,West,alice@x.com,5\n"     # dup
    "Carol White,,2023-02-10,North,carol@x.com,2\n"
    "Dave Brown,\"$3,400.00\",March 5 2023,,,10\n"
    "Eve Davis,$920.75,2023-03-20,South,eve@x.com,4\n"
)

CHURN = (
    "customer_id,age,tenure_months,monthly_charges,plan,churned\n"
    "1001,34,12,59.9,basic,0\n1002,52,3,89.1,premium,1\n1003,29,44,45.0,basic,0\n"
    "1004,41,8,72.5,premium,1\n1005,60,60,30.2,basic,0\n1006,23,1,99.9,premium,1\n"
)

ALLNULL = (
    "id,notes,value\n1,,10\n2,,20\n3,,30\n4,,40\n"    # notes entirely empty
)

HIGHCARD = (
    "user_id,country,signups\n"
    + "".join(f"u{i},{['US','UK','DE','FR'][i%4]},{i}\n" for i in range(30))
)


# ---------------------------------------------------------------------------
# scenario definitions:  (name, files, turns, invariant_fn)
# invariant_fn(rr) -> list[str] of violations (empty = ok)
# ---------------------------------------------------------------------------

def _logs(rr):
    return rr["state"].transformation_logs if rr["state"] else []


def _no_column_removed(rr):
    bad = []
    for log in _logs(rr):
        removed = log.column_lineage.columns_removed if log.column_lineage else []
        if removed:
            bad.append(f"{log.step_name} removed columns {removed} (should keep all)")
    return bad


def _final_mentions_error(rr):
    txt = rr["final_text"].lower()
    return [] if any(w in txt for w in _ERR_WORDS) else \
        ["final answer does not surface the failure/inability to the user"]


# The orchestrator's own rule: "If a specialist reports low confidence (< 0.7) ...
# pause, explain, and ask the user how to proceed." Every mutating tool sets
# TransformationLog.confidence, and D15/D17 deliberately drop it to 0.6 on a
# destructive/uncertain step (column drop, ≥5% coercion loss). So: whenever a
# transform logs confidence < 0.7, the agent's final answer must acknowledge/surface
# it — not silently continue or claim a clean result.
_LOW_CONF = 0.7
_SURFACE_WORDS = (
    "warn", "caution", "note that", "heads up", "dropped", "drop", "remov", "empty",
    "as planned", "couldn't", "could not", "unable", "confirm", "review", "careful",
    "lost", "attention", "⚠", "however", "issue", "concern", "please", "would you",
    "should i", "flag", "not applied", "skipped", "unparseable", "coerced",
    "low confidence", "check",
)


def inv_low_confidence_surfaced(rr):
    low = [lg for lg in _logs(rr) if lg.confidence < _LOW_CONF]
    if not low:
        return []
    txt = rr["final_text"].lower()
    if any(w in txt for w in _SURFACE_WORDS):
        return []
    steps = [f"{lg.step_name}(conf={lg.confidence})" for lg in low]
    return [f"low-confidence transform(s) {steps} not surfaced/paused in final answer"]


def inv_clean_keep_all(rr):
    v = _no_column_removed(rr)
    if not _logs(rr):
        v.append("no transformation applied — expected cleaning steps to run")
    return v


def inv_fill_emails_only(rr):
    # D15 class: 'only fill emails' must not drop or destroy other columns (esp. dates).
    # (A refuse-and-warn on a rigid date parse is correct behavior, not a violation —
    # _no_column_removed only flags actual column loss.)
    return _no_column_removed(rr)


def inv_error_recovery(rr):
    return _final_mentions_error(rr)


def inv_no_dataset(rr):
    v = _final_mentions_error(rr)
    if _logs(rr):
        v.append("transformations ran despite no dataset being loaded")
    return v


def inv_missing_not_annihilated(rr):
    v = []
    for log in _logs(rr):
        if log.cols_after == 0:
            v.append(f"{log.step_name} annihilated the dataset (0 columns)")
    if rr["state"] and not rr["state"].current_dataset_key:
        v.append("no current dataset after a load+clean request")
    return v


def inv_terminates(rr):
    return []   # generic checks (loop/budget) cover this; nothing scenario-specific


SCENARIOS = [
    ("messy_sales_clean_keep_all",
     {"sales": MESSY_SALES},
     ["I have a messy sales CSV at {sales}. Load it, then clean it: convert the "
      "Sale Amount currency to numbers, parse Order Date to real dates, and remove "
      "duplicate rows. Keep every column. Go ahead and do it."],
     inv_clean_keep_all),

    ("fill_emails_only_d15",
     {"sales": MESSY_SALES},
     ["Load the CSV at {sales}. Then ONLY fill the missing Email values with "
      "'unknown@example.com'. Do not change or drop any other column. Do it now."],
     inv_fill_emails_only),

    ("bad_column_reference",
     {"sales": MESSY_SALES},
     ["Load {sales} and then compute the correlation between the columns 'revenue' "
      "and 'profit'."],
     inv_error_recovery),

    ("bad_file_path",
     {},
     ["Load the dataset at /nope/does/not/exist/data.csv and profile it."],
     inv_error_recovery),

    ("transform_without_dataset",
     {},
     ["Standardize the formats and scale all the numeric features in my dataset."],
     inv_no_dataset),

    ("all_null_column_pressure",
     {"an": ALLNULL},
     ["Load {an} and handle the missing values so the data is clean for modeling. "
      "Then tell me the final shape."],
     inv_missing_not_annihilated),

    ("high_cardinality_encode",
     {"hc": HIGHCARD},
     ["Load {hc} and one-hot encode every categorical column, including user_id. "
      "Then report the number of columns."],
     inv_terminates),

    ("vague_delete_bad_rows",
     {"sales": MESSY_SALES},
     ["Load {sales} and delete the bad rows."],
     inv_terminates),

    ("multiturn_plan_then_approve",
     {"sales": MESSY_SALES},
     ["I have a sales CSV at {sales}. I want it cleaned and deduplicated — what's "
      "your plan?",
      "Yes, that plan looks good. Please execute it now and keep all columns."],
     inv_clean_keep_all),

    ("churn_eda",
     {"churn": CHURN},
     ["Load {churn}. Which features correlate most with churn? Give me the top "
      "correlations."],
     inv_terminates),
]


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

class _S:
    def __init__(self, state):
        self.state = state


async def run_once(root_agent, scenario, tmp: Path):
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    from google.adk.artifacts import InMemoryArtifactService
    from google.genai import types
    from datascience_agent.tools.artifact_utils import get_session_state

    name, files, turns, _ = scenario
    paths = {}
    for k, content in files.items():
        p = tmp / f"{name}_{k}.csv"
        p.write_text(content)
        paths[k] = str(p)

    app, uid, sid = "bugbash", "u", f"s_{name}"
    ss = InMemorySessionService()
    await ss.create_session(app_name=app, user_id=uid, session_id=sid)
    runner = Runner(agent=root_agent, app_name=app, session_service=ss,
                    artifact_service=InMemoryArtifactService())

    delegations, loops, errors, transcript = [], {}, [], []
    over_budget = False
    final_text = ""
    for turn in turns:
        prompt = turn.format(**paths)
        transcript.append({"user": prompt})
        msg = types.Content(role="user", parts=[types.Part(text=prompt)])
        events = 0
        turn_final = ""
        try:
            async for event in runner.run_async(user_id=uid, session_id=sid, new_message=msg):
                events += 1
                if events > MAX_EVENTS:
                    over_budget = True
                    break
                if getattr(event, "error_message", None) or getattr(event, "error_code", None):
                    errors.append(str(getattr(event, "error_message", "") or event.error_code))
                for part in (event.content.parts or []) if event.content else []:
                    fc = getattr(part, "function_call", None)
                    if fc:
                        delegations.append(fc.name)
                        key = f"{fc.name}:{fc.args}"
                        loops[key] = loops.get(key, 0) + 1
                if event.is_final_response() and event.content:
                    turn_final = "".join(p.text or "" for p in (event.content.parts or []) if p.text)
        except Exception as e:
            # A hallucinated tool call etc. makes ADK raise mid-turn — record it as an
            # agent finding (with the transcript so far), don't crash the harness.
            errors.append(f"run raised: {type(e).__name__}: {str(e).splitlines()[0]}")
        final_text = turn_final or final_text
        transcript.append({"agent": turn_final[:600]})
        if over_budget:
            # a runaway turn taints the scenario — don't run later turns on top of it
            break

    session = await ss.get_session(app_name=app, user_id=uid, session_id=sid)
    try:
        state = get_session_state(_S(session.state))
    except Exception as e:
        state = None
        errors.append(f"state parse error: {e}")

    return {
        "final_text": final_text, "delegations": delegations,
        "loops": {k: c for k, c in loops.items() if c > LOOP_REPEAT},
        "over_budget": over_budget, "errors": errors, "state": state,
        "transcript": transcript,
    }


def generic_violations(rr):
    v = []
    if not rr["final_text"].strip():
        v.append("no final answer produced")
    if rr["over_budget"]:
        v.append(f"exceeded {MAX_EVENTS}-event budget in a turn (possible loop/runaway)")
    if rr["loops"]:
        v.append(f"repeated identical delegation (loop): {list(rr['loops'])[:2]}")
    if rr["errors"]:
        v.append(f"error events: {rr['errors'][:2]}")
    for log in (rr["state"].transformation_logs if rr["state"] else []):
        if log.cols_after == 0:
            v.append(f"{log.step_name} produced a 0-column dataset")
    v += inv_low_confidence_surfaced(rr)
    return v


async def main():
    repeats = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    pkg = Path(__file__).resolve().parent.parent
    load_dotenv(pkg / ".env")
    from datascience_agent.agent import root_agent, MODEL
    print(f"model: {MODEL!r} | {len(SCENARIOS)} scenarios × {repeats} repeats\n")

    findings = []
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        for scenario in SCENARIOS:
            name, _, _, inv = scenario
            fails = 0
            for r in range(repeats):
                try:
                    rr = await run_once(root_agent, scenario, tmp)
                    viol = generic_violations(rr) + inv(rr)
                except Exception as e:
                    import traceback
                    rr = {"transcript": [], "delegations": []}
                    viol = [f"HARNESS/AGENT EXCEPTION: {e}"]
                    traceback.print_exc()
                status = "ok" if not viol else "FAIL"
                if viol:
                    fails += 1
                    findings.append({"scenario": name, "repeat": r, "violations": viol,
                                     "delegations": rr.get("delegations"),
                                     "transcript": rr.get("transcript")})
                print(f"  {name}[{r}] {status}" + (f"  {viol}" if viol else ""))
            print(f"= {name}: {repeats-fails}/{repeats} clean\n")

    out = SCRATCH / "live_bug_bash_findings.json"
    out.write_text(json.dumps(findings, indent=2, default=str))
    print(f"\n{len(findings)} failing runs across {len(SCENARIOS)*repeats} total. → {out}")


if __name__ == "__main__":
    asyncio.run(main())

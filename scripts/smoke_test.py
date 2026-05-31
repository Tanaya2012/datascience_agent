"""
Interactive smoke test: drive the root agent through the ADK Runner with a real
LLM call, exercising the load → profile path on a tiny CSV.

Run: conda run -n dsagent python -m datascience_agent.scripts.smoke_test
(must be run from /Users/tushar/interests so the package is importable)
"""

from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path

from dotenv import load_dotenv


async def main() -> int:
    # Load .env from the package dir (GOOGLE_API_KEY, AGENT_MODEL, ...).
    pkg_dir = Path(__file__).resolve().parent.parent
    load_dotenv(pkg_dir / ".env")

    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    from google.adk.artifacts import InMemoryArtifactService
    from google.genai import types

    from datascience_agent.agent import root_agent, MODEL

    print(f"Resolved model: {MODEL!r}")

    # Tiny dataset with a missing value.
    csv = "name,age,city\nAlice,30,NYC\nBob,,LA\nCarol,25,SF\n"
    tmp = Path(tempfile.gettempdir()) / "ds_smoke.csv"
    tmp.write_text(csv)
    print(f"Wrote test CSV: {tmp}")

    app, uid, sid = "smoke", "u1", "s1"
    session_service = InMemorySessionService()
    await session_service.create_session(app_name=app, user_id=uid, session_id=sid)
    runner = Runner(
        agent=root_agent,
        app_name=app,
        session_service=session_service,
        artifact_service=InMemoryArtifactService(),
    )

    prompt = (
        f"I have a local CSV at {tmp}. Load it with source_type='local', then "
        f"profile it. Tell me the shape and which columns have missing values."
    )
    msg = types.Content(role="user", parts=[types.Part(text=prompt)])

    tool_calls: list[str] = []
    final_text = ""
    async for event in runner.run_async(user_id=uid, session_id=sid, new_message=msg):
        for part in (event.content.parts if event.content else []) or []:
            if getattr(part, "function_call", None):
                tool_calls.append(part.function_call.name)
                print(f"  → tool call: {part.function_call.name}({dict(part.function_call.args or {})})")
            if getattr(part, "function_response", None):
                resp = part.function_response.response
                ok = resp.get("success") if isinstance(resp, dict) else None
                print(f"  ← tool result: {part.function_response.name} success={ok}")
        if event.is_final_response() and event.content:
            final_text = "".join(p.text or "" for p in event.content.parts)

    print("\n=== FINAL RESPONSE ===")
    print(final_text.strip()[:1500])
    print("\n=== SUMMARY ===")
    print(f"tool calls: {tool_calls}")
    ok = "dataset_loader" in tool_calls and "profile_dataset" in tool_calls
    print(f"SMOKE TEST {'PASSED' if ok else 'INCOMPLETE'} "
          f"(expected dataset_loader + profile_dataset)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

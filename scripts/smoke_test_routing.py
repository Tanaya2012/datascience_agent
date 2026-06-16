"""
Routing smoke test (M2): verify the orchestrator delegates to the right
specialist sub-agent via AgentTool, using a real LLM call.

A load+profile request should route to `data_steward`. We assert at least the
data_steward specialist is invoked; nested tool calls (dataset_loader,
profile_dataset) run inside that sub-agent invocation.

Run: conda run -n dsagent python -m datascience_agent.scripts.smoke_test_routing
(must be run from /Users/tushar/interests so the package is importable)
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

from dotenv import load_dotenv


async def main() -> int:
    pkg_dir = Path(__file__).resolve().parent.parent
    load_dotenv(pkg_dir / ".env")

    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    from google.adk.artifacts import InMemoryArtifactService
    from google.genai import types

    from datascience_agent.agent import root_agent, MODEL

    print(f"Resolved model: {MODEL!r}")

    csv = "name,age,city\nAlice,30,NYC\nBob,,LA\nCarol,25,SF\n"
    tmp = Path(tempfile.gettempdir()) / "ds_routing_smoke.csv"
    tmp.write_text(csv)
    print(f"Wrote test CSV: {tmp}")

    app, uid, sid = "routing", "u1", "s1"
    session_service = InMemorySessionService()
    await session_service.create_session(app_name=app, user_id=uid, session_id=sid)
    runner = Runner(
        agent=root_agent,
        app_name=app,
        session_service=session_service,
        artifact_service=InMemoryArtifactService(),
    )

    prompt = (
        f"I have a local CSV at {tmp}. Please load it and profile it — tell me the "
        f"shape and which columns have missing values."
    )
    msg = types.Content(role="user", parts=[types.Part(text=prompt)])

    tool_calls: list[str] = []
    final_text = ""
    async for event in runner.run_async(user_id=uid, session_id=sid, new_message=msg):
        for part in (event.content.parts if event.content else []) or []:
            if getattr(part, "function_call", None):
                tool_calls.append(part.function_call.name)
                print(f"  → call: {part.function_call.name}")
        if event.is_final_response() and event.content:
            final_text = "".join(p.text or "" for p in event.content.parts)

    print("\n=== FINAL RESPONSE ===")
    print(final_text.strip()[:1200])
    print("\n=== SUMMARY ===")
    print(f"all calls observed: {tool_calls}")
    routed = "data_steward" in tool_calls
    print(f"ROUTING SMOKE TEST {'PASSED' if routed else 'INCOMPLETE'} "
          f"(expected delegation to data_steward)")
    return 0 if routed else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

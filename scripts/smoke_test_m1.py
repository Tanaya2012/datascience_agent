"""
M1 smoke test: drive the agent to use run_python for a derived column + commit,
through the real LLM + ADK Runner. Validates the code-execution escape hatch
end-to-end (acceptance: derive `margin = revenue - cost`, commit a new version).

Run from /Users/tushar/interests:
  conda run -n dsagent python -m datascience_agent.scripts.smoke_test_m1
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

    from datascience_agent.agent import root_agent
    from datascience_agent.tools.code_exec.run_python import reset_kernels

    csv = "product,revenue,cost\nA,100,60\nB,200,150\nC,50,30\n"
    tmp = Path(tempfile.gettempdir()) / "ds_m1.csv"
    tmp.write_text(csv)

    app, uid, sid = "m1", "u1", "s1"
    session_service = InMemorySessionService()
    await session_service.create_session(app_name=app, user_id=uid, session_id=sid)
    runner = Runner(agent=root_agent, app_name=app, session_service=session_service,
                    artifact_service=InMemoryArtifactService())

    prompt = (
        f"Load the local CSV at {tmp}. Then use run_python to add a new column "
        f"'margin' = revenue - cost, and commit it. Finally show the column names "
        f"and the first 3 rows."
    )
    msg = types.Content(role="user", parts=[types.Part(text=prompt)])

    tool_calls: list[str] = []
    committed = False
    final = ""
    async for event in runner.run_async(user_id=uid, session_id=sid, new_message=msg):
        for part in (event.content.parts if event.content else []) or []:
            if getattr(part, "function_call", None):
                tool_calls.append(part.function_call.name)
                print(f"  → {part.function_call.name}({dict(part.function_call.args or {})})")
            if getattr(part, "function_response", None):
                resp = part.function_response.response
                if isinstance(resp, dict) and resp.get("committed"):
                    committed = True
                ok = resp.get("success") if isinstance(resp, dict) else None
                print(f"  ← {part.function_response.name} success={ok} "
                      f"committed={resp.get('committed') if isinstance(resp, dict) else '-'}")
        if event.is_final_response() and event.content:
            final = "".join(p.text or "" for p in event.content.parts)

    reset_kernels()
    print("\n=== FINAL ===")
    print(final.strip()[:1200])
    print("\n=== SUMMARY ===")
    print(f"tool calls: {tool_calls}")
    ok = "run_python" in tool_calls and committed and "margin" in final.lower()
    print(f"M1 SMOKE {'PASSED' if ok else 'INCOMPLETE'} "
          f"(want run_python + commit + 'margin' in answer)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

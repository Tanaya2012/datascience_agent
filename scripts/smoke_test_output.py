"""
Smoke test for the generate_output disk-path fix: the agent should save real
files and report their absolute paths (the behavior that previously failed).

Run from /Users/tushar/interests:
  conda run -n dsagent python -m datascience_agent.scripts.smoke_test_output
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

    csv = "name,age,city\nAlice,30,NYC\nBob,,LA\nCarol,25,SF\n"
    tmp = Path(tempfile.gettempdir()) / "ds_out.csv"
    tmp.write_text(csv)

    app, uid, sid = "out", "u1", "s1"
    ss = InMemorySessionService()
    await ss.create_session(app_name=app, user_id=uid, session_id=sid)
    runner = Runner(agent=root_agent, app_name=app, session_service=ss,
                    artifact_service=InMemoryArtifactService())

    prompt = (
        f"Load the local CSV at {tmp}, then save/export the dataset with "
        f"generate_output and tell me the exact file path of the saved CSV."
    )
    msg = types.Content(role="user", parts=[types.Part(text=prompt)])

    tool_calls: list[str] = []
    csv_path = None
    final = ""
    async for event in runner.run_async(user_id=uid, session_id=sid, new_message=msg):
        for part in (event.content.parts if event.content else []) or []:
            if getattr(part, "function_call", None):
                tool_calls.append(part.function_call.name)
            if getattr(part, "function_response", None):
                resp = part.function_response.response
                if isinstance(resp, dict) and resp.get("csv_path"):
                    csv_path = resp["csv_path"]
        if event.is_final_response() and event.content:
            final = "".join(p.text or "" for p in event.content.parts)

    reset_kernels()
    print("tool calls:", tool_calls)
    print("csv_path returned by tool:", csv_path)
    print("file actually exists:", bool(csv_path) and Path(csv_path).exists())
    print("\n=== FINAL ===\n" + final.strip()[:900])
    ok = ("generate_output" in tool_calls and csv_path and Path(csv_path).exists()
          and csv_path.split("/")[-1] in final)
    print(f"\nOUTPUT SMOKE {'PASSED' if ok else 'INCOMPLETE'} "
          "(want generate_output + real csv_path that exists + path shared in answer)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

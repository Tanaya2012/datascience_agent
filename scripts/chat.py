"""
Interactive REPL for the data-science agent (a clear, dependency-light way to
test it by hand). Prints each tool call and whether a step committed a new
dataset version, then the agent's reply. Keeps one session across turns.

Run from /Users/tushar/interests:
  conda run -n dsagent python -m datascience_agent.scripts.chat

Type your messages; 'exit' / 'quit' (or Ctrl-D) to leave.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from dotenv import load_dotenv


async def main() -> None:
    pkg_dir = Path(__file__).resolve().parent.parent
    load_dotenv(pkg_dir / ".env")

    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    from google.adk.artifacts import InMemoryArtifactService
    from google.genai import types

    from datascience_agent.agent import root_agent, MODEL
    from datascience_agent.tools.code_exec.run_python import reset_kernels

    app, uid, sid = "chat", "user", "session"
    session_service = InMemorySessionService()
    await session_service.create_session(app_name=app, user_id=uid, session_id=sid)
    runner = Runner(agent=root_agent, app_name=app, session_service=session_service,
                    artifact_service=InMemoryArtifactService())

    print(f"data-science agent ready (model={MODEL}). Type 'exit' to quit.\n")
    try:
        while True:
            try:
                user = input("you > ").strip()
            except EOFError:
                break
            if user.lower() in {"exit", "quit"}:
                break
            if not user:
                continue

            msg = types.Content(role="user", parts=[types.Part(text=user)])
            async for event in runner.run_async(user_id=uid, session_id=sid, new_message=msg):
                for part in (event.content.parts if event.content else []) or []:
                    fc = getattr(part, "function_call", None)
                    fr = getattr(part, "function_response", None)
                    if fc:
                        print(f"   · calling {fc.name}({_brief(dict(fc.args or {}))})")
                    if fr and isinstance(fr.response, dict):
                        flags = []
                        if fr.response.get("committed"):
                            flags.append("committed new version")
                        if fr.response.get("error_type"):
                            flags.append(f"error={fr.response['error_type']}")
                        suffix = f" [{', '.join(flags)}]" if flags else ""
                        print(f"   · {fr.name} → success={fr.response.get('success')}{suffix}")
                if event.is_final_response() and event.content:
                    text = "".join(p.text or "" for p in event.content.parts)
                    print(f"\nagent > {text.strip()}\n")
    finally:
        reset_kernels()
        print("bye.")


def _brief(args: dict, limit: int = 80) -> str:
    s = ", ".join(f"{k}={v!r}" for k, v in args.items())
    return s if len(s) <= limit else s[:limit] + "…"


if __name__ == "__main__":
    asyncio.run(main())

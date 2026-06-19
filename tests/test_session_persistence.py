"""
Tests for resumable sessions via DatabaseSessionService (M2c).

Verify the `configs/session.py` helpers and that pipeline state written under a
session id survives a process-like restart — i.e. a *fresh* service instance
pointed at the same SQLite file reads the prior state back.
"""

from __future__ import annotations

import pytest
from google.adk.events import Event, EventActions
from google.adk.sessions import DatabaseSessionService, InMemorySessionService

from datascience_agent.configs import session as session_cfg


def test_default_uri_uses_async_sqlite_driver():
    # ADK's async engine requires an async driver; plain sqlite:// is rejected.
    uri = session_cfg.default_session_db_uri()
    assert uri.startswith("sqlite+aiosqlite:///")
    assert session_cfg.DEFAULT_SESSIONS_DIR.exists()


def test_make_session_service_kinds():
    assert isinstance(session_cfg.make_session_service(persistent=False), InMemorySessionService)
    # Persistent service builds against an isolated temp DB (no project-file writes).


@pytest.mark.asyncio
async def test_ensure_session_get_or_create(tmp_path):
    uri = f"sqlite+aiosqlite:///{tmp_path/'s.db'}"
    svc = session_cfg.make_session_service(db_uri=uri)
    app, uid, sid = "t", "u", "sess"

    created = await session_cfg.ensure_session(svc, app, uid, sid)
    assert created is not None
    # Second call resumes the same session rather than creating a duplicate.
    resumed = await session_cfg.ensure_session(svc, app, uid, sid)
    assert resumed.id == created.id


@pytest.mark.asyncio
async def test_pipeline_state_survives_restart(tmp_path):
    uri = f"sqlite+aiosqlite:///{tmp_path/'sessions.db'}"
    app, uid, sid = "dsagent", "user", "s1"
    payload = {"current_dataset_key": "ingest__v1__dataset", "current_task_index": 2}

    # First "process": write pipeline_state via a state-delta event.
    svc1 = DatabaseSessionService(db_url=uri)
    sess = await svc1.create_session(app_name=app, user_id=uid, session_id=sid)
    await svc1.append_event(
        sess, Event(author="user", actions=EventActions(state_delta={"pipeline_state": payload}))
    )

    # Second "process": fresh service against the same file resumes the state.
    svc2 = DatabaseSessionService(db_url=uri)
    restored = await svc2.get_session(app_name=app, user_id=uid, session_id=sid)
    assert restored is not None
    assert restored.state["pipeline_state"] == payload

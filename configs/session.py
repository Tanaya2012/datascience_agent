"""
Session- and artifact-service construction for the agent's script entry points (M2c).

ADK's `adk web` / `adk run` wire these via the ``--session_service_uri`` /
``--artifact_service_uri`` CLI flags, but our own ``Runner``-based scripts
(``scripts/chat.py``, smoke tests) build the services themselves. These helpers
return a **persistent** ``DatabaseSessionService`` (SQLite under ``sessions/``)
and ``FileArtifactService`` (under ``artifacts/``) by default so a re-run with the
same ``session_id`` resumes prior state, or in-memory services for throwaway
runs/tests.

**Pair them.** Session state stores artifact *keys*, not bytes, so a persistent
session backed by an in-memory artifact store resumes pointing at artifacts that no
longer exist — every tool then fails with "Artifact '<key>' not found" (D28).
"""

from __future__ import annotations

from pathlib import Path

# Default on-disk session DB lives under the project's gitignored sessions/ dir.
_PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_SESSIONS_DIR = _PROJECT_DIR / "sessions"
DEFAULT_DB_PATH = DEFAULT_SESSIONS_DIR / "sessions.db"
# Absolute, so it does not move with the caller's cwd (scripts run from the parent dir).
DEFAULT_ARTIFACTS_DIR = _PROJECT_DIR / "artifacts"


def default_session_db_uri() -> str:
    """Async-SQLite URI for the default on-disk session DB (creates sessions/ if needed).

    ADK's ``DatabaseSessionService`` builds an *async* SQLAlchemy engine, which
    requires an async driver — hence the ``sqlite+aiosqlite`` scheme (plain
    ``sqlite://`` / pysqlite is sync-only and is rejected).
    """
    DEFAULT_SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    return f"sqlite+aiosqlite:///{DEFAULT_DB_PATH}"


def make_session_service(persistent: bool = True, db_uri: str | None = None):
    """
    Build a session service.

    Args:
        persistent: True → ``DatabaseSessionService`` (resumable across restarts);
            False → ``InMemorySessionService`` (ephemeral; for tests/throwaway runs).
        db_uri: Override the SQLite/DB URI (defaults to the on-disk sessions.db).

    Returns:
        An ADK session service instance.
    """
    if not persistent:
        from google.adk.sessions import InMemorySessionService

        return InMemorySessionService()

    from google.adk.sessions import DatabaseSessionService

    return DatabaseSessionService(db_url=db_uri or default_session_db_uri())


def make_artifact_service(persistent: bool = True, root_dir: str | Path | None = None):
    """
    Build an artifact service to match the session service (see the module note).

    Args:
        persistent: True → ``FileArtifactService`` rooted at ``artifacts/`` (survives a
            restart); False → ``InMemoryArtifactService`` (ephemeral; single-process
            smoke tests, where the session store is in-memory too).
        root_dir: Override the on-disk root (defaults to the project's ``artifacts/``).

    Returns:
        An ADK artifact service instance.
    """
    if not persistent:
        from google.adk.artifacts import InMemoryArtifactService

        return InMemoryArtifactService()

    from google.adk.artifacts import FileArtifactService

    root = Path(root_dir) if root_dir is not None else DEFAULT_ARTIFACTS_DIR
    root.mkdir(parents=True, exist_ok=True)
    return FileArtifactService(root_dir=root)


async def ensure_session(service, app_name: str, user_id: str, session_id: str):
    """
    Return an existing session (resume) or create a new one (fresh start).

    A persistent service keeps sessions across process restarts, so calling
    ``create_session`` blindly would clobber/duplicate; we get-or-create instead.
    """
    existing = await service.get_session(
        app_name=app_name, user_id=user_id, session_id=session_id
    )
    if existing is not None:
        return existing
    return await service.create_session(
        app_name=app_name, user_id=user_id, session_id=session_id
    )

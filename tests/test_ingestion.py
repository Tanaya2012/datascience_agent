"""
Tests for ingest_uploaded_file (M2b).

Cover the two ways an upload reaches a tool — an inline Part on the user message
and a named ADK artifact — plus secondary registration and a bad-format error.
We use lightweight fake contexts (the shared mock_ctx has no user_content /
working load_artifact) and let artifact *saving* fall back to ARTIFACTS_DIR.
"""

from __future__ import annotations

import io
import os

import pandas as pd
import pytest
from google.genai import types

from tools.ingestion import ingest_uploaded_file
from tools.upload_callback import ingest_upload_callback
from tools.artifact_utils import load_artifact, parquet_bytes_to_df


def _csv_bytes() -> bytes:
    return b"name,age,city\nAlice,30,NYC\nBob,,LA\nCarol,25,SF\n"


def _parquet_bytes() -> bytes:
    buf = io.BytesIO()
    pd.DataFrame({"x": [1, 2], "y": [3, 4]}).to_parquet(buf)
    return buf.getvalue()


class _InlineCtx:
    """Fake ToolContext exposing an inline upload Part via user_content."""

    def __init__(self, data: bytes, filename: str | None, mime: str):
        blob = types.Blob(data=data, mime_type=mime, display_name=filename)
        self.user_content = types.Content(role="user", parts=[types.Part(inline_data=blob)])
        self.state: dict = {}

    async def save_artifact(self, **_):
        raise RuntimeError("force filesystem fallback")  # artifact_utils falls back to disk

    async def load_artifact(self, **_):
        raise RuntimeError("no artifact in this ctx")


class _ArtifactCtx:
    """Fake ToolContext where the upload is fetched via load_artifact(filename)."""

    def __init__(self, data: bytes, filename: str, mime: str):
        self._artifacts = {
            filename: types.Part(inline_data=types.Blob(data=data, mime_type=mime, display_name=filename))
        }
        self.user_content = None
        self.state: dict = {}

    async def load_artifact(self, filename=None, **_):
        return self._artifacts.get(filename)

    async def save_artifact(self, **_):
        raise RuntimeError("force filesystem fallback")


@pytest.mark.asyncio
async def test_ingest_from_inline_part():
    ctx = _InlineCtx(_csv_bytes(), "data.csv", "text/csv")
    res = await ingest_uploaded_file(filename="data.csv", tool_context=ctx)
    assert res["success"] is True
    assert res["shape_after"] == {"rows": 3, "cols": 3}
    assert res["output_artifact_key"].startswith("ingest__v1__dataset")
    # current_dataset_key was set in session state.
    assert ctx.state["pipeline_state"]["current_dataset_key"] == res["output_artifact_key"]


@pytest.mark.asyncio
async def test_ingest_from_inline_part_without_filename():
    # No filename given → use the first inline part; mime drives the reader.
    ctx = _InlineCtx(_csv_bytes(), None, "text/csv")
    res = await ingest_uploaded_file(tool_context=ctx)
    assert res["success"] is True
    assert res["shape_after"]["rows"] == 3


@pytest.mark.asyncio
async def test_ingest_from_artifact():
    ctx = _ArtifactCtx(_csv_bytes(), "upload.csv", "text/csv")
    res = await ingest_uploaded_file(filename="upload.csv", tool_context=ctx)
    assert res["success"] is True
    assert res["shape_after"] == {"rows": 3, "cols": 3}


@pytest.mark.asyncio
async def test_ingest_parquet_from_inline_part():
    ctx = _InlineCtx(_parquet_bytes(), "data.parquet", "application/x-parquet")
    res = await ingest_uploaded_file(filename="data.parquet", tool_context=ctx)
    assert res["success"] is True
    assert res["shape_after"] == {"rows": 2, "cols": 2}


@pytest.mark.asyncio
async def test_ingest_secondary_registration():
    ctx = _InlineCtx(_csv_bytes(), "lookup.csv", "text/csv")
    res = await ingest_uploaded_file(
        filename="lookup.csv", is_secondary=True, secondary_name="lookup", tool_context=ctx
    )
    assert res["success"] is True
    state = ctx.state["pipeline_state"]
    assert "lookup" in state["secondary_datasets"]
    assert state["secondary_datasets"]["lookup"]["artifact_key"].startswith("ingest__v1__dataset")
    # A secondary upload must NOT become the current dataset.
    assert state["current_dataset_key"] is None


@pytest.mark.asyncio
async def test_ingest_unsupported_format_errors():
    ctx = _InlineCtx(b"\x00\x01rubbish", "notes.txt", "application/octet-stream")
    res = await ingest_uploaded_file(filename="notes.txt", tool_context=ctx)
    assert res["success"] is False
    assert "Unsupported upload format" in res["error_message"]


@pytest.mark.asyncio
async def test_ingest_no_file_present_errors():
    ctx = _InlineCtx(_csv_bytes(), "data.csv", "text/csv")
    ctx.user_content = None  # nothing inline, no artifact
    res = await ingest_uploaded_file(filename="missing.csv", tool_context=ctx)
    assert res["success"] is False
    assert "No uploaded file found" in res["error_message"]


# ---------------------------------------------------------------------------
# Upload auto-ingest callback (D20) — CallbackContext is duck-typed the same way
# (user_content / state / save_artifact / load_artifact), so _InlineCtx doubles
# as a fake CallbackContext.
# ---------------------------------------------------------------------------

class _TextOnlyCtx:
    """Fake callback context whose user message has no inline upload (text only)."""

    def __init__(self):
        self.user_content = types.Content(role="user", parts=[types.Part(text="profile it")])
        self.state: dict = {}

    async def save_artifact(self, **_):
        raise RuntimeError("force filesystem fallback")

    async def load_artifact(self, **_):
        raise RuntimeError("no artifact in this ctx")


@pytest.mark.asyncio
async def test_upload_callback_ingests_inline_upload():
    ctx = _InlineCtx(_csv_bytes(), "data.csv", "text/csv")
    result = await ingest_upload_callback(ctx)
    assert result is None  # proceed normally; the upload is now the current dataset
    state = ctx.state["pipeline_state"]
    key = state["current_dataset_key"]
    assert key and key.startswith("ingest__v1__dataset")
    assert any(log["step_name"] == "ingest" for log in state["transformation_logs"])
    df = parquet_bytes_to_df(await load_artifact(key, ctx))
    assert df.shape == (3, 3)


@pytest.mark.asyncio
async def test_upload_callback_noop_without_upload():
    ctx = _TextOnlyCtx()
    result = await ingest_upload_callback(ctx)
    assert result is None
    st = ctx.state.get("pipeline_state")
    assert st is None or st.get("current_dataset_key") is None  # nothing ingested


@pytest.mark.asyncio
async def test_upload_callback_swallows_bad_upload():
    ctx = _InlineCtx(b"\x00\x01rubbish", "notes.txt", "application/octet-stream")
    result = await ingest_upload_callback(ctx)  # must not raise
    assert result is None
    st = ctx.state.get("pipeline_state")
    assert st is None or st.get("current_dataset_key") is None  # failed parse → unloaded


def _llm_evals_enabled() -> bool:
    return os.environ.get("RUN_LLM_EVALS") == "1" and bool(
        os.environ.get("GOOGLE_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
    )


@pytest.mark.llm
@pytest.mark.skipif(not _llm_evals_enabled(), reason="set RUN_LLM_EVALS=1 + API key to run LLM evals")
@pytest.mark.asyncio
async def test_upload_ingested_and_profiled_through_orchestrator():
    """D20 acceptance (live): an inline upload reaches the agent through the *real
    orchestrator* — 0/3 before the before_agent_callback — and gets profiled, not
    reported back to the user as a failed upload. The committed successor to
    scratchpad/probe_upload.py."""
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    from google.adk.artifacts import InMemoryArtifactService
    from datascience_agent.agent import root_agent
    from datascience_agent.tools.artifact_utils import get_session_state

    class _S:
        def __init__(self, state):
            self.state = state

    app, uid, sid = "d20", "u", "s"
    ss = InMemorySessionService()
    await ss.create_session(app_name=app, user_id=uid, session_id=sid)
    runner = Runner(agent=root_agent, app_name=app, session_service=ss,
                    artifact_service=InMemoryArtifactService())
    blob = types.Blob(data=_csv_bytes(), mime_type="text/csv", display_name="data.csv")
    msg = types.Content(role="user", parts=[
        types.Part(text="I uploaded a CSV file. Profile it — tell me the shape and which "
                        "columns have missing values."),
        types.Part(inline_data=blob),
    ])
    final = ""
    async for ev in runner.run_async(user_id=uid, session_id=sid, new_message=msg):
        if ev.is_final_response() and ev.content:
            final = "".join(p.text or "" for p in (ev.content.parts or []) if p.text)

    sess = await ss.get_session(app_name=app, user_id=uid, session_id=sid)
    st = get_session_state(_S(sess.state))
    steps = [lg.step_name for lg in st.transformation_logs]
    assert "ingest" in steps, f"upload not auto-ingested; steps={steps}"
    assert st.current_dataset_key, "upload did not become the current dataset"
    assert not any(w in final.lower() for w in
                   ("wasn't uploaded", "not uploaded", "no file", "couldn't find", "no uploaded")), \
        f"agent reported a failed upload despite ingesting: {final!r}"

"""
Tests for ingest_uploaded_file (M2b).

Cover the two ways an upload reaches a tool — an inline Part on the user message
and a named ADK artifact — plus secondary registration and a bad-format error.
We use lightweight fake contexts (the shared mock_ctx has no user_content /
working load_artifact) and let artifact *saving* fall back to ARTIFACTS_DIR.
"""

from __future__ import annotations

import io

import pandas as pd
import pytest
from google.genai import types

from tools.ingestion import ingest_uploaded_file


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

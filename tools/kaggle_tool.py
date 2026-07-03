"""
Kaggle access as deterministic Data Steward tools (D16).

Replaces the `kaggle-mcp` MCP server, which exposed only one hardcoded tool, dumped
files inside its own uv-cache dir, and returned no path (forcing cache-scraping).
Here we use the official ``kaggle`` library directly: authenticate from
``~/.kaggle/kaggle.json`` (or KAGGLE_USERNAME/KAGGLE_KEY), download/unzip to a
**controlled** path (``artifacts/kaggle/<slug>/``), and **return the file paths** so
the agent can hand a chosen file to ``dataset_loader``. Covers datasets AND
competitions AND search — capabilities the MCP wrapper lacked.

Design: MCP suits the *control plane* (search/metadata that travels through the
protocol); bulk local-file delivery is *data plane* → an in-process tool + local
files (see ARCHITECTURE "MCP connectors — selection criterion", D16).
"""

from __future__ import annotations

import os
import zipfile
from pathlib import Path
from typing import Optional

from google.adk.tools import ToolContext  # type: ignore[import]

from .schemas import (
    KaggleDownloadResult,
    KaggleHit,
    KaggleSearchResult,
    KaggleSource,
)

# Controlled download root, anchored to the project (not cwd) so `adk web` — which
# runs from the parent dir — still writes under the project's gitignored artifacts/.
_KAGGLE_DIR = Path(__file__).resolve().parent.parent / "artifacts" / "kaggle"

_AUTH_HINT = (
    "Kaggle credentials not found. Add ~/.kaggle/kaggle.json (kaggle.com → Settings "
    "→ API → Create New Token) or set KAGGLE_USERNAME / KAGGLE_KEY."
)


def kaggle_credentials_available() -> bool:
    """True if Kaggle creds are present (``~/.kaggle/kaggle.json`` or env vars)."""
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True
    return (Path.home() / ".kaggle" / "kaggle.json").is_file()


def _get_api():
    """Return an authenticated KaggleApi. Raises if the lib/creds are unavailable.

    Imported lazily so the package imports without the optional ``kaggle`` dep, and
    so tests can monkeypatch this function to avoid any network/credentials.
    """
    from kaggle.api.kaggle_api_extended import KaggleApi  # type: ignore[import]

    api = KaggleApi()
    api.authenticate()
    return api


def _comp_id(ref: str) -> str:
    """Competition id from a ref that may be a full URL or a bare id."""
    return ref.rstrip("/").split("/")[-1]


def _slug(source: KaggleSource, ref: str) -> str:
    """Filesystem-safe, unique folder name for a download."""
    if source == KaggleSource.competition:
        return f"comp__{_comp_id(ref)}"
    return ref.replace("/", "__")


def _unzip_all(dest: Path) -> None:
    """Extract and remove any .zip files in dest (competition downloads are zipped)."""
    for zpath in list(dest.glob("*.zip")):
        try:
            with zipfile.ZipFile(zpath) as zf:
                zf.extractall(dest)
            zpath.unlink()
        except zipfile.BadZipFile:
            continue


async def search_kaggle(
    query: str,
    source: str = "dataset",
    limit: int = 10,
    tool_context: Optional[ToolContext] = None,
) -> dict:
    """
    Search Kaggle for datasets or competitions.

    Args:
        query: Search terms (e.g. "titanic", "house prices").
        source: "dataset" (default) or "competition".
        limit: Max results to return.
        tool_context: Injected by ADK at runtime.

    Returns:
        Serialized KaggleSearchResult dict. Pass a hit's `ref` to `download_kaggle`.
    """
    try:
        src = KaggleSource(source)
    except ValueError:
        return KaggleSearchResult(
            success=False, step_name="search_kaggle",
            error_message=f"Unknown source '{source}'. Expected 'dataset' or 'competition'.",
        ).model_dump(mode="json")

    try:
        api = _get_api()
    except Exception as exc:
        return KaggleSearchResult(
            success=False, step_name="search_kaggle", error_message=_auth_error(exc),
        ).model_dump(mode="json")

    try:
        if src == KaggleSource.dataset:
            raw = api.dataset_list(search=query) or []
            hits = [
                KaggleHit(
                    ref=d.ref, title=getattr(d, "title", None),
                    subtitle=getattr(d, "subtitle", None), url=getattr(d, "url", None),
                    size_bytes=getattr(d, "total_bytes", None),
                )
                for d in raw[:limit]
            ]
        else:
            raw = (api.competitions_list(search=query) or []).competitions or []
            hits = [
                KaggleHit(ref=_comp_id(c.ref), title=getattr(c, "title", None), url=getattr(c, "ref", None))
                for c in raw[:limit]
            ]
    except Exception as exc:
        return KaggleSearchResult(
            success=False, step_name="search_kaggle", error_message=str(exc),
        ).model_dump(mode="json")

    return KaggleSearchResult(
        success=True, step_name="search_kaggle", source=src, query=query, hits=hits,
    ).model_dump(mode="json")


async def download_kaggle(
    ref: str,
    source: str = "dataset",
    tool_context: Optional[ToolContext] = None,
) -> dict:
    """
    Download a Kaggle dataset or competition to a local folder and return file paths.

    Files land under ``artifacts/kaggle/<slug>/`` (unzipped). Then load a chosen file
    with ``dataset_loader(source_type="local", dataset_identifier=<path>)``.

    Args:
        ref: Dataset ref "owner/slug" (source="dataset") or competition id
            (source="competition"), e.g. from `search_kaggle`.
        source: "dataset" (default) or "competition".
        tool_context: Injected by ADK at runtime.

    Returns:
        Serialized KaggleDownloadResult dict (with `files` + `download_dir`).
    """
    try:
        src = KaggleSource(source)
    except ValueError:
        return KaggleDownloadResult(
            success=False, step_name="download_kaggle",
            error_message=f"Unknown source '{source}'. Expected 'dataset' or 'competition'.",
        ).model_dump(mode="json")

    try:
        api = _get_api()
    except Exception as exc:
        return KaggleDownloadResult(
            success=False, step_name="download_kaggle", error_message=_auth_error(exc),
        ).model_dump(mode="json")

    dest = _KAGGLE_DIR / _slug(src, ref)
    dest.mkdir(parents=True, exist_ok=True)
    try:
        if src == KaggleSource.dataset:
            api.dataset_download_files(ref, path=str(dest), unzip=True, quiet=True)
        else:
            api.competition_download_files(_comp_id(ref), path=str(dest), quiet=True)
            _unzip_all(dest)
    except Exception as exc:
        return KaggleDownloadResult(
            success=False, step_name="download_kaggle", source=src, ref=ref,
            error_message=f"Download failed: {exc}",
        ).model_dump(mode="json")

    files = sorted(str(p) for p in dest.rglob("*") if p.is_file() and p.suffix != ".zip")
    if not files:
        return KaggleDownloadResult(
            success=False, step_name="download_kaggle", source=src, ref=ref,
            download_dir=str(dest), error_message="Download produced no files.",
        ).model_dump(mode="json")

    return KaggleDownloadResult(
        success=True, step_name="download_kaggle", source=src, ref=ref,
        download_dir=str(dest), files=files,
    ).model_dump(mode="json")


def _auth_error(exc: Exception) -> str:
    """Turn an import/auth failure into an actionable message."""
    if isinstance(exc, ModuleNotFoundError):
        return "The 'kaggle' package is not installed (pip install kaggle)."
    if not kaggle_credentials_available():
        return _AUTH_HINT
    return f"Kaggle authentication failed: {exc}"

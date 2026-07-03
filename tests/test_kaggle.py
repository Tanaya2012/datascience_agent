"""
Tests for the kaggle-library Data Steward tools (D16): search_kaggle + download_kaggle.

`_get_api` is monkeypatched to a fake KaggleApi so nothing touches the network or
credentials; `_KAGGLE_DIR` is redirected to tmp_path so downloads don't touch the repo.
"""

from __future__ import annotations

import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import kaggle_tool
from tools.kaggle_tool import download_kaggle, search_kaggle


class _FakeApi:
    def dataset_list(self, search=None):
        return [
            SimpleNamespace(ref="owner/titanic", title="Titanic", subtitle="sub", url="u", total_bytes=11090),
            SimpleNamespace(ref="a/b", title="B", subtitle=None, url=None, total_bytes=None),
        ]

    def competitions_list(self, search=None):
        return SimpleNamespace(
            competitions=[SimpleNamespace(ref="https://www.kaggle.com/competitions/titanic", title="Titanic ML")]
        )

    def dataset_download_files(self, ref, path=None, unzip=False, quiet=True):
        Path(path).mkdir(parents=True, exist_ok=True)
        (Path(path) / "train.csv").write_text("a,b\n1,2\n")
        (Path(path) / "test.csv").write_text("a,b\n3,4\n")

    def competition_download_files(self, competition, path=None, quiet=True):
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(p / f"{competition}.zip", "w") as zf:
            zf.writestr("train.csv", "a,b\n1,2\n")


@pytest.fixture()
def fake_kaggle(monkeypatch, tmp_path):
    monkeypatch.setattr(kaggle_tool, "_get_api", lambda: _FakeApi())
    monkeypatch.setattr(kaggle_tool, "_KAGGLE_DIR", tmp_path / "kaggle")
    return tmp_path


# --- search --------------------------------------------------------------------

@pytest.mark.asyncio
async def test_search_datasets(fake_kaggle):
    res = await search_kaggle("titanic", source="dataset")
    assert res["success"] is True
    assert res["source"] == "dataset"
    assert res["hits"][0]["ref"] == "owner/titanic"
    assert res["hits"][0]["size_bytes"] == 11090


@pytest.mark.asyncio
async def test_search_competitions_extracts_id(fake_kaggle):
    res = await search_kaggle("titanic", source="competition")
    assert res["success"] is True
    # Competition ref is normalized from the URL to the bare id.
    assert res["hits"][0]["ref"] == "titanic"


@pytest.mark.asyncio
async def test_search_limit(fake_kaggle):
    res = await search_kaggle("x", source="dataset", limit=1)
    assert len(res["hits"]) == 1


@pytest.mark.asyncio
async def test_bad_source(fake_kaggle):
    res = await search_kaggle("x", source="notebook")
    assert res["success"] is False
    assert "Unknown source" in res["error_message"]


@pytest.mark.asyncio
async def test_auth_failure_is_clean(monkeypatch):
    def boom():
        raise OSError("Could not find kaggle.json")

    monkeypatch.setattr(kaggle_tool, "_get_api", boom)
    monkeypatch.setattr(kaggle_tool, "kaggle_credentials_available", lambda: False)
    res = await search_kaggle("x")
    assert res["success"] is False
    assert "credentials not found" in res["error_message"].lower()


# --- download ------------------------------------------------------------------

@pytest.mark.asyncio
async def test_download_dataset_returns_files(fake_kaggle):
    res = await download_kaggle("owner/titanic", source="dataset")
    assert res["success"] is True
    names = {Path(f).name for f in res["files"]}
    assert names == {"train.csv", "test.csv"}
    assert res["download_dir"].endswith("owner__titanic")  # slashes → __


@pytest.mark.asyncio
async def test_download_competition_unzips(fake_kaggle):
    res = await download_kaggle("titanic", source="competition")
    assert res["success"] is True
    names = {Path(f).name for f in res["files"]}
    assert "train.csv" in names
    assert not any(f.endswith(".zip") for f in res["files"])  # zip removed after extract
    assert Path(res["download_dir"]).name == "comp__titanic"


@pytest.mark.asyncio
async def test_download_no_files_errors(fake_kaggle, monkeypatch):
    monkeypatch.setattr(_FakeApi, "dataset_download_files", lambda self, ref, path=None, **k: None)
    res = await download_kaggle("owner/empty", source="dataset")
    assert res["success"] is False
    assert "no files" in res["error_message"].lower()

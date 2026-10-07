"""
Shared pytest fixtures for the datascience_agent test suite.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import pytest

import datascience_agent.tools.artifact_utils as au
from datascience_agent.tools import kaggle_tool, output_generator

_PROJECT_ROOT = Path(__file__).resolve().parent
_WORKER_VENV = _PROJECT_ROOT / ".worker-venv"


def _is_project_file(path: str | None) -> bool:
    # Pseudo-paths such as "<frozen ...>" or "<string>" are relative; resolving them
    # would anchor them to the cwd (the project) and report false positives.
    if not path or not Path(path).is_absolute():
        return False
    p = Path(path).resolve()
    return p.is_relative_to(_PROJECT_ROOT) and not p.is_relative_to(_WORKER_VENV)


@pytest.fixture(scope="session", autouse=True)
def _one_module_name_per_file():
    """Every module loaded from this project must be loaded as ``datascience_agent.*``.

    Importing the same file under a second name (e.g. a flat top-level ``tools``)
    creates a second, *distinct* module: separate classes, separate globals, and a
    monkeypatch on one copy silently misses the other (D31). ``python -m pytest`` puts
    the project dir on ``sys.path``, so nothing stops such an import except this check.
    It runs once, after the whole session, against everything the suite imported.
    """
    yield
    stray = sorted(
        name for name, mod in list(sys.modules.items())
        if _is_project_file(getattr(mod, "__file__", None))
        and name != "datascience_agent" and not name.startswith("datascience_agent.")
    )
    assert not stray, (
        "Project modules were imported under a non-package name, creating duplicate "
        f"module copies: {stray}. Import them as datascience_agent.<...> instead (D31)."
    )


@pytest.fixture(autouse=True)
def _isolate_local_storage(tmp_path, monkeypatch):
    """Point every on-disk store the code under test writes to at a per-test temp dir.

    The fixture this replaced *deleted* the real ``artifacts/`` before every test — the
    directory FileArtifactService persists ``chat.py`` sessions into (D28) and Kaggle
    downloads land in — so a test run wiped them (D32). Tests must never touch the
    project tree; ``_project_tree_untouched`` enforces that.

    Read these paths at call time (``artifact_utils.ARTIFACTS_DIR``), never via
    ``from … import ARTIFACTS_DIR`` — that binds the original path and bypasses the patch.
    """
    monkeypatch.setattr(au, "ARTIFACTS_DIR", tmp_path / "artifacts")
    monkeypatch.setattr(output_generator, "_DEFAULT_OUTPUTS_ROOT", tmp_path / "outputs")
    monkeypatch.setattr(kaggle_tool, "_KAGGLE_DIR", tmp_path / "kaggle")


def _project_files() -> set[str]:
    """Every file in the project tree, minus dot-dirs/dot-files (VCS, venvs, caches,
    editor swap files) and bytecode."""
    files = set()
    for dirpath, dirnames, filenames in os.walk(_PROJECT_ROOT):
        dirnames[:] = [d for d in dirnames if not d.startswith(".") and d != "__pycache__"]
        for f in filenames:
            if not f.startswith(".") and not f.endswith("~"):
                files.add(os.path.relpath(os.path.join(dirpath, f), _PROJECT_ROOT))
    return files


@pytest.fixture(scope="session", autouse=True)
def _project_tree_untouched():
    """The suite must neither create nor delete files in the project tree (D32).

    Catches both failure modes F8 found: litter (``outputs/run_*`` piling up) and
    destruction (a fixture wiping ``artifacts/``). If you were running the agent in
    parallel and it wrote an export, that file will be listed here too.
    """
    before = _project_files()
    yield
    after = _project_files()
    created, deleted = sorted(after - before), sorted(before - after)
    assert not created and not deleted, (
        "The test session changed the project tree (D32) — "
        f"created: {created[:20]}, deleted: {deleted[:20]}"
    )


@pytest.fixture()
def sample_df() -> pd.DataFrame:
    """Small DataFrame with missing values and a duplicate row."""
    return pd.DataFrame(
        {
            "Name": ["Alice", "Bob", "Charlie", "Alice", None],
            "Age": [30.0, None, 25.0, 30.0, 28.0],
            "Salary": [50_000.0, 60_000.0, None, 50_000.0, 45_000.0],
            "JoinDate": ["2020-01-15", "2019-06-01", "2021-03-20", "2020-01-15", "2022-11-30"],
        }
    )


@pytest.fixture()
def csv_file(tmp_path, sample_df) -> Path:
    """Write sample_df to a temporary CSV file and return its path."""
    p = tmp_path / "employees.csv"
    sample_df.to_csv(p, index=False)
    return p


@pytest.fixture()
def secondary_csv(tmp_path) -> Path:
    """A secondary lookup CSV for merge tests."""
    df = pd.DataFrame({"Name": ["Alice", "Bob", "Charlie"], "Dept": ["Eng", "HR", "Eng"]})
    p = tmp_path / "depts.csv"
    df.to_csv(p, index=False)
    return p


@pytest.fixture()
def empty_csv(tmp_path) -> Path:
    """CSV with headers but zero data rows."""
    p = tmp_path / "empty.csv"
    pd.DataFrame(columns=["Name", "Age", "Salary"]).to_csv(p, index=False)
    return p


@pytest.fixture()
def single_row_csv(tmp_path) -> Path:
    """CSV with exactly one data row."""
    p = tmp_path / "single.csv"
    pd.DataFrame({"Name": ["Alice"], "Age": [30.0], "Salary": [50_000.0]}).to_csv(p, index=False)
    return p


@pytest.fixture()
def all_null_col_csv(tmp_path) -> Path:
    """CSV where one column is entirely NaN."""
    p = tmp_path / "allnull.csv"
    pd.DataFrame({"id": [1, 2, 3], "value": [None, None, None], "label": ["a", "b", "c"]}).to_csv(
        p, index=False
    )
    return p


@pytest.fixture()
def sales_csv(tmp_path) -> Path:
    """Realistic sales dataset with currency strings, mixed dates, duplicates, and missing values."""
    import numpy as np

    df = pd.DataFrame(
        {
            "Customer Name": [
                "Alice Johnson", "Bob Smith", "Alice Johnson",  # duplicate
                "Carol White", "Dave Brown", "Eve Davis",
            ],
            "Sale Amount": ["$1,200.50", "$850.00", "$1,200.50", None, "$3,400.00", "$920.75"],
            "Sale Date": [
                "2023-01-15", "01/22/2023", "2023-01-15",
                "2023-02-10", "03/05/2023", "2023-03-20",
            ],
            "Region": ["West", "East", "West", "North", None, "South"],
            "Units Sold": ["5", "3", "5", "2", "10", "4"],
        }
    )
    p = tmp_path / "sales.csv"
    df.to_csv(p, index=False)
    return p


@pytest.fixture()
def mock_ctx(tmp_path):
    """
    A lightweight stand-in for ADK ToolContext.

    - Stores session state in an in-memory dict.
    - Delegates artifact I/O to the local filesystem fallback by always raising,
      so artifact_utils falls through to ARTIFACTS_DIR — a per-test temp dir
      (see `_isolate_local_storage`).
    """

    class _MockCtx:
        def __init__(self):
            self.state: dict = {}

        async def save_artifact(self, **_):
            raise RuntimeError("no ADK in tests")

        async def load_artifact(self, **_):
            raise RuntimeError("no ADK in tests")

    return _MockCtx()

"""
Shared pytest fixtures for the datascience_agent test suite.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
import pytest

import tools.artifact_utils as au


@pytest.fixture(autouse=True)
def clean_artifacts():
    """Remove the local artifacts directory before each test to ensure isolation."""
    if au.ARTIFACTS_DIR.exists():
        shutil.rmtree(au.ARTIFACTS_DIR)
    yield
    # Leave artifacts in place after the test (easier debugging).
    # They will be wiped at the start of the next test.


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
      so artifact_utils falls through to ARTIFACTS_DIR.
    """

    class _MockCtx:
        def __init__(self):
            self.state: dict = {}

        async def save_artifact(self, **_):
            raise RuntimeError("no ADK in tests")

        async def load_artifact(self, **_):
            raise RuntimeError("no ADK in tests")

    return _MockCtx()

"""
Integration tests for all 8 pipeline tools.

Each test calls the async tool function directly (no ADK runtime) and
validates the returned result dict.  Artifacts are written to and read
from the local `artifacts/` directory via the filesystem fallback in
artifact_utils.  The `clean_artifacts` autouse fixture (conftest.py)
wipes that directory before each test.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from tools.artifact_utils import (
    df_to_parquet_bytes,
    load_artifact,
    make_artifact_key,
    parquet_bytes_to_df,
    ARTIFACTS_DIR,
)
from tools.schemas import (
    AgentSessionState,
    DatasetVersion,
    MissingStrategy,
    PipelineStatus,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_local(key: str, data: bytes) -> None:
    """Write bytes to the local artifacts fallback without a tool_context."""
    path = ARTIFACTS_DIR / key
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


async def _put_df(df: pd.DataFrame, key: str) -> None:
    """Persist a DataFrame as a parquet artifact at the given key."""
    _save_local(key, df_to_parquet_bytes(df))


def _state_with_secondary(secondary_key: str, name: str = "lookup") -> AgentSessionState:
    """Build a session state that has one secondary dataset registered."""
    state = AgentSessionState()
    state.secondary_datasets[name] = DatasetVersion(
        artifact_key=secondary_key,
        step_name="dataset_loader",
        version=1,
        shape=(3, 2),
        checksum="dummy",
        schema_digest="dummy",
        created_at=datetime.utcnow(),
    )
    return state


class _MockCtxWithState:
    """MockCtx whose state is pre-seeded with an AgentSessionState."""

    def __init__(self, state: AgentSessionState):
        from tools.artifact_utils import SESSION_STATE_KEY
        self.state = {SESSION_STATE_KEY: state.model_dump(mode="json")}

    async def save_artifact(self, **_):
        raise RuntimeError("no ADK")

    async def load_artifact(self, **_):
        raise RuntimeError("no ADK")


# ===========================================================================
# 1. dataset_loader
# ===========================================================================

class TestDatasetLoader:
    async def test_load_csv_success(self, csv_file):
        from tools.dataset_loader import dataset_loader

        result = await dataset_loader("local", str(csv_file))

        assert result["success"] is True
        assert result["shape_after"]["rows"] == 5
        assert result["shape_after"]["cols"] == 4
        assert result["output_artifact_key"] == "dataset_loader/v1/dataset"
        assert len(result["schema_summary"]) == 4

    async def test_artifact_persisted_to_disk(self, csv_file):
        from tools.dataset_loader import dataset_loader

        await dataset_loader("local", str(csv_file))

        path = ARTIFACTS_DIR / "dataset_loader/v1/dataset"
        assert path.exists()
        df = parquet_bytes_to_df(path.read_bytes())
        assert list(df.columns) == ["Name", "Age", "Salary", "JoinDate"]

    async def test_load_excel(self, tmp_path, sample_df):
        from tools.dataset_loader import dataset_loader

        p = tmp_path / "data.xlsx"
        sample_df.to_excel(p, index=False)
        result = await dataset_loader("local", str(p))

        assert result["success"] is True
        assert result["shape_after"]["rows"] == 5

    async def test_missing_file_returns_failure(self):
        from tools.dataset_loader import dataset_loader

        result = await dataset_loader("local", "/nonexistent/path/data.csv")

        assert result["success"] is False
        assert result["error_message"] is not None

    async def test_unsupported_format_returns_failure(self, tmp_path):
        from tools.dataset_loader import dataset_loader

        p = tmp_path / "data.json"
        p.write_text('{"a": 1}')
        result = await dataset_loader("local", str(p))

        assert result["success"] is False

    async def test_unknown_source_type_returns_failure(self, csv_file):
        from tools.dataset_loader import dataset_loader

        result = await dataset_loader("s3", str(csv_file))

        assert result["success"] is False

    async def test_kaggle_source_type_rejected(self):
        from tools.dataset_loader import dataset_loader

        result = await dataset_loader("kaggle", "owner/dataset-name")

        assert result["success"] is False
        assert "local" in result["error_message"].lower()

    async def test_secondary_dataset_flag(self, secondary_csv):
        from tools.dataset_loader import dataset_loader

        result = await dataset_loader(
            "local", str(secondary_csv), is_secondary=True, secondary_name="depts"
        )

        assert result["success"] is True
        # Output artifact key should still be set
        assert result["output_artifact_key"] is not None

    async def test_schema_summary_contains_sample(self, csv_file):
        from tools.dataset_loader import dataset_loader

        result = await dataset_loader("local", str(csv_file))

        cols = {s["col"] for s in result["schema_summary"]}
        assert "Name" in cols
        assert "Age" in cols

    async def test_version_increments_on_second_call(self, csv_file, mock_ctx):
        from tools.dataset_loader import dataset_loader

        # Both calls share mock_ctx so state (artifact manifest) accumulates.
        r1 = await dataset_loader("local", str(csv_file), tool_context=mock_ctx)
        r2 = await dataset_loader("local", str(csv_file), tool_context=mock_ctx)

        assert r1["output_artifact_key"] == "dataset_loader/v1/dataset"
        assert r2["output_artifact_key"] == "dataset_loader/v2/dataset"


# ===========================================================================
# 2. profile_dataset
# ===========================================================================

class TestDataProfiler:
    async def test_profile_success(self, csv_file):
        from tools.dataset_loader import dataset_loader
        from tools.data_profiler import profile_dataset

        load_r = await dataset_loader("local", str(csv_file))
        result = await profile_dataset(load_r["output_artifact_key"])

        assert result["success"] is True
        assert result["profile"] is not None
        assert result["profile"]["shape"] == [5, 4]

    async def test_profile_contains_all_columns(self, csv_file):
        from tools.dataset_loader import dataset_loader
        from tools.data_profiler import profile_dataset

        load_r = await dataset_loader("local", str(csv_file))
        result = await profile_dataset(load_r["output_artifact_key"])

        col_names = {c["name"] for c in result["profile"]["column_profiles"]}
        assert col_names == {"Name", "Age", "Salary", "JoinDate"}

    async def test_profile_detects_missing_values(self, csv_file):
        from tools.dataset_loader import dataset_loader
        from tools.data_profiler import profile_dataset

        load_r = await dataset_loader("local", str(csv_file))
        result = await profile_dataset(load_r["output_artifact_key"])

        age_profile = next(
            c for c in result["profile"]["column_profiles"] if c["name"] == "Age"
        )
        assert age_profile["missing_count"] == 1

    async def test_profile_quality_score_in_range(self, csv_file):
        from tools.dataset_loader import dataset_loader
        from tools.data_profiler import profile_dataset

        load_r = await dataset_loader("local", str(csv_file))
        result = await profile_dataset(load_r["output_artifact_key"])

        score = result["profile"]["quality_score_estimate"]
        assert 0.0 <= score <= 100.0

    async def test_profile_artifact_key_saved(self, csv_file):
        from tools.dataset_loader import dataset_loader
        from tools.data_profiler import profile_dataset

        load_r = await dataset_loader("local", str(csv_file))
        result = await profile_dataset(load_r["output_artifact_key"])

        assert result["profile_artifact_key"] == "profile_dataset/v1/profile"
        assert (ARTIFACTS_DIR / "profile_dataset/v1/profile").exists()

    async def test_profile_missing_artifact_returns_failure(self):
        from tools.data_profiler import profile_dataset

        result = await profile_dataset("nonexistent/v99/dataset")

        assert result["success"] is False
        assert result["error_message"] is not None

    async def test_profile_dataset_unchanged(self, csv_file):
        """Profiling should never modify the dataset artifact."""
        from tools.dataset_loader import dataset_loader
        from tools.data_profiler import profile_dataset

        load_r = await dataset_loader("local", str(csv_file))
        key = load_r["output_artifact_key"]
        result = await profile_dataset(key)

        # output_artifact_key should point to the same dataset (unchanged)
        assert result["output_artifact_key"] == key


# ===========================================================================
# 3. handle_missing_values
# ===========================================================================

class TestMissingHandler:
    async def test_median_imputation(self, csv_file):
        from tools.dataset_loader import dataset_loader
        from tools.cleaning.missing_handler import handle_missing_values

        load_r = await dataset_loader("local", str(csv_file))
        result = await handle_missing_values(
            load_r["output_artifact_key"],
            strategy_config={"Age": "median"},
        )

        assert result["success"] is True
        assert "Age" in result["columns_imputed"]

        df = parquet_bytes_to_df((ARTIFACTS_DIR / result["output_artifact_key"]).read_bytes())
        assert df["Age"].isna().sum() == 0

    async def test_mean_imputation(self, csv_file):
        from tools.dataset_loader import dataset_loader
        from tools.cleaning.missing_handler import handle_missing_values

        load_r = await dataset_loader("local", str(csv_file))
        result = await handle_missing_values(
            load_r["output_artifact_key"],
            strategy_config={"Salary": "mean"},
        )

        assert result["success"] is True
        df = parquet_bytes_to_df((ARTIFACTS_DIR / result["output_artifact_key"]).read_bytes())
        assert df["Salary"].isna().sum() == 0

    async def test_mode_imputation(self, csv_file):
        from tools.dataset_loader import dataset_loader
        from tools.cleaning.missing_handler import handle_missing_values

        load_r = await dataset_loader("local", str(csv_file))
        result = await handle_missing_values(
            load_r["output_artifact_key"],
            strategy_config={"Name": "mode"},
        )

        assert result["success"] is True

    async def test_constant_imputation(self, csv_file):
        from tools.dataset_loader import dataset_loader
        from tools.cleaning.missing_handler import handle_missing_values

        load_r = await dataset_loader("local", str(csv_file))
        result = await handle_missing_values(
            load_r["output_artifact_key"],
            strategy_config={"Name": "constant"},
            constant_fill_values={"Name": "Unknown"},
        )

        assert result["success"] is True
        df = parquet_bytes_to_df((ARTIFACTS_DIR / result["output_artifact_key"]).read_bytes())
        assert "Unknown" in df["Name"].values

    async def test_constant_missing_fill_value_produces_warning(self, csv_file):
        from tools.dataset_loader import dataset_loader
        from tools.cleaning.missing_handler import handle_missing_values

        load_r = await dataset_loader("local", str(csv_file))
        result = await handle_missing_values(
            load_r["output_artifact_key"],
            strategy_config={"Name": "constant"},
            # constant_fill_values intentionally omitted
        )

        assert result["success"] is True
        assert any("constant" in w.lower() for w in result["warnings"])

    async def test_drop_row_strategy(self, csv_file):
        from tools.dataset_loader import dataset_loader
        from tools.cleaning.missing_handler import handle_missing_values

        load_r = await dataset_loader("local", str(csv_file))
        result = await handle_missing_values(
            load_r["output_artifact_key"],
            strategy_config={"Age": "drop_row"},
        )

        assert result["success"] is True
        assert result["rows_removed"] == 1  # one row has missing Age
        assert result["shape_after"]["rows"] == 4

    async def test_ffill_strategy(self, tmp_path):
        from tools.dataset_loader import dataset_loader
        from tools.cleaning.missing_handler import handle_missing_values

        df = pd.DataFrame({"val": [1.0, None, None, 4.0]})
        p = tmp_path / "fill.csv"
        df.to_csv(p, index=False)
        load_r = await dataset_loader("local", str(p))
        result = await handle_missing_values(
            load_r["output_artifact_key"],
            strategy_config={"val": "ffill"},
        )

        assert result["success"] is True
        df_out = parquet_bytes_to_df((ARTIFACTS_DIR / result["output_artifact_key"]).read_bytes())
        assert df_out["val"].isna().sum() == 0

    async def test_bfill_strategy(self, tmp_path):
        from tools.dataset_loader import dataset_loader
        from tools.cleaning.missing_handler import handle_missing_values

        df = pd.DataFrame({"val": [None, None, 3.0, 4.0]})
        p = tmp_path / "bfill.csv"
        df.to_csv(p, index=False)
        load_r = await dataset_loader("local", str(p))
        result = await handle_missing_values(
            load_r["output_artifact_key"],
            strategy_config={"val": "bfill"},
        )

        assert result["success"] is True
        df_out = parquet_bytes_to_df((ARTIFACTS_DIR / result["output_artifact_key"]).read_bytes())
        assert df_out["val"].isna().sum() == 0

    async def test_drop_threshold_removes_column(self, tmp_path):
        from tools.dataset_loader import dataset_loader
        from tools.cleaning.missing_handler import handle_missing_values

        # 'junk' column is 80% missing → should be dropped at threshold=0.5
        df = pd.DataFrame({"id": [1, 2, 3, 4, 5], "junk": [None, None, None, None, 1.0]})
        p = tmp_path / "thresh.csv"
        df.to_csv(p, index=False)
        load_r = await dataset_loader("local", str(p))
        result = await handle_missing_values(
            load_r["output_artifact_key"],
            strategy_config={},
            drop_threshold=0.5,
        )

        assert result["success"] is True
        assert "junk" in result["columns_dropped"]
        df_out = parquet_bytes_to_df((ARTIFACTS_DIR / result["output_artifact_key"]).read_bytes())
        assert "junk" not in df_out.columns

    async def test_nonexistent_column_produces_warning(self, csv_file):
        from tools.dataset_loader import dataset_loader
        from tools.cleaning.missing_handler import handle_missing_values

        load_r = await dataset_loader("local", str(csv_file))
        result = await handle_missing_values(
            load_r["output_artifact_key"],
            strategy_config={"ghost_column": "median"},
        )

        assert result["success"] is True
        assert any("ghost_column" in w for w in result["warnings"])

    async def test_missing_artifact_returns_failure(self):
        from tools.cleaning.missing_handler import handle_missing_values

        result = await handle_missing_values(
            "nonexistent/v1/dataset",
            strategy_config={"col": "mean"},
        )
        assert result["success"] is False

    async def test_cells_modified_count(self, csv_file):
        from tools.dataset_loader import dataset_loader
        from tools.cleaning.missing_handler import handle_missing_values

        load_r = await dataset_loader("local", str(csv_file))
        result = await handle_missing_values(
            load_r["output_artifact_key"],
            strategy_config={"Age": "median", "Salary": "mean", "Name": "mode"},
        )

        # Age has 1 missing, Salary has 1, Name has 1 → 3 cells total
        assert result["cells_modified"] == 3


# ===========================================================================
# 4. standardize_formats
# ===========================================================================

class TestStandardizer:
    async def test_snake_case_headers(self, tmp_path):
        from tools.dataset_loader import dataset_loader
        from tools.cleaning.standardizer import standardize_formats

        df = pd.DataFrame({"First Name": ["a"], "Last-Name": ["b"], "AgeYears": [1]})
        p = tmp_path / "messy.csv"
        df.to_csv(p, index=False)
        load_r = await dataset_loader("local", str(p))
        result = await standardize_formats(load_r["output_artifact_key"], normalize_headers=True)

        assert result["success"] is True
        df_out = parquet_bytes_to_df((ARTIFACTS_DIR / result["output_artifact_key"]).read_bytes())
        assert "first_name" in df_out.columns
        assert "last_name" in df_out.columns
        assert "age_years" in df_out.columns

    async def test_rename_tracked_in_column_lineage(self, tmp_path):
        from tools.dataset_loader import dataset_loader
        from tools.cleaning.standardizer import standardize_formats

        df = pd.DataFrame({"First Name": [1], "Second Name": [2]})
        p = tmp_path / "names.csv"
        df.to_csv(p, index=False)
        load_r = await dataset_loader("local", str(p))
        result = await standardize_formats(load_r["output_artifact_key"])

        renames = result["log"]["column_lineage"]["columns_renamed"]
        assert "First Name" in renames
        assert renames["First Name"] == "first_name"

    async def test_currency_stripped_to_float(self, tmp_path):
        from tools.dataset_loader import dataset_loader
        from tools.cleaning.standardizer import standardize_formats

        df = pd.DataFrame({"price": ["$1,200.50", "$999.00", None]})
        p = tmp_path / "prices.csv"
        df.to_csv(p, index=False)
        load_r = await dataset_loader("local", str(p))
        result = await standardize_formats(
            load_r["output_artifact_key"], parse_currency=True
        )

        assert result["success"] is True
        df_out = parquet_bytes_to_df((ARTIFACTS_DIR / result["output_artifact_key"]).read_bytes())
        assert pd.api.types.is_numeric_dtype(df_out["price"])
        assert df_out["price"].dropna().iloc[0] == pytest.approx(1200.50)

    async def test_numeric_string_coercion(self, tmp_path):
        from tools.dataset_loader import dataset_loader
        from tools.cleaning.standardizer import standardize_formats

        df = pd.DataFrame({"score": ["10", "20", "30", "40", "50"]})
        p = tmp_path / "scores.csv"
        df.to_csv(p, index=False)
        load_r = await dataset_loader("local", str(p))
        result = await standardize_formats(
            load_r["output_artifact_key"], parse_numerics=True, normalize_headers=False
        )

        assert result["success"] is True
        df_out = parquet_bytes_to_df((ARTIFACTS_DIR / result["output_artifact_key"]).read_bytes())
        assert pd.api.types.is_numeric_dtype(df_out["score"])

    async def test_date_parsing(self, tmp_path):
        from tools.dataset_loader import dataset_loader
        from tools.cleaning.standardizer import standardize_formats

        df = pd.DataFrame({"event_date": ["2020-01-01", "2021-06-15", "2022-12-31"]})
        p = tmp_path / "dates.csv"
        df.to_csv(p, index=False)
        load_r = await dataset_loader("local", str(p))
        result = await standardize_formats(
            load_r["output_artifact_key"], parse_dates=True, normalize_headers=False
        )

        assert result["success"] is True
        df_out = parquet_bytes_to_df((ARTIFACTS_DIR / result["output_artifact_key"]).read_bytes())
        assert pd.api.types.is_datetime64_any_dtype(df_out["event_date"])

    async def test_no_changes_when_all_disabled(self, csv_file):
        from tools.dataset_loader import dataset_loader
        from tools.cleaning.standardizer import standardize_formats

        load_r = await dataset_loader("local", str(csv_file))
        result = await standardize_formats(
            load_r["output_artifact_key"],
            normalize_headers=False,
            parse_dates=False,
            parse_currency=False,
            parse_numerics=False,
        )

        assert result["success"] is True
        assert result["cells_modified"] == 0

    async def test_missing_artifact_returns_failure(self):
        from tools.cleaning.standardizer import standardize_formats

        result = await standardize_formats("nonexistent/v1/dataset")

        assert result["success"] is False


# ===========================================================================
# 5. deduplicate_dataset
# ===========================================================================

class TestDeduplicator:
    async def test_exact_dedup_removes_duplicate_row(self, csv_file):
        from tools.dataset_loader import dataset_loader
        from tools.cleaning.deduplicator import deduplicate_dataset

        load_r = await dataset_loader("local", str(csv_file))
        result = await deduplicate_dataset(load_r["output_artifact_key"], exact_dedup=True)

        assert result["success"] is True
        assert result["exact_duplicates_removed"] == 1
        assert result["shape_after"]["rows"] == 4

    async def test_exact_dedup_keep_last(self, csv_file):
        from tools.dataset_loader import dataset_loader
        from tools.cleaning.deduplicator import deduplicate_dataset

        load_r = await dataset_loader("local", str(csv_file))
        result = await deduplicate_dataset(
            load_r["output_artifact_key"], exact_dedup=True, dedup_keep="last"
        )

        assert result["success"] is True
        assert result["exact_duplicates_removed"] == 1

    async def test_no_duplicates_no_rows_removed(self, tmp_path):
        from tools.dataset_loader import dataset_loader
        from tools.cleaning.deduplicator import deduplicate_dataset

        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        p = tmp_path / "unique.csv"
        df.to_csv(p, index=False)
        load_r = await dataset_loader("local", str(p))
        result = await deduplicate_dataset(load_r["output_artifact_key"])

        assert result["success"] is True
        assert result["exact_duplicates_removed"] == 0
        assert result["rows_removed"] == 0

    async def test_fuzzy_dedup_removes_near_duplicates(self, tmp_path):
        from tools.dataset_loader import dataset_loader
        from tools.cleaning.deduplicator import deduplicate_dataset

        df = pd.DataFrame({"name": ["Alice Smith", "Alice Smyth", "Bob Jones"]})
        p = tmp_path / "fuzzy.csv"
        df.to_csv(p, index=False)
        load_r = await dataset_loader("local", str(p))
        result = await deduplicate_dataset(
            load_r["output_artifact_key"],
            exact_dedup=False,
            fuzzy_dedup=True,
            fuzzy_columns=["name"],
            fuzzy_threshold=0.8,
        )

        assert result["success"] is True
        assert result["fuzzy_duplicates_removed"] >= 1

    async def test_fuzzy_dedup_without_columns_produces_warning(self, csv_file):
        from tools.dataset_loader import dataset_loader
        from tools.cleaning.deduplicator import deduplicate_dataset

        load_r = await dataset_loader("local", str(csv_file))
        result = await deduplicate_dataset(
            load_r["output_artifact_key"],
            fuzzy_dedup=True,
            fuzzy_columns=None,
        )

        assert result["success"] is True
        assert any("fuzzy" in w.lower() for w in result["warnings"])

    async def test_missing_artifact_returns_failure(self):
        from tools.cleaning.deduplicator import deduplicate_dataset

        result = await deduplicate_dataset("nonexistent/v1/dataset")

        assert result["success"] is False

    async def test_confidence_lower_for_fuzzy(self, tmp_path):
        from tools.dataset_loader import dataset_loader
        from tools.cleaning.deduplicator import deduplicate_dataset

        df = pd.DataFrame({"name": ["Alice", "Bob"]})
        p = tmp_path / "c.csv"
        df.to_csv(p, index=False)
        load_r = await dataset_loader("local", str(p))

        exact_r = await deduplicate_dataset(load_r["output_artifact_key"], exact_dedup=True)
        assert exact_r["confidence"] == 1.0

        # Reload for next call
        load_r2 = await dataset_loader("local", str(p))
        fuzzy_r = await deduplicate_dataset(
            load_r2["output_artifact_key"],
            exact_dedup=False,
            fuzzy_dedup=True,
            fuzzy_columns=["name"],
        )
        assert fuzzy_r["confidence"] < 1.0


# ===========================================================================
# 6. merge_datasets
# ===========================================================================

class TestMerge:
    async def _load_two(self, csv_file, secondary_csv, ctx):
        """
        Load primary + secondary datasets into the same mock_ctx so that:
        - Primary is saved at dataset_loader/v1/dataset
        - Secondary is saved at dataset_loader/v2/dataset
        - ctx.state has secondary_datasets["depts"] registered
        Returns the primary artifact key.
        """
        from tools.dataset_loader import dataset_loader

        r1 = await dataset_loader("local", str(csv_file), tool_context=ctx)
        await dataset_loader(
            "local", str(secondary_csv),
            is_secondary=True, secondary_name="depts",
            tool_context=ctx,
        )
        return r1["output_artifact_key"]

    async def test_left_join_success(self, csv_file, secondary_csv, mock_ctx):
        from tools.merge_tool import merge_datasets

        primary_key = await self._load_two(csv_file, secondary_csv, mock_ctx)
        result = await merge_datasets(primary_key, "depts", "Name", "left", tool_context=mock_ctx)

        assert result["success"] is True
        assert result["merged_shape"]["cols"] > 4  # new column Dept added

    async def test_inner_join_drops_unmatched(self, csv_file, secondary_csv, mock_ctx):
        from tools.merge_tool import merge_datasets

        primary_key = await self._load_two(csv_file, secondary_csv, mock_ctx)
        result = await merge_datasets(primary_key, "depts", "Name", "inner", tool_context=mock_ctx)

        assert result["success"] is True
        df = parquet_bytes_to_df((ARTIFACTS_DIR / result["output_artifact_key"]).read_bytes())
        assert len(df) <= 5

    async def test_match_rate_computed(self, csv_file, secondary_csv, mock_ctx):
        from tools.merge_tool import merge_datasets

        primary_key = await self._load_two(csv_file, secondary_csv, mock_ctx)
        result = await merge_datasets(primary_key, "depts", "Name", tool_context=mock_ctx)

        assert 0.0 <= result["match_rate"] <= 1.0

    async def test_missing_secondary_name_returns_failure(self, csv_file, secondary_csv, mock_ctx):
        from tools.merge_tool import merge_datasets

        primary_key = await self._load_two(csv_file, secondary_csv, mock_ctx)
        # Use a fresh ctx with no secondary datasets registered
        empty_ctx = _MockCtxWithState(AgentSessionState())
        result = await merge_datasets(primary_key, "no_such_name", "Name", tool_context=empty_ctx)

        assert result["success"] is False
        assert "not found" in result["error_message"].lower()

    async def test_missing_join_key_in_primary_returns_failure(self, csv_file, secondary_csv, mock_ctx):
        from tools.merge_tool import merge_datasets

        primary_key = await self._load_two(csv_file, secondary_csv, mock_ctx)
        result = await merge_datasets(primary_key, "depts", "nonexistent_col", tool_context=mock_ctx)

        assert result["success"] is False

    async def test_invalid_join_type_returns_failure(self, csv_file, secondary_csv, mock_ctx):
        from tools.merge_tool import merge_datasets

        primary_key = await self._load_two(csv_file, secondary_csv, mock_ctx)
        result = await merge_datasets(
            primary_key, "depts", "Name", join_type="cross", tool_context=mock_ctx
        )

        assert result["success"] is False

    async def test_new_columns_tracked_in_lineage(self, csv_file, secondary_csv, mock_ctx):
        from tools.merge_tool import merge_datasets

        primary_key = await self._load_two(csv_file, secondary_csv, mock_ctx)
        result = await merge_datasets(primary_key, "depts", "Name", tool_context=mock_ctx)

        assert "Dept" in result["log"]["column_lineage"]["columns_added"]


# ===========================================================================
# 7. validate_dataset
# ===========================================================================

class TestValidator:
    async def test_clean_dataset_gets_high_score(self, tmp_path):
        from tools.dataset_loader import dataset_loader
        from tools.validator import validate_dataset

        df = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": ["x", "y", "z", "w", "v"]})
        p = tmp_path / "clean.csv"
        df.to_csv(p, index=False)
        load_r = await dataset_loader("local", str(p))
        result = await validate_dataset(load_r["output_artifact_key"])

        assert result["success"] is True
        assert result["quality_score"] == 100.0
        assert result["passed"] is True

    async def test_dirty_dataset_gets_lower_score(self, csv_file):
        from tools.dataset_loader import dataset_loader
        from tools.validator import validate_dataset

        load_r = await dataset_loader("local", str(csv_file))
        result = await validate_dataset(load_r["output_artifact_key"])

        assert result["success"] is True
        assert result["quality_score"] < 100.0

    async def test_issues_list_populated_for_dirty_data(self, tmp_path):
        from tools.dataset_loader import dataset_loader
        from tools.validator import validate_dataset

        df = pd.DataFrame(
            {
                "col_a": [None, None, None, None, 1],  # 80% missing
                "col_b": [1, 1, 1, 1, 1],              # constant
            }
        )
        p = tmp_path / "dirty.csv"
        df.to_csv(p, index=False)
        load_r = await dataset_loader("local", str(p))
        result = await validate_dataset(load_r["output_artifact_key"])

        assert result["success"] is True
        assert len(result["issues"]) > 0

    async def test_duplicate_rows_flagged(self, csv_file):
        from tools.dataset_loader import dataset_loader
        from tools.validator import validate_dataset

        load_r = await dataset_loader("local", str(csv_file))
        result = await validate_dataset(load_r["output_artifact_key"])

        issue_types = {i["issue_type"] for i in result["issues"]}
        assert "near_duplicates" in issue_types

    async def test_passed_threshold_is_70(self, tmp_path):
        from tools.dataset_loader import dataset_loader
        from tools.validator import validate_dataset

        # 75% of values missing in one column  → big penalty → score < 70 → not passed
        df = pd.DataFrame({"x": [None, None, None, 1.0]})
        p = tmp_path / "lowqual.csv"
        df.to_csv(p, index=False)
        load_r = await dataset_loader("local", str(p))
        result = await validate_dataset(load_r["output_artifact_key"])

        # Score should be < 70 → passed=False
        if result["quality_score"] < 70:
            assert result["passed"] is False
        else:
            assert result["passed"] is True

    async def test_report_artifact_saved(self, csv_file):
        from tools.dataset_loader import dataset_loader
        from tools.validator import validate_dataset

        load_r = await dataset_loader("local", str(csv_file))
        result = await validate_dataset(load_r["output_artifact_key"])

        assert result["success"] is True
        report_path = ARTIFACTS_DIR / "validate_dataset/v1/report"
        assert report_path.exists()

    async def test_missing_artifact_returns_failure(self):
        from tools.validator import validate_dataset

        result = await validate_dataset("nonexistent/v1/dataset")

        assert result["success"] is False

    async def test_dataset_unchanged_after_validation(self, csv_file):
        from tools.dataset_loader import dataset_loader
        from tools.validator import validate_dataset

        load_r = await dataset_loader("local", str(csv_file))
        key = load_r["output_artifact_key"]
        result = await validate_dataset(key)

        # output_artifact_key should be the same (validation is read-only)
        assert result["output_artifact_key"] == key


# ===========================================================================
# 8. generate_output
# ===========================================================================

class TestOutputGenerator:
    async def test_output_success(self, csv_file, mock_ctx):
        from tools.dataset_loader import dataset_loader
        from tools.output_generator import generate_output

        load_r = await dataset_loader("local", str(csv_file))
        result = await generate_output(load_r["output_artifact_key"])

        assert result["success"] is True
        assert result["csv_artifact_key"] is not None
        assert result["log_artifact_key"] is not None
        assert result["report_artifact_key"] is not None

    async def test_csv_artifact_is_valid_csv(self, csv_file):
        from tools.dataset_loader import dataset_loader
        from tools.output_generator import generate_output

        load_r = await dataset_loader("local", str(csv_file))
        result = await generate_output(load_r["output_artifact_key"])

        csv_bytes = (ARTIFACTS_DIR / result["csv_artifact_key"]).read_bytes()
        df = pd.read_csv(__import__("io").BytesIO(csv_bytes))
        assert len(df) == 5
        assert "Name" in df.columns

    async def test_log_artifact_is_valid_json(self, csv_file):
        from tools.dataset_loader import dataset_loader
        from tools.output_generator import generate_output
        import json

        load_r = await dataset_loader("local", str(csv_file))
        result = await generate_output(load_r["output_artifact_key"])

        log_bytes = (ARTIFACTS_DIR / result["log_artifact_key"]).read_bytes()
        logs = json.loads(log_bytes)
        assert isinstance(logs, list)

    async def test_report_artifact_is_markdown(self, csv_file):
        from tools.dataset_loader import dataset_loader
        from tools.output_generator import generate_output

        load_r = await dataset_loader("local", str(csv_file))
        result = await generate_output(load_r["output_artifact_key"])

        report_bytes = (ARTIFACTS_DIR / result["report_artifact_key"]).read_bytes()
        report_text = report_bytes.decode("utf-8")
        assert "# Data Cleaning Quality Report" in report_text
        assert "## Transformation Log" in report_text

    async def test_report_includes_summary_stats(self, csv_file):
        from tools.dataset_loader import dataset_loader
        from tools.output_generator import generate_output

        load_r = await dataset_loader("local", str(csv_file))
        result = await generate_output(
            load_r["output_artifact_key"], include_summary_stats=True
        )

        report_bytes = (ARTIFACTS_DIR / result["report_artifact_key"]).read_bytes()
        assert "## Summary Statistics" in report_bytes.decode("utf-8")

    async def test_report_omits_summary_stats_when_disabled(self, csv_file):
        from tools.dataset_loader import dataset_loader
        from tools.output_generator import generate_output

        load_r = await dataset_loader("local", str(csv_file))
        result = await generate_output(
            load_r["output_artifact_key"], include_summary_stats=False
        )

        report_bytes = (ARTIFACTS_DIR / result["report_artifact_key"]).read_bytes()
        assert "## Summary Statistics" not in report_bytes.decode("utf-8")

    async def test_log_contains_loader_entry(self, csv_file, mock_ctx):
        from tools.dataset_loader import dataset_loader
        from tools.output_generator import generate_output
        import json

        # Both calls share mock_ctx so the loader's TransformationLog is visible
        # to generate_output via the accumulated session state.
        load_r = await dataset_loader("local", str(csv_file), tool_context=mock_ctx)
        result = await generate_output(load_r["output_artifact_key"], tool_context=mock_ctx)

        log_bytes = (ARTIFACTS_DIR / result["log_artifact_key"]).read_bytes()
        logs = json.loads(log_bytes)
        step_names = [l["step_name"] for l in logs]
        assert "dataset_loader" in step_names

    async def test_missing_artifact_returns_failure(self):
        from tools.output_generator import generate_output

        result = await generate_output("nonexistent/v1/dataset")

        assert result["success"] is False


# ===========================================================================
# 9. End-to-end pipeline smoke test
# ===========================================================================

class TestEndToEndPipeline:
    async def test_full_pipeline(self, csv_file, mock_ctx):
        """Run all 7 tool steps sharing one mock_ctx and verify the final output."""
        from tools.dataset_loader import dataset_loader
        from tools.data_profiler import profile_dataset
        from tools.cleaning.missing_handler import handle_missing_values
        from tools.cleaning.standardizer import standardize_formats
        from tools.cleaning.deduplicator import deduplicate_dataset
        from tools.validator import validate_dataset
        from tools.output_generator import generate_output
        import json

        ctx = mock_ctx  # shared context accumulates state across all steps

        # Step 1: Load
        r = await dataset_loader("local", str(csv_file), tool_context=ctx)
        assert r["success"], r["error_message"]
        key = r["output_artifact_key"]

        # Step 2: Profile
        r = await profile_dataset(key, tool_context=ctx)
        assert r["success"], r["error_message"]

        # Step 3: Handle missing
        r = await handle_missing_values(
            key,
            strategy_config={"Age": "median", "Salary": "mean", "Name": "mode"},
            tool_context=ctx,
        )
        assert r["success"], r["error_message"]
        key = r["output_artifact_key"]

        # Step 4: Standardize
        r = await standardize_formats(key, tool_context=ctx)
        assert r["success"], r["error_message"]
        key = r["output_artifact_key"]

        # Step 5: Deduplicate
        r = await deduplicate_dataset(key, tool_context=ctx)
        assert r["success"], r["error_message"]
        key = r["output_artifact_key"]
        assert r["exact_duplicates_removed"] == 1

        # Step 6: Validate
        r = await validate_dataset(key, tool_context=ctx)
        assert r["success"], r["error_message"]
        assert r["quality_score"] == 100.0

        # Step 7: Generate output
        r = await generate_output(key, tool_context=ctx)
        assert r["success"], r["error_message"]

        # Final CSV should have 4 rows (1 deduped) with snake_case headers
        csv_bytes = (ARTIFACTS_DIR / r["csv_artifact_key"]).read_bytes()
        df_final = pd.read_csv(__import__("io").BytesIO(csv_bytes))
        assert len(df_final) == 4
        assert "name" in df_final.columns  # snake_case applied

        # Logs JSON must be parseable and contain all step entries
        log_bytes = (ARTIFACTS_DIR / r["log_artifact_key"]).read_bytes()
        logs = json.loads(log_bytes)
        assert len(logs) >= 6  # loader + profiler + missing + std + dedup + validate
        step_names = {l["step_name"] for l in logs}
        assert "dataset_loader" in step_names
        assert "deduplicate_dataset" in step_names

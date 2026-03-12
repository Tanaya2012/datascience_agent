"""
Edge-case and boundary tests for all 8 pipeline tools.

These tests supplement test_tools.py by targeting untested code paths:
- Degenerate inputs (empty datasets, single-row datasets, all-null columns)
- Join type variations (right, outer) in merge_datasets
- Deduplicator: all-rows-identical, alternative fuzzy algorithms
- Missing handler: drop_threshold boundary values (0.0 and 1.0)
- Standardizer: snake_case idempotence, mixed date formats
- Validator: constant column detection, quality score clamping
- Realistic pipeline on a sales dataset
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import pandas as pd
import pytest

from tools.artifact_utils import (
    ARTIFACTS_DIR,
    df_to_parquet_bytes,
    parquet_bytes_to_df,
)
from tools.schemas import AgentSessionState, DatasetVersion
from datetime import datetime


# ---------------------------------------------------------------------------
# Helpers (copied from test_tools.py to keep this file self-contained)
# ---------------------------------------------------------------------------

def _save_local(key: str, data: bytes) -> None:
    path = ARTIFACTS_DIR / key
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


class _MockCtxWithState:
    """MockCtx pre-seeded with an AgentSessionState."""

    def __init__(self, state: AgentSessionState):
        from tools.artifact_utils import SESSION_STATE_KEY
        self.state = {SESSION_STATE_KEY: state.model_dump(mode="json")}

    async def save_artifact(self, **_):
        raise RuntimeError("no ADK")

    async def load_artifact(self, **_):
        raise RuntimeError("no ADK")


# ===========================================================================
# 1. dataset_loader — degenerate inputs
# ===========================================================================

class TestDatasetLoaderEdgeCases:
    async def test_load_empty_csv(self, empty_csv):
        from tools.dataset_loader import dataset_loader

        result = await dataset_loader("local", str(empty_csv))

        assert result["success"] is True
        assert result["shape_after"]["rows"] == 0
        assert result["shape_after"]["cols"] == 3

    async def test_load_single_row_csv(self, single_row_csv):
        from tools.dataset_loader import dataset_loader

        result = await dataset_loader("local", str(single_row_csv))

        assert result["success"] is True
        assert result["shape_after"]["rows"] == 1

    async def test_secondary_state_registered(self, csv_file, secondary_csv, mock_ctx):
        """Verify that loading as secondary actually registers the dataset in session state."""
        from tools.dataset_loader import dataset_loader
        from tools.artifact_utils import get_session_state

        await dataset_loader(
            "local", str(secondary_csv),
            is_secondary=True, secondary_name="lookup",
            tool_context=mock_ctx,
        )

        state = get_session_state(mock_ctx)
        assert "lookup" in state.secondary_datasets


# ===========================================================================
# 2. profile_dataset — edge cases
# ===========================================================================

class TestDataProfilerEdgeCases:
    async def test_profile_empty_dataset(self, empty_csv):
        from tools.dataset_loader import dataset_loader
        from tools.data_profiler import profile_dataset

        load_r = await dataset_loader("local", str(empty_csv))
        result = await profile_dataset(load_r["output_artifact_key"])

        assert result["success"] is True
        assert result["profile"]["shape"][0] == 0  # 0 rows

    async def test_profile_single_row(self, single_row_csv):
        from tools.dataset_loader import dataset_loader
        from tools.data_profiler import profile_dataset

        load_r = await dataset_loader("local", str(single_row_csv))
        result = await profile_dataset(load_r["output_artifact_key"])

        assert result["success"] is True
        assert result["profile"]["shape"][0] == 1

    async def test_profile_all_null_column(self, all_null_col_csv):
        from tools.dataset_loader import dataset_loader
        from tools.data_profiler import profile_dataset

        load_r = await dataset_loader("local", str(all_null_col_csv))
        result = await profile_dataset(load_r["output_artifact_key"])

        assert result["success"] is True
        value_profile = next(
            c for c in result["profile"]["column_profiles"] if c["name"] == "value"
        )
        assert value_profile["missing_count"] == 3  # all 3 rows are null

    async def test_profile_duplicate_count_accuracy(self, csv_file):
        from tools.dataset_loader import dataset_loader
        from tools.data_profiler import profile_dataset

        # sample_df has 1 exact duplicate row (Alice, 30.0, 50000.0, 2020-01-15)
        load_r = await dataset_loader("local", str(csv_file))
        result = await profile_dataset(load_r["output_artifact_key"])

        assert result["profile"]["duplicate_count"] >= 1

    async def test_profile_outlier_detection(self, tmp_path):
        from tools.dataset_loader import dataset_loader
        from tools.data_profiler import profile_dataset

        # Salary column with one extreme outlier (10× others)
        df = pd.DataFrame({"salary": [50_000, 52_000, 49_000, 51_000, 500_000]})
        p = tmp_path / "outliers.csv"
        df.to_csv(p, index=False)
        load_r = await dataset_loader("local", str(p))
        result = await profile_dataset(load_r["output_artifact_key"])

        assert result["success"] is True
        salary_profile = result["profile"]["column_profiles"][0]
        # Outlier detection may surface in anomalies or outlier_count
        assert "outlier_count" in salary_profile or "anomalies" in salary_profile


# ===========================================================================
# 3. handle_missing_values — boundary values
# ===========================================================================

class TestMissingHandlerEdgeCases:
    async def test_drop_threshold_zero_drops_all_cols_with_any_missing(self, csv_file):
        """threshold=0.0 → drop every column that has ≥0% missing (i.e., any missing at all)."""
        from tools.dataset_loader import dataset_loader
        from tools.cleaning.missing_handler import handle_missing_values

        load_r = await dataset_loader("local", str(csv_file))
        result = await handle_missing_values(
            load_r["output_artifact_key"],
            strategy_config={},
            drop_threshold=0.0,
        )

        assert result["success"] is True
        df_out = parquet_bytes_to_df((ARTIFACTS_DIR / result["output_artifact_key"]).read_bytes())
        # Name, Age, Salary all have ≥1 missing value — must be dropped
        assert "Age" not in df_out.columns
        assert "Salary" not in df_out.columns

    async def test_drop_threshold_one_keeps_partial_null_columns(self, csv_file):
        """threshold=1.0 → only drop columns that are 100% null; partial-null columns stay."""
        from tools.dataset_loader import dataset_loader
        from tools.cleaning.missing_handler import handle_missing_values

        load_r = await dataset_loader("local", str(csv_file))
        result = await handle_missing_values(
            load_r["output_artifact_key"],
            strategy_config={},
            drop_threshold=1.0,
        )

        assert result["success"] is True
        df_out = parquet_bytes_to_df((ARTIFACTS_DIR / result["output_artifact_key"]).read_bytes())
        # Age and Salary are only partially null — must NOT be dropped
        assert "Age" in df_out.columns
        assert "Salary" in df_out.columns

    async def test_all_null_column_drops_with_threshold(self, all_null_col_csv):
        """A 100%-null column (fraction=1.0) should be dropped when threshold < 1.0.
        The implementation uses strict `>` comparison, so threshold must be < 1.0."""
        from tools.dataset_loader import dataset_loader
        from tools.cleaning.missing_handler import handle_missing_values

        load_r = await dataset_loader("local", str(all_null_col_csv))
        result = await handle_missing_values(
            load_r["output_artifact_key"],
            strategy_config={},
            drop_threshold=0.9,  # 1.0 > 0.9 → column gets dropped
        )

        assert result["success"] is True
        assert "value" in result["columns_dropped"]

    async def test_mode_on_all_null_column_no_error(self, all_null_col_csv):
        """mode imputation on a column with no data should not crash; column stays null.
        drop_threshold > 1.0 prevents the column from being auto-dropped first."""
        from tools.dataset_loader import dataset_loader
        from tools.cleaning.missing_handler import handle_missing_values

        load_r = await dataset_loader("local", str(all_null_col_csv))
        result = await handle_missing_values(
            load_r["output_artifact_key"],
            strategy_config={"value": "mode"},
            drop_threshold=1.5,  # bypass auto-drop so strategy is applied to all-null column
        )

        assert result["success"] is True
        df_out = parquet_bytes_to_df((ARTIFACTS_DIR / result["output_artifact_key"]).read_bytes())
        # Column is all-null; mode is undefined (empty Series) → column remains null
        assert df_out["value"].isna().all()

    async def test_ffill_all_null_column_no_error(self, all_null_col_csv):
        """ffill on all-null column should not crash; column stays null.
        drop_threshold > 1.0 prevents the column from being auto-dropped first."""
        from tools.dataset_loader import dataset_loader
        from tools.cleaning.missing_handler import handle_missing_values

        load_r = await dataset_loader("local", str(all_null_col_csv))
        result = await handle_missing_values(
            load_r["output_artifact_key"],
            strategy_config={"value": "ffill"},
            drop_threshold=1.5,  # bypass auto-drop
        )

        assert result["success"] is True
        df_out = parquet_bytes_to_df((ARTIFACTS_DIR / result["output_artifact_key"]).read_bytes())
        assert df_out["value"].isna().all()

    async def test_drop_row_multiple_columns_removes_each_row_once(self, tmp_path):
        """A row missing in two strategy columns should be removed only once.
        drop_threshold=0.9 prevents column 'b' (67% null) from being auto-dropped."""
        from tools.dataset_loader import dataset_loader
        from tools.cleaning.missing_handler import handle_missing_values

        df = pd.DataFrame({"a": [1.0, None, 3.0], "b": [None, None, 6.0]})
        p = tmp_path / "multi_null.csv"
        df.to_csv(p, index=False)
        load_r = await dataset_loader("local", str(p))
        result = await handle_missing_values(
            load_r["output_artifact_key"],
            strategy_config={"a": "drop_row", "b": "drop_row"},
            drop_threshold=0.9,  # 0.67 < 0.9 → 'b' kept for strategy, not auto-dropped
        )

        assert result["success"] is True
        # drop_row on 'a' removes row 1 (a=None); drop_row on 'b' removes row 0 (b=None)
        # row 2 survives both; rows_removed = 3 - 1 = 2
        assert result["rows_removed"] == 2
        assert result["shape_after"]["rows"] == 1


# ===========================================================================
# 4. standardize_formats — edge cases
# ===========================================================================

class TestStandardizerEdgeCases:
    async def test_snake_case_already_snake_idempotent(self, tmp_path):
        """Headers already in snake_case → no renames, cells_modified=0."""
        from tools.dataset_loader import dataset_loader
        from tools.cleaning.standardizer import standardize_formats

        df = pd.DataFrame({"first_name": ["a"], "last_name": ["b"], "age_years": [1]})
        p = tmp_path / "already_snake.csv"
        df.to_csv(p, index=False)
        load_r = await dataset_loader("local", str(p))
        result = await standardize_formats(load_r["output_artifact_key"], normalize_headers=True)

        assert result["success"] is True
        df_out = parquet_bytes_to_df((ARTIFACTS_DIR / result["output_artifact_key"]).read_bytes())
        assert list(df_out.columns) == ["first_name", "last_name", "age_years"]
        assert result["log"]["column_lineage"]["columns_renamed"] == {}

    async def test_date_parsing_mixed_formats(self, tmp_path):
        """Column with both ISO and US date formats should parse to datetime."""
        from tools.dataset_loader import dataset_loader
        from tools.cleaning.standardizer import standardize_formats

        df = pd.DataFrame({"event_date": ["2023-01-15", "01/22/2023", "03/05/2023"]})
        p = tmp_path / "mixed_dates.csv"
        df.to_csv(p, index=False)
        load_r = await dataset_loader("local", str(p))
        result = await standardize_formats(
            load_r["output_artifact_key"], parse_dates=True, normalize_headers=False
        )

        assert result["success"] is True
        df_out = parquet_bytes_to_df((ARTIFACTS_DIR / result["output_artifact_key"]).read_bytes())
        assert pd.api.types.is_datetime64_any_dtype(df_out["event_date"])

    async def test_all_null_column_unchanged_by_standardizer(self, all_null_col_csv):
        """An all-null column should not cause errors and should remain null."""
        from tools.dataset_loader import dataset_loader
        from tools.cleaning.standardizer import standardize_formats

        load_r = await dataset_loader("local", str(all_null_col_csv))
        result = await standardize_formats(
            load_r["output_artifact_key"],
            parse_dates=True,
            parse_numerics=True,
            parse_currency=True,
        )

        assert result["success"] is True
        df_out = parquet_bytes_to_df((ARTIFACTS_DIR / result["output_artifact_key"]).read_bytes())
        assert df_out["value"].isna().all()

    async def test_numeric_coercion_on_already_numeric_column(self, tmp_path):
        """parse_numerics=True on a column already stored as float → cells_modified=0."""
        from tools.dataset_loader import dataset_loader
        from tools.cleaning.standardizer import standardize_formats

        df = pd.DataFrame({"score": [1.0, 2.0, 3.0]})
        p = tmp_path / "already_numeric.csv"
        df.to_csv(p, index=False)
        load_r = await dataset_loader("local", str(p))
        result = await standardize_formats(
            load_r["output_artifact_key"],
            parse_numerics=True,
            normalize_headers=False,
        )

        assert result["success"] is True
        assert result["cells_modified"] == 0

    async def test_currency_with_no_decimal(self, tmp_path):
        """Currency strings without decimal places should parse correctly."""
        from tools.dataset_loader import dataset_loader
        from tools.cleaning.standardizer import standardize_formats

        df = pd.DataFrame({"price": ["$500", "$1,000", "$250"]})
        p = tmp_path / "nodecimal.csv"
        df.to_csv(p, index=False)
        load_r = await dataset_loader("local", str(p))
        result = await standardize_formats(
            load_r["output_artifact_key"], parse_currency=True, normalize_headers=False
        )

        assert result["success"] is True
        df_out = parquet_bytes_to_df((ARTIFACTS_DIR / result["output_artifact_key"]).read_bytes())
        assert pd.api.types.is_numeric_dtype(df_out["price"])
        assert df_out["price"].iloc[1] == pytest.approx(1000.0)


# ===========================================================================
# 5. deduplicate_dataset — edge cases
# ===========================================================================

class TestDeduplicatorEdgeCases:
    async def test_exact_dedup_all_rows_identical(self, tmp_path):
        """5 identical rows → 1 row remains with dedup_keep='first'."""
        from tools.dataset_loader import dataset_loader
        from tools.cleaning.deduplicator import deduplicate_dataset

        df = pd.DataFrame({"a": [1, 1, 1, 1, 1], "b": ["x", "x", "x", "x", "x"]})
        p = tmp_path / "alldup.csv"
        df.to_csv(p, index=False)
        load_r = await dataset_loader("local", str(p))
        result = await deduplicate_dataset(load_r["output_artifact_key"], exact_dedup=True)

        assert result["success"] is True
        assert result["exact_duplicates_removed"] == 4
        assert result["shape_after"]["rows"] == 1

    async def test_fuzzy_dedup_jaro_winkler(self, tmp_path):
        """jaro_winkler algorithm should identify near-duplicate names."""
        from tools.dataset_loader import dataset_loader
        from tools.cleaning.deduplicator import deduplicate_dataset

        df = pd.DataFrame({"name": ["Jonathan", "Johnathan", "Bob"]})
        p = tmp_path / "jaro.csv"
        df.to_csv(p, index=False)
        load_r = await dataset_loader("local", str(p))
        result = await deduplicate_dataset(
            load_r["output_artifact_key"],
            exact_dedup=False,
            fuzzy_dedup=True,
            fuzzy_columns=["name"],
            fuzzy_threshold=0.85,
            fuzzy_algorithm="jaro_winkler",
        )

        assert result["success"] is True
        assert result["fuzzy_duplicates_removed"] >= 1

    async def test_fuzzy_dedup_partial_ratio(self, tmp_path):
        """partial_ratio algorithm should work without errors."""
        from tools.dataset_loader import dataset_loader
        from tools.cleaning.deduplicator import deduplicate_dataset

        df = pd.DataFrame({"name": ["Alice Smith", "Smith Alice", "Bob Jones"]})
        p = tmp_path / "partial.csv"
        df.to_csv(p, index=False)
        load_r = await dataset_loader("local", str(p))
        result = await deduplicate_dataset(
            load_r["output_artifact_key"],
            exact_dedup=False,
            fuzzy_dedup=True,
            fuzzy_columns=["name"],
            fuzzy_threshold=0.8,
            fuzzy_algorithm="partial_ratio",
        )

        assert result["success"] is True

    async def test_rows_removed_equals_exact_plus_fuzzy(self, tmp_path):
        """rows_removed should equal exact_duplicates_removed + fuzzy_duplicates_removed."""
        from tools.dataset_loader import dataset_loader
        from tools.cleaning.deduplicator import deduplicate_dataset

        df = pd.DataFrame({"name": ["Alice", "Alice", "Alice Smith", "Alice Smyth", "Bob"]})
        p = tmp_path / "mixed_dup.csv"
        df.to_csv(p, index=False)
        load_r = await dataset_loader("local", str(p))
        result = await deduplicate_dataset(
            load_r["output_artifact_key"],
            exact_dedup=True,
            fuzzy_dedup=True,
            fuzzy_columns=["name"],
            fuzzy_threshold=0.85,
        )

        assert result["success"] is True
        assert result["rows_removed"] == (
            result["exact_duplicates_removed"] + result["fuzzy_duplicates_removed"]
        )

    async def test_deduplicate_single_row(self, single_row_csv):
        """Single-row dataset has no duplicates to remove."""
        from tools.dataset_loader import dataset_loader
        from tools.cleaning.deduplicator import deduplicate_dataset

        load_r = await dataset_loader("local", str(single_row_csv))
        result = await deduplicate_dataset(load_r["output_artifact_key"], exact_dedup=True)

        assert result["success"] is True
        assert result["exact_duplicates_removed"] == 0
        assert result["shape_after"]["rows"] == 1


# ===========================================================================
# 6. merge_datasets — right and outer joins
# ===========================================================================

class TestMergeEdgeCases:
    async def _load_two(self, csv_file, secondary_csv, ctx):
        from tools.dataset_loader import dataset_loader

        r1 = await dataset_loader("local", str(csv_file), tool_context=ctx)
        await dataset_loader(
            "local", str(secondary_csv),
            is_secondary=True, secondary_name="depts",
            tool_context=ctx,
        )
        return r1["output_artifact_key"]

    async def test_right_join_keeps_all_secondary_rows(self, csv_file, secondary_csv, mock_ctx):
        """Right join: all rows from secondary must appear in output."""
        from tools.merge_tool import merge_datasets

        primary_key = await self._load_two(csv_file, secondary_csv, mock_ctx)
        result = await merge_datasets(
            primary_key, "depts", "Name", join_type="right", tool_context=mock_ctx
        )

        assert result["success"] is True
        df = parquet_bytes_to_df((ARTIFACTS_DIR / result["output_artifact_key"]).read_bytes())
        # secondary has 3 rows; right join keeps all 3
        assert len(df) >= 3

    async def test_outer_join_keeps_all_rows_with_nulls(self, csv_file, secondary_csv, mock_ctx):
        """Outer join: output must have at least as many rows as both sides combined."""
        from tools.merge_tool import merge_datasets

        primary_key = await self._load_two(csv_file, secondary_csv, mock_ctx)
        result = await merge_datasets(
            primary_key, "depts", "Name", join_type="outer", tool_context=mock_ctx
        )

        assert result["success"] is True
        df = parquet_bytes_to_df((ARTIFACTS_DIR / result["output_artifact_key"]).read_bytes())
        # primary has 5 rows, secondary 3 rows; outer keeps all unique keys
        assert len(df) >= 3

    async def test_merge_no_matching_keys_left_join(self, tmp_path, mock_ctx):
        """Left join with no key overlap → all primary rows kept, secondary cols all NaN."""
        from tools.dataset_loader import dataset_loader
        from tools.merge_tool import merge_datasets

        primary = pd.DataFrame({"id": [1, 2, 3], "val": ["a", "b", "c"]})
        secondary = pd.DataFrame({"id": [10, 20], "label": ["x", "y"]})
        p1 = tmp_path / "primary.csv"
        p2 = tmp_path / "secondary.csv"
        primary.to_csv(p1, index=False)
        secondary.to_csv(p2, index=False)

        r1 = await dataset_loader("local", str(p1), tool_context=mock_ctx)
        await dataset_loader(
            "local", str(p2),
            is_secondary=True, secondary_name="lookup",
            tool_context=mock_ctx,
        )

        result = await merge_datasets(
            r1["output_artifact_key"], "lookup", "id", join_type="left", tool_context=mock_ctx
        )

        assert result["success"] is True
        assert result["match_rate"] == pytest.approx(0.0)
        df = parquet_bytes_to_df((ARTIFACTS_DIR / result["output_artifact_key"]).read_bytes())
        assert len(df) == 3  # all primary rows preserved

    async def test_merge_perfect_match_rate(self, tmp_path, mock_ctx):
        """When all primary keys match secondary, match_rate should be 1.0."""
        from tools.dataset_loader import dataset_loader
        from tools.merge_tool import merge_datasets

        primary = pd.DataFrame({"id": [1, 2, 3], "val": ["a", "b", "c"]})
        secondary = pd.DataFrame({"id": [1, 2, 3], "label": ["x", "y", "z"]})
        p1 = tmp_path / "primary.csv"
        p2 = tmp_path / "secondary.csv"
        primary.to_csv(p1, index=False)
        secondary.to_csv(p2, index=False)

        r1 = await dataset_loader("local", str(p1), tool_context=mock_ctx)
        await dataset_loader(
            "local", str(p2),
            is_secondary=True, secondary_name="labels",
            tool_context=mock_ctx,
        )

        result = await merge_datasets(
            r1["output_artifact_key"], "labels", "id", join_type="left", tool_context=mock_ctx
        )

        assert result["success"] is True
        assert result["match_rate"] == pytest.approx(1.0)


# ===========================================================================
# 7. validate_dataset — edge cases
# ===========================================================================

class TestValidatorEdgeCases:
    async def test_constant_column_detected(self, tmp_path):
        """A column with all identical values should be flagged as an issue."""
        from tools.dataset_loader import dataset_loader
        from tools.validator import validate_dataset

        df = pd.DataFrame({"id": [1, 2, 3, 4, 5], "status": ["active"] * 5})
        p = tmp_path / "constant.csv"
        df.to_csv(p, index=False)
        load_r = await dataset_loader("local", str(p))
        result = await validate_dataset(load_r["output_artifact_key"])

        assert result["success"] is True
        issue_types = {i["issue_type"] for i in result["issues"]}
        assert "constant_column" in issue_types

    async def test_quality_score_not_negative(self, tmp_path):
        """Massive missingness should clamp quality_score to 0.0, not go negative."""
        from tools.dataset_loader import dataset_loader
        from tools.validator import validate_dataset

        # 4 columns all 80% null → enormous penalty
        df = pd.DataFrame(
            {
                "a": [None, None, None, None, 1.0],
                "b": [None, None, None, None, 1.0],
                "c": [None, None, None, None, 1.0],
                "d": [None, None, None, None, 1.0],
            }
        )
        p = tmp_path / "very_dirty.csv"
        df.to_csv(p, index=False)
        load_r = await dataset_loader("local", str(p))
        result = await validate_dataset(load_r["output_artifact_key"])

        assert result["success"] is True
        assert result["quality_score"] >= 0.0

    async def test_quality_score_100_for_perfect_data(self, tmp_path):
        """No missing, no duplicates, no anomalies → quality_score == 100."""
        from tools.dataset_loader import dataset_loader
        from tools.validator import validate_dataset

        df = pd.DataFrame(
            {"id": range(10), "name": [f"Person_{i}" for i in range(10)], "value": range(10)}
        )
        p = tmp_path / "perfect.csv"
        df.to_csv(p, index=False)
        load_r = await dataset_loader("local", str(p))
        result = await validate_dataset(load_r["output_artifact_key"])

        assert result["success"] is True
        assert result["quality_score"] == 100.0
        assert result["passed"] is True

    async def test_validate_single_row(self, single_row_csv):
        """Single-row dataset should validate successfully with a high score."""
        from tools.dataset_loader import dataset_loader
        from tools.validator import validate_dataset

        load_r = await dataset_loader("local", str(single_row_csv))
        result = await validate_dataset(load_r["output_artifact_key"])

        assert result["success"] is True
        assert result["quality_score"] > 0.0

    async def test_validate_empty_dataset(self, empty_csv):
        """Empty dataset (0 rows) should not crash the validator."""
        from tools.dataset_loader import dataset_loader
        from tools.validator import validate_dataset

        load_r = await dataset_loader("local", str(empty_csv))
        result = await validate_dataset(load_r["output_artifact_key"])

        assert result["success"] is True


# ===========================================================================
# 8. generate_output — edge cases
# ===========================================================================

class TestOutputGeneratorEdgeCases:
    async def test_csv_export_preserves_nan_as_empty_cell(self, tmp_path):
        """NaN values in the DataFrame should appear as empty cells in the CSV export."""
        from tools.dataset_loader import dataset_loader
        from tools.output_generator import generate_output

        df = pd.DataFrame({"a": [1.0, None, 3.0], "b": ["x", "y", None]})
        p = tmp_path / "withnan.csv"
        df.to_csv(p, index=False)
        load_r = await dataset_loader("local", str(p))
        result = await generate_output(load_r["output_artifact_key"])

        assert result["success"] is True
        csv_bytes = (ARTIFACTS_DIR / result["csv_artifact_key"]).read_bytes()
        df_out = pd.read_csv(io.BytesIO(csv_bytes))
        assert df_out["a"].isna().sum() == 1
        assert df_out["b"].isna().sum() == 1

    async def test_output_with_empty_dataset(self, empty_csv):
        """Generating output on a 0-row dataset should succeed."""
        from tools.dataset_loader import dataset_loader
        from tools.output_generator import generate_output

        load_r = await dataset_loader("local", str(empty_csv))
        result = await generate_output(load_r["output_artifact_key"])

        assert result["success"] is True
        csv_bytes = (ARTIFACTS_DIR / result["csv_artifact_key"]).read_bytes()
        df_out = pd.read_csv(io.BytesIO(csv_bytes))
        assert len(df_out) == 0

    async def test_log_json_is_list_after_no_cleaning_steps(self, csv_file):
        """When no cleaning tools ran, the logs JSON should still be a valid (possibly empty) list."""
        from tools.dataset_loader import dataset_loader
        from tools.output_generator import generate_output

        load_r = await dataset_loader("local", str(csv_file))
        result = await generate_output(load_r["output_artifact_key"])

        log_bytes = (ARTIFACTS_DIR / result["log_artifact_key"]).read_bytes()
        logs = json.loads(log_bytes)
        assert isinstance(logs, list)

    async def test_report_contains_timestamp(self, csv_file):
        """Quality report markdown should include a UTC timestamp."""
        from tools.dataset_loader import dataset_loader
        from tools.output_generator import generate_output

        load_r = await dataset_loader("local", str(csv_file))
        result = await generate_output(load_r["output_artifact_key"])

        report_bytes = (ARTIFACTS_DIR / result["report_artifact_key"]).read_bytes()
        report_text = report_bytes.decode("utf-8")
        assert "UTC" in report_text


# ===========================================================================
# 9. Realistic pipeline: sales dataset
# ===========================================================================

class TestSalesPipeline:
    async def test_full_sales_pipeline(self, sales_csv, mock_ctx):
        """
        End-to-end pipeline on a realistic sales dataset with:
        - Currency strings → numeric
        - Mixed date formats → datetime
        - One exact duplicate row
        - Missing values in Region and Sale Amount
        - Numeric string (Units Sold) → integer
        """
        from tools.dataset_loader import dataset_loader
        from tools.data_profiler import profile_dataset
        from tools.cleaning.standardizer import standardize_formats
        from tools.cleaning.missing_handler import handle_missing_values
        from tools.cleaning.deduplicator import deduplicate_dataset
        from tools.validator import validate_dataset
        from tools.output_generator import generate_output

        ctx = mock_ctx

        # Load
        r = await dataset_loader("local", str(sales_csv), tool_context=ctx)
        assert r["success"], r.get("error_message")
        key = r["output_artifact_key"]
        assert r["shape_after"]["rows"] == 6

        # Profile
        r = await profile_dataset(key, tool_context=ctx)
        assert r["success"]
        assert r["profile"]["duplicate_count"] >= 1

        # Standardize: snake_case headers + parse currency + parse dates + parse numerics
        r = await standardize_formats(
            key,
            normalize_headers=True,
            parse_currency=True,
            parse_dates=True,
            parse_numerics=True,
            tool_context=ctx,
        )
        assert r["success"], r.get("error_message")
        key = r["output_artifact_key"]

        df = parquet_bytes_to_df((ARTIFACTS_DIR / key).read_bytes())
        assert "customer_name" in df.columns  # snake_case applied
        assert pd.api.types.is_numeric_dtype(df["sale_amount"])
        assert pd.api.types.is_numeric_dtype(df["units_sold"])

        # Handle missing values
        r = await handle_missing_values(
            key,
            strategy_config={"region": "mode", "sale_amount": "median"},
            tool_context=ctx,
        )
        assert r["success"], r.get("error_message")
        key = r["output_artifact_key"]

        df = parquet_bytes_to_df((ARTIFACTS_DIR / key).read_bytes())
        assert df["region"].isna().sum() == 0
        assert df["sale_amount"].isna().sum() == 0

        # Deduplicate
        r = await deduplicate_dataset(key, exact_dedup=True, tool_context=ctx)
        assert r["success"], r.get("error_message")
        key = r["output_artifact_key"]
        assert r["exact_duplicates_removed"] >= 1  # Alice Johnson duplicate row

        # Validate
        r = await validate_dataset(key, tool_context=ctx)
        assert r["success"], r.get("error_message")
        assert r["quality_score"] > 70.0

        # Generate output
        r = await generate_output(key, include_summary_stats=True, tool_context=ctx)
        assert r["success"], r.get("error_message")

        csv_bytes = (ARTIFACTS_DIR / r["csv_artifact_key"]).read_bytes()
        df_final = pd.read_csv(io.BytesIO(csv_bytes))
        assert len(df_final) == 5  # 6 rows - 1 duplicate
        assert "customer_name" in df_final.columns

        report_bytes = (ARTIFACTS_DIR / r["report_artifact_key"]).read_bytes()
        assert "## Summary Statistics" in report_bytes.decode("utf-8")

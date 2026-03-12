"""
Tests for tools/schemas.py and tools/artifact_utils.py.

Verifies:
  - All models instantiate with valid sample data
  - round-trip: model_dump() → model_validate() produces identical output
  - Enum values are correct
  - Pydantic validation catches invalid data
  - artifact_utils profile builders work on a sample DataFrame
"""

from __future__ import annotations

import hashlib
from datetime import datetime

import pandas as pd
import pytest

from tools.schemas import (
    AgentSessionState,
    ArtifactManifest,
    BaseToolResult,
    CategoricalStats,
    ColumnAnomaly,
    ColumnLineage,
    ColumnProfile,
    DataProfilerResult,
    DataQualityIssue,
    DatasetLoaderConfig,
    DatasetLoaderResult,
    DatasetProfile,
    DatasetVersion,
    DatetimeStats,
    DeduplicatorConfig,
    DeduplicatorResult,
    FormatStandardizerConfig,
    FormatStandardizerResult,
    FuzzyAlgorithm,
    InferredDataType,
    IssueSeverity,
    IssueType,
    JoinType,
    MergeConfig,
    MergeResult,
    MissingHandlerConfig,
    MissingHandlerResult,
    MissingStrategy,
    NumericStats,
    OutputGeneratorConfig,
    OutputGeneratorResult,
    PipelineStatus,
    PlannedTask,
    ShapeInfo,
    TaskStatus,
    TaskType,
    TransformationLog,
    ValidatorConfig,
    ValidatorResult,
    DataProfilerConfig,
)
from tools.artifact_utils import (
    build_column_profile,
    build_dataset_profile,
    compute_checksum,
    df_to_parquet_bytes,
    make_artifact_key,
    make_schema_digest,
    next_version,
    parquet_bytes_to_df,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [25, 30, None, 40, 25],
            "name": ["Alice", "Bob", "Charlie", "Alice", None],
            "salary": [50000.0, 60000.0, 70000.0, None, 50000.0],
            "joined": ["2020-01-01", "2019-06-15", "2021-03-20", "2018-11-01", "2020-01-01"],
        }
    )


def _sample_dataset_version() -> DatasetVersion:
    return DatasetVersion(
        artifact_key="load/v1/dataset",
        step_name="dataset_loader",
        version=1,
        shape=(100, 5),
        checksum="abc123",
        schema_digest="def456",
        created_at=datetime(2024, 1, 1, 12, 0, 0),
        input_artifact_key=None,
    )


def _sample_transformation_log() -> TransformationLog:
    return TransformationLog(
        step_name="handle_missing_values",
        task_type=TaskType.handle_missing_values,
        rows_before=100,
        rows_after=95,
        cols_before=5,
        cols_after=5,
        rows_removed=5,
        cells_modified=10,
        checksum_before="aaa",
        checksum_after="bbb",
        confidence=0.9,
    )


# ---------------------------------------------------------------------------
# A. Enum tests
# ---------------------------------------------------------------------------

class TestEnums:
    def test_task_type_values(self):
        assert TaskType.dataset_loader == "dataset_loader"
        assert TaskType.generate_output == "generate_output"
        assert len(TaskType) == 8

    def test_pipeline_status_values(self):
        assert set(PipelineStatus) == {
            PipelineStatus.pending, PipelineStatus.running, PipelineStatus.paused,
            PipelineStatus.completed, PipelineStatus.failed, PipelineStatus.aborted,
        }

    def test_missing_strategy_values(self):
        assert MissingStrategy.drop_row == "drop_row"
        assert MissingStrategy.constant == "constant"

    def test_join_type_values(self):
        assert set(JoinType) == {JoinType.left, JoinType.right, JoinType.inner, JoinType.outer}

    def test_fuzzy_algorithm_values(self):
        assert FuzzyAlgorithm.jaro_winkler == "jaro_winkler"


# ---------------------------------------------------------------------------
# B. Artifact & Version Models
# ---------------------------------------------------------------------------

class TestArtifactModels:
    def test_dataset_version_roundtrip(self):
        v = _sample_dataset_version()
        dumped = v.model_dump()
        restored = DatasetVersion.model_validate(dumped)
        assert restored == v

    def test_artifact_manifest_latest(self):
        v1 = _sample_dataset_version()
        v2 = DatasetVersion(
            artifact_key="load/v2/dataset",
            step_name="dataset_loader",
            version=2,
            shape=(90, 5),
            checksum="xyz",
            schema_digest="uvw",
            created_at=datetime(2024, 1, 2),
        )
        manifest = ArtifactManifest(versions={"dataset_loader": [v1, v2]})
        assert manifest.latest("dataset_loader") == v2
        assert manifest.latest("nonexistent") is None

    def test_artifact_manifest_all_keys(self):
        v = _sample_dataset_version()
        manifest = ArtifactManifest(versions={"dataset_loader": [v]})
        assert "load/v1/dataset" in manifest.all_keys()

    def test_empty_manifest(self):
        manifest = ArtifactManifest()
        assert manifest.all_keys() == []
        assert manifest.latest("any") is None


# ---------------------------------------------------------------------------
# C. Task Plan Models
# ---------------------------------------------------------------------------

class TestTaskPlanModels:
    def test_dataset_loader_config(self):
        cfg = DatasetLoaderConfig(source_type="local", dataset_identifier="data/file.csv")
        assert cfg.task_type == TaskType.dataset_loader
        assert not cfg.is_secondary

    def test_dataset_loader_config_secondary(self):
        cfg = DatasetLoaderConfig(
            source_type="local",
            dataset_identifier="datasets/titanic/train.csv",
            is_secondary=True,
            secondary_name="extra",
        )
        assert cfg.secondary_name == "extra"

    def test_data_profiler_config(self):
        cfg = DataProfilerConfig()
        assert cfg.task_type == TaskType.profile_dataset

    def test_missing_handler_config(self):
        cfg = MissingHandlerConfig(
            strategy_config={"age": MissingStrategy.median, "name": MissingStrategy.mode},
            drop_threshold=0.6,
        )
        assert cfg.drop_threshold == 0.6
        assert cfg.strategy_config["age"] == MissingStrategy.median

    def test_missing_handler_config_drop_threshold_bounds(self):
        with pytest.raises(Exception):
            MissingHandlerConfig(strategy_config={}, drop_threshold=1.5)

    def test_format_standardizer_config_defaults(self):
        cfg = FormatStandardizerConfig()
        assert cfg.normalize_headers is True
        assert cfg.parse_dates is True

    def test_deduplicator_config(self):
        cfg = DeduplicatorConfig(
            fuzzy_dedup=True,
            fuzzy_columns=["name"],
            fuzzy_threshold=0.9,
            fuzzy_algorithm=FuzzyAlgorithm.jaro_winkler,
        )
        assert cfg.fuzzy_threshold == 0.9

    def test_merge_config(self):
        cfg = MergeConfig(secondary_name="lookup", join_key="id", join_type=JoinType.inner)
        assert cfg.join_type == JoinType.inner

    def test_validator_config(self):
        cfg = ValidatorConfig()
        assert cfg.task_type == TaskType.validate_dataset

    def test_output_generator_config(self):
        cfg = OutputGeneratorConfig(include_summary_stats=False)
        assert cfg.output_format == "csv"

    def test_planned_task_roundtrip(self):
        task = PlannedTask(
            task_id=1,
            task_type=TaskType.handle_missing_values,
            description="Impute missing values",
            config=MissingHandlerConfig(
                strategy_config={"age": MissingStrategy.median},
            ),
        )
        dumped = task.model_dump()
        restored = PlannedTask.model_validate(dumped)
        assert restored.task_id == 1
        assert restored.status == TaskStatus.pending

    def test_planned_task_discriminated_union(self):
        """TaskConfig discriminated union must pick the right model from task_type."""
        task = PlannedTask(
            task_id=2,
            task_type=TaskType.deduplicate_dataset,
            description="Remove duplicates",
            config={
                "task_type": "deduplicate_dataset",
                "exact_dedup": True,
                "fuzzy_dedup": False,
            },
        )
        assert isinstance(task.config, DeduplicatorConfig)


# ---------------------------------------------------------------------------
# D. Data Profile Models
# ---------------------------------------------------------------------------

class TestDataProfileModels:
    def test_numeric_stats(self):
        s = NumericStats(mean=5.0, median=4.5, std=1.2, min=1.0, max=10.0, q25=3.0, q75=7.0)
        assert s.mean == 5.0
        assert s.model_dump()["median"] == 4.5

    def test_categorical_stats(self):
        s = CategoricalStats(top_values=[("a", 10), ("b", 5)], entropy=1.5)
        dumped = s.model_dump()
        restored = CategoricalStats.model_validate(dumped)
        assert restored.top_values[0] == ("a", 10)

    def test_datetime_stats(self):
        s = DatetimeStats(min_date="2020-01-01", max_date="2023-12-31", inferred_format="%Y-%m-%d")
        assert s.inferred_format == "%Y-%m-%d"

    def test_column_anomaly(self):
        a = ColumnAnomaly(
            anomaly_type=IssueType.high_missingness,
            description="50% missing",
            affected_count=50,
            affected_pct=50.0,
        )
        assert a.anomaly_type == IssueType.high_missingness

    def test_column_profile_roundtrip(self):
        cp = ColumnProfile(
            name="age",
            inferred_type=InferredDataType.numeric,
            dtype_raw="float64",
            missing_count=5,
            missing_pct=10.0,
            unique_count=45,
            unique_pct=90.0,
            numeric_stats=NumericStats(mean=30.0),
            recommended_strategy=MissingStrategy.median,
        )
        dumped = cp.model_dump()
        restored = ColumnProfile.model_validate(dumped)
        assert restored.name == "age"
        assert restored.recommended_strategy == MissingStrategy.median

    def test_dataset_profile_roundtrip(self):
        dp = DatasetProfile(
            artifact_key="load/v1/dataset",
            shape=(100, 5),
            total_missing_pct=5.0,
            duplicate_count=3,
            duplicate_pct=3.0,
            column_profiles=[],
            quality_score_estimate=85.0,
            high_priority_issues=["Column 'age' has 23% missing"],
        )
        dumped = dp.model_dump()
        restored = DatasetProfile.model_validate(dumped)
        assert restored.quality_score_estimate == 85.0

    def test_dataset_profile_quality_score_bounds(self):
        with pytest.raises(Exception):
            DatasetProfile(
                artifact_key="x",
                shape=(10, 2),
                total_missing_pct=0.0,
                duplicate_count=0,
                duplicate_pct=0.0,
                column_profiles=[],
                quality_score_estimate=150.0,  # invalid: > 100
            )


# ---------------------------------------------------------------------------
# E. Transformation Log Models
# ---------------------------------------------------------------------------

class TestTransformationLog:
    def test_log_has_auto_id(self):
        log = _sample_transformation_log()
        assert len(log.log_id) == 36  # UUID4 string

    def test_log_roundtrip(self):
        log = _sample_transformation_log()
        dumped = log.model_dump()
        restored = TransformationLog.model_validate(dumped)
        assert restored.log_id == log.log_id
        assert restored.confidence == 0.9

    def test_column_lineage(self):
        lineage = ColumnLineage(
            columns_added=["new_col"],
            columns_removed=["old_col"],
            columns_renamed={"First Name": "first_name"},
            type_changes={"salary": ("object", "float64")},
        )
        dumped = lineage.model_dump()
        restored = ColumnLineage.model_validate(dumped)
        assert restored.columns_renamed["First Name"] == "first_name"


# ---------------------------------------------------------------------------
# F. Tool Result Models
# ---------------------------------------------------------------------------

class TestToolResults:
    def test_base_tool_result(self):
        r = BaseToolResult(success=True, step_name="test_step")
        assert r.confidence == 1.0
        assert r.rows_removed == 0

    def test_dataset_loader_result(self):
        r = DatasetLoaderResult(
            success=True,
            step_name="dataset_loader",
            schema_summary=[{"col": "age", "dtype": "float64", "sample": "25"}],
        )
        assert len(r.schema_summary) == 1

    def test_data_profiler_result(self):
        r = DataProfilerResult(
            success=True,
            step_name="profile_dataset",
            profile_artifact_key="profile/v1/profile",
        )
        assert r.profile is None

    def test_missing_handler_result(self):
        r = MissingHandlerResult(
            success=True,
            step_name="handle_missing_values",
            columns_imputed={"age": MissingStrategy.median},
            columns_dropped=["junk_col"],
        )
        assert r.columns_dropped == ["junk_col"]

    def test_format_standardizer_result(self):
        r = FormatStandardizerResult(
            success=True,
            step_name="standardize_formats",
            format_report={"First Name": ["renamed to first_name"]},
        )
        assert "First Name" in r.format_report

    def test_deduplicator_result(self):
        r = DeduplicatorResult(
            success=True,
            step_name="deduplicate_dataset",
            exact_duplicates_removed=5,
            fuzzy_duplicates_removed=2,
        )
        assert r.exact_duplicates_removed == 5

    def test_merge_result(self):
        r = MergeResult(
            success=True,
            step_name="merge_datasets",
            match_rate=0.95,
            left_unmatched=5,
            merged_shape=ShapeInfo(rows=200, cols=8),
        )
        assert r.match_rate == 0.95

    def test_merge_result_match_rate_bounds(self):
        with pytest.raises(Exception):
            MergeResult(success=True, step_name="x", match_rate=1.5)

    def test_validator_result(self):
        r = ValidatorResult(
            success=True,
            step_name="validate_dataset",
            quality_score=82.5,
            passed=True,
            issues=[
                DataQualityIssue(
                    issue_type=IssueType.near_duplicates,
                    severity=IssueSeverity.warning,
                    affected_columns=["name"],
                    description="Near-duplicate rows detected.",
                )
            ],
        )
        assert r.passed is True
        assert r.issues[0].severity == IssueSeverity.warning

    def test_output_generator_result(self):
        r = OutputGeneratorResult(
            success=True,
            step_name="generate_output",
            csv_artifact_key="output/v1/dataset",
            log_artifact_key="output/v1/log",
            report_artifact_key="output/v1/report",
        )
        assert r.csv_artifact_key is not None

    def test_error_result(self):
        r = BaseToolResult(
            success=False,
            step_name="dataset_loader",
            error_message="File not found: data.csv",
        )
        assert not r.success
        assert "not found" in r.error_message


# ---------------------------------------------------------------------------
# G. AgentSessionState
# ---------------------------------------------------------------------------

class TestAgentSessionState:
    def test_default_state(self):
        state = AgentSessionState()
        assert state.pipeline_status == PipelineStatus.pending
        assert state.current_dataset_key is None
        assert state.task_plan == []

    def test_state_roundtrip(self):
        state = AgentSessionState(
            pipeline_status=PipelineStatus.running,
            current_dataset_key="load/v1/dataset",
            initial_dataset_key="load/v1/dataset",
            current_task_index=2,
        )
        dumped = state.model_dump(mode="json")
        restored = AgentSessionState.model_validate(dumped)
        assert restored.pipeline_status == PipelineStatus.running
        assert restored.current_dataset_key == "load/v1/dataset"

    def test_state_with_plan(self):
        task = PlannedTask(
            task_id=1,
            task_type=TaskType.dataset_loader,
            description="Load CSV",
            config=DatasetLoaderConfig(source_type="local", dataset_identifier="data.csv"),
        )
        state = AgentSessionState(task_plan=[task])
        dumped = state.model_dump(mode="json")
        restored = AgentSessionState.model_validate(dumped)
        assert len(restored.task_plan) == 1
        assert restored.task_plan[0].task_type == TaskType.dataset_loader

    def test_state_with_transformation_log(self):
        log = _sample_transformation_log()
        state = AgentSessionState(transformation_logs=[log])
        dumped = state.model_dump(mode="json")
        restored = AgentSessionState.model_validate(dumped)
        assert len(restored.transformation_logs) == 1

    def test_state_secondary_datasets(self):
        v = _sample_dataset_version()
        state = AgentSessionState(secondary_datasets={"lookup": v})
        dumped = state.model_dump(mode="json")
        restored = AgentSessionState.model_validate(dumped)
        assert "lookup" in restored.secondary_datasets


# ---------------------------------------------------------------------------
# artifact_utils: serialization
# ---------------------------------------------------------------------------

class TestArtifactUtilsSerialization:
    def test_parquet_roundtrip(self):
        df = _sample_df()
        data = df_to_parquet_bytes(df)
        restored = parquet_bytes_to_df(data)
        assert list(restored.columns) == list(df.columns)
        assert len(restored) == len(df)

    def test_compute_checksum(self):
        df = _sample_df()
        cs1 = compute_checksum(df)
        cs2 = compute_checksum(df)
        assert cs1 == cs2
        assert len(cs1) == 32  # MD5 hex

    def test_checksum_changes_on_modification(self):
        df = _sample_df()
        cs1 = compute_checksum(df)
        df2 = df.copy()
        df2.loc[0, "age"] = 999
        cs2 = compute_checksum(df2)
        assert cs1 != cs2

    def test_schema_digest(self):
        df = _sample_df()
        d1 = make_schema_digest(df)
        d2 = make_schema_digest(df)
        assert d1 == d2

    def test_schema_digest_changes_on_rename(self):
        df = _sample_df()
        d1 = make_schema_digest(df)
        df2 = df.rename(columns={"age": "years"})
        d2 = make_schema_digest(df2)
        assert d1 != d2


# ---------------------------------------------------------------------------
# artifact_utils: key generation
# ---------------------------------------------------------------------------

class TestArtifactKeyGeneration:
    def test_make_artifact_key(self):
        key = make_artifact_key("handle_missing_values", 2, "dataset")
        assert key == "handle_missing_values/v2/dataset"

    def test_make_artifact_key_profile(self):
        key = make_artifact_key("profile_dataset", 1, "profile")
        assert key == "profile_dataset/v1/profile"

    def test_next_version_empty_manifest(self):
        manifest = ArtifactManifest()
        assert next_version(manifest, "any_step") == 1

    def test_next_version_increments(self):
        v = _sample_dataset_version()
        manifest = ArtifactManifest(versions={"dataset_loader": [v]})
        assert next_version(manifest, "dataset_loader") == 2

    def test_next_version_new_step(self):
        v = _sample_dataset_version()
        manifest = ArtifactManifest(versions={"dataset_loader": [v]})
        assert next_version(manifest, "handle_missing_values") == 1


# ---------------------------------------------------------------------------
# artifact_utils: profile builders
# ---------------------------------------------------------------------------

class TestProfileBuilders:
    def test_build_column_profile_numeric(self):
        df = _sample_df()
        cp = build_column_profile(df, "age")
        assert cp.name == "age"
        assert cp.inferred_type == InferredDataType.numeric
        assert cp.missing_count == 1
        assert 0.0 <= cp.missing_pct <= 100.0
        assert cp.numeric_stats is not None
        assert cp.numeric_stats.mean is not None

    def test_build_column_profile_categorical(self):
        df = _sample_df()
        cp = build_column_profile(df, "name")
        assert cp.name == "name"
        assert cp.inferred_type in (InferredDataType.categorical, InferredDataType.text)
        assert cp.missing_count == 1

    def test_build_column_profile_no_crash_on_all_null(self):
        df = pd.DataFrame({"col": [None, None, None]})
        cp = build_column_profile(df, "col")
        assert cp.missing_pct == 100.0

    def test_build_dataset_profile(self):
        df = _sample_df()
        profile = build_dataset_profile(df, "load/v1/dataset")
        assert profile.artifact_key == "load/v1/dataset"
        assert profile.shape == (5, 4)
        assert len(profile.column_profiles) == 4
        assert 0.0 <= profile.quality_score_estimate <= 100.0

    def test_build_dataset_profile_detects_duplicates(self):
        df = pd.DataFrame({"a": [1, 1, 2], "b": ["x", "x", "y"]})
        profile = build_dataset_profile(df, "test/v1/dataset")
        assert profile.duplicate_count == 1

    def test_build_dataset_profile_high_priority_issues(self):
        df = pd.DataFrame(
            {
                "good": [1, 2, 3, 4, 5],
                "bad": [None, None, None, None, 1],  # 80% missing
            }
        )
        profile = build_dataset_profile(df, "x")
        # Should flag 'bad' column's high missingness
        issues_text = " ".join(profile.high_priority_issues)
        assert "bad" in issues_text

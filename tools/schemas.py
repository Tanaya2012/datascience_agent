"""
Data models for the data cleaning agent pipeline.

All inter-tool contracts use Pydantic v2. Structured in sections:
  A. Core Enums
  B. Artifact & Version Models
  C. Task Plan Models
  D. Data Profile Models
  E. Transformation Log Models
  F. Tool Result Models
  G. AgentSessionState
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)


# ---------------------------------------------------------------------------
# A. Core Enums
# ---------------------------------------------------------------------------

class TaskType(str, Enum):
    dataset_loader = "dataset_loader"
    profile_dataset = "profile_dataset"
    handle_missing_values = "handle_missing_values"
    standardize_formats = "standardize_formats"
    deduplicate_dataset = "deduplicate_dataset"
    merge_datasets = "merge_datasets"
    validate_dataset = "validate_dataset"
    generate_output = "generate_output"


class PipelineStatus(str, Enum):
    pending = "pending"
    running = "running"
    paused = "paused"
    completed = "completed"
    failed = "failed"
    aborted = "aborted"


class TaskStatus(str, Enum):
    pending = "pending"
    in_progress = "in_progress"
    completed = "completed"
    failed = "failed"
    skipped = "skipped"


class MissingStrategy(str, Enum):
    mean = "mean"
    median = "median"
    mode = "mode"
    ffill = "ffill"
    bfill = "bfill"
    drop_row = "drop_row"
    constant = "constant"


class JoinType(str, Enum):
    left = "left"
    right = "right"
    inner = "inner"
    outer = "outer"


class InferredDataType(str, Enum):
    numeric = "numeric"
    categorical = "categorical"
    datetime = "datetime"
    text = "text"
    boolean = "boolean"
    mixed = "mixed"


class IssueSeverity(str, Enum):
    info = "info"
    warning = "warning"
    error = "error"


class IssueType(str, Enum):
    high_missingness = "high_missingness"
    high_cardinality = "high_cardinality"
    mixed_types = "mixed_types"
    constant_column = "constant_column"
    possible_outliers = "possible_outliers"
    format_inconsistency = "format_inconsistency"
    near_duplicates = "near_duplicates"
    schema_conflict = "schema_conflict"


class FuzzyAlgorithm(str, Enum):
    token_set_ratio = "token_set_ratio"
    partial_ratio = "partial_ratio"
    jaro_winkler = "jaro_winkler"


# ---------------------------------------------------------------------------
# B. Artifact & Version Models
# ---------------------------------------------------------------------------

class DatasetVersion(BaseModel):
    artifact_key: str
    step_name: str
    version: int
    shape: tuple[int, int]          # (rows, cols)
    checksum: str                   # MD5 of parquet bytes
    schema_digest: str              # hash of col names + dtypes
    created_at: datetime
    input_artifact_key: str | None = None   # lineage: what this was made from


class ArtifactManifest(BaseModel):
    versions: dict[str, list[DatasetVersion]] = Field(default_factory=dict)
    # step_name → ordered list of DatasetVersion

    def latest(self, step_name: str) -> DatasetVersion | None:
        """Return the most recent DatasetVersion for a given step, or None."""
        entries = self.versions.get(step_name, [])
        return entries[-1] if entries else None

    def all_keys(self) -> list[str]:
        """Return all artifact keys across every step."""
        keys: list[str] = []
        for entries in self.versions.values():
            keys.extend(v.artifact_key for v in entries)
        return keys


# ---------------------------------------------------------------------------
# C. Task Plan Models
# ---------------------------------------------------------------------------

class DatasetLoaderConfig(BaseModel):
    task_type: Literal[TaskType.dataset_loader] = TaskType.dataset_loader
    source_type: Literal["local"]
    dataset_identifier: str             # local file path
    sheet_name: str | None = None
    is_secondary: bool = False
    secondary_name: str | None = None   # key in AgentSessionState.secondary_datasets


class DataProfilerConfig(BaseModel):
    task_type: Literal[TaskType.profile_dataset] = TaskType.profile_dataset
    # no extra config; reads current_dataset_key at runtime


class MissingHandlerConfig(BaseModel):
    task_type: Literal[TaskType.handle_missing_values] = TaskType.handle_missing_values
    strategy_config: dict[str, MissingStrategy]             # col → strategy
    constant_fill_values: dict[str, Any] | None = None
    drop_threshold: Annotated[float, Field(ge=0.0, le=1.0)] = 0.5
    # columns with > drop_threshold missing fraction are dropped entirely


class FormatStandardizerConfig(BaseModel):
    task_type: Literal[TaskType.standardize_formats] = TaskType.standardize_formats
    normalize_headers: bool = True      # → snake_case
    parse_dates: bool = True
    parse_currency: bool = True
    parse_numerics: bool = True
    column_overrides: dict[str, str] | None = None  # col → explicit format string


class DeduplicatorConfig(BaseModel):
    task_type: Literal[TaskType.deduplicate_dataset] = TaskType.deduplicate_dataset
    exact_dedup: bool = True
    fuzzy_dedup: bool = False
    fuzzy_columns: list[str] | None = None
    fuzzy_threshold: Annotated[float, Field(ge=0.0, le=1.0)] = 0.85
    fuzzy_algorithm: FuzzyAlgorithm = FuzzyAlgorithm.token_set_ratio
    dedup_keep: Literal["first", "last"] = "first"


class MergeConfig(BaseModel):
    task_type: Literal[TaskType.merge_datasets] = TaskType.merge_datasets
    secondary_name: str                 # key in AgentSessionState.secondary_datasets
    join_key: str
    join_type: JoinType = JoinType.left
    validate_key_uniqueness: bool = True


class ValidatorConfig(BaseModel):
    task_type: Literal[TaskType.validate_dataset] = TaskType.validate_dataset


class OutputGeneratorConfig(BaseModel):
    task_type: Literal[TaskType.generate_output] = TaskType.generate_output
    include_summary_stats: bool = True
    output_format: Literal["csv"] = "csv"


TaskConfig = Annotated[
    DatasetLoaderConfig
    | DataProfilerConfig
    | MissingHandlerConfig
    | FormatStandardizerConfig
    | DeduplicatorConfig
    | MergeConfig
    | ValidatorConfig
    | OutputGeneratorConfig,
    Field(discriminator="task_type"),
]


class PlannedTask(BaseModel):
    task_id: int
    task_type: TaskType
    description: str                    # human-readable; shown during plan confirmation
    config: TaskConfig
    status: TaskStatus = TaskStatus.pending
    output_artifact_key: str | None = None
    error: str | None = None
    # on_failure is always "halt" (user decision), so not a per-task field


# ---------------------------------------------------------------------------
# D. Data Profile Models
# ---------------------------------------------------------------------------

class NumericStats(BaseModel):
    mean: float | None = None
    median: float | None = None
    std: float | None = None
    min: float | None = None
    max: float | None = None
    q25: float | None = None
    q75: float | None = None


class CategoricalStats(BaseModel):
    top_values: list[tuple[str, int]] = []  # top 10 (value, count) pairs
    entropy: float = 0.0                    # distribution uniformity


class DatetimeStats(BaseModel):
    min_date: str
    max_date: str
    inferred_format: str                    # e.g. "%Y-%m-%d"


class ColumnAnomaly(BaseModel):
    anomaly_type: IssueType
    description: str
    affected_count: int
    affected_pct: float


class ColumnProfile(BaseModel):
    name: str
    inferred_type: InferredDataType
    dtype_raw: str                          # pandas dtype string, e.g. "float64"
    missing_count: int
    missing_pct: Annotated[float, Field(ge=0.0, le=100.0)]
    unique_count: int
    unique_pct: Annotated[float, Field(ge=0.0, le=100.0)]
    numeric_stats: NumericStats | None = None
    categorical_stats: CategoricalStats | None = None
    datetime_stats: DatetimeStats | None = None
    anomalies: list[ColumnAnomaly] = []
    recommended_strategy: MissingStrategy | None = None  # profiler's suggestion


class DatasetProfile(BaseModel):
    artifact_key: str
    shape: tuple[int, int]
    total_missing_pct: float
    duplicate_count: int
    duplicate_pct: float
    column_profiles: list[ColumnProfile]
    quality_score_estimate: Annotated[float, Field(ge=0.0, le=100.0)]
    high_priority_issues: list[str] = []   # plain English for LLM


# ---------------------------------------------------------------------------
# E. Transformation Log Models
# ---------------------------------------------------------------------------

class ColumnLineage(BaseModel):
    columns_added: list[str] = []
    columns_removed: list[str] = []
    columns_renamed: dict[str, str] = {}        # old_name → new_name
    type_changes: dict[str, tuple[str, str]] = {}   # col → (old_dtype, new_dtype)


class TransformationLog(BaseModel):
    log_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    step_name: str
    task_type: TaskType
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    rows_before: int
    rows_after: int
    cols_before: int
    cols_after: int
    rows_removed: int = 0
    cells_modified: int = 0
    column_lineage: ColumnLineage = Field(default_factory=ColumnLineage)
    checksum_before: str
    checksum_after: str
    confidence: Annotated[float, Field(ge=0.0, le=1.0)]
    operation_detail: dict[str, Any] = {}   # tool-specific detail
    warnings: list[str] = []


# ---------------------------------------------------------------------------
# F. Tool Result Models
# ---------------------------------------------------------------------------

class ShapeInfo(BaseModel):
    rows: int
    cols: int


class BaseToolResult(BaseModel):
    success: bool
    step_name: str
    output_artifact_key: str | None = None
    shape_before: ShapeInfo | None = None
    shape_after: ShapeInfo | None = None
    rows_removed: int = 0
    cells_modified: int = 0
    confidence: Annotated[float, Field(ge=0.0, le=1.0)] = 1.0
    warnings: list[str] = []
    log: TransformationLog | None = None
    error_message: str | None = None


class DatasetLoaderResult(BaseToolResult):
    schema_summary: list[dict[str, str]] = []   # [{col, dtype, sample}]


class DataProfilerResult(BaseToolResult):
    profile: DatasetProfile | None = None
    profile_artifact_key: str | None = None


class MissingHandlerResult(BaseToolResult):
    columns_imputed: dict[str, MissingStrategy] = {}
    columns_dropped: list[str] = []    # columns dropped due to drop_threshold


class FormatStandardizerResult(BaseToolResult):
    format_report: dict[str, list[str]] = {}    # col → list of changes made


class DeduplicatorResult(BaseToolResult):
    exact_duplicates_removed: int = 0
    fuzzy_duplicates_removed: int = 0


class MergeResult(BaseToolResult):
    match_rate: Annotated[float, Field(ge=0.0, le=1.0)] | None = None
    left_unmatched: int = 0
    right_unmatched: int = 0
    merged_shape: ShapeInfo | None = None


class DataQualityIssue(BaseModel):
    issue_type: IssueType
    severity: IssueSeverity
    affected_columns: list[str]
    description: str
    suggested_fix: str | None = None


class ValidatorResult(BaseToolResult):
    quality_score: Annotated[float, Field(ge=0.0, le=100.0)] | None = None
    issues: list[DataQualityIssue] = []
    passed: bool = False


class OutputGeneratorResult(BaseToolResult):
    csv_artifact_key: str | None = None
    log_artifact_key: str | None = None
    report_artifact_key: str | None = None


# ---------------------------------------------------------------------------
# G. AgentSessionState
# ---------------------------------------------------------------------------

class AgentSessionState(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    pipeline_status: PipelineStatus = PipelineStatus.pending
    current_dataset_key: str | None = None      # always the latest transformed dataset
    initial_dataset_key: str | None = None      # original load; never overwritten
    profile_artifact_key: str | None = None     # latest profile
    quality_report_artifact_key: str | None = None

    task_plan: list[PlannedTask] = []
    current_task_index: int = 0

    artifact_manifest: ArtifactManifest = Field(default_factory=ArtifactManifest)
    secondary_datasets: dict[str, DatasetVersion] = {}   # name → DatasetVersion
    transformation_logs: list[TransformationLog] = []
    user_clarifications: dict[str, Any] = {}             # freeform intake metadata

"""
run_python — the code-execution escape hatch (M1).

Runs LLM-authored Python against the current dataset in a persistent, isolated
kernel. The current dataset is preloaded as a pandas DataFrame named ``df``.
Variables and imports persist across calls within a session, so analysis can be
built up incrementally.

Hybrid contract: the kernel is the fast working store; the artifact layer stays
canonical. With ``commit=True`` the (possibly mutated) ``df`` is checkpointed as
a new versioned Parquet artifact plus a TransformationLog, and becomes the
session's current dataset — so code-driven mutations are audited exactly like
deterministic tools. Read-only exploration (``commit=False``) creates no version.
"""

from __future__ import annotations

import hashlib
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from google.adk.tools import ToolContext  # type: ignore[import]

from ..artifact_utils import (
    get_session_state,
    load_artifact,
    make_artifact_key,
    next_version,
    save_artifact,
    set_session_state,
)
from ..schemas import (
    AgentSessionState,
    CodeExecResult,
    ColumnLineage,
    DatasetVersion,
    ShapeInfo,
    TaskType,
    TransformationLog,
)
from .executor import CodeExecutor, ExecResult, KernelStartError, make_default_executor

STEP_NAME = "run_python"


# ---------------------------------------------------------------------------
# Per-session kernel registry (executors are live processes, not serializable,
# so they live at module scope keyed by session — never in session state).
# ---------------------------------------------------------------------------

@dataclass
class _Kernel:
    executor: CodeExecutor
    loaded_key: str | None = None        # dataset artifact key currently in `df`
    advertised: bool = False             # have we sent the package list to the model yet
    packages: list[str] = field(default_factory=list)


_KERNELS: dict[str, _Kernel] = {}


def _session_key(tool_context: Optional[ToolContext]) -> str:
    if tool_context is None:
        return "default"
    try:
        sid = tool_context._invocation_context.session.id  # type: ignore[attr-defined]
        if sid:
            return str(sid)
    except Exception:
        pass
    return "default"


def _get_kernel(tool_context: Optional[ToolContext]) -> _Kernel:
    key = _session_key(tool_context)
    kernel = _KERNELS.get(key)
    if kernel is None:
        kernel = _Kernel(executor=make_default_executor())
        _KERNELS[key] = kernel
    return kernel


def reset_kernels() -> None:
    """Shut down all kernels (used by tests / teardown)."""
    for kernel in _KERNELS.values():
        try:
            kernel.executor.shutdown()
        except Exception:
            pass
    _KERNELS.clear()


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------

async def run_python(
    code: str,
    commit: bool = False,
    commit_step_name: str = STEP_NAME,
    tool_context: Optional[ToolContext] = None,
) -> dict:
    """
    Execute Python against the current dataset in a persistent sandboxed kernel.

    The current dataset is preloaded as a pandas DataFrame named ``df`` (pandas is
    imported as ``pd``). ``print(...)`` what you want to see; the value of the
    cell's final expression is also echoed. matplotlib is available (Agg backend) —
    just create figures and they are captured automatically.

    Args:
        code: Python source to run. Operates on ``df``; persists across calls.
        commit: If True, save the resulting ``df`` as a new versioned dataset
            (with an audit log) and make it the session's current dataset.
            Use for real transformations; leave False for read-only analysis.
        commit_step_name: Step label for the committed version (e.g. "feature_engineering").
        tool_context: Injected by ADK at runtime.

    Returns:
        Serialized CodeExecResult dict (stdout, result_repr, error/traceback,
        committed, plot_artifact_keys, ...).
    """
    state = get_session_state(tool_context) if tool_context else AgentSessionState()
    kernel = _get_kernel(tool_context)
    warnings: list[str] = []

    # 1. Ensure the kernel is up.
    try:
        kernel.executor.execute("None")  # triggers lazy start
    except KernelStartError as exc:
        return CodeExecResult(
            success=False, step_name=STEP_NAME,
            error_message=str(exc), error_type="KernelStartError",
        ).model_dump(mode="json")

    scratch = getattr(kernel.executor, "scratch_dir", os.environ.get("TMPDIR", "/tmp"))

    # 2. Hydrate `df` from the canonical artifact if the current key changed.
    current_key = state.current_dataset_key
    if current_key and current_key != kernel.loaded_key:
        try:
            raw = await load_artifact(current_key, tool_context)
            hydrate_path = os.path.join(scratch, "hydrate.parquet")
            with open(hydrate_path, "wb") as fh:
                fh.write(raw)
            hres = kernel.executor.hydrate_dataframe(hydrate_path)
            if not hres.ok:
                warnings.append(f"Failed to load df into kernel: {hres.error_type}")
            else:
                kernel.loaded_key = current_key
        except Exception as exc:
            warnings.append(f"Could not hydrate dataset '{current_key}': {exc}")
    elif not current_key:
        warnings.append("No dataset loaded yet — `df` is not defined.")

    # 3. Capture shape before (for the audit log on commit).
    shape_before = kernel.executor.df_shape() if kernel.loaded_key else None

    # 4. Run the user code.
    plot_dir = os.path.join(scratch, "plots", uuid.uuid4().hex[:8])
    os.makedirs(plot_dir, exist_ok=True)
    res: ExecResult = kernel.executor.execute(code, plot_dir=plot_dir)

    # A timeout kills the kernel → its `df` is gone; force re-hydrate next call.
    if res.timed_out:
        kernel.loaded_key = None

    # 5. Persist any plots as artifacts.
    plot_keys = await _save_plots(res.plots, tool_context)

    # 6. Commit the mutated df, if requested and possible.
    committed = False
    if commit and res.ok:
        if not kernel.executor.var_exists("df"):
            warnings.append("commit=True but `df` is not defined — nothing committed.")
        else:
            committed = await _commit_df(
                kernel, state, commit_step_name, code, shape_before, warnings, tool_context
            )

    # 7. Advertise available packages once per session.
    available: list[str] = []
    if not kernel.advertised:
        kernel.packages = kernel.executor.installed_packages()
        kernel.advertised = True
        available = kernel.packages

    if tool_context:
        set_session_state(state, tool_context)

    return CodeExecResult(
        success=res.ok,
        step_name=STEP_NAME,
        output_artifact_key=state.current_dataset_key,
        confidence=0.9,
        warnings=warnings,
        error_message=(res.error_type if not res.ok else None),
        stdout=res.stdout,
        result_repr=res.result_repr,
        error_type=res.error_type,
        traceback=res.error,
        timed_out=res.timed_out,
        committed=committed,
        plot_artifact_keys=plot_keys,
        available_packages=available,
    ).model_dump(mode="json")


async def _save_plots(paths: list[str], tool_context: Optional[ToolContext]) -> list[str]:
    keys: list[str] = []
    for i, path in enumerate(paths):
        try:
            with open(path, "rb") as fh:
                data = fh.read()
            key = f"{STEP_NAME}__plot__{uuid.uuid4().hex[:8]}_{i}.png"
            await save_artifact(key, data, tool_context)
            keys.append(key)
        except Exception:
            continue
    return keys


async def _commit_df(
    kernel: _Kernel,
    state: AgentSessionState,
    commit_step_name: str,
    code: str,
    shape_before: tuple[int, int] | None,
    warnings: list[str],
    tool_context: Optional[ToolContext],
) -> bool:
    """Snapshot the kernel's df to a new versioned artifact + TransformationLog."""
    scratch = getattr(kernel.executor, "scratch_dir", "/tmp")
    out_path = os.path.join(scratch, "commit.parquet")
    snap = kernel.executor.snapshot_dataframe(out_path)
    if not snap.ok:
        warnings.append(f"Commit failed while writing df: {snap.error_type}")
        return False
    try:
        with open(out_path, "rb") as fh:
            data = fh.read()
    except OSError as exc:
        warnings.append(f"Commit failed reading snapshot: {exc}")
        return False

    version = next_version(state.artifact_manifest, commit_step_name)
    artifact_key = make_artifact_key(commit_step_name, version, "dataset")
    await save_artifact(artifact_key, data, tool_context)

    checksum_after = hashlib.md5(data).hexdigest()
    shape_after = kernel.executor.df_shape() or (0, 0)
    rows_before, cols_before = shape_before or (0, 0)
    rows_after, cols_after = shape_after

    prev_key = state.current_dataset_key
    checksum_before = ""
    for versions in state.artifact_manifest.versions.values():
        for dv in versions:
            if dv.artifact_key == prev_key:
                checksum_before = dv.checksum

    dataset_version = DatasetVersion(
        artifact_key=artifact_key,
        step_name=commit_step_name,
        version=version,
        shape=(rows_after, cols_after),
        checksum=checksum_after,
        schema_digest="",  # df schema lives in the kernel; not recomputed in-parent
        created_at=datetime.now(timezone.utc),
        input_artifact_key=prev_key,
    )
    state.artifact_manifest.versions.setdefault(commit_step_name, []).append(dataset_version)
    state.current_dataset_key = artifact_key
    kernel.loaded_key = artifact_key  # kernel df now matches the committed version

    log = TransformationLog(
        step_name=commit_step_name,
        task_type=TaskType.run_python,
        rows_before=rows_before,
        rows_after=rows_after,
        cols_before=cols_before,
        cols_after=cols_after,
        rows_removed=max(rows_before - rows_after, 0),
        column_lineage=ColumnLineage(),
        checksum_before=checksum_before,
        checksum_after=checksum_after,
        confidence=0.9,
        operation_detail={"code": code[:2000], "committed": True},
        warnings=warnings,
    )
    state.transformation_logs.append(log)
    return True

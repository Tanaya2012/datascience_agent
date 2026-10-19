"""
Tests for the M1 code-execution kernel: SubprocessKernelExecutor + run_python tool.

These exercise the real worker (via make_default_executor → .worker-venv if
bootstrapped, else the current interpreter). The run_python tool tests use the
mock_ctx fixture and reset the per-session kernel registry between tests.
"""

from __future__ import annotations

import os

import pytest

from datascience_agent.tools.artifact_utils import (
    get_session_state,
    load_artifact,
    parquet_bytes_to_df,
)
from datascience_agent.tools.code_exec.executor import (
    make_default_executor,
)
from datascience_agent.tools.code_exec.run_python import reset_kernels, run_python
from datascience_agent.tools.dataset_loader import dataset_loader


@pytest.fixture
def worker():
    ex = make_default_executor()
    ex.start()
    yield ex
    ex.shutdown()


@pytest.fixture(autouse=True)
def _reset_kernels():
    reset_kernels()
    yield
    reset_kernels()


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------

class TestExecutor:
    def test_simple_print(self, worker):
        r = worker.execute("print('hello kernel')")
        assert r.ok
        assert "hello kernel" in r.stdout

    def test_result_repr_echoes_last_expression(self, worker):
        assert worker.execute("21 * 2").result_repr == "42"

    def test_statements_have_no_repr(self, worker):
        assert worker.execute("a = 5").result_repr is None

    def test_traceback_on_error(self, worker):
        r = worker.execute("1 / 0")
        assert not r.ok
        assert r.error_type == "ZeroDivisionError"
        assert "ZeroDivisionError" in (r.error or "")

    def test_syntax_error(self, worker):
        r = worker.execute("def (:")
        assert not r.ok
        assert r.error_type == "SyntaxError"

    def test_state_persists_across_calls(self, worker):
        worker.execute("counter = 7")
        assert worker.execute("counter + 1").result_repr == "8"

    def test_installed_packages_includes_pandas(self, worker):
        pkgs = [p.lower() for p in worker.installed_packages()]
        assert "pandas" in pkgs

    def test_dataframe_roundtrip(self, worker, tmp_path):
        src = str(tmp_path / "in.parquet")
        worker.execute(
            "import pandas as pd\n"
            f"pd.DataFrame({{'a': [1, 2, 3]}}).to_parquet({src!r}, index=False)"
        )
        assert worker.hydrate_dataframe(src).ok
        assert worker.df_shape() == (3, 1)
        worker.execute("df['b'] = df['a'] * 10")
        out = str(tmp_path / "out.parquet")
        assert worker.snapshot_dataframe(out).ok
        import pandas as pd
        back = pd.read_parquet(out)
        assert list(back.columns) == ["a", "b"]
        assert back["b"].tolist() == [10, 20, 30]

    def test_plots_are_harvested(self, worker, tmp_path):
        if "matplotlib" not in [p.lower() for p in worker.installed_packages()]:
            pytest.skip("matplotlib not available in worker env")
        r = worker.execute(
            "import matplotlib.pyplot as plt\nplt.plot([1, 2, 3])\nNone",
            plot_dir=str(tmp_path),
        )
        assert r.ok
        assert r.plots, "expected at least one harvested figure"
        assert os.path.exists(r.plots[0])

    def test_dead_worker_is_reported_fast_as_a_crash_and_recovers(self):
        """A worker that dies mid-cell (native abort, OOM kill) used to be reported only
        after the full timeout, as a misleading TimeoutError (D36)."""
        import time

        ex = make_default_executor()
        ex.start()
        try:
            t0 = time.monotonic()
            r = ex.execute("import os; os.write(2, b'boom\\n'); os._exit(3)",  # fd 2, like a native abort
                           timeout=20)
            elapsed = time.monotonic() - t0
            assert not r.ok and not r.timed_out
            assert r.error_type == "KernelCrashed"
            assert "exit code 3" in r.error and "boom" in r.error
            assert elapsed < 5, f"crash took {elapsed:.1f}s to report"
            assert ex.execute("21 + 21").result_repr == "42"   # fresh kernel
        finally:
            ex.shutdown()

    def test_memory_cap_is_rlimit_data_not_rlimit_as(self):
        """RLIMIT_AS caps reserved address space (~3 GB for ~210 MB used with pyarrow,
        growing with CPU count) and crashed parquet I/O on Linux; RLIMIT_DATA caps
        memory actually allocated (D36)."""
        ex = make_default_executor()
        assert "WORKER_RLIMIT_DATA" in ex._build_env()
        assert "WORKER_RLIMIT_AS" not in ex._build_env()
        worker_src = (__import__("pathlib").Path(__file__).resolve().parent.parent
                      / "tools" / "code_exec" / "worker.py").read_text()
        assert "resource.RLIMIT_DATA" in worker_src
        assert "resource.RLIMIT_AS" not in worker_src

    def test_wall_clock_timeout_and_recovery(self):
        ex = make_default_executor()
        ex.start()
        try:
            r = ex.execute("import time; time.sleep(30)", timeout=2)
            assert r.timed_out
            assert not r.ok
            assert not ex.is_alive()  # runaway kernel was killed
            # Next call auto-restarts a fresh kernel.
            assert ex.execute("21 + 21").result_repr == "42"
        finally:
            ex.shutdown()

    def test_secrets_not_leaked_to_worker(self, monkeypatch):
        monkeypatch.setenv("SUPER_SECRET_TOKEN", "leak-me")
        ex = make_default_executor()
        ex.start()
        try:
            r = ex.execute("import os; print(repr(os.environ.get('SUPER_SECRET_TOKEN')))")
            assert r.ok
            assert "leak-me" not in r.stdout
            assert "None" in r.stdout
        finally:
            ex.shutdown()


# ---------------------------------------------------------------------------
# run_python tool
# ---------------------------------------------------------------------------

class TestRunPythonTool:
    async def test_runs_without_dataset(self, mock_ctx):
        r = await run_python("print(6 * 7)", tool_context=mock_ctx)
        assert r["success"]
        assert "42" in r["stdout"]
        assert any("No dataset" in w for w in r["warnings"])

    async def test_error_is_fed_back(self, mock_ctx):
        r = await run_python("undefined_thing + 1", tool_context=mock_ctx)
        assert not r["success"]
        assert r["error_type"] == "NameError"
        assert "NameError" in (r["traceback"] or "")

    async def test_available_packages_advertised_once(self, mock_ctx):
        r1 = await run_python("1 + 1", tool_context=mock_ctx)
        assert any(p.lower() == "pandas" for p in r1["available_packages"])
        r2 = await run_python("2 + 2", tool_context=mock_ctx)
        assert r2["available_packages"] == []

    async def test_kernel_crash_forces_rehydrate(self, mock_ctx, csv_file):
        """After a crash the new kernel has no `df`; the next call must reload it."""
        await dataset_loader(
            source_type="local", dataset_identifier=csv_file, tool_context=mock_ctx
        )
        crashed = await run_python("import os; os._exit(1)", tool_context=mock_ctx)
        assert crashed["error_type"] == "KernelCrashed" and not crashed["timed_out"]
        r = await run_python("df.shape", tool_context=mock_ctx)
        assert r["success"], r
        assert r["result_repr"] == "(5, 4)"

    async def test_commit_creates_versioned_dataset(self, mock_ctx, csv_file):
        await dataset_loader(
            source_type="local", dataset_identifier=csv_file, tool_context=mock_ctx
        )
        r = await run_python(
            "df['flag'] = 1",
            commit=True,
            commit_step_name="feature_engineering",
            tool_context=mock_ctx,
        )
        assert r["success"]
        assert r["committed"] is True

        state = get_session_state(mock_ctx)
        key = state.current_dataset_key
        assert "feature_engineering" in key
        df = parquet_bytes_to_df(await load_artifact(key, mock_ctx))
        assert "flag" in df.columns
        assert (df["flag"] == 1).all()
        # A TransformationLog was recorded for the mutation.
        assert any(log.task_type.value == "run_python" for log in state.transformation_logs)

    async def test_no_commit_does_not_version(self, mock_ctx, csv_file):
        await dataset_loader(
            source_type="local", dataset_identifier=csv_file, tool_context=mock_ctx
        )
        before = get_session_state(mock_ctx).current_dataset_key
        r = await run_python("df.shape", tool_context=mock_ctx)
        assert r["success"]
        assert r["committed"] is False
        assert get_session_state(mock_ctx).current_dataset_key == before


# ---------------------------------------------------------------------------
# Event-loop friendliness (audit F3 / D29)
# ---------------------------------------------------------------------------

def _ctx_for(session_id: str):
    """A ToolContext stand-in reporting a distinct session id (→ its own kernel)."""
    import types

    ctx = types.SimpleNamespace(state={})
    ctx._invocation_context = types.SimpleNamespace(
        session=types.SimpleNamespace(id=session_id))

    async def _no_adk(**_):
        raise RuntimeError("no ADK in tests")

    ctx.save_artifact = _no_adk
    ctx.load_artifact = _no_adk
    return ctx


class TestEventLoopIsNotBlocked:
    async def test_other_tasks_keep_running_during_slow_code(self, mock_ctx):
        """The kernel call must happen off-loop: a co-tenant task has to keep ticking
        while user code sleeps, or one session's slow cell freezes the whole server."""
        import asyncio

        await run_python("1", tool_context=mock_ctx)          # warm the kernel

        ticks = 0
        stop = asyncio.Event()

        async def ticker():
            nonlocal ticks
            while not stop.is_set():
                await asyncio.sleep(0.05)
                ticks += 1

        task = asyncio.create_task(ticker())
        r = await run_python("import time; time.sleep(1.0)", tool_context=mock_ctx)
        stop.set()
        await task

        assert r["success"]
        # ~20 ticks if the loop is free; ~1 if it was blocked for the full second.
        assert ticks >= 5, f"event loop was blocked — only {ticks} ticks in 1s"

    async def test_separate_sessions_execute_concurrently(self):
        """Distinct sessions own distinct kernels, so their calls must overlap.
        (Same-session calls still serialize on the kernel lock — that is correct:
        one kernel, one `df` namespace.)"""
        import asyncio
        import time

        a, b = _ctx_for("sess-a"), _ctx_for("sess-b")
        await asyncio.gather(run_python("1", tool_context=a),
                             run_python("1", tool_context=b))   # warm both

        t0 = time.monotonic()
        await asyncio.gather(
            run_python("import time; time.sleep(1.5)", tool_context=a),
            run_python("import time; time.sleep(1.5)", tool_context=b),
        )
        elapsed = time.monotonic() - t0
        # ~1.5s concurrent vs ~3.0s serialized.
        assert elapsed < 2.5, f"sessions serialized ({elapsed:.2f}s) — loop likely blocked"

    async def test_async_and_sync_helper_forms_agree(self, worker, tmp_path):
        """The a*/sync pairs share their source builders, so they must not drift."""
        src = str(tmp_path / "in.parquet")
        worker.execute(
            "import pandas as pd; pd.DataFrame({'a':[1,2,3]}).to_parquet(%r, index=False)" % src
        )
        assert (await worker.ahydrate_dataframe(src)).ok
        assert await worker.adf_shape() == worker.df_shape() == (3, 1)
        assert await worker.avar_exists("df") is worker.var_exists("df") is True
        assert await worker.avar_exists("nope") is worker.var_exists("nope") is False
        assert (await worker.aexecute("21 * 2")).result_repr == "42"
        assert sorted(await worker.ainstalled_packages()) == sorted(worker.installed_packages())
        out = str(tmp_path / "out.parquet")
        assert (await worker.asnapshot_dataframe(out)).ok
        assert os.path.exists(out)

    def test_run_python_makes_no_sync_executor_calls(self):
        """Guards the specific regression: a sync executor call inside the async tool."""
        import re
        from pathlib import Path

        src = (Path(__file__).resolve().parent.parent
               / "tools" / "code_exec" / "run_python.py").read_text()
        sync_calls = re.findall(
            r"(?<!await )kernel\.executor\.(?!shutdown)(\w+)\(", src)
        assert not sync_calls, f"un-awaited executor calls in async tool: {sync_calls}"


# ---------------------------------------------------------------------------
# Scratch-dir isolation between sessions (audit F4 / D30)
# ---------------------------------------------------------------------------

async def _seed_session(ctx, who: str, version: int):
    """Give `ctx` its own dataset, every row tagged with `who`."""
    import pandas as pd

    from datascience_agent.tools.artifact_utils import (
        df_to_parquet_bytes, make_artifact_key, save_artifact, set_session_state,
    )
    from datascience_agent.tools.schemas import AgentSessionState

    df = pd.DataFrame({"who": [who] * 5_000, "n": range(5_000)})
    key = make_artifact_key("dataset_loader", version, "dataset")
    await save_artifact(key, df_to_parquet_bytes(df), ctx)
    set_session_state(AgentSessionState(current_dataset_key=key), ctx)


class TestScratchIsolation:
    async def test_each_session_gets_its_own_scratch_dir(self):
        import asyncio

        from datascience_agent.tools.code_exec.run_python import _KERNELS

        a, b = _ctx_for("scratch-a"), _ctx_for("scratch-b")
        await asyncio.gather(run_python("1", tool_context=a),
                             run_python("1", tool_context=b))
        dirs = {k.executor.scratch_dir for k in _KERNELS.values()}
        assert len(dirs) == 2, f"sessions shared a scratch dir: {dirs}"

    async def test_concurrent_sessions_do_not_swap_datasets(self):
        """The parent writes the hydrate parquet, then awaits while the kernel reads it.
        With a shared filename the other session overwrites it inside that gap."""
        import asyncio

        a, b = _ctx_for("hydrate-a"), _ctx_for("hydrate-b")
        await _seed_session(a, "A", 1)
        await _seed_session(b, "B", 2)

        ra, rb = await asyncio.gather(
            run_python("print(df['who'].iloc[0])", tool_context=a),
            run_python("print(df['who'].iloc[0])", tool_context=b),
        )
        assert ra["stdout"].strip() == "A", "session A hydrated another session's dataset"
        assert rb["stdout"].strip() == "B", "session B hydrated another session's dataset"

    async def test_concurrent_commits_do_not_cross_contaminate(self):
        """The kernel writes the commit parquet, then the parent awaits before reading —
        a shared filename commits the other session's data into this session's lineage."""
        import asyncio

        from datascience_agent.tools.artifact_utils import parquet_bytes_to_df

        a, b = _ctx_for("commit-a"), _ctx_for("commit-b")
        await _seed_session(a, "A", 1)
        await _seed_session(b, "B", 2)
        await run_python("1", tool_context=a)          # hydrate separately …
        await run_python("1", tool_context=b)

        # Distinct step names so the artifact *keys* differ: the local fallback store
        # is not session-namespaced, and identical keys would collide independently
        # of the bug under test.
        await asyncio.gather(
            run_python("df.shape", commit=True, commit_step_name="sess_a", tool_context=a),
            run_python("df.shape", commit=True, commit_step_name="sess_b", tool_context=b),
        )
        for name, ctx in (("A", a), ("B", b)):
            key = get_session_state(ctx).current_dataset_key
            df = parquet_bytes_to_df(await load_artifact(key, ctx))
            assert df["who"].unique().tolist() == [name], \
                f"session {name} committed another session's data"

    async def test_scratch_parquets_are_cleaned_up(self):
        from datascience_agent.tools.code_exec.run_python import _KERNELS

        ctx = _ctx_for("cleanup-one")
        await _seed_session(ctx, "A", 1)
        await run_python("df.shape", commit=True, tool_context=ctx)

        scratch = list(_KERNELS.values())[0].executor.scratch_dir
        leftovers = [f for f in os.listdir(scratch) if f.endswith(".parquet")]
        assert leftovers == [], f"scratch parquets left behind: {leftovers}"

    async def test_reset_removes_generated_scratch_dirs(self):
        from datascience_agent.tools.code_exec.run_python import _KERNELS

        await run_python("1", tool_context=_ctx_for("cleanup-two"))
        dirs = [k.executor.scratch_dir for k in _KERNELS.values()]
        assert all(os.path.isdir(d) for d in dirs)
        reset_kernels()
        assert not any(os.path.isdir(d) for d in dirs)

    def test_caller_supplied_scratch_dir_is_not_deleted(self, tmp_path):
        """We only clean up directories we generated."""
        mine = tmp_path / "caller-owned"
        mine.mkdir()
        ex = make_default_executor(scratch_dir=str(mine))
        ex.start()
        ex.shutdown()
        assert mine.is_dir(), "shutdown deleted a caller-supplied scratch dir"

"""
Tests for the M1 code-execution kernel: SubprocessKernelExecutor + run_python tool.

These exercise the real worker (via make_default_executor → .worker-venv if
bootstrapped, else the current interpreter). The run_python tool tests use the
mock_ctx fixture and reset the per-session kernel registry between tests.
"""

from __future__ import annotations

import os
import sys

import pytest

from datascience_agent.tools.artifact_utils import (
    get_session_state,
    load_artifact,
    parquet_bytes_to_df,
)
from datascience_agent.tools.code_exec.executor import (
    SubprocessKernelExecutor,
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

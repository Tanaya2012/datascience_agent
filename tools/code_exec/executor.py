"""
Code executors — the swappable sandbox behind the run_python escape hatch.

`CodeExecutor` is the abstract interface; `SubprocessKernelExecutor` is the L2
implementation: a persistent worker process running an isolated Python (the
worker venv) with a scrubbed environment, best-effort resource limits, and a
wall-clock timeout enforced by the parent. The interface is intentionally small
so an L4 container backend can be dropped in later without touching callers.

The worker keeps a persistent namespace, so `df` and intermediate variables
survive across `execute` calls within a session.
"""

from __future__ import annotations

import ast
import json
import os
import queue
import subprocess
import sys
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

_WORKER_SCRIPT = str(Path(__file__).resolve().parent / "worker.py")

# Default L2 limits.
_DEFAULT_CPU_SECONDS = 20
_DEFAULT_MEM_BYTES = 2 * 1024 * 1024 * 1024  # 2 GiB (not enforced on macOS)
_DEFAULT_TIMEOUT = 30.0
_STARTUP_TIMEOUT = 30.0

# Env var names we must NEVER leak into the sandbox.
_SECRET_HINTS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL")


@dataclass
class ExecResult:
    ok: bool
    stdout: str = ""
    stderr: str = ""
    result_repr: str | None = None
    error: str | None = None          # full traceback text
    error_type: str | None = None
    timed_out: bool = False
    plots: list[str] = field(default_factory=list)  # filesystem paths to saved figures
    duration: float = 0.0


class KernelStartError(RuntimeError):
    """Raised when the worker process cannot be started (e.g. venv missing)."""


class CodeExecutor(ABC):
    """Abstract code-execution backend."""

    @abstractmethod
    def execute(self, code: str, timeout: float | None = None,
                plot_dir: str | None = None) -> ExecResult: ...

    @abstractmethod
    def is_alive(self) -> bool: ...

    @abstractmethod
    def shutdown(self) -> None: ...

    # --- helpers built on execute() (shared across backends) ----------------

    def hydrate_dataframe(self, parquet_path: str, var: str = "df") -> ExecResult:
        """Load a parquet file into `var` inside the kernel."""
        return self.execute(
            f"import pandas as pd\n{var} = pd.read_parquet({parquet_path!r})\ntuple({var}.shape)"
        )

    def snapshot_dataframe(self, parquet_path: str, var: str = "df") -> ExecResult:
        """Write `var` from the kernel to a parquet file."""
        return self.execute(f"{var}.to_parquet({parquet_path!r}, index=False)")

    def var_exists(self, var: str = "df") -> bool:
        res = self.execute(f"{var!r} in globals()")
        return res.ok and res.result_repr == "True"

    def df_shape(self, var: str = "df") -> tuple[int, int] | None:
        res = self.execute(f"tuple({var}.shape)")
        if not res.ok or not res.result_repr:
            return None
        try:
            shape = ast.literal_eval(res.result_repr)
            return (int(shape[0]), int(shape[1]))
        except Exception:
            return None

    def installed_packages(self) -> list[str]:
        code = (
            "import importlib.metadata as _m, json as _j\n"
            "print(_j.dumps(sorted({d.metadata['Name'] for d in _m.distributions() "
            "if d.metadata['Name']})))"
        )
        res = self.execute(code)
        if not res.ok or not res.stdout.strip():
            return []
        try:
            return list(json.loads(res.stdout.strip().splitlines()[-1]))
        except Exception:
            return []


class SubprocessKernelExecutor(CodeExecutor):
    """L2 executor: persistent subprocess kernel in the worker venv."""

    def __init__(
        self,
        python_executable: str | None = None,
        scratch_dir: str | None = None,
        cpu_seconds: int = _DEFAULT_CPU_SECONDS,
        mem_bytes: int = _DEFAULT_MEM_BYTES,
        default_timeout: float = _DEFAULT_TIMEOUT,
    ):
        self.python_executable = python_executable or sys.executable
        self.scratch_dir = scratch_dir or os.path.join(
            os.environ.get("TMPDIR", "/tmp"), "ds_worker_scratch"
        )
        os.makedirs(self.scratch_dir, exist_ok=True)
        self.cpu_seconds = cpu_seconds
        self.mem_bytes = mem_bytes
        self.default_timeout = default_timeout

        self._proc: subprocess.Popen | None = None
        self._stdout_q: queue.Queue[str] = queue.Queue()
        self._stderr_buf: list[str] = []
        self._req_id = 0
        self._lock = threading.Lock()

    # --- environment --------------------------------------------------------

    def _build_env(self) -> dict[str, str]:
        """Minimal, secret-scrubbed environment for the worker."""
        mpl_dir = os.path.join(self.scratch_dir, ".mpl")
        os.makedirs(mpl_dir, exist_ok=True)
        env = {
            "PATH": os.environ.get("PATH", ""),
            "HOME": self.scratch_dir,
            "TMPDIR": self.scratch_dir,
            "MPLCONFIGDIR": mpl_dir,
            "MPLBACKEND": "Agg",            # headless plotting
            "LANG": os.environ.get("LANG", "C.UTF-8"),
            "PYTHONUNBUFFERED": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "WORKER_RLIMIT_CPU": str(self.cpu_seconds),
            "WORKER_RLIMIT_AS": str(self.mem_bytes),
        }
        # Defence in depth: never inherit anything that looks like a secret.
        return {k: v for k, v in env.items()
                if not any(h in k.upper() for h in _SECRET_HINTS)}

    # --- lifecycle ----------------------------------------------------------

    def start(self) -> None:
        if self.is_alive():
            return
        if not os.path.exists(self.python_executable):
            raise KernelStartError(
                f"Worker python not found: {self.python_executable}. "
                "Bootstrap it with scripts/bootstrap_worker.sh or set WORKER_PYTHON."
            )
        try:
            self._proc = subprocess.Popen(
                [self.python_executable, "-u", _WORKER_SCRIPT],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=self._build_env(),
                cwd=self.scratch_dir,
                text=True,
                bufsize=1,
            )
        except OSError as exc:
            raise KernelStartError(f"Failed to launch worker: {exc}") from exc

        self._stdout_q = queue.Queue()
        self._stderr_buf = []
        threading.Thread(target=self._drain_stdout, daemon=True).start()
        threading.Thread(target=self._drain_stderr, daemon=True).start()

        # Wait for the worker's ready handshake.
        try:
            msg = self._stdout_q.get(timeout=_STARTUP_TIMEOUT)
        except queue.Empty:
            self._kill()
            raise KernelStartError(
                "Worker did not signal ready. stderr:\n" + "".join(self._stderr_buf)
            )
        if not (isinstance(msg, dict) and msg.get("ready")):
            self._kill()
            raise KernelStartError(f"Unexpected worker handshake: {msg!r}")

    def _drain_stdout(self) -> None:
        assert self._proc and self._proc.stdout
        for line in self._proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                self._stdout_q.put(json.loads(line))
            except json.JSONDecodeError:
                # Non-protocol noise on stdout — keep for diagnostics.
                self._stderr_buf.append(line + "\n")

    def _drain_stderr(self) -> None:
        assert self._proc and self._proc.stderr
        for line in self._proc.stderr:
            self._stderr_buf.append(line)

    def is_alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def _kill(self) -> None:
        if self._proc is not None:
            try:
                self._proc.kill()
            except Exception:
                pass
            self._proc = None

    def shutdown(self) -> None:
        if self.is_alive():
            try:
                self._send({"id": -1, "action": "shutdown"})
                self._proc.wait(timeout=5)  # type: ignore[union-attr]
            except Exception:
                self._kill()
        self._proc = None

    # --- execution ----------------------------------------------------------

    def _send(self, obj: dict) -> None:
        assert self._proc and self._proc.stdin
        self._proc.stdin.write(json.dumps(obj) + "\n")
        self._proc.stdin.flush()

    def execute(self, code: str, timeout: float | None = None,
                plot_dir: str | None = None) -> ExecResult:
        timeout = timeout or self.default_timeout
        with self._lock:
            if not self.is_alive():
                self.start()
            self._req_id += 1
            rid = self._req_id
            start = time.monotonic()
            try:
                self._send({"id": rid, "action": "exec", "code": code,
                            "plot_dir": plot_dir})
            except (BrokenPipeError, OSError) as exc:
                self._kill()
                return ExecResult(ok=False, error=f"Kernel pipe broken: {exc}",
                                  error_type="KernelError")

            # Read until we get the response for this request id.
            while True:
                remaining = timeout - (time.monotonic() - start)
                if remaining <= 0:
                    self._kill()  # runaway code — kernel state is lost
                    return ExecResult(
                        ok=False, timed_out=True, duration=timeout,
                        error=f"Execution exceeded {timeout:.0f}s and was terminated. "
                              "The kernel was restarted, so prior in-memory state is lost.",
                        error_type="TimeoutError",
                    )
                try:
                    msg = self._stdout_q.get(timeout=remaining)
                except queue.Empty:
                    continue
                if not isinstance(msg, dict) or msg.get("id") != rid:
                    continue  # skip stray handshake / mismatched ids
                return ExecResult(
                    ok=bool(msg.get("ok")),
                    stdout=msg.get("stdout", "") or "",
                    stderr=msg.get("stderr", "") or "",
                    result_repr=msg.get("result_repr"),
                    error=msg.get("error"),
                    error_type=msg.get("error_type"),
                    plots=list(msg.get("plots", []) or []),
                    duration=time.monotonic() - start,
                )


def make_default_executor(scratch_dir: str | None = None) -> SubprocessKernelExecutor:
    """
    Build the production executor, resolving the worker interpreter:
      WORKER_PYTHON env  →  repo .worker-venv  →  fallback to current interpreter.

    The fallback keeps the agent working before the worker venv is bootstrapped
    (and lets tests run), at the cost of running in the agent's own venv.
    """
    py = os.environ.get("WORKER_PYTHON")
    if not py:
        repo_root = Path(__file__).resolve().parents[2]
        candidate = repo_root / ".worker-venv" / "bin" / "python"
        py = str(candidate) if candidate.exists() else sys.executable
    return SubprocessKernelExecutor(python_executable=py, scratch_dir=scratch_dir)

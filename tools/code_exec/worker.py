"""
Persistent code-execution worker (runs inside the isolated worker venv).

This is a standalone script — it must NOT import the agent package. It speaks a
newline-delimited JSON protocol over stdin/stdout with the parent
SubprocessKernelExecutor:

    parent → worker   {"id": 1, "action": "exec", "code": "...", "plot_dir": "..."}
    worker → parent   {"id": 1, "ok": true, "stdout": "...", "result_repr": "...",
                       "error": null, "error_type": null, "plots": ["...png"]}

A single persistent namespace (`_NS`) is reused across `exec` requests, so
variables, imports, and the loaded `df` survive between calls — the kernel is the
fast working store for the session. User `print()` output is captured via
redirect; the real stdout carries only protocol JSON (one line per response).

Resource limits (best-effort) are applied at startup from env vars:
  WORKER_RLIMIT_CPU  — CPU seconds (RLIMIT_CPU)
  WORKER_RLIMIT_AS   — address-space bytes (RLIMIT_AS; not enforced on macOS)
"""

from __future__ import annotations

import ast
import contextlib
import io
import json
import os
import sys
import traceback

# Persistent namespace shared across all exec requests in this kernel's lifetime.
_NS: dict = {"__name__": "__ds_worker__"}

# Caps so a giant DataFrame repr / runaway print can't flood the parent's context.
_MAX_STDOUT = 20_000
_MAX_REPR = 8_000


def _truncate(text: str, limit: int) -> str:
    if text is None:
        return ""
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... [truncated {len(text) - limit} chars]"


def _apply_limits() -> None:
    """Best-effort resource limits. Silently skip what the OS doesn't support."""
    try:
        import resource
    except Exception:
        return
    cpu = os.environ.get("WORKER_RLIMIT_CPU")
    if cpu:
        try:
            resource.setrlimit(resource.RLIMIT_CPU, (int(cpu), int(cpu) + 1))
        except Exception:
            pass
    mem = os.environ.get("WORKER_RLIMIT_AS")
    if mem:
        try:
            resource.setrlimit(resource.RLIMIT_AS, (int(mem), int(mem)))
        except Exception:
            pass  # macOS commonly ignores RLIMIT_AS


def _harvest_plots(plot_dir: str | None) -> list[str]:
    """Save any open matplotlib figures to plot_dir; return their paths."""
    if not plot_dir:
        return []
    plt = sys.modules.get("matplotlib.pyplot")
    if plt is None:
        return []
    paths: list[str] = []
    try:
        for i, num in enumerate(plt.get_fignums()):
            fig = plt.figure(num)
            path = os.path.join(plot_dir, f"plot_{i}.png")
            try:
                fig.savefig(path, bbox_inches="tight", dpi=100)
                paths.append(path)
            except Exception:
                pass
        plt.close("all")
    except Exception:
        pass
    return paths


def _run_code(code: str, plot_dir: str | None) -> dict:
    """
    Execute `code` in the persistent namespace, capturing output.

    Mimics a REPL: if the final statement is an expression, its value's repr is
    returned as `result_repr` (like a notebook cell's auto-echo).
    """
    out, err = io.StringIO(), io.StringIO()
    result_repr = None
    error = None
    error_type = None

    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError:
        return {
            "ok": False,
            "stdout": "",
            "stderr": "",
            "result_repr": None,
            "error": traceback.format_exc(),
            "error_type": "SyntaxError",
            "plots": [],
        }

    last_expr = None
    if tree.body and isinstance(tree.body[-1], ast.Expr):
        last_expr = ast.Expression(tree.body.pop().value)

    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            exec(compile(tree, "<cell>", "exec"), _NS)
            if last_expr is not None:
                value = eval(compile(last_expr, "<cell>", "eval"), _NS)
                if value is not None:
                    result_repr = repr(value)
    except BaseException as exc:  # noqa: BLE001 — report everything, incl. SystemExit
        error = traceback.format_exc()
        error_type = type(exc).__name__

    return {
        "ok": error is None,
        "stdout": _truncate(out.getvalue(), _MAX_STDOUT),
        "stderr": _truncate(err.getvalue(), _MAX_STDOUT),
        "result_repr": _truncate(result_repr, _MAX_REPR) if result_repr else None,
        "error": error,
        "error_type": error_type,
        "plots": _harvest_plots(plot_dir),
    }


def _send(obj: dict) -> None:
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def main() -> None:
    _apply_limits()
    _send({"id": 0, "ready": True})  # handshake

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            continue
        rid = req.get("id")
        action = req.get("action")

        if action == "shutdown":
            _send({"id": rid, "ok": True})
            break
        if action == "ping":
            _send({"id": rid, "ok": True})
            continue
        if action == "exec":
            resp = _run_code(req.get("code", ""), req.get("plot_dir"))
            resp["id"] = rid
            _send(resp)
            continue

        _send({"id": rid, "ok": False, "error": f"unknown action: {action!r}",
               "error_type": "ProtocolError"})


if __name__ == "__main__":
    main()

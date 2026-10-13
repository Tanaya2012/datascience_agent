"""
Guard against the double import fixed in D31 (audit F7).

The repo root *is* the ``datascience_agent`` package, so its subpackages are also
reachable as flat top-level names whenever the project dir is on ``sys.path`` —
which ``python -m pytest`` always arranges. A flat import loads a second, distinct
copy of the module. ``conftest._one_module_name_per_file`` catches that at runtime
for whatever the suite imports; this test also covers files the suite never imports
(the scripts) and points at the offending line.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_FLAT = "tools|configs|sub_agents|scripts"
_FLAT_IMPORT = re.compile(rf"^\s*(from|import)\s+({_FLAT})([. ]|$)")
# String import targets, e.g. monkeypatch.setattr / mock.patch / importlib.import_module.
_FLAT_STRING_TARGET = re.compile(rf"[\"']({_FLAT})\.[A-Za-z_]")
# Pruned while walking — .worker-venv alone holds tens of thousands of files.
_SKIP_DIRS = {"artifacts", "outputs", "data", "sessions", "__pycache__"}


def _project_files(name_pattern: str):
    """Project files matching *name_pattern*, never descending into dot-dirs
    (.worker-venv, .git, .context, …) or the runtime-output directories."""
    for dirpath, dirnames, filenames in os.walk(_ROOT):
        dirnames[:] = [d for d in dirnames if not d.startswith(".") and d not in _SKIP_DIRS]
        for f in filenames:
            if Path(f).match(name_pattern):
                path = Path(dirpath, f)
                yield path, path.relative_to(_ROOT)


def test_no_flat_imports_anywhere_in_the_project():
    offenders = []
    for path, rel in _project_files("*.py"):
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if _FLAT_IMPORT.match(line) or _FLAT_STRING_TARGET.search(line):
                offenders.append(f"{rel}:{n}: {line.strip()}")
    assert not offenders, (
        "Flat (non-package) imports load a second copy of the module; use "
        "datascience_agent.<subpackage> instead:\n" + "\n".join(offenders)
    )


# Namespace packages share a top-level name across distributions (google.adk,
# google.genai, google.auth, …), so they are resolved one level deeper. An unmapped
# google.* import fails the test below — add it here deliberately.
_NAMESPACE_DISTS = {"google.adk": "google-adk", "google.genai": "google-genai"}

# Needed at runtime but never imported by our code, so no import scan can see them —
# litellm is exactly how F5 went unnoticed. Each must stay declared, for the reason given.
_INDIRECT_RUNTIME = {
    "litellm": "ADK's LiteLlm wrapper — every non-Gemini provider (configs/model_config.py, D3)",
    "sqlalchemy": "DatabaseSessionService — persistent sessions (D11)",
    "aiosqlite": "the sqlite+aiosqlite:// async driver for persistent sessions (D11)",
    "greenlet": "SQLAlchemy's async engine (D11)",
    "pyarrow": "pandas' parquet engine — every dataset artifact is parquet",
    "openpyxl": "pandas' .xlsx engine (dataset_loader)",
    "xlrd": "pandas' .xls engine (dataset_loader)",
}
_INDIRECT_DEV = {"pytest-asyncio": "asyncio_mode = auto in pytest.ini"}


def _norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _declared_names(specs: list[str]) -> set[str]:
    return {_norm(re.split(r"[<>=!~;\[ ]", s, maxsplit=1)[0]) for s in specs}


def test_every_directly_imported_distribution_is_declared():
    """An import that works only because something else happened to install the
    package breaks on a fresh clone — litellm arrived via the eval extra, numpy and
    google-genai transitively (audit F5, D33). Runtime code may import only
    `dependencies`; tests may also use the `dev` extra."""
    import ast
    import sys
    import tomllib
    from importlib.metadata import packages_distributions

    project = tomllib.loads((_ROOT / "pyproject.toml").read_text())["project"]
    runtime = _declared_names(project["dependencies"])
    with_dev = runtime | _declared_names(project["optional-dependencies"]["dev"])
    top_to_dists = packages_distributions()

    problems = set()
    for path, rel in _project_files("*.py"):
        is_test = rel.parts[0] == "tests" or rel.name == "conftest.py"
        allowed = with_dev if is_test else runtime
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"), str(rel))):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                modules = [node.module]
            else:
                continue
            for mod in modules:
                top = mod.split(".")[0]
                if top in sys.stdlib_module_names or top in {"datascience_agent", "__future__"}:
                    continue
                if top in {"google"}:
                    dist = _NAMESPACE_DISTS.get(".".join(mod.split(".")[:2]))
                    if dist is None:
                        problems.add(f"{rel}: {mod} — unmapped namespace import; "
                                     "add it to _NAMESPACE_DISTS")
                        continue
                    candidates = {dist}
                else:
                    candidates = {_norm(d) for d in top_to_dists.get(top, [])}
                    if not candidates:
                        problems.add(f"{rel}: {mod} — not installed (and not declared)")
                        continue
                if not candidates & allowed:
                    where = "[project.dependencies]" + (" or [dev]" if is_test else "")
                    problems.add(f"{rel}: {mod} -> {sorted(candidates)} not in {where}")
    for dist, why in _INDIRECT_RUNTIME.items():
        if dist not in runtime:
            problems.add(f"{dist} missing from [project.dependencies] — needed for {why}")
    for dist, why in _INDIRECT_DEV.items():
        if dist not in with_dev:
            problems.add(f"{dist} missing from [dev] — needed for {why}")
    assert not problems, (
        "Directly imported but undeclared distributions (pyproject.toml):\n"
        + "\n".join(sorted(problems))
    )


def test_pyproject_lists_every_package_on_disk():
    """The package list in pyproject.toml is explicit (automatic discovery would
    install the subpackages as top-level names), so a new subpackage must be added
    to it by hand — this catches the omission."""
    import tomllib

    listed = set(tomllib.loads((_ROOT / "pyproject.toml").read_text())["tool"]
                 ["setuptools"]["packages"])
    on_disk = {"datascience_agent"}
    for _, rel in _project_files("__init__.py"):
        pkg = rel.parent
        if not pkg.parts or pkg.parts[0] == "tests":      # tests are not distributed
            continue
        on_disk.add("datascience_agent." + ".".join(pkg.parts))
    assert listed == on_disk, (
        f"pyproject.toml packages out of sync — missing: {sorted(on_disk - listed)}, "
        f"stale: {sorted(listed - on_disk)}"
    )

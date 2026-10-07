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

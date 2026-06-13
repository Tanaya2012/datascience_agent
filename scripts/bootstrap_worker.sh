#!/usr/bin/env bash
#
# Build the code-execution worker venv (.worker-venv) from requirements-worker.txt.
# The venv is gitignored; this script reconstructs it on any machine.
#
# Usage:
#   bash scripts/bootstrap_worker.sh
#   WORKER_BASE_PYTHON=python3.12 bash scripts/bootstrap_worker.sh   # choose base interpreter
#
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$HERE/.worker-venv"
PYBASE="${WORKER_BASE_PYTHON:-python3}"

echo "Creating worker venv at $VENV using '$PYBASE' ..."
"$PYBASE" -m venv "$VENV"
"$VENV/bin/python" -m pip install --upgrade pip >/dev/null
"$VENV/bin/python" -m pip install -r "$HERE/requirements-worker.txt"

echo "Verifying worker deps ..."
"$VENV/bin/python" -c "import pandas, numpy, pyarrow, sklearn, matplotlib, scipy, statsmodels; print('worker deps OK')"

echo
echo "Worker venv ready: $VENV/bin/python"
echo "Point the executor at it with:  export WORKER_PYTHON=\"$VENV/bin/python\""

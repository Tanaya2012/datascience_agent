"""Planted-truth dataset generation for agent capability testing.

Each scenario writes a CSV plus a machine-checkable ``.truth.json`` answer key,
so a test can ask whether the agent produced the *right* answer rather than
merely whether it avoided crashing. See ``_common.py`` for the fact/answer-key
contract and ``__main__.py`` for the CLI.

Importing this package registers every scenario in ``SCENARIOS``.
"""

from __future__ import annotations

from . import analysis, modeling, traps  # noqa: F401 — imported for registration
from ._common import (
    DEFAULT_SEED,
    GENERATOR_VERSION,
    SCENARIOS,
    Fact,
    Scenario,
    measure,
    verify_truth,
    write_scenario,
)

__all__ = [
    "DEFAULT_SEED",
    "GENERATOR_VERSION",
    "SCENARIOS",
    "Fact",
    "Scenario",
    "measure",
    "verify_truth",
    "write_scenario",
]

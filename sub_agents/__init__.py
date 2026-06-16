"""
Specialist sub-agents for the data-science orchestrator (M2).

Each specialist is a focused ``LlmAgent`` owning a small relevant toolset; the
orchestrator (``agent.py``) wraps them via ``AgentTool`` and routes work to them.
Four specialists exist today — Data Steward, Cleaning, Analysis, Reporting;
Feature-Engineering and Modeling specialists arrive in M4/M5 when they gain tools.
"""

from __future__ import annotations

from .data_steward import build_data_steward, data_steward
from .cleaning import build_cleaning_specialist, cleaning_specialist
from .analysis import build_analysis_specialist, analysis_specialist
from .reporting import build_reporting_specialist, reporting_specialist

__all__ = [
    "build_data_steward",
    "data_steward",
    "build_cleaning_specialist",
    "cleaning_specialist",
    "build_analysis_specialist",
    "analysis_specialist",
    "build_reporting_specialist",
    "reporting_specialist",
]

"""
core/pdf/sections.py — SHIM (backwards compatibility only).

Split into:
    core/pdf/narrative_sections.py  — exec summary, insights, DQ, benchmark, attrition
    core/pdf/data_sections.py       — dataset overview, stats, BI, charts, recommendations
"""
import logging
logger = logging.getLogger(__name__)

from core.pdf.narrative_sections import (   # noqa: F401
    _exec_summary, _top_insights, _dq_note,
    _benchmark_section, _attrition_page,
)
from core.pdf.data_sections import (        # noqa: F401
    _dataset_overview, _stats_section, _bi_section,
    _chart_page, _recommendations,
)

__all__ = [
    "_exec_summary", "_top_insights", "_dq_note",
    "_benchmark_section", "_attrition_page",
    "_dataset_overview", "_stats_section", "_bi_section",
    "_chart_page", "_recommendations",
]

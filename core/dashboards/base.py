"""
core/dashboards/base.py — Shared theme helper and column finder.
"""
from __future__ import annotations

"""
core/domain_dashboards.py  — DataForge AI
Domain-specific KPI cards and chart sets for HR, Finance, E-Commerce, Sales.

Usage in pages/3_Dashboard.py:
    from core.domain_dashboards import get_domain_kpis, get_domain_charts
    kpis   = get_domain_kpis(df, domain)
    charts = get_domain_charts(df, domain)
"""
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import List, Dict, Optional, Tuple
import logging
logger = logging.getLogger(__name__)

from core.dashboards._utils import _apply_theme, _find_col  # noqa: F401

# ── Import domain implementations (split into separate files) ─────────────────
from core.dashboards.hr        import _hr_kpis, _hr_charts
from core.dashboards.finance   import _finance_kpis, _finance_charts
from core.dashboards.ecommerce import _ecommerce_kpis, _ecommerce_charts
from core.dashboards.sales     import _sales_kpis, _sales_charts
from core.dashboards.general   import _general_kpis, _general_charts


# ── Palette (matches chart_theme.py) ─────────────────────────────────────────
C_NAVY   = "#1B2A4A"
C_BLUE   = "#1B4FD8"
C_TEAL   = "#0D9488"
C_GREEN  = "#059669"
C_AMBER  = "#D97706"
C_RED    = "#DC2626"
C_SLATE  = "#475569"
C_LGRAY  = "#F8FAFC"
C_MGRAY  = "#E2E8F0"
SEQ_BLUE = [C_NAVY, C_BLUE, "#1976D2", "#1565C0", "#0D47A1"]


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN PUBLIC API
# ═══════════════════════════════════════════════════════════════════════════════

def get_domain_kpis(df: pd.DataFrame, domain: str) -> List[Dict]:
    """Returns list of KPI dicts: {label, value, sub, color, delta}"""
    try:
        if domain == "hr":          return _hr_kpis(df)
        if domain == "finance":     return _finance_kpis(df)
        if domain == "ecommerce":   return _ecommerce_kpis(df)
        if domain == "sales":       return _sales_kpis(df)
    except Exception:
        logger.warning("%s unexpected failure", exc_info=True)
    return _general_kpis(df)


def get_domain_charts(df: pd.DataFrame, domain: str) -> List[Tuple[str, go.Figure]]:
    """Returns list of (chart_title, plotly_figure) tuples."""
    try:
        if domain == "hr":          return _hr_charts(df)
        if domain == "finance":     return _finance_charts(df)
        if domain == "ecommerce":   return _ecommerce_charts(df)
        if domain == "sales":       return _sales_charts(df)
    except Exception:
        logger.warning("%s unexpected failure", exc_info=True)
    return _general_charts(df)


# ═══════════════════════════════════════════════════════════════════════════════
#  HR DOMAIN
# ═══════════════════════════════════════════════════════════════════════════════


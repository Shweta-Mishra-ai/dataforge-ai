"""
core/dashboards/general.py — General/unknown domain KPIs and charts.
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


from core.dashboards._utils import _apply_theme, _find_col

def _general_kpis(df: pd.DataFrame) -> List[Dict]:
    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
    miss_pct = float(df.isna().mean().mean() * 100)
    dup_n    = int(df.duplicated().sum())
    kpis = [
        {"label": "Total Records",    "value": f"{len(df):,}",
         "sub": f"{len(df.columns)} columns", "color": C_NAVY, "delta": None},
        {"label": "Numeric Columns",  "value": str(len(num_cols)),
         "sub": "available for analysis", "color": C_BLUE, "delta": None},
        {"label": "Categorical Cols", "value": str(len(cat_cols)),
         "sub": "for segmentation", "color": C_TEAL, "delta": None},
        {"label": "Missing Data",     "value": f"{miss_pct:.1f}%",
         "sub": "across all columns",
         "color": C_RED if miss_pct > 10 else C_AMBER if miss_pct > 2 else C_GREEN,
         "delta": None},
        {"label": "Duplicate Rows",   "value": f"{dup_n:,}",
         "sub": f"{dup_n/len(df)*100:.1f}% of total",
         "color": C_AMBER if dup_n > 0 else C_GREEN, "delta": None},
    ]
    return kpis


def _general_charts(df: pd.DataFrame) -> List[Tuple[str, go.Figure]]:
    charts = []
    num_cols = df.select_dtypes(include="number").columns.tolist()[:3]
    for col in num_cols:
        try:
            fig = px.histogram(df, x=col, nbins=20,
                               color_discrete_sequence=[C_BLUE])
            _apply_theme(fig, f"Distribution: {col}", "")
            fig.update_layout(height=300)
            charts.append((f"Distribution: {col}", fig))
        except Exception:
            logger.warning("%s unexpected failure", exc_info=True)
    return charts

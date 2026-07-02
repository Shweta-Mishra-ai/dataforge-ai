"""
core/dashboards/_utils.py
Shared palette, theme helper, and column finder.
Imported by BOTH base.py (dispatcher) and domain files (hr, finance, etc.)
to avoid circular imports.
"""
from __future__ import annotations
from typing import List, Optional
import plotly.graph_objects as go
import pandas as pd
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


def _apply_theme(fig: go.Figure, title: str, subtitle: str = "") -> go.Figure:
    title_text = f"<b>{title}</b>"
    if subtitle:
        title_text += f"<br><span style='font-size:11px;color:{C_SLATE}'>{subtitle}</span>"
    fig.update_layout(
        title=dict(text=title_text, font=dict(size=15, color="#0A1628", family="Arial Black, Arial"),
                   x=0, xanchor="left", pad=dict(l=4)),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, Arial", size=11, color="#0F172A"),
        margin=dict(l=55, r=20, t=72, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    font=dict(size=10), bgcolor="rgba(255,255,255,0.9)",
                    bordercolor=C_MGRAY, borderwidth=1),
        xaxis=dict(showgrid=False, showline=True, linecolor=C_MGRAY,
                   tickfont=dict(size=10, color='#0F172A'), title_font=dict(size=11, color='#0F172A')),
        yaxis=dict(showgrid=True, gridcolor=C_MGRAY, gridwidth=0.8,
                   showline=False, tickfont=dict(size=10, color='#0F172A'), title_font=dict(size=11, color='#0F172A')),
        hoverlabel=dict(bgcolor="white", bordercolor=C_MGRAY,
                        font=dict(family="Inter, Arial", size=11)),
    )
    return fig


def _find_col(df: pd.DataFrame, keywords: List[str], exclude: List[str] = None) -> Optional[str]:
    excl = exclude or []
    for c in df.columns:
        cl = c.lower()
        if any(k in cl for k in keywords) and not any(e in cl for e in excl):
            return c
    return None


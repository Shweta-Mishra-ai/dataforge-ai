"""
core/dashboards/sales.py — Sales KPIs and charts.
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

def _sales_kpis(df: pd.DataFrame) -> List[Dict]:
    kpis = []
    rev_col    = _find_col(df, ["revenue","sales","amount","value"], exclude=["budget","target"])
    target_col = _find_col(df, ["target","quota","budget","plan"])
    rep_col    = _find_col(df, ["rep","salesperson","agent","executive","owner"])
    _region_col = _find_col(df, ["region","territory","area","zone"])
    margin_col = _find_col(df, ["margin","profit","gross"])
    won_col    = _find_col(df, ["won","closed","status","result"])
    _prod_col  = _find_col(df, ["product","item","sku","service"])

    kpis.append({"label": "Total Records", "value": f"{len(df):,}",
                 "sub": "sales transactions / opportunities",
                 "color": C_NAVY, "delta": None})

    if rev_col:
        total_rev = float(df[rev_col].sum())
        avg_deal  = float(df[rev_col].mean())
        kpis.append({"label": "Total Revenue", "value": f"{total_rev:,.0f}",
                     "sub": f"Avg deal: {avg_deal:,.0f} | Median: {float(df[rev_col].median()):,.0f}",
                     "color": C_NAVY, "delta": None})

    if rev_col and target_col:
        try:
            total_r = float(df[rev_col].sum())
            total_t = float(df[target_col].sum())
            ach     = total_r / total_t * 100 if total_t else 0
            color   = C_GREEN if ach >= 100 else C_AMBER if ach >= 80 else C_RED
            kpis.append({"label": "Quota Achievement",
                         "value": f"{ach:.1f}%",
                         "sub": f"Actual: {total_r:,.0f} vs Target: {total_t:,.0f}",
                         "color": color, "delta": f"{ach - 100:+.1f}pp vs target"})
        except Exception:
            logger.warning("%s unexpected failure", exc_info=True)

    if rep_col and rev_col:
        try:
            rep_rev    = df.groupby(rep_col)[rev_col].sum()
            top_rep    = str(rep_rev.idxmax())
            top_pct    = float(rep_rev.max() / rep_rev.sum() * 100)
            n_reps     = len(rep_rev)
            color = C_AMBER if top_pct > 40 else C_GREEN
            kpis.append({"label": "Top Rep Concentration",
                         "value": f"{top_pct:.1f}%",
                         "sub": f"'{top_rep[:18]}' drives {top_pct:.1f}% of revenue · {n_reps} reps",
                         "color": color, "delta": None})
        except Exception:
            logger.warning("%s unexpected failure", exc_info=True)

    if margin_col:
        mean_m = float(df[margin_col].mean())
        neg_n  = int((df[margin_col] < 0).sum())
        color  = C_GREEN if mean_m > 30 else C_AMBER if mean_m > 15 else C_RED
        kpis.append({"label": "Avg Margin", "value": f"{mean_m:.1f}%",
                     "sub": f"{neg_n} negative-margin deals" if neg_n else "All deals profitable",
                     "color": color, "delta": None})

    if won_col:
        try:
            won_vals = df[won_col].astype(str).str.lower()
            won_n    = int(won_vals.isin(["won","closed","closed won","1","true","yes","win"]).sum())
            if won_n > 0:
                win_rate = won_n / len(df) * 100
                color    = C_GREEN if win_rate > 50 else C_AMBER if win_rate > 30 else C_RED
                kpis.append({"label": "Win Rate",
                             "value": f"{win_rate:.1f}%",
                             "sub": f"{won_n:,} won of {len(df):,} total",
                             "color": color, "delta": None})
        except Exception:
            logger.warning("%s unexpected failure", exc_info=True)

    return kpis[:6]


def _sales_charts(df: pd.DataFrame) -> List[Tuple[str, go.Figure]]:
    charts = []
    rev_col    = _find_col(df, ["revenue","sales","amount","value"], exclude=["budget","target"])
    target_col = _find_col(df, ["target","quota","budget","plan"])
    rep_col    = _find_col(df, ["rep","salesperson","agent","executive","owner"])
    region_col = _find_col(df, ["region","territory","area","zone"])
    margin_col = _find_col(df, ["margin","profit","gross"])
    period_col = _find_col(df, ["month","quarter","date","period","year"])
    prod_col   = _find_col(df, ["product","item","sku","service"])

    # Chart 1: Revenue by Rep (quintile ranking)
    if rep_col and rev_col:
        try:
            rep_data = df.groupby(rep_col)[rev_col].sum().reset_index()
            rep_data.columns = ["Rep", "Revenue"]
            total_r  = float(rep_data["Revenue"].sum())
            rep_data["pct"] = rep_data["Revenue"] / total_r * 100
            rep_data = rep_data.sort_values("Revenue", ascending=True).tail(20)

            # Color by quintile
            n = len(rep_data)
            q_colors = []
            for i, _ in enumerate(rep_data.itertuples()):
                rank_pct = i / n
                if rank_pct >= 0.8:   q_colors.append(C_NAVY)
                elif rank_pct >= 0.6: q_colors.append(C_BLUE)
                elif rank_pct >= 0.4: q_colors.append("#3B82F6")
                elif rank_pct >= 0.2: q_colors.append(C_AMBER)
                else:                 q_colors.append(C_RED)

            fig = go.Figure(go.Bar(
                x=rep_data["Revenue"], y=rep_data["Rep"],
                orientation="h", marker_color=q_colors,
                text=[f"{p:.1f}%" for p in rep_data["pct"]],
                textposition="outside",
            ))
            avg_r = float(rep_data["Revenue"].mean())
            fig.add_vline(x=avg_r, line_dash="dash", line_color=C_SLATE,
                          line_width=1.5, annotation_text=f"Avg {avg_r:,.0f}",
                          annotation_font_size=10)
            _apply_theme(fig, "Revenue by Sales Rep (Top 20)",
                         "Dark blue = top quintile · Red = bottom quintile · % = revenue share")
            fig.update_layout(height=max(300, min(len(rep_data), 20) * 36 + 80),
                              xaxis_title="Revenue")
            charts.append(("Revenue by Rep", fig))
        except Exception:
            logger.warning("%s unexpected failure", exc_info=True)

    # Chart 2: Revenue vs Target by Region
    if region_col and rev_col and target_col:
        try:
            region_data = df.groupby(region_col)[[rev_col, target_col]].sum().reset_index()
            region_data.columns = ["Region", "Revenue", "Target"]
            region_data["Achievement"] = region_data["Revenue"] / region_data["Target"] * 100
            region_data = region_data.sort_values("Achievement", ascending=False)

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=region_data["Region"], y=region_data["Target"],
                name="Target", marker_color=C_SLATE, opacity=0.6,
            ))
            fig.add_trace(go.Bar(
                x=region_data["Region"], y=region_data["Revenue"],
                name="Actual",
                marker_color=[C_GREEN if a >= 100 else C_AMBER if a >= 80 else C_RED
                              for a in region_data["Achievement"]],
            ))
            fig.update_layout(barmode="group")
            _apply_theme(fig, "Revenue vs Target by Region",
                         "Green = ≥100% · Amber = 80–100% · Red = <80%")
            fig.update_layout(height=360, yaxis_title="Amount")
            charts.append(("Revenue vs Target by Region", fig))
        except Exception:
            logger.warning("%s unexpected failure", exc_info=True)

    # Chart 3: Revenue Trend
    if period_col and rev_col:
        try:
            trend = df.groupby(period_col)
            if target_col:
                trend = trend[[rev_col, target_col]].sum().reset_index()
            else:
                trend = trend[rev_col].sum().reset_index()
            try:
                trend = trend.sort_values(period_col)
            except Exception:
                logger.warning("%s unexpected failure", exc_info=True)

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=trend[period_col], y=trend[rev_col],
                mode="lines+markers", name="Revenue",
                line=dict(color=C_BLUE, width=2.5),
                marker=dict(size=7),
            ))
            if target_col and target_col in trend.columns:
                fig.add_trace(go.Scatter(
                    x=trend[period_col], y=trend[target_col],
                    mode="lines", name="Target",
                    line=dict(color=C_SLATE, width=1.5, dash="dash"),
                ))
            _apply_theme(fig, "Revenue Trend by Period",
                         "Dashed = target if available")
            fig.update_layout(height=340, yaxis_title="Revenue")
            charts.append(("Revenue Trend", fig))
        except Exception:
            logger.warning("%s unexpected failure", exc_info=True)

    # Chart 4: Deal Size Distribution
    if rev_col:
        try:
            deal_vals = df[rev_col].dropna()
            p80 = float(deal_vals.quantile(0.80))
            _p20 = float(deal_vals.quantile(0.20))
            fig = px.histogram(deal_vals, nbins=25, color_discrete_sequence=[C_BLUE])
            fig.add_vline(x=float(deal_vals.median()), line_dash="solid",
                          line_color=C_RED, line_width=2,
                          annotation_text=f"Median {deal_vals.median():,.0f}",
                          annotation_font_color=C_RED)
            fig.add_vline(x=p80, line_dash="dash", line_color=C_GREEN, line_width=1.5,
                          annotation_text=f"P80 {p80:,.0f}", annotation_font_color=C_GREEN)
            _apply_theme(fig, "Deal Size Distribution",
                         "Use median for reporting — likely right-skewed · P80 marks large-deal threshold")
            fig.update_layout(height=320, xaxis_title=rev_col, yaxis_title="Count")
            charts.append(("Deal Size Distribution", fig))
        except Exception:
            logger.warning("%s unexpected failure", exc_info=True)

    # Chart 5: Margin by Product / Category
    if margin_col and (prod_col or region_col):
        grp_col = prod_col or region_col
        try:
            margin_data = df.groupby(grp_col)[margin_col].mean().reset_index()
            margin_data.columns = ["Group", "Avg Margin"]
            margin_data = margin_data.sort_values("Avg Margin").tail(15)
            colors_m = [C_RED if m < 0 else C_AMBER if m < 20 else C_GREEN
                        for m in margin_data["Avg Margin"]]
            fig = go.Figure(go.Bar(
                x=margin_data["Avg Margin"], y=margin_data["Group"],
                orientation="h", marker_color=colors_m,
                text=[f"{m:.1f}%" for m in margin_data["Avg Margin"]],
                textposition="outside",
            ))
            fig.add_vline(x=0, line_dash="solid", line_color=C_NAVY, line_width=1.5)
            _apply_theme(fig, f"Avg Margin by {grp_col.replace('_',' ').title()}",
                         "Red = negative margin (selling at loss) · Green = healthy margin")
            fig.update_layout(height=max(280, len(margin_data) * 36 + 80),
                              xaxis_title="Average Margin (%)")
            charts.append(("Margin by Segment", fig))
        except Exception:
            logger.warning("%s unexpected failure", exc_info=True)

    return charts


# ═══════════════════════════════════════════════════════════════════════════════
#  GENERAL FALLBACK
# ═══════════════════════════════════════════════════════════════════════════════


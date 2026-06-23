"""
core/dashboards/finance.py — Finance KPIs and charts.
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


from core.dashboards.base import _apply_theme, _find_col

def _finance_kpis(df: pd.DataFrame) -> List[Dict]:
    kpis = []
    rev_col    = _find_col(df, ["revenue","total_revenue","income","turnover","sales_amount"])
    cost_col   = _find_col(df, ["cost","cogs","cost_of_goods","direct_cost"])
    profit_col = _find_col(df, ["net_profit","profit","net_income"])
    budget_col = _find_col(df, ["budget","plan","target","forecast"])
    actual_col = _find_col(df, ["actual","actuals"], exclude=["target","budget"])
    opex_col   = _find_col(df, ["opex","operating_expense","overhead"])

    if rev_col:
        total_rev = float(df[rev_col].sum())
        kpis.append({"label": "Total Revenue", "value": f"{total_rev:,.0f}",
                     "sub": f"Median period: {float(df[rev_col].median()):,.0f}",
                     "color": C_NAVY, "delta": None})

    if rev_col and cost_col:
        try:
            total_r = float(df[rev_col].sum())
            total_c = float(df[cost_col].sum())
            gm = (total_r - total_c) / total_r * 100 if total_r else 0
            color = C_GREEN if gm > 40 else C_AMBER if gm > 20 else C_RED
            kpis.append({"label": "Gross Margin", "value": f"{gm:.1f}%",
                         "sub": f"Revenue: {total_r:,.0f} | COGS: {total_c:,.0f}",
                         "color": color, "delta": None})
        except Exception:
            logger.warning("%s unexpected failure", exc_info=True)

    if profit_col:
        total_p   = float(df[profit_col].sum())
        neg_count = int((df[profit_col] < 0).sum())
        color = C_GREEN if total_p > 0 else C_RED
        kpis.append({"label": "Total Profit / Income", "value": f"{total_p:,.0f}",
                     "sub": f"{neg_count} loss-making rows" if neg_count else "All rows profitable",
                     "color": color, "delta": None})

    if budget_col and actual_col and budget_col != actual_col:
        try:
            common = df[[budget_col, actual_col]].dropna()
            if len(common) > 2:
                var_pct = float(((common[actual_col] - common[budget_col]) /
                                  common[budget_col].replace(0, np.nan)).mean() * 100)
                color   = C_GREEN if abs(var_pct) < 10 else C_AMBER if abs(var_pct) < 20 else C_RED
                kpis.append({"label": "Budget Variance", "value": f"{var_pct:+.1f}%",
                             "sub": "Avg actual vs budget across all rows",
                             "color": color, "delta": f"{var_pct:+.1f}%"})
        except Exception:
            logger.warning("%s unexpected failure", exc_info=True)

    if opex_col and rev_col:
        try:
            opex_ratio = float(df[opex_col].sum()) / float(df[rev_col].sum()) * 100
            color = C_GREEN if opex_ratio < 30 else C_AMBER if opex_ratio < 50 else C_RED
            kpis.append({"label": "OpEx Ratio", "value": f"{opex_ratio:.1f}%",
                         "sub": "Operating expenses as % of revenue",
                         "color": color, "delta": None})
        except Exception:
            logger.warning("%s unexpected failure", exc_info=True)

    return kpis[:6]


def _finance_charts(df: pd.DataFrame) -> List[Tuple[str, go.Figure]]:
    charts = []
    rev_col    = _find_col(df, ["revenue","total_revenue","income","turnover","sales_amount"])
    cost_col   = _find_col(df, ["cost","cogs","cost_of_goods","direct_cost"])
    profit_col = _find_col(df, ["net_profit","profit","net_income"])
    budget_col = _find_col(df, ["budget","plan","target","forecast"])
    actual_col = _find_col(df, ["actual","actuals"], exclude=["target","budget"])
    period_col = _find_col(df, ["month","quarter","period","year","date"])
    cat_col    = _find_col(df, ["category","department","cost_center","account","segment"])
    opex_col   = _find_col(df, ["opex","operating_expense","overhead"])
    expense_col= _find_col(df, ["expense","spend","expenditure"])
    val_col    = cost_col or expense_col or opex_col

    # Chart 1: Revenue vs Cost over periods
    if period_col and rev_col:
        try:
            agg_cols = {rev_col: "sum"}
            if cost_col: agg_cols[cost_col] = "sum"
            period_data = df.groupby(period_col).agg(agg_cols).reset_index()
            try:
                period_data = period_data.sort_values(period_col)
            except Exception:
                logger.warning("%s unexpected failure", exc_info=True)

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=period_data[period_col], y=period_data[rev_col],
                name="Revenue", marker_color=C_BLUE, opacity=0.85,
            ))
            if cost_col and cost_col in period_data.columns:
                fig.add_trace(go.Bar(
                    x=period_data[period_col], y=period_data[cost_col],
                    name="Cost", marker_color=C_AMBER, opacity=0.85,
                ))
                # Margin line
                period_data["_margin"] = (
                    (period_data[rev_col] - period_data[cost_col]) /
                    period_data[rev_col].replace(0, np.nan) * 100
                )
                fig.add_trace(go.Scatter(
                    x=period_data[period_col], y=period_data["_margin"],
                    name="Gross Margin %", yaxis="y2",
                    line=dict(color=C_GREEN, width=2.5, dash="solid"),
                    mode="lines+markers", marker=dict(size=6),
                ))
                fig.update_layout(yaxis2=dict(
                    title="Gross Margin %", overlaying="y", side="right",
                    showgrid=False, tickfont=dict(size=10, color=C_GREEN),
                    title_font=dict(size=11, color=C_GREEN),
                ))
            fig.update_layout(barmode="group")
            _apply_theme(fig, "Revenue vs Cost by Period",
                         "Bars = absolute values · Green line = gross margin %")
            fig.update_layout(height=380, xaxis_title=period_col, yaxis_title="Amount")
            charts.append(("Revenue vs Cost by Period", fig))
        except Exception:
            logger.warning("%s unexpected failure", exc_info=True)

    # Chart 2: Budget vs Actual
    if budget_col and actual_col and budget_col != actual_col:
        try:
            comp_col = period_col or cat_col
            if comp_col:
                agg = df.groupby(comp_col)[[budget_col, actual_col]].sum().reset_index()
                try:
                    agg = agg.sort_values(actual_col, ascending=False).head(15)
                except Exception:
                    logger.warning("%s unexpected failure", exc_info=True)
                agg["variance_pct"] = ((agg[actual_col] - agg[budget_col]) /
                                        agg[budget_col].replace(0, np.nan) * 100)

                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=agg[comp_col], y=agg[budget_col],
                    name="Budget / Plan", marker_color=C_SLATE, opacity=0.65,
                ))
                fig.add_trace(go.Bar(
                    x=agg[comp_col], y=agg[actual_col],
                    name="Actual",
                    marker_color=[C_RED if v < 0 else C_GREEN if abs(v) < 10 else C_AMBER
                                  for v in agg["variance_pct"]],
                    opacity=0.9,
                ))
                fig.update_layout(barmode="group")
                _apply_theme(fig, "Budget vs Actual Comparison",
                             "Green = within 10% variance · Amber = 10–20% · Red = >20% or loss")
                fig.update_layout(height=360, yaxis_title="Amount")
                charts.append(("Budget vs Actual", fig))
        except Exception:
            logger.warning("%s unexpected failure", exc_info=True)

    # Chart 3: Cost / Expense by Category
    if cat_col and val_col:
        try:
            cat_data = df.groupby(cat_col)[val_col].sum().reset_index()
            cat_data.columns = ["Category", "Amount"]
            cat_data = cat_data.sort_values("Amount", ascending=False).head(12)
            total    = float(cat_data["Amount"].sum())
            cat_data["pct"] = cat_data["Amount"] / total * 100

            fig = go.Figure(go.Bar(
                x=cat_data["Amount"], y=cat_data["Category"],
                orientation="h",
                marker_color=px.colors.sequential.Blues_r[:len(cat_data)],
                text=[f"{p:.1f}%" for p in cat_data["pct"]],
                textposition="outside",
            ))
            _apply_theme(fig, "Cost/Expense by Category",
                         "% labels show share of total")
            fig.update_layout(height=max(300, len(cat_data) * 38 + 80),
                              xaxis_title="Total Amount")
            charts.append(("Cost by Category", fig))
        except Exception:
            logger.warning("%s unexpected failure", exc_info=True)

    # Chart 4: Profit / Margin Trend
    if period_col and profit_col:
        try:
            trend = df.groupby(period_col)[profit_col].sum().reset_index()
            try:
                trend = trend.sort_values(period_col)
            except Exception:
                logger.warning("%s unexpected failure", exc_info=True)
            colors_t = [C_GREEN if v >= 0 else C_RED for v in trend[profit_col]]
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=trend[period_col], y=trend[profit_col],
                marker_color=colors_t, name="Profit",
            ))
            fig.add_hline(y=0, line_dash="dash", line_color=C_NAVY, line_width=1.5,
                          annotation_text="Break-even", annotation_font_size=10)
            _apply_theme(fig, "Profit by Period",
                         "Green = profit · Red = loss · Dashed = break-even line")
            fig.update_layout(height=340, yaxis_title="Profit Amount")
            charts.append(("Profit by Period", fig))
        except Exception:
            logger.warning("%s unexpected failure", exc_info=True)

    # Chart 5: Revenue Waterfall (top categories)
    if cat_col and rev_col:
        try:
            cat_rev = df.groupby(cat_col)[rev_col].sum().sort_values(ascending=False).head(8)
            total_r = float(cat_rev.sum())
            fig = go.Figure(go.Bar(
                x=cat_rev.index, y=cat_rev.values,
                marker_color=SEQ_BLUE[:len(cat_rev)],
                text=[f"{v/total_r*100:.1f}%" for v in cat_rev.values],
                textposition="outside",
            ))
            _apply_theme(fig, "Revenue by Category (Top 8)",
                         "% = share of total revenue")
            fig.update_layout(height=340, yaxis_title="Revenue")
            charts.append(("Revenue by Category", fig))
        except Exception:
            logger.warning("%s unexpected failure", exc_info=True)

    return charts


# ═══════════════════════════════════════════════════════════════════════════════
#  E-COMMERCE DOMAIN
# ═══════════════════════════════════════════════════════════════════════════════


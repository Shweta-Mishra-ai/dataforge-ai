"""
core/dashboards/hr.py — HR KPIs and charts.
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

def _hr_kpis(df: pd.DataFrame) -> List[Dict]:
    kpis = []
    atr_col  = _find_col(df, ["left","attrition","churned","exited"])
    sat_col  = _find_col(df, ["satisfaction"])
    _dept_col = _find_col(df, ["department","dept"])
    sal_col  = _find_col(df, ["salary"])
    ten_col  = _find_col(df, ["tenure","time_spend","years_at"])
    eval_col = _find_col(df, ["evaluation","performance","last_eval"])

    kpis.append({"label": "Total Records", "value": f"{len(df):,}",
                 "sub": "employees analysed", "color": C_NAVY, "delta": None})

    if atr_col:
        rate = float(df[atr_col].mean()) * 100
        n    = int(df[atr_col].sum())
        color = C_RED if rate > 20 else C_AMBER if rate > 12 else C_GREEN
        kpis.append({"label": "Attrition Rate", "value": f"{rate:.1f}%",
                     "sub": f"{n:,} employees departed",
                     "color": color, "delta": None})

    if sat_col:
        mean_s = float(df[sat_col].mean())
        target = 0.70
        delta  = mean_s - target
        color  = C_GREEN if mean_s >= 0.70 else C_AMBER if mean_s >= 0.55 else C_RED
        kpis.append({"label": "Avg Satisfaction", "value": f"{mean_s:.2f}/1.0",
                     "sub": f"Target: 0.70 | Gap: {delta:+.2f}",
                     "color": color, "delta": f"{delta:+.2f}"})

    if ten_col:
        med_t = float(df[ten_col].median())
        kpis.append({"label": "Median Tenure", "value": f"{med_t:.1f} yrs",
                     "sub": "Use median — right-skewed distribution",
                     "color": C_BLUE, "delta": None})

    if eval_col:
        mean_e = float(df[eval_col].mean())
        kpis.append({"label": "Avg Evaluation", "value": f"{mean_e:.2f}/1.0",
                     "sub": f"{int((df[eval_col] > 0.70).sum()):,} high performers (>0.70)",
                     "color": C_TEAL, "delta": None})

    if atr_col and sal_col:
        try:
            lo_rate = float(df[df[sal_col].str.lower() == "low"][atr_col].mean()) * 100
            hi_rate = float(df[df[sal_col].str.lower() == "high"][atr_col].mean()) * 100
            kpis.append({"label": "Salary Attrition Gap",
                         "value": f"{lo_rate - hi_rate:.1f}pp",
                         "sub": f"Low: {lo_rate:.1f}% | High: {hi_rate:.1f}%",
                         "color": C_AMBER if lo_rate - hi_rate > 10 else C_SLATE, "delta": None})
        except Exception:
            logger.warning("%s unexpected failure", exc_info=True)

    return kpis[:6]


def _hr_charts(df: pd.DataFrame) -> List[Tuple[str, go.Figure]]:
    charts = []
    atr_col  = _find_col(df, ["left","attrition","churned","exited"])
    sat_col  = _find_col(df, ["satisfaction"])
    dept_col = _find_col(df, ["department","dept"])
    sal_col  = _find_col(df, ["salary"])
    ten_col  = _find_col(df, ["tenure","time_spend","years_at"])
    _eval_col = _find_col(df, ["evaluation","performance","last_eval"])

    # Chart 1: Attrition by Department
    if dept_col and atr_col:
        try:
            d = df.groupby(dept_col)[atr_col].mean().reset_index()
            d.columns = ["Department", "Attrition Rate"]
            d["Attrition Rate"] *= 100
            d = d.sort_values("Attrition Rate", ascending=True)
            avg_rate = float(df[atr_col].mean()) * 100

            colors_bar = [C_RED if v > 25 else C_AMBER if v > 15 else C_GREEN
                          for v in d["Attrition Rate"]]
            fig = go.Figure(go.Bar(
                x=d["Attrition Rate"], y=d["Department"],
                orientation="h", marker_color=colors_bar,
                text=[f"{v:.1f}%" for v in d["Attrition Rate"]],
                textposition="outside",
            ))
            fig.add_vline(x=avg_rate, line_dash="dash", line_color=C_NAVY,
                          line_width=1.5,
                          annotation_text=f"Avg {avg_rate:.1f}%",
                          annotation_font_size=10)
            _apply_theme(fig, "Attrition Rate by Department",
                         "Red ≥25% · Amber 15–25% · Green <15%")
            fig.update_layout(height=max(300, len(d) * 42 + 80),
                              xaxis_title="Attrition Rate (%)")
            charts.append(("Attrition by Department", fig))
        except Exception:
            logger.warning("%s unexpected failure", exc_info=True)

    # Chart 2: Tenure Cohort Attrition
    if ten_col and atr_col:
        try:
            bins   = [0, 2, 4, 6, 8, 99]
            labels = ["1–2 yrs", "3–4 yrs", "5–6 yrs", "7–8 yrs", "9+ yrs"]
            df_t   = df.copy()
            df_t["_band"] = pd.cut(df_t[ten_col], bins=bins, labels=labels)
            cohort = df_t.groupby("_band", observed=True).agg(
                n=(atr_col, "count"), left=(atr_col, "sum")).reset_index()
            cohort["rate"] = cohort["left"] / cohort["n"] * 100
            avg_rate = float(df[atr_col].mean()) * 100
            colors_t = [C_RED if r > avg_rate * 1.5 else
                        C_AMBER if r > avg_rate else C_GREEN
                        for r in cohort["rate"]]

            fig = go.Figure(go.Bar(
                x=cohort["_band"], y=cohort["rate"],
                marker_color=colors_t,
                text=[f"{r:.1f}%\n(n={n:,})" for r, n in zip(cohort["rate"], cohort["n"])],
                textposition="outside",
            ))
            fig.add_hline(y=avg_rate, line_dash="dash", line_color=C_NAVY,
                          line_width=1.5,
                          annotation_text=f"Company avg {avg_rate:.1f}%",
                          annotation_position="bottom right")
            _apply_theme(fig, "Attrition by Tenure Cohort",
                         "The 5–6 year window is the highest-risk zone in most HR datasets")
            fig.update_layout(height=380, yaxis_title="Attrition Rate (%)")
            charts.append(("Tenure Cohort Attrition", fig))
        except Exception:
            logger.warning("%s unexpected failure", exc_info=True)

    # Chart 3: Satisfaction Distribution
    if sat_col:
        try:
            fig = px.histogram(df, x=sat_col, nbins=20,
                               color_discrete_sequence=[C_BLUE])
            fig.add_vline(x=float(df[sat_col].mean()), line_dash="solid",
                          line_color=C_RED, line_width=2,
                          annotation_text=f"Mean {df[sat_col].mean():.2f}",
                          annotation_font_color=C_RED)
            fig.add_vline(x=0.70, line_dash="dash", line_color=C_GREEN,
                          line_width=1.5,
                          annotation_text="Target 0.70",
                          annotation_font_color=C_GREEN)
            _apply_theme(fig, "Satisfaction Score Distribution",
                         f"{int((df[sat_col] < 0.40).sum()):,} employees below 0.40 (high-risk zone)")
            fig.update_layout(height=340, xaxis_title="Satisfaction Score",
                              yaxis_title="Employee Count")
            charts.append(("Satisfaction Distribution", fig))
        except Exception:
            logger.warning("%s unexpected failure", exc_info=True)

    # Chart 4: Satisfaction vs Attrition scatter
    if sat_col and atr_col:
        try:
            bins = np.arange(0.05, 1.05, 0.05)
            df_s = df.copy()
            df_s["_sbin"] = pd.cut(df_s[sat_col], bins=bins)
            agg = df_s.groupby("_sbin", observed=True).agg(
                n=(atr_col, "count"), rate=(atr_col, "mean")).reset_index()
            agg["mid"]  = agg["_sbin"].apply(lambda x: x.mid)
            agg["rate"] *= 100

            fig = go.Figure(go.Scatter(
                x=agg["mid"], y=agg["rate"],
                mode="markers",
                marker=dict(
                    size=[max(8, n / 5) for n in agg["n"]],
                    color=agg["rate"], colorscale="RdYlGn_r",
                    cmin=0, cmax=70,
                    showscale=True,
                    colorbar=dict(title="Attrition %", thickness=12),
                    line=dict(width=0.5, color="white"),
                ),
                text=[f"Sat: {m:.2f}<br>Attrition: {r:.1f}%<br>n={n:,}"
                      for m, r, n in zip(agg["mid"], agg["rate"], agg["n"])],
                hovertemplate="%{text}<extra></extra>",
            ))
            fig.add_vline(x=0.40, line_dash="dot", line_color=C_RED,
                          line_width=1.5, annotation_text="Risk threshold 0.40",
                          annotation_font_size=9)
            _apply_theme(fig, "Satisfaction Score vs Attrition Rate",
                         "Bubble size = employee count · Colour = attrition %")
            fig.update_layout(height=360, xaxis_title="Satisfaction Score",
                              yaxis_title="Attrition Rate (%)")
            charts.append(("Satisfaction vs Attrition", fig))
        except Exception:
            logger.warning("%s unexpected failure", exc_info=True)

    # Chart 5: Tenure × Salary Heatmap
    if ten_col and sal_col and atr_col:
        try:
            bins   = [0, 2, 4, 6, 8, 99]
            labels = ["1–2yr", "3–4yr", "5–6yr", "7–8yr", "9+yr"]
            df_h   = df.copy()
            df_h["_band"] = pd.cut(df_h[ten_col], bins=bins, labels=labels)
            heat = pd.crosstab(df_h["_band"], df_h[sal_col],
                               values=df_h[atr_col], aggfunc="mean") * 100
            heat = heat.fillna(0)
            # Reorder salary columns
            for order in [["low","medium","high"], ["Low","Medium","High"]]:
                if all(c in heat.columns for c in order):
                    heat = heat[order]
                    break

            fig = go.Figure(go.Heatmap(
                z=heat.values,
                x=[str(c) for c in heat.columns],
                y=[str(i) for i in heat.index],
                colorscale="RdYlGn_r", zmin=0, zmax=60,
                text=np.round(heat.values, 1),
                texttemplate="%{text:.1f}%",
                textfont=dict(size=12, color="white"),
                colorbar=dict(title="Attrition %", thickness=12),
            ))
            _apply_theme(fig, "Attrition Heatmap: Tenure × Salary",
                         "Darkest cell = highest attrition. Target intervention at 5–6yr + low salary.")
            fig.update_layout(height=320)
            charts.append(("Tenure × Salary Heatmap", fig))
        except Exception:
            logger.warning("%s unexpected failure", exc_info=True)

    return charts


# ═══════════════════════════════════════════════════════════════════════════════
#  FINANCE DOMAIN
# ═══════════════════════════════════════════════════════════════════════════════


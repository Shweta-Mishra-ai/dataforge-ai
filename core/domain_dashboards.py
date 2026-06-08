"""
core/domain_dashboards.py  — DataForge AI
Domain-specific KPI cards and chart sets for HR, Finance, E-Commerce, Sales.

Usage in pages/3_Dashboard.py:
    from core.domain_dashboards import get_domain_kpis, get_domain_charts
    kpis   = get_domain_kpis(df, domain)
    charts = get_domain_charts(df, domain)
"""
from __future__ import annotations
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import List, Dict, Optional, Tuple


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
SEQ_BLUE = [C_NAVY, C_BLUE, "#3B82F6", "#93C5FD", "#DBEAFE"]


def _apply_theme(fig: go.Figure, title: str, subtitle: str = "") -> go.Figure:
    title_text = f"<b>{title}</b>"
    if subtitle:
        title_text += f"<br><span style='font-size:11px;color:{C_SLATE}'>{subtitle}</span>"
    fig.update_layout(
        title=dict(text=title_text, font=dict(size=15, color=C_NAVY, family="Inter, Arial"),
                   x=0, xanchor="left", pad=dict(l=4)),
        paper_bgcolor="white", plot_bgcolor="white",
        font=dict(family="Inter, Arial", size=11, color=C_SLATE),
        margin=dict(l=55, r=20, t=72, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    font=dict(size=10), bgcolor="rgba(255,255,255,0.9)",
                    bordercolor=C_MGRAY, borderwidth=1),
        xaxis=dict(showgrid=False, showline=True, linecolor=C_MGRAY,
                   tickfont=dict(size=10), title_font=dict(size=11, color=C_SLATE)),
        yaxis=dict(showgrid=True, gridcolor=C_MGRAY, gridwidth=0.8,
                   showline=False, tickfont=dict(size=10), title_font=dict(size=11, color=C_SLATE)),
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
        pass
    return _general_kpis(df)


def get_domain_charts(df: pd.DataFrame, domain: str) -> List[Tuple[str, go.Figure]]:
    """Returns list of (chart_title, plotly_figure) tuples."""
    try:
        if domain == "hr":          return _hr_charts(df)
        if domain == "finance":     return _finance_charts(df)
        if domain == "ecommerce":   return _ecommerce_charts(df)
        if domain == "sales":       return _sales_charts(df)
    except Exception:
        pass
    return _general_charts(df)


# ═══════════════════════════════════════════════════════════════════════════════
#  HR DOMAIN
# ═══════════════════════════════════════════════════════════════════════════════

def _hr_kpis(df: pd.DataFrame) -> List[Dict]:
    kpis = []
    atr_col  = _find_col(df, ["left","attrition","churned","exited"])
    sat_col  = _find_col(df, ["satisfaction"])
    dept_col = _find_col(df, ["department","dept"])
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
            pass

    return kpis[:6]


def _hr_charts(df: pd.DataFrame) -> List[Tuple[str, go.Figure]]:
    charts = []
    atr_col  = _find_col(df, ["left","attrition","churned","exited"])
    sat_col  = _find_col(df, ["satisfaction"])
    dept_col = _find_col(df, ["department","dept"])
    sal_col  = _find_col(df, ["salary"])
    ten_col  = _find_col(df, ["tenure","time_spend","years_at"])
    eval_col = _find_col(df, ["evaluation","performance","last_eval"])

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
            pass

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
            pass

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
            pass

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
            pass

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
                    heat = heat[order]; break

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
            pass

    return charts


# ═══════════════════════════════════════════════════════════════════════════════
#  FINANCE DOMAIN
# ═══════════════════════════════════════════════════════════════════════════════

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
            pass

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
            pass

    if opex_col and rev_col:
        try:
            opex_ratio = float(df[opex_col].sum()) / float(df[rev_col].sum()) * 100
            color = C_GREEN if opex_ratio < 30 else C_AMBER if opex_ratio < 50 else C_RED
            kpis.append({"label": "OpEx Ratio", "value": f"{opex_ratio:.1f}%",
                         "sub": "Operating expenses as % of revenue",
                         "color": color, "delta": None})
        except Exception:
            pass

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
                pass

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
            pass

    # Chart 2: Budget vs Actual
    if budget_col and actual_col and budget_col != actual_col:
        try:
            comp_col = period_col or cat_col
            if comp_col:
                agg = df.groupby(comp_col)[[budget_col, actual_col]].sum().reset_index()
                try:
                    agg = agg.sort_values(actual_col, ascending=False).head(15)
                except Exception:
                    pass
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
            pass

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
            pass

    # Chart 4: Profit / Margin Trend
    if period_col and profit_col:
        try:
            trend = df.groupby(period_col)[profit_col].sum().reset_index()
            try:
                trend = trend.sort_values(period_col)
            except Exception:
                pass
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
            pass

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
            pass

    return charts


# ═══════════════════════════════════════════════════════════════════════════════
#  E-COMMERCE DOMAIN
# ═══════════════════════════════════════════════════════════════════════════════

def _ecommerce_kpis(df: pd.DataFrame) -> List[Dict]:
    kpis = []
    rating_col = _find_col(df, ["rating"], exclude=["count","num"])
    price_col  = _find_col(df, ["discounted_price","selling_price","price"], exclude=["actual","mrp"])
    disc_col   = _find_col(df, ["discount"])
    rev_col    = _find_col(df, ["revenue","sales","amount"])
    cat_col    = _find_col(df, ["category"])
    review_col = _find_col(df, ["review","review_count","no_of_rating"])

    kpis.append({"label": "Total Products", "value": f"{len(df):,}",
                 "sub": f"{df[cat_col].nunique()} categories" if cat_col else "all products",
                 "color": C_NAVY, "delta": None})

    if rating_col:
        mean_r = float(df[rating_col].mean())
        low_n  = int((df[rating_col] < 3.0).sum())
        color  = C_GREEN if mean_r >= 4.0 else C_AMBER if mean_r >= 3.5 else C_RED
        kpis.append({"label": "Avg Rating", "value": f"{mean_r:.2f}/5",
                     "sub": f"{low_n:,} products below 3.0",
                     "color": color, "delta": None})

    if price_col:
        mean_p  = float(df[price_col].mean())
        median_p= float(df[price_col].median())
        kpis.append({"label": "Avg Selling Price", "value": f"{mean_p:.0f}",
                     "sub": f"Median: {median_p:.0f} (use median — right-skewed)",
                     "color": C_BLUE, "delta": None})

    if disc_col:
        avg_d = float(df[disc_col].mean())
        max_d = float(df[disc_col].max())
        color = C_RED if avg_d > 50 else C_AMBER if avg_d > 30 else C_GREEN
        kpis.append({"label": "Avg Discount", "value": f"{avg_d:.1f}%",
                     "sub": f"Max: {max_d:.0f}% | High discount = margin risk",
                     "color": color, "delta": None})

    if rev_col:
        total_rev = float(df[rev_col].sum())
        kpis.append({"label": "Total Revenue", "value": f"{total_rev:,.0f}",
                     "sub": f"Avg per product: {total_rev/len(df):,.0f}",
                     "color": C_TEAL, "delta": None})

    if review_col:
        total_rev_count = int(df[review_col].sum())
        avg_rev         = float(df[review_col].mean())
        kpis.append({"label": "Total Reviews", "value": f"{total_rev_count:,}",
                     "sub": f"Avg per product: {avg_rev:.0f}",
                     "color": C_SLATE, "delta": None})

    return kpis[:6]


def _ecommerce_charts(df: pd.DataFrame) -> List[Tuple[str, go.Figure]]:
    charts = []
    rating_col = _find_col(df, ["rating"], exclude=["count","num"])
    price_col  = _find_col(df, ["discounted_price","selling_price","price"], exclude=["actual","mrp"])
    actual_col = _find_col(df, ["actual_price","mrp","original_price"])
    disc_col   = _find_col(df, ["discount"])
    cat_col    = _find_col(df, ["category"])
    rev_col    = _find_col(df, ["revenue","sales","amount"])

    # Chart 1: Rating by Category
    if cat_col and rating_col:
        try:
            cat_agg = (df.groupby(cat_col)
                         .agg(avg_rating=(rating_col, "mean"),
                              count=(rating_col, "count"))
                         .query("count >= 5")
                         .sort_values("avg_rating", ascending=True)
                         .tail(15)
                         .reset_index())
            colors_r = [C_RED if r < 3.5 else C_AMBER if r < 4.0 else C_GREEN
                        for r in cat_agg["avg_rating"]]
            fig = go.Figure(go.Bar(
                x=cat_agg["avg_rating"], y=cat_agg[cat_col],
                orientation="h", marker_color=colors_r,
                text=[f"{r:.2f} (n={n:,})" for r, n in zip(cat_agg["avg_rating"], cat_agg["count"])],
                textposition="outside",
            ))
            fig.add_vline(x=4.0, line_dash="dash", line_color=C_GREEN, line_width=1.5,
                          annotation_text="Target 4.0", annotation_font_size=10)
            _apply_theme(fig, "Average Rating by Category",
                         "Red <3.5 · Amber 3.5–4.0 · Green ≥4.0 · Min 5 products per category")
            fig.update_layout(height=max(280, len(cat_agg) * 38 + 80),
                              xaxis_title="Average Rating", xaxis=dict(range=[0, 5.5]))
            charts.append(("Rating by Category", fig))
        except Exception:
            pass

    # Chart 2: Discount vs Rating scatter
    if disc_col and rating_col:
        try:
            plot_df = df[[disc_col, rating_col]].dropna().copy()
            if cat_col:
                plot_df[cat_col] = df[cat_col]
            fig = px.scatter(plot_df, x=disc_col, y=rating_col,
                             color=cat_col if cat_col else None,
                             opacity=0.55,
                             color_discrete_sequence=px.colors.qualitative.Set2,
                             trendline="ols")
            fig.add_hline(y=4.0, line_dash="dash", line_color=C_GREEN, line_width=1.5,
                          annotation_text="Rating target 4.0")
            _apply_theme(fig, "Discount % vs Product Rating",
                         "Do higher discounts correlate with lower ratings? — OLS trendline shown")
            fig.update_layout(height=360,
                              xaxis_title="Discount (%)", yaxis_title="Rating")
            charts.append(("Discount vs Rating", fig))
        except Exception:
            pass

    # Chart 3: Price distribution
    if price_col:
        try:
            plot_df = df[price_col].dropna()
            fig = px.histogram(plot_df, nbins=30, color_discrete_sequence=[C_BLUE])
            fig.add_vline(x=float(plot_df.median()), line_dash="solid",
                          line_color=C_RED, line_width=2,
                          annotation_text=f"Median {plot_df.median():.0f}",
                          annotation_font_color=C_RED)
            fig.add_vline(x=float(plot_df.mean()), line_dash="dash",
                          line_color=C_AMBER, line_width=1.5,
                          annotation_text=f"Mean {plot_df.mean():.0f}",
                          annotation_font_color=C_AMBER)
            _apply_theme(fig, "Price Distribution",
                         "Use median for reporting — right-skewed distribution")
            fig.update_layout(height=320, xaxis_title=price_col, yaxis_title="Product Count")
            charts.append(("Price Distribution", fig))
        except Exception:
            pass

    # Chart 4: Revenue by Category
    if cat_col and rev_col:
        try:
            cat_rev = (df.groupby(cat_col)[rev_col]
                         .sum().sort_values(ascending=False).head(10).reset_index())
            cat_rev.columns = ["Category", "Revenue"]
            total_r = cat_rev["Revenue"].sum()
            cat_rev["pct"] = cat_rev["Revenue"] / total_r * 100
            fig = px.pie(cat_rev, names="Category", values="Revenue",
                         color_discrete_sequence=px.colors.qualitative.Set2,
                         hole=0.4)
            fig.update_traces(texttemplate="<b>%{label}</b><br>%{percent:.1%}",
                              textposition="outside")
            _apply_theme(fig, "Revenue Share by Category (Top 10)", "")
            fig.update_layout(height=380)
            charts.append(("Revenue by Category", fig))
        except Exception:
            pass

    # Chart 5: Price band heatmap (price range vs rating)
    if price_col and rating_col:
        try:
            df_pb = df[[price_col, rating_col]].dropna().copy()
            df_pb["price_band"] = pd.qcut(df_pb[price_col], q=5,
                                           labels=["Bottom 20%","20–40%","40–60%","60–80%","Top 20%"],
                                           duplicates="drop")
            df_pb["rating_band"] = pd.cut(df_pb[rating_col],
                                           bins=[0,2,3,3.5,4,5],
                                           labels=["<2","2–3","3–3.5","3.5–4","4–5"])
            heat = pd.crosstab(df_pb["price_band"], df_pb["rating_band"])
            fig  = go.Figure(go.Heatmap(
                z=heat.values,
                x=[str(c) for c in heat.columns],
                y=[str(i) for i in heat.index],
                colorscale="Blues",
                text=heat.values,
                texttemplate="%{text}",
                textfont=dict(size=11),
                colorbar=dict(title="Count", thickness=12),
            ))
            _apply_theme(fig, "Price Band vs Rating Distribution",
                         "How are products distributed across price tier and rating?")
            fig.update_layout(height=320,
                              xaxis_title="Rating Band", yaxis_title="Price Tier")
            charts.append(("Price Band vs Rating", fig))
        except Exception:
            pass

    return charts


# ═══════════════════════════════════════════════════════════════════════════════
#  SALES DOMAIN
# ═══════════════════════════════════════════════════════════════════════════════

def _sales_kpis(df: pd.DataFrame) -> List[Dict]:
    kpis = []
    rev_col    = _find_col(df, ["revenue","sales","amount","value"], exclude=["budget","target"])
    target_col = _find_col(df, ["target","quota","budget","plan"])
    rep_col    = _find_col(df, ["rep","salesperson","agent","executive","owner"])
    region_col = _find_col(df, ["region","territory","area","zone"])
    margin_col = _find_col(df, ["margin","profit","gross"])
    won_col    = _find_col(df, ["won","closed","status","result"])
    prod_col   = _find_col(df, ["product","item","sku","service"])

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
            pass

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
            pass

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
            pass

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
            pass

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
            pass

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
                pass

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
            pass

    # Chart 4: Deal Size Distribution
    if rev_col:
        try:
            deal_vals = df[rev_col].dropna()
            p80 = float(deal_vals.quantile(0.80))
            p20 = float(deal_vals.quantile(0.20))
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
            pass

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
            pass

    return charts


# ═══════════════════════════════════════════════════════════════════════════════
#  GENERAL FALLBACK
# ═══════════════════════════════════════════════════════════════════════════════

def _general_kpis(df: pd.DataFrame) -> List[Dict]:
    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include="object").columns.tolist()
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
            pass
    return charts


"""
core/dashboards/ecommerce.py — E-Commerce KPIs and charts.
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
    _actual_col = _find_col(df, ["actual_price","mrp","original_price"])
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
            logger.warning("%s unexpected failure", exc_info=True)

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
            logger.warning("%s unexpected failure", exc_info=True)

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
            logger.warning("%s unexpected failure", exc_info=True)

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
            logger.warning("%s unexpected failure", exc_info=True)

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
            logger.warning("%s unexpected failure", exc_info=True)

    return charts


# ═══════════════════════════════════════════════════════════════════════════════
#  SALES DOMAIN
# ═══════════════════════════════════════════════════════════════════════════════


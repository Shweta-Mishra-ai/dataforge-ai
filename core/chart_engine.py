"""
core/chart_engine.py — DataForge AI
=====================================
FIX v2.0 — Column Intelligence + Smart Chart Selection

CHANGES FROM v1:
  FIX-001: Column role detection — index/ID columns NEVER used as metrics
  FIX-002: Chart aggregation uses .mean() for scores, .sum() for revenue/qty
  FIX-003: Pie chart only for genuine part-of-whole (headcount, revenue share)
  FIX-004: Bar chart for averaged scores (satisfaction, rating) — never pie
  FIX-005: Time series only when real datetime column exists
  FIX-006: Sanity check — percentage outputs capped, absurd gaps flagged
  FIX-007: Domain-aware metric selection per domain contract
  FIX-008: Correlation matrix excludes ID columns

NO Streamlit imports — core layer rule.
"""

import re
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Tuple, Optional, Dict

# Human-readable column name translator
def _ht(col: str) -> str:
    """Convert raw column name to human-readable title for chart labels."""
    _MAP = {
        "satisfaction_level":    "Employee Satisfaction Score",
        "last_evaluation":       "Last Performance Evaluation",
        "number_project":        "Number of Active Projects",
        "average_montly_hours":  "Avg Monthly Hours Worked",
        "average_monthly_hours": "Avg Monthly Hours Worked",
        "time_spend_company":    "Employee Tenure (Years)",
        "work_accident":         "Work Accident Rate",
        "left":                  "Employee Attrition",
        "attrition":             "Attrition Rate",
        "promotion_last_5years": "Promoted in Last 5 Years",
        "department":            "Department",
        "dept":                  "Department",
        "salary":                "Salary Band",
        "discounted_price":      "Selling Price",
        "actual_price":          "Original Price (MRP)",
        "discount_percentage":   "Discount Applied (%)",
        "rating_count":          "Number of Reviews",
        "rating":                "Customer Rating",
        "amount":                "Order Revenue",
        "qty":                   "Order Quantity",
        "fulfilment":            "Fulfilment Method",
        "status":                "Order Status",
        "revenue":               "Revenue",
        "sales":                 "Sales Amount",
        "profit":                "Profit",
        "margin":                "Profit Margin (%)",
        "region":                "Sales Region",
        "pcs":                   "Units Sold (PCS)",
        "gross amt":             "Gross Revenue",
        "gross_amt":             "Gross Revenue",
        "rate":                  "Unit Rate",
    }
    low = col.lower().strip()
    if low in _MAP:
        return _MAP[low]
    # Fallback: clean up underscores and capitalise
    return " ".join(w.capitalize() for w in col.replace("_", " ").split())

PALETTE  = ["#4f8ef7", "#22d3a5", "#f7934f", "#a78bfa",
            "#f77070", "#ffd43b", "#38bdf8", "#fb7185"]
TEMPLATE = "plotly_dark"


# ══════════════════════════════════════════════════════════
#  FIX-001: COLUMN ROLE DETECTION
# ══════════════════════════════════════════════════════════

_IDENTIFIER_NAMES = {
    "index", "idx", "id", "row", "rowid", "row_id", "row_num",
    "rownum", "serial", "sr", "sr_no", "sno", "s_no",
    "order_id", "orderid", "customer_id", "customerid",
    "user_id", "userid", "emp_id", "empid", "employee_id",
    "product_id", "productid", "item_id", "itemid", "sku_id",
    "transaction_id", "txn_id", "record_id", "entry_id",
    "asin", "uuid", "guid",
}

_METRIC_NAMES = {
    "amount", "revenue", "sales", "price", "cost", "profit",
    "margin", "salary", "income", "spend", "budget", "expense",
    "qty", "quantity", "units", "volume", "count",
    "score", "rating", "satisfaction", "evaluation", "performance",
    "rate", "percentage", "pct", "percent",
    "hours", "days", "tenure", "age",
}

_PIE_VALID_METRICS = {
    "revenue", "sales", "amount", "profit", "spend",
    "qty", "quantity", "units", "volume", "count", "headcount",
}

_SCORE_METRICS = {
    "satisfaction", "rating", "score", "evaluation",
    "performance", "satisfaction_level", "last_evaluation",
}


def _is_identifier(col_name: str, series: pd.Series, n_rows: int) -> bool:
    col_lower = col_name.lower().strip()

    if col_lower in _IDENTIFIER_NAMES:
        return True

    if re.search(r'\bid\b|\bindex\b|\bidx\b', col_lower):
        return True

    if not pd.api.types.is_numeric_dtype(series):
        return False

    if any(kw in col_lower for kw in _METRIC_NAMES):
        return False

    n_unique = series.nunique()
    uniqueness_ratio = n_unique / max(n_rows, 1)

    if uniqueness_ratio > 0.95 and n_rows > 100:
        return True

    try:
        clean = series.dropna().sort_values().reset_index(drop=True)
        if len(clean) > 10:
            diffs = clean.diff().dropna()
            if (diffs == 1).mean() > 0.95:
                return True
    except Exception:
        pass

    return False


def _get_analysis_columns(df: pd.DataFrame) -> Dict:
    n_rows = len(df)
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    date_cols = df.select_dtypes(include="datetime").columns.tolist()

    id_cols = []
    metrics = []
    score_metrics = []

    for col in numeric_cols:
        if _is_identifier(col, df[col], n_rows):
            id_cols.append(col)
            continue
        col_lower = col.lower()
        if any(kw in col_lower for kw in _SCORE_METRICS):
            score_metrics.append(col)
        else:
            metrics.append(col)

    all_metrics = score_metrics + metrics
    dimensions = [
        c for c in cat_cols
        if 2 <= df[c].nunique() <= 30
        and not _is_identifier(c, df[c], n_rows)
    ]

    return {
        "metrics": metrics,
        "score_metrics": score_metrics,
        "all_metrics": all_metrics,
        "dimensions": dimensions,
        "date_cols": date_cols,
        "id_cols": id_cols,
        "cat_cols": cat_cols,
    }


def _pick_primary_metric(cols: Dict, domain: str = "general") -> Optional[str]:
    domain_priority = {
        "ecommerce": ["amount", "revenue", "sales", "price", "qty", "quantity"],
        "hr":        ["satisfaction_level", "satisfaction", "last_evaluation",
                      "salary", "average_montly_hours", "number_project"],
        "sales":     ["revenue", "sales", "amount", "profit", "margin"],
        "finance":   ["revenue", "profit", "amount", "income", "expense"],
        "marketing": ["revenue", "spend", "impressions", "clicks", "conversions"],
        "general":   [],
    }

    priority = domain_priority.get(domain, [])
    all_metrics = cols["all_metrics"]

    if not all_metrics:
        return None

    for preferred in priority:
        for col in all_metrics:
            if preferred in col.lower():
                return col

    if cols["score_metrics"]:
        return cols["score_metrics"][0]
    if cols["metrics"]:
        return cols["metrics"][0]

    return None


def _pick_best_dimension(cols: Dict, metric_col: Optional[str]) -> Optional[str]:
    dims = cols["dimensions"]
    if not dims:
        return None
    return dims[0]


def _is_score_metric(col_name: str) -> bool:
    col_lower = col_name.lower()
    return any(kw in col_lower for kw in _SCORE_METRICS)


def _is_pie_valid(metric_col: str) -> bool:
    col_lower = metric_col.lower()
    return any(kw in col_lower for kw in _PIE_VALID_METRICS)


# ══════════════════════════════════════════════════════════
#  STYLING
# ══════════════════════════════════════════════════════════

def _style(fig: go.Figure) -> go.Figure:
    fig.update_layout(
        paper_bgcolor="#07080f",
        plot_bgcolor="#0e0f1a",
        font=dict(family="JetBrains Mono, monospace", color="#dde1f5"),
        margin=dict(l=20, r=20, t=50, b=20),
    )
    fig.update_xaxes(gridcolor="#1e2035", zeroline=False)
    fig.update_yaxes(gridcolor="#1e2035", zeroline=False)
    return fig


# ══════════════════════════════════════════════════════════
#  FIX-002 thru FIX-008: SMART CHART RECOMMENDER
# ══════════════════════════════════════════════════════════

def recommend_charts(
    df: pd.DataFrame,
    domain: str = "general"
) -> List[Tuple[str, go.Figure]]:
    """
    5 meaningful charts using column role detection.
    ID columns excluded. Aggregation method matches metric type.
    """
    cols = _get_analysis_columns(df)
    charts = []

    primary_metric = _pick_primary_metric(cols, domain)
    best_dim = _pick_best_dimension(cols, primary_metric)
    all_metrics = cols["all_metrics"]
    date_cols = cols["date_cols"]

    # Chart 1: Primary metric by dimension (BAR)
    if primary_metric and best_dim:
        is_score = _is_score_metric(primary_metric)
        agg_func = "mean" if is_score else "sum"
        label = "Avg" if is_score else "Total"

        agg = (
            df.groupby(best_dim)[primary_metric]
            .agg(agg_func)
            .reset_index()
            .sort_values(primary_metric, ascending=False)
            .head(15)
        )
        agg[primary_metric] = agg[primary_metric].round(4 if is_score else 0)

        fig = px.bar(
            agg, x=best_dim, y=primary_metric,
            title=f"{label} {_ht(primary_metric)} by {_ht(best_dim)}",
            template=TEMPLATE,
            color=primary_metric,
            color_continuous_scale="Blues",
            text=primary_metric,
        )
        fig.update_traces(
            texttemplate="%{text:.3f}" if is_score else "%{text:,.0f}",
            textposition="outside"
        )
        charts.append((_ht(primary_metric) + " by " + _ht(best_dim), _style(fig)))

    # Chart 2: Trend (LINE) — only real datetime, never order_id
    if date_cols and primary_metric:
        date_col = date_cols[0]
        try:
            trend = (
                df.groupby(pd.Grouper(key=date_col, freq="M"))[primary_metric]
                .mean()
                .reset_index()
            )
            trend.columns = [date_col, primary_metric]
            trend = trend.dropna()

            if len(trend) >= 2:
                fig = px.line(
                    trend, x=date_col, y=primary_metric,
                    title=f"{_ht(primary_metric)} Trend Over Time",
                    template=TEMPLATE,
                    color_discrete_sequence=PALETTE,
                    markers=True,
                )
                charts.append(("Trend Over Time", _style(fig)))
        except Exception:
            pass

    # Chart 3: Distribution histogram
    if primary_metric:
        fig = px.histogram(
            df, x=primary_metric, nbins=40, marginal="box",
            title=f"Distribution: {_ht(primary_metric)}",
            template=TEMPLATE,
            color_discrete_sequence=PALETTE,
        )
        charts.append(("Distribution", _style(fig)))

    # Chart 4: Correlation matrix — ID columns excluded
    if len(all_metrics) >= 2:
        corr = df[all_metrics].corr().round(2)
        fig = px.imshow(
            corr, text_auto=True,
            title="Correlation Matrix",
            template=TEMPLATE,
            color_continuous_scale="RdBu_r",
            zmin=-1, zmax=1,
        )
        charts.append(("Correlation Matrix", _style(fig)))

    # Chart 5: Share or Ranking
    if primary_metric and best_dim:
        if _is_pie_valid(primary_metric):
            agg = (
                df.groupby(best_dim)[primary_metric]
                .sum().reset_index()
                .sort_values(primary_metric, ascending=False).head(8)
            )
            fig = px.pie(
                agg, names=best_dim, values=primary_metric,
                title=f"{_ht(primary_metric)} Share by {_ht(best_dim)}",
                template=TEMPLATE,
                color_discrete_sequence=PALETTE,
            )
            charts.append(("Share by " + _ht(best_dim), _style(fig)))
        else:
            # Score metric — horizontal bar ranked chart
            agg = (
                df.groupby(best_dim)[primary_metric]
                .mean().reset_index()
                .sort_values(primary_metric, ascending=True)
            )
            agg[primary_metric] = agg[primary_metric].round(3)
            fig = px.bar(
                agg, x=primary_metric, y=best_dim, orientation="h",
                title=f"{_ht(primary_metric)} Ranking by {_ht(best_dim)}",
                template=TEMPLATE,
                color=primary_metric,
                color_continuous_scale="Blues",
                text=primary_metric,
            )
            fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
            charts.append((_ht(primary_metric) + " Ranking by " + _ht(best_dim), _style(fig)))

    return charts[:5]


# ══════════════════════════════════════════════════════════
#  INDIVIDUAL CHART BUILDERS
# ══════════════════════════════════════════════════════════

def make_bar(df: pd.DataFrame, x: str, y: str, title: str = "") -> go.Figure:
    is_score = _is_score_metric(y)
    agg_func = "mean" if is_score else "sum"
    agg = (
        df.groupby(x)[y].agg(agg_func)
        .reset_index()
        .sort_values(y, ascending=False)
        .head(20)
    )
    agg[y] = agg[y].round(4 if is_score else 0)
    return _style(px.bar(
        agg, x=x, y=y,
        title=title or f"{'Avg' if is_score else 'Total'} {y} by {x}",
        template=TEMPLATE, color=y, color_continuous_scale="Blues",
    ))


def make_horizontal_bar(
    df: pd.DataFrame, x: str, y: str, title: str = ""
) -> go.Figure:
    is_score = _is_score_metric(y)
    agg = (
        df.groupby(x)[y]
        .agg("mean" if is_score else "sum")
        .reset_index()
        .sort_values(y, ascending=True).head(20)
    )
    return _style(px.bar(
        agg, x=y, y=x, orientation="h",
        title=title or f"{y} Ranking by {x}",
        template=TEMPLATE, color=y, color_continuous_scale="Blues",
    ))


def make_line(df: pd.DataFrame, x: str, y: str, title: str = "") -> go.Figure:
    return _style(px.line(
        df.sort_values(x), x=x, y=y,
        title=title or f"{y} over {x}",
        template=TEMPLATE, color_discrete_sequence=PALETTE, markers=True,
    ))


def make_scatter(
    df: pd.DataFrame, x: str, y: str,
    color: Optional[str] = None, title: str = ""
) -> go.Figure:
    return _style(px.scatter(
        df.head(3000), x=x, y=y, color=color,
        title=title or f"{x} vs {y}",
        template=TEMPLATE, color_discrete_sequence=PALETTE, opacity=0.7,
    ))


def make_histogram(
    df: pd.DataFrame, col: str, nbins: int = 40, title: str = ""
) -> go.Figure:
    return _style(px.histogram(
        df, x=col, nbins=nbins, marginal="box",
        title=title or f"Distribution: {col}",
        template=TEMPLATE, color_discrete_sequence=PALETTE,
    ))


def make_pie(
    df: pd.DataFrame, names_col: str, values_col: str, title: str = ""
) -> go.Figure:
    """Guard: redirects score metrics to horizontal bar."""
    if not _is_pie_valid(values_col):
        return make_horizontal_bar(
            df, names_col, values_col,
            title=title or f"{values_col} by {names_col}"
        )
    agg = df.groupby(names_col)[values_col].sum().reset_index().head(10)
    return _style(px.pie(
        agg, names=names_col, values=values_col,
        title=title or f"{values_col} Share by {names_col}",
        template=TEMPLATE, color_discrete_sequence=PALETTE,
    ))


def make_heatmap(df: pd.DataFrame, domain: str = "general") -> go.Figure:
    """Correlation matrix with ID columns excluded."""
    cols = _get_analysis_columns(df)
    metric_cols = cols["all_metrics"]
    if len(metric_cols) < 2:
        metric_cols = df.select_dtypes(include="number").columns.tolist()
    corr = df[metric_cols].corr().round(2)
    return _style(px.imshow(
        corr, text_auto=True, title="Correlation Matrix",
        template=TEMPLATE, color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
    ))


# ══════════════════════════════════════════════════════════
#  FIX-006: SANITY CHECK
# ══════════════════════════════════════════════════════════

def safe_pct_gap(val_a: float, val_b: float) -> str:
    """Cap absurd percentage gaps — anything >999% = column selection error."""
    if val_b == 0:
        return "N/A (baseline is zero)"
    gap = abs(val_a - val_b) / abs(val_b) * 100
    if gap > 999:
        return ">999% (verify column selection)"
    return f"{gap:.1f}%"

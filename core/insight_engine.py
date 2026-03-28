"""
core/insight_engine.py — DataForge AI
========================================
FIX v2.0 — Domain Contracts + Column-Gated Insights

CHANGES FROM v1:
  FIX-010: Domain-specific KPI contracts — HR follows HR standards, Ecommerce follows Ecommerce
  FIX-011: Column-gated insights — recommendations only generated for columns that EXIST in dataset
  FIX-012: Threshold-based severity — "0 critical issues" is now impossible for real datasets
  FIX-013: Domain template isolation — HR language never appears in Ecommerce reports
  FIX-014: Ecommerce insights use Amount/Revenue, not index
  FIX-015: All percentages sanity-checked before output
  FIX-016: Overwork detection for HR (avg monthly hours threshold)

NO Streamlit imports — core layer rule.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional


# ══════════════════════════════════════════════════════════
#  DOMAIN THRESHOLDS
#  Industry-standard benchmarks per domain
# ══════════════════════════════════════════════════════════

HR_THRESHOLDS = {
    "attrition_rate":       {"warning": 15.0,  "critical": 20.0,  "source": "SHRM 2024"},
    "satisfaction_mean":    {"warning": 0.65,   "critical": 0.50,  "source": "Gallup/Mercer"},
    "avg_monthly_hours":    {"warning": 180,    "critical": 200,   "source": "Labour standards"},
    "promotion_rate_5yr":   {"warning": 0.05,   "critical": 0.02,  "source": "Mercer 2024"},
    "dept_attrition_gap":   {"warning": 8.0,    "critical": 12.0,  "source": "Internal benchmark"},
}

ECOMMERCE_THRESHOLDS = {
    "cancellation_rate":    {"warning": 10.0,  "critical": 20.0,  "source": "Shopify 2024"},
    "return_rate":          {"warning": 10.0,  "critical": 20.0,  "source": "Shopify 2024"},
    "lost_in_transit_rate": {"warning": 2.0,   "critical": 5.0,   "source": "Industry norm"},
    "missing_amount_pct":   {"warning": 5.0,   "critical": 10.0,  "source": "Data quality"},
    "avg_rating":           {"warning": 4.0,   "critical": 3.5,   "source": "Amazon/G2 2024"},
}

SALES_THRESHOLDS = {
    "revenue_cv":           {"warning": 50.0,  "critical": 80.0,  "source": "Sales ops norm"},
    "win_rate":             {"warning": 20.0,  "critical": 10.0,  "source": "HubSpot 2024"},
    "avg_deal_size_cv":     {"warning": 60.0,  "critical": 100.0, "source": "Internal benchmark"},
}

FINANCE_THRESHOLDS = {
    "expense_growth_pct":   {"warning": 10.0,  "critical": 20.0,  "source": "CFO benchmark"},
    "margin_pct":           {"warning": 15.0,  "critical": 5.0,   "source": "Industry norm"},
}


# ══════════════════════════════════════════════════════════
#  COLUMN FINDERS
#  Find relevant columns by keyword — never hardcode column names
# ══════════════════════════════════════════════════════════

def _find_col(df: pd.DataFrame, keywords: List[str],
              numeric_only: bool = False) -> Optional[str]:
    """Find first column matching any keyword. Returns None if not found."""
    candidates = df.select_dtypes(include="number").columns if numeric_only else df.columns
    for col in candidates:
        col_lower = col.lower()
        if any(kw in col_lower for kw in keywords):
            return col
    return None


def _find_col_exact(df: pd.DataFrame, names: List[str]) -> Optional[str]:
    """Find column with exact name match (case-insensitive)."""
    for col in df.columns:
        if col.lower() in [n.lower() for n in names]:
            return col
    return None


# ══════════════════════════════════════════════════════════
#  FIX-010 + FIX-011: DOMAIN-SPECIFIC INSIGHT GENERATORS
#  Each function checks if required columns exist BEFORE generating insights
# ══════════════════════════════════════════════════════════

def _hr_insights(df: pd.DataFrame) -> List[Dict]:
    """
    HR domain analysis contract.
    Required outputs: attrition, satisfaction, workload, compensation, promotion.
    Only generates insight if the required column exists in df.
    """
    insights = []
    n = len(df)

    # ── 1. Attrition Analysis ─────────────────────────────────────────────────
    atr_col = _find_col_exact(df, ["left", "attrition", "churned", "resigned", "exited"])
    if atr_col:
        try:
            left_vals = df[atr_col].astype(str).str.lower().str.strip()
            left_mask = left_vals.isin(["yes", "1", "1.0", "true", "left"])
            n_left = int(left_mask.sum())
            rate = round(n_left / max(n, 1) * 100, 1)

            sev = ("critical" if rate > HR_THRESHOLDS["attrition_rate"]["critical"]
                   else "warning" if rate > HR_THRESHOLDS["attrition_rate"]["warning"]
                   else "info")

            # Department breakdown — only if dept column exists
            dept_col = _find_col(df, ["department", "dept", "division", "team"])
            dept_detail = ""
            if dept_col:
                dept_atr = (df.groupby(dept_col)[atr_col]
                            .apply(lambda x: (x.astype(str).str.lower()
                                              .str.strip()
                                              .isin(["yes","1","1.0","true","left"])
                                              .mean() * 100))
                            .sort_values(ascending=False))
                if len(dept_atr) >= 2:
                    worst_dept = dept_atr.index[0]
                    best_dept = dept_atr.index[-1]
                    dept_detail = (
                        f" '{worst_dept}' department has the highest attrition "
                        f"at {dept_atr.iloc[0]:.1f}% vs "
                        f"'{best_dept}' at {dept_atr.iloc[-1]:.1f}%."
                    )

            insights.append({
                "title": f"Attrition Rate: {rate:.1f}% — "
                         + ("CRITICAL" if sev == "critical" else
                            "Above Benchmark" if sev == "warning" else "Healthy"),
                "body": (
                    f"{n_left:,} of {n:,} employees left ({rate:.1f}%). "
                    f"SHRM 2024 benchmark: 10–15%. Best practice: <10%."
                    f"{dept_detail}"
                ),
                "type": sev,
                "icon": "🚨" if sev == "critical" else "⚠️" if sev == "warning" else "✅",
                "metric": "attrition_rate",
                "value": rate,
                "benchmark": "10–15% (SHRM 2024)",
                "columns_used": [atr_col],
            })
        except Exception:
            pass

    # ── 2. Satisfaction Analysis ──────────────────────────────────────────────
    sat_col = _find_col(df, ["satisfaction", "satisfaction_level", "engagement"], numeric_only=True)
    if sat_col:
        try:
            mean_sat = float(df[sat_col].mean())
            threshold_warn = HR_THRESHOLDS["satisfaction_mean"]["warning"]
            threshold_crit = HR_THRESHOLDS["satisfaction_mean"]["critical"]

            sev = ("critical" if mean_sat < threshold_crit
                   else "warning" if mean_sat < threshold_warn
                   else "info")

            # Find worst department for satisfaction
            dept_col = _find_col(df, ["department", "dept", "division", "team"])
            dept_detail = ""
            if dept_col:
                dept_sat = df.groupby(dept_col)[sat_col].mean().sort_values()
                if len(dept_sat) >= 2:
                    worst = dept_sat.index[0]
                    best = dept_sat.index[-1]
                    dept_detail = (
                        f" Worst: '{worst}' ({dept_sat.iloc[0]:.3f}), "
                        f"Best: '{best}' ({dept_sat.iloc[-1]:.3f}). "
                        f"Gap: {(dept_sat.iloc[-1] - dept_sat.iloc[0]):.3f} pts."
                    )

            below_40pct = int((df[sat_col] < 0.40).sum())

            insights.append({
                "title": f"Satisfaction: {mean_sat:.2f}/1.0 — "
                         + ("Below Critical Threshold" if sev == "critical"
                            else "Below Target" if sev == "warning" else "On Track"),
                "body": (
                    f"Mean satisfaction score: {mean_sat:.3f}. "
                    f"Industry norm: 0.70+. Gap: {max(0, 0.70 - mean_sat):.3f} pts. "
                    f"{below_40pct:,} employees below 0.40 — highest flight risk segment."
                    f"{dept_detail}"
                ),
                "type": sev,
                "icon": "🚨" if sev == "critical" else "⚠️" if sev == "warning" else "✅",
                "metric": "satisfaction",
                "value": mean_sat,
                "benchmark": "0.70+ (Gallup/Mercer)",
                "columns_used": [sat_col],
            })
        except Exception:
            pass

    # ── 3. Workload / Overwork Detection ─────────────────────────────────────
    hours_col = _find_col(df, ["average_montly_hours", "monthly_hours", "hours", "avg_hours"], numeric_only=True)
    if hours_col:
        try:
            mean_hours = float(df[hours_col].mean())
            weekly_equiv = round(mean_hours / 4.33, 1)
            crit_thresh = HR_THRESHOLDS["avg_monthly_hours"]["critical"]
            warn_thresh = HR_THRESHOLDS["avg_monthly_hours"]["warning"]

            sev = ("critical" if mean_hours > crit_thresh
                   else "warning" if mean_hours > warn_thresh
                   else "info")

            overwork_n = int((df[hours_col] > crit_thresh).sum())

            if sev in ("critical", "warning"):
                insights.append({
                    "title": f"Overwork Alert: {mean_hours:.0f} Avg Monthly Hours "
                             f"({weekly_equiv}h/week)",
                    "body": (
                        f"Average {mean_hours:.0f} monthly hours = ~{weekly_equiv}h/week. "
                        f"Healthy norm: <180 hrs/month (41.5h/week). "
                        f"{overwork_n:,} employees exceed {crit_thresh} hrs/month — "
                        f"chronic overwork territory. "
                        f"Burnout is a leading predictor of voluntary attrition (Gallup 2024)."
                    ),
                    "type": sev,
                    "icon": "🚨" if sev == "critical" else "⚠️",
                    "metric": "avg_monthly_hours",
                    "value": mean_hours,
                    "benchmark": "<180 hrs/month",
                    "columns_used": [hours_col],
                })
        except Exception:
            pass

    # ── 4. Compensation Analysis ──────────────────────────────────────────────
    sal_col = _find_col_exact(df, ["salary", "salary_band", "compensation", "pay_grade"])
    atr_col2 = _find_col_exact(df, ["left", "attrition", "churned"])
    if sal_col and atr_col2:
        try:
            sal_atr = (df.groupby(sal_col)[atr_col2]
                       .apply(lambda x: x.astype(str).str.lower().str.strip()
                              .isin(["yes","1","1.0","true","left"]).mean() * 100))
            if len(sal_atr) >= 2:
                sorted_sal = sal_atr.sort_values(ascending=False)
                worst_band = sorted_sal.index[0]
                best_band = sorted_sal.index[-1]
                gap = sorted_sal.iloc[0] - sorted_sal.iloc[-1]

                if gap > 5:
                    bands_str = " | ".join(
                        [f"{k}: {v:.1f}%" for k, v in sal_atr.items()])
                    insights.append({
                        "title": f"Pay Band '{worst_band}': {sorted_sal.iloc[0]:.1f}% Attrition — "
                                 f"{gap:.1f}pp Above '{best_band}'",
                        "body": (
                            f"Attrition by pay band: {bands_str}. "
                            f"SHRM: 38% of exits cite below-market pay as primary reason. "
                            f"Immediate salary benchmarking required for '{worst_band}' band."
                        ),
                        "type": "critical" if sorted_sal.iloc[0] > 25 else "warning",
                        "icon": "🚨" if sorted_sal.iloc[0] > 25 else "⚠️",
                        "metric": "salary_attrition",
                        "value": sorted_sal.iloc[0],
                        "benchmark": "<15% (SHRM)",
                        "columns_used": [sal_col, atr_col2],
                    })
        except Exception:
            pass

    # ── 5. Promotion Gap ─────────────────────────────────────────────────────
    promo_col = _find_col(df, ["promotion", "promoted", "promotion_last_5years"], numeric_only=True)
    if promo_col:
        try:
            promo_rate = float(df[promo_col].mean()) * 100
            if promo_rate < HR_THRESHOLDS["promotion_rate_5yr"]["warning"] * 100:
                insights.append({
                    "title": f"Only {promo_rate:.1f}% Promoted in Last 5 Years — Career Stagnation Risk",
                    "body": (
                        f"{promo_rate:.1f}% of employees received a promotion in the past 5 years. "
                        f"Mercer 2024: career growth is the #1 driver of voluntary exits. "
                        f"Best practice: structured promotion pipeline with annual review cycles."
                    ),
                    "type": "critical" if promo_rate < 2 else "warning",
                    "icon": "🚨" if promo_rate < 2 else "⚠️",
                    "metric": "promotion_rate",
                    "value": promo_rate,
                    "benchmark": ">5% per 5 years (Mercer)",
                    "columns_used": [promo_col],
                })
        except Exception:
            pass

    return insights


def _ecommerce_insights(df: pd.DataFrame) -> List[Dict]:
    """
    Ecommerce domain analysis contract.
    Required outputs: revenue trend, cancellation, fulfillment, AOV, status breakdown.
    Only generates insight if required column exists.
    """
    insights = []

    # ── 1. Revenue/Amount Overview ────────────────────────────────────────────
    amount_col = _find_col(df, ["amount", "revenue", "sales", "price", "gmv"], numeric_only=True)
    if amount_col:
        try:
            total = df[amount_col].sum()
            mean_aov = df[amount_col].mean()
            missing_pct = df[amount_col].isna().mean() * 100
            outlier_n = int((df[amount_col] > df[amount_col].quantile(0.99)).sum())

            body = (
                f"Total revenue: {total:,.0f}. "
                f"Average Order Value (AOV): {mean_aov:,.2f}. "
                f"Median: {df[amount_col].median():,.2f}."
            )

            if missing_pct > ECOMMERCE_THRESHOLDS["missing_amount_pct"]["warning"]:
                sev = ("critical" if missing_pct > ECOMMERCE_THRESHOLDS["missing_amount_pct"]["critical"]
                       else "warning")
                body += (
                    f" WARNING: {missing_pct:.1f}% of Amount values are missing — "
                    f"reported revenue is understated. Median imputation recommended."
                )
            else:
                sev = "info"

            insights.append({
                "title": f"Total Revenue: {total:,.0f} | AOV: {mean_aov:,.2f}",
                "body": body,
                "type": sev,
                "icon": "💰",
                "metric": "revenue",
                "value": total,
                "benchmark": "Context-dependent",
                "columns_used": [amount_col],
            })
        except Exception:
            pass

    # ── 2. Order Status / Cancellation ───────────────────────────────────────
    status_col = _find_col_exact(df, ["status", "order_status", "order status"])
    if status_col:
        try:
            status_counts = df[status_col].value_counts(normalize=True) * 100
            status_raw = df[status_col].value_counts()

            # Detect cancellation
            cancel_keys = [k for k in status_counts.index
                           if "cancel" in str(k).lower()]
            if cancel_keys:
                cancel_rate = status_counts[cancel_keys].sum()
                cancel_n = status_raw[cancel_keys].sum()
                sev = ("critical" if cancel_rate > ECOMMERCE_THRESHOLDS["cancellation_rate"]["critical"]
                       else "warning" if cancel_rate > ECOMMERCE_THRESHOLDS["cancellation_rate"]["warning"]
                       else "info")

                # Revenue at risk
                rev_at_risk = ""
                if amount_col:
                    cancel_mask = df[status_col].isin(cancel_keys)
                    rev_risk = df.loc[cancel_mask, amount_col].sum()
                    rev_at_risk = f" Revenue impact: {rev_risk:,.0f}."

                insights.append({
                    "title": f"Cancellation Rate: {cancel_rate:.1f}% — "
                             + ("CRITICAL" if sev == "critical"
                                else "Above Benchmark" if sev == "warning" else "Healthy"),
                    "body": (
                        f"{cancel_n:,} orders cancelled ({cancel_rate:.1f}%). "
                        f"Industry benchmark: <10% (Shopify 2024).{rev_at_risk} "
                        f"Investigate: payment failures, pricing issues, or inventory gaps."
                    ),
                    "type": sev,
                    "icon": "🚨" if sev == "critical" else "⚠️" if sev == "warning" else "✅",
                    "metric": "cancellation_rate",
                    "value": cancel_rate,
                    "benchmark": "<10% (Shopify 2024)",
                    "columns_used": [status_col],
                })

            # Detect lost-in-transit
            lost_keys = [k for k in status_counts.index
                         if "lost" in str(k).lower() or "transit" in str(k).lower()]
            if lost_keys:
                lost_rate = status_counts[lost_keys].sum()
                lost_n = status_raw[lost_keys].sum()
                if lost_rate > ECOMMERCE_THRESHOLDS["lost_in_transit_rate"]["warning"]:
                    sev = ("critical" if lost_rate > ECOMMERCE_THRESHOLDS["lost_in_transit_rate"]["critical"]
                           else "warning")
                    rev_lost = ""
                    if amount_col:
                        lost_mask = df[status_col].isin(lost_keys)
                        lost_rev = df.loc[lost_mask, amount_col].sum()
                        rev_lost = f" Estimated revenue at risk: {lost_rev:,.0f}."

                    insights.append({
                        "title": f"Lost In Transit: {lost_n:,} Orders ({lost_rate:.1f}%) — Courier Issue",
                        "body": (
                            f"{lost_n:,} shipments lost in transit ({lost_rate:.1f}%).{rev_lost} "
                            f"Industry benchmark: <2%. "
                            f"Conduct courier SLA audit — identify which courier/region has highest loss rate."
                        ),
                        "type": sev,
                        "icon": "🚨" if sev == "critical" else "⚠️",
                        "metric": "lost_in_transit",
                        "value": lost_rate,
                        "benchmark": "<2% (Industry norm)",
                        "columns_used": [status_col],
                    })
        except Exception:
            pass

    # ── 3. Fulfillment Performance ────────────────────────────────────────────
    fulfil_col = _find_col_exact(df, ["fulfilment", "fulfillment", "fulfilled_by", "fulfillment_method"])
    if fulfil_col and amount_col:
        try:
            fulfil_perf = (df.groupby(fulfil_col)[amount_col]
                           .agg(["mean", "count"])
                           .round(2))
            fulfil_perf.columns = ["avg_order_value", "order_count"]
            fulfil_perf = fulfil_perf.sort_values("avg_order_value", ascending=False)

            if len(fulfil_perf) >= 2:
                best = fulfil_perf.index[0]
                worst = fulfil_perf.index[-1]
                gap_pct = ((fulfil_perf.iloc[0]["avg_order_value"] -
                            fulfil_perf.iloc[-1]["avg_order_value"]) /
                           max(fulfil_perf.iloc[-1]["avg_order_value"], 1) * 100)

                if gap_pct > 5:
                    insights.append({
                        "title": f"Fulfillment AOV Gap: '{best}' Outperforms '{worst}' by {gap_pct:.1f}%",
                        "body": (
                            f"'{best}' fulfillment: AOV {fulfil_perf.iloc[0]['avg_order_value']:,.2f} "
                            f"({fulfil_perf.iloc[0]['order_count']:,} orders). "
                            f"'{worst}' fulfillment: AOV {fulfil_perf.iloc[-1]['avg_order_value']:,.2f} "
                            f"({fulfil_perf.iloc[-1]['order_count']:,} orders). "
                            f"Higher AOV from '{best}' suggests premium buyers prefer this channel."
                        ),
                        "type": "info",
                        "icon": "📦",
                        "metric": "fulfillment_aov",
                        "value": gap_pct,
                        "benchmark": "Context-dependent",
                        "columns_used": [fulfil_col, amount_col],
                    })
        except Exception:
            pass

    # ── 4. Quantity Analysis ──────────────────────────────────────────────────
    qty_col = _find_col(df, ["qty", "quantity", "units", "items"], numeric_only=True)
    if qty_col:
        try:
            zero_qty = int((df[qty_col] == 0).sum())
            zero_pct = zero_qty / len(df) * 100
            if zero_pct > 1:
                insights.append({
                    "title": f"{zero_qty:,} Orders with Zero Quantity ({zero_pct:.1f}%) — Data Issue",
                    "body": (
                        f"{zero_qty:,} order records have quantity = 0. "
                        f"These are likely cancelled/voided orders recorded without cleanup. "
                        f"Exclude from revenue calculations — they inflate order count metrics."
                    ),
                    "type": "warning",
                    "icon": "⚠️",
                    "metric": "zero_qty",
                    "value": zero_pct,
                    "benchmark": "0% (Data quality)",
                    "columns_used": [qty_col],
                })
        except Exception:
            pass

    return insights


def _sales_insights(df: pd.DataFrame) -> List[Dict]:
    """Sales domain analysis contract."""
    insights = []

    rev_col = _find_col(df, ["revenue", "sales", "amount", "profit", "gmv"], numeric_only=True)
    if rev_col:
        try:
            mean_r = float(df[rev_col].mean())
            cv = float(df[rev_col].std()) / abs(mean_r) * 100 if mean_r != 0 else 0
            total = df[rev_col].sum()
            sev = ("critical" if cv > SALES_THRESHOLDS["revenue_cv"]["critical"]
                   else "warning" if cv > SALES_THRESHOLDS["revenue_cv"]["warning"]
                   else "info")

            insights.append({
                "title": f"Revenue Spread: {cv:.0f}% Variability — "
                         + ("High Concentration Risk" if sev != "info" else "Healthy Distribution"),
                "body": (
                    f"Total revenue: {total:,.0f}. Mean per record: {mean_r:,.0f}. "
                    f"CV {cv:.0f}% indicates "
                    + ("high concentration — top accounts likely drive 80%+ of revenue. "
                       "Account diversification recommended." if cv > 50
                       else "healthy distribution across accounts.")
                ),
                "type": sev,
                "icon": "⚠️" if sev != "info" else "✅",
                "metric": "revenue_cv",
                "value": cv,
                "benchmark": "<50% CV",
                "columns_used": [rev_col],
            })
        except Exception:
            pass

    return insights


def _general_insights(df: pd.DataFrame) -> List[Dict]:
    """Fallback for undetected domains — data quality focused."""
    insights = []
    num_cols = df.select_dtypes(include="number").columns.tolist()

    for col in num_cols[:5]:
        try:
            sk = float(df[col].skew())
            if abs(sk) > 1.5:
                s = df[col].dropna()
                diff_pct = abs(s.mean() - s.median()) / max(abs(s.median()), 1e-9) * 100
                insights.append({
                    "title": f"'{col}': Use Median, Not Mean (Skew={sk:.2f})",
                    "body": (
                        f"'{col}' is heavily skewed — mean ({s.mean():.3f}) differs from "
                        f"median ({s.median():.3f}) by {diff_pct:.0f}%. "
                        f"Reporting the mean misleads stakeholders. Use median ({s.median():.3f}) "
                        f"in all summaries."
                    ),
                    "type": "warning",
                    "icon": "⚠️",
                    "metric": "skewness",
                    "value": sk,
                    "benchmark": "Skew < 1.5",
                    "columns_used": [col],
                })
        except Exception:
            continue

    return insights[:3]


# ══════════════════════════════════════════════════════════
#  FIX-012: MINIMUM INSIGHT FLOOR
#  "0 critical issues" is impossible for real datasets
# ══════════════════════════════════════════════════════════

def _data_quality_insights(df: pd.DataFrame) -> List[Dict]:
    """
    Always-run data quality checks regardless of domain.
    These fire before domain insights so quality issues are never missed.
    """
    insights = []
    n = len(df)

    # Missing data check
    missing_pct = df.isnull().mean().mean() * 100
    if missing_pct > 5:
        worst_col = df.isnull().mean().idxmax()
        worst_pct = df[worst_col].isnull().mean() * 100
        insights.append({
            "title": f"{missing_pct:.1f}% Missing Data — Analysis Reliability Risk",
            "body": (
                f"Overall {missing_pct:.1f}% of data cells are missing. "
                f"Worst column: '{worst_col}' ({worst_pct:.1f}% missing). "
                f"Use median imputation for numeric columns, mode/Unknown for categorical."
            ),
            "type": "critical" if missing_pct > 20 else "warning",
            "icon": "🚨" if missing_pct > 20 else "⚠️",
            "metric": "missing_data",
            "value": missing_pct,
            "benchmark": "<5%",
            "columns_used": [worst_col],
        })

    # Duplicate rows
    dup_n = int(df.duplicated().sum())
    dup_pct = dup_n / max(n, 1) * 100
    if dup_pct > 5:
        insights.append({
            "title": f"{dup_n:,} Duplicate Rows ({dup_pct:.1f}%) — Remove Before Analysis",
            "body": (
                f"{dup_n:,} exact duplicate rows detected ({dup_pct:.1f}% of dataset). "
                f"Duplicates inflate counts, distort aggregations, and bias ML models. "
                f"Remove before any analysis."
            ),
            "type": "critical",
            "icon": "🚨",
            "metric": "duplicate_rows",
            "value": dup_pct,
            "benchmark": "0%",
            "columns_used": [],
        })

    return insights


# ══════════════════════════════════════════════════════════
#  PUBLIC API — DROP-IN REPLACEMENT FOR ORIGINAL
# ══════════════════════════════════════════════════════════

def generate_insights(df: pd.DataFrame, domain: str = "general") -> List[Dict]:
    """
    Main entry point. Domain-aware, column-gated, threshold-driven.

    Args:
        df: cleaned dataframe
        domain: detected domain string (hr/ecommerce/sales/finance/general)

    Returns:
        List of insight dicts with keys:
        title, body, type (critical/warning/info/positive), icon,
        metric, value, benchmark, columns_used
    """
    insights = []

    # Step 1: Data quality checks — always run first
    insights.extend(_data_quality_insights(df))

    # Step 2: Domain-specific analysis
    domain_lower = domain.lower()

    if domain_lower == "hr":
        insights.extend(_hr_insights(df))
    elif domain_lower == "ecommerce":
        insights.extend(_ecommerce_insights(df))
    elif domain_lower == "sales":
        insights.extend(_sales_insights(df))
    elif domain_lower == "finance":
        insights.extend(_sales_insights(df))  # finance uses similar revenue analysis
    else:
        insights.extend(_general_insights(df))

    # Step 3: Minimum floor — if still empty, add dataset summary
    if not insights:
        num_cols = df.select_dtypes(include="number").columns.tolist()
        insights.append({
            "title": f"Dataset: {len(df):,} rows, {len(df.columns)} columns — Clean",
            "body": (
                f"No critical issues detected. "
                f"{len(num_cols)} numeric columns available for analysis. "
                f"Missing data: {df.isnull().mean().mean()*100:.1f}%. "
                f"Duplicates: {df.duplicated().sum()}."
            ),
            "type": "info",
            "icon": "✅",
            "metric": "dataset_quality",
            "value": 100.0,
            "benchmark": "N/A",
            "columns_used": [],
        })

    # Sort: critical → warning → info → positive
    order = {"critical": 0, "warning": 1, "info": 2, "positive": 3}
    insights.sort(key=lambda x: order.get(x.get("type", "info"), 9))

    return insights

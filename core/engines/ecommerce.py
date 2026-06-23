"""
core/engines/ecommerce.py — E-Commerce domain engine.
Single responsibility: rating, pricing, discount, and category insights.
"""
from __future__ import annotations
import logging
from typing import Dict, List

import numpy as np
import pandas as pd

from core.engines.base import Insight, build_insight, col_stats

logger = logging.getLogger(__name__)


def _insights_ecommerce(df: pd.DataFrame, stats: Dict, corrs: List) -> Dict:
    findings, risks, opps, actions = [], [], [], []
    insights = []

    rating_col = next((c for c in df.columns if "rating" in c.lower()
                       and "count" not in c.lower()), None)
    price_col  = next((c for c in df.columns
                       if any(k in c.lower() for k in ["discounted_price","selling_price","price"])
                       and c in stats), None)
    _actual_col = next((c for c in df.columns
                       if "actual_price" in c.lower() or "mrp" in c.lower()), None)
    disc_col   = next((c for c in df.columns if "discount" in c.lower() and c in stats), None)
    cat_col    = next((c for c in df.select_dtypes(include=["object", "string"]).columns
                       if "category" in c.lower() and df[c].nunique()<=30), None)
    rev_col    = next((c for c in df.columns
                       if any(k in c.lower() for k in ["revenue","sales","amount"]) and c in stats), None)

    # ── Rating Analysis ────────────────────────────────────
    if rating_col and rating_col in stats:
        st     = stats[rating_col]
        mean_r = st.get("mean",0)
        low_n  = int((df[rating_col].dropna()<3.0).sum()) if rating_col in df.columns else 0
        out_ct = st.get("outliers",0)
        q1     = st.get("q1",0)

        if mean_r < 3.5:
            insights.append(build_insight(
                title="Rating Emergency: {:.2f}/5 Average — {:,} Products Below 3.0".format(mean_r, low_n),
                problem="Average {:.2f}/5 with {:,} products rated below 3.0 (unacceptable threshold)".format(
                    mean_r, low_n),
                cause="Products failing to meet customer expectations — quality, description, or delivery mismatch",
                evidence="Mean={:.2f}. Benchmark: 4.0+. Bottom 25% rated below {:.1f}. "
                         "{:,} critically low-rated products.".format(mean_r, q1, low_n),
                action="1. Immediate audit of all products rated below 3.0  "
                       "2. Customer feedback analysis for bottom-rated items  "
                       "3. Supplier quality review for failing products  "
                       "4. Remove or improve within 14 days",
                impact="Ratings below 3.5 cause 40-60% lower purchase probability. "
                       "Each negative review reduces future sales by ~1%.",
                severity="critical", category="rating"
            ))
            risks.append("Rating {:.2f}/5 — {:,} products below 3.0 causing significant revenue loss".format(
                mean_r, low_n))
        elif mean_r < 4.0:
            insights.append(build_insight(
                title="Rating Below Target: {:.2f}/5 (Target 4.0+)".format(mean_r),
                problem="{:.2f}/5 average. Bottom 25% rated below {:.1f}. {:,} critical products.".format(
                    mean_r, q1, low_n),
                cause="Bottom quartile products dragging overall performance",
                evidence="Mean={:.2f}. 25th percentile={:.1f}. Target: 4.0+.".format(mean_r, q1),
                action="1. Fix or remove bottom quartile products  "
                       "2. Improve product descriptions and images  "
                       "3. Category-level quality audit",
                impact="Reaching 4.0+ rating = estimated 15-20% conversion improvement",
                severity="warning", category="rating"
            ))
        else:
            insights.append(build_insight(
                title="Strong Ratings: {:.2f}/5 — Competitive Advantage".format(mean_r),
                problem="N/A — ratings are strong",
                cause="Quality products meeting customer expectations",
                evidence="Mean={:.2f}/5. Above 4.0 benchmark. Only {:,} products below 3.0.".format(
                    mean_r, low_n),
                action="Leverage high ratings in all marketing. "
                       "Use as social proof in product listings.",
                impact="4.0+ ratings enable 10-15% premium pricing vs competitors",
                severity="positive", category="rating"
            ))
            opps.append("Rating {:.2f}/5 enables premium pricing — test 5-10% price increase on top-rated items".format(mean_r))

        if out_ct > 0:
            pct = st.get("outlier_pct",0)
            findings.append(
                "{:,} products have outlier ratings ({:.1f}% of catalog) — "
                "investigate immediately for quality or fraud issues".format(out_ct, pct))

    # ── Category Performance ───────────────────────────────
    if cat_col and rating_col and rating_col in df.columns:
        cat_perf = df.groupby(cat_col)[rating_col].agg(["mean","count"]).sort_values("mean")
        cat_perf = cat_perf[cat_perf["count"]>=5]
        if len(cat_perf)>=2:
            worst_c = cat_perf.index[0]
            best_c  = cat_perf.index[-1]
            gap     = cat_perf.loc[best_c,"mean"] - cat_perf.loc[worst_c,"mean"]
            if gap > 0.3:
                insights.append(build_insight(
                    title="Category Gap: '{}' ({:.2f}) vs '{}' ({:.2f})".format(
                        worst_c, cat_perf.loc[worst_c,"mean"],
                        best_c, cat_perf.loc[best_c,"mean"]),
                    problem="'{}' category underperforms by {:.1f} rating points ({:.0f}% gap)".format(
                        worst_c, gap, gap/cat_perf.loc[worst_c,"mean"]*100),
                    cause="Supplier quality, product complexity, or customer expectation mismatch by category",
                    evidence="{:.1f} point gap across {} categories. "
                             "'{}' avg={:.2f}, '{}' avg={:.2f}".format(
                        gap, len(cat_perf), worst_c, cat_perf.loc[worst_c,"mean"],
                        best_c, cat_perf.loc[best_c,"mean"]),
                    action="1. Quality audit of '{}' category suppliers  "
                           "2. Customer complaint analysis for '{}' products  "
                           "3. Apply '{}' category best practices to '{}' category".format(
                        worst_c, worst_c, best_c, worst_c),
                    impact="Closing gap by 50% = +{:.1f} rating points overall. "
                           "Estimated 10-15% revenue uplift in '{}' category.".format(
                        gap*0.5, worst_c),
                    severity="warning" if gap<0.8 else "critical",
                    category="category_performance"
                ))
                findings.append("Category range: {} ({:.2f}) to {} ({:.2f})".format(
                    worst_c, cat_perf.loc[worst_c,"mean"],
                    best_c, cat_perf.loc[best_c,"mean"]))

    # ── Pricing Analysis ───────────────────────────────────
    if price_col and price_col in stats:
        st   = stats[price_col]
        skew = st.get("skew",0)
        if skew > 1.5:
            findings.append(
                "Price distribution right-skewed (skew={:.1f}) — median {:.0f} vs mean {:.0f}. "
                "Most products are budget-range with few premium items. "
                "Consider expanding mid-market range.".format(
                    skew, st["median"], st["mean"]))
            opps.append("Mid-market price gap detected — products between median ({:.0f}) "
                        "and 75th percentile ({:.0f}) are underrepresented".format(
                            st["median"], st["q3"]))

    # ── Discount Analysis ──────────────────────────────────
    if disc_col and disc_col in stats:
        st       = stats[disc_col]
        avg_disc = st.get("mean",0)
        max_disc = st.get("max",0)
        if avg_disc > 40:
            insights.append(build_insight(
                title="High Avg Discount {:.0f}% — Potential Margin Erosion".format(avg_disc),
                problem="Average discount {:.0f}% with some products at {:.0f}% — profitability at risk".format(
                    avg_disc, max_disc),
                cause="Competitive pressure or unstrategic discounting without margin analysis",
                evidence="Mean discount={:.0f}%, max={:.0f}%. "
                         "Benchmark: 15-25% is healthy for most categories.".format(avg_disc, max_disc),
                action="1. Margin analysis per category — identify below-cost discounts  "
                       "2. Reduce discounts on 4.0+ rated products (they sell without discounts)  "
                       "3. Strategic discounting only: new launch, clearance, seasonal",
                impact="Every 10% unnecessary discount = direct margin loss. "
                       "High discounts also train customers to wait for sales.",
                severity="warning" if avg_disc<55 else "critical",
                category="pricing"
            ))
            risks.append("Avg discount {:.0f}% may be eroding margins — review per-product profitability".format(avg_disc))

    # ── Price-Rating Correlation ───────────────────────────
    for corr in corrs:
        cols = [corr["col_a"], corr["col_b"]]
        has_rating = any("rating" in c.lower() for c in cols)
        has_price  = any("price" in c.lower() for c in cols)
        if has_rating and has_price and abs(corr["r"])>=0.3:
            if corr["r"] < 0:
                risks.append(
                    "Higher-priced products have LOWER ratings (r={:.2f}) — "
                    "premium pricing not matching perceived value. Review quality.".format(corr["r"]))
            else:
                opps.append(
                    "Higher-priced products have HIGHER ratings (r={:.2f}) — "
                    "quality-price alignment confirmed. Safe to test premium pricing.".format(corr["r"]))

    # FIX-011: Column-gated actions — only recommend for columns that exist in dataset
    # Never generate recommendations for columns that are not present
    _ec_cols = [c.lower() for c in df.columns]

    if rating_col:
        actions.append("Weekly rating monitoring — alert if any product drops below 3.5")
        actions.append("Remove products with <3.0 rating and <50 reviews — they damage brand perception")
        if price_col:
            actions.append("A/B test 5% price increase on products with rating above 4.3 — high ratings justify premium")
    if rev_col or any(k in _ec_cols for k in ["amount","sales","revenue"]):
        actions.append("Customer feedback loop — auto-survey buyers 7 days post-delivery to track satisfaction vs revenue")
    if cat_col:
        actions.append("Category manager review — monthly revenue and rating performance vs category target")
    else:
        actions.append("Segment your data by product type or channel — subgroup performance often tells a different story than averages")

    return {"findings":findings, "risks":risks, "opportunities":opps,
            "actions":actions, "insights":insights}


# ══════════════════════════════════════════════════════════
#  SALES INSIGHTS
# ══════════════════════════════════════════════════════════


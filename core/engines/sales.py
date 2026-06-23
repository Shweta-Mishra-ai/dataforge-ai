"""
core/engines/sales.py — Sales Performance domain engine.
Single responsibility: revenue, quota, margin, and rep performance insights.
"""
from __future__ import annotations
import logging
from typing import Dict, List

import numpy as np
import pandas as pd

from core.engines.base import Insight, build_insight, col_stats

logger = logging.getLogger(__name__)


def _insights_sales(df: pd.DataFrame, stats: Dict, corrs: List) -> Dict:
    findings, risks, opps, actions = [], [], [], []
    insights = []

    rev_col    = next((c for c in df.columns
                       if any(k in c.lower() for k in ["revenue","sales","amount","total"])
                       and c in stats), None)
    profit_col = next((c for c in df.columns
                       if any(k in c.lower() for k in ["profit","margin","net"])
                       and c in stats), None)
    target_col = next((c for c in df.columns
                       if any(k in c.lower() for k in ["target","quota","goal"])
                       and c in stats), None)
    region_col = next((c for c in df.select_dtypes(include=["object", "string"]).columns
                       if any(k in c.lower() for k in ["region","territory","zone","area"])
                       and df[c].nunique()<=25), None)
    product_col= next((c for c in df.select_dtypes(include=["object", "string"]).columns
                       if any(k in c.lower() for k in ["product","category","segment"])
                       and df[c].nunique()<=30), None)
    _rep_col    = next((c for c in df.select_dtypes(include=["object", "string"]).columns
                       if any(k in c.lower() for k in ["rep","salesperson","agent","owner"])
                       and df[c].nunique()<=50), None)

    # ── Revenue Analysis ───────────────────────────────────
    if rev_col and rev_col in stats:
        st   = stats[rev_col]
        skew = st.get("skew",0)
        cv   = st.get("cv",0)
        mean = st.get("mean",0)
        med  = st.get("median",0)

        insights.append(build_insight(
            title="Revenue Overview: Mean {:.0f} | Median {:.0f} | Range {:.0f}-{:.0f}".format(
                mean, med, st["min"], st["max"]),
            problem="Revenue distribution analysis" + (" — high variability detected" if cv>0.5 else ""),
            cause="Skewness={:.1f} indicates {}".format(
                skew, "few large deals driving disproportionate revenue (Pareto effect)" if skew>1
                else "revenue is relatively evenly distributed"),
            evidence="Mean={:.0f}, Median={:.0f} ({:.0f}% difference). "
                     "Top 25% of transactions above {:.0f}.".format(
                mean, med, abs(mean-med)/max(med,1)*100, st["q3"]),
            action="1. Identify top 20% revenue drivers — protect and replicate  "
                   "2. Analyze bottom 20% — cut or transform low performers  "
                   "3. Revenue concentration risk assessment",
            impact="If top 20% drives 80% of revenue, losing 1 key client/product = severe impact",
            severity="info" if cv<0.5 else "warning",
            category="revenue"
        ))

        if skew > 1.5:
            opps.append(
                "Revenue is right-skewed — small number of high-value transactions. "
                "Focus on replicating conditions for top transactions.")
            findings.append(
                "Revenue Pareto effect detected: median {:.0f} vs mean {:.0f} — "
                "few large deals driving disproportionate revenue".format(med, mean))

    # ── Target/Quota Analysis ──────────────────────────────
    if target_col and rev_col and target_col in stats and rev_col in stats:
        target_mean = stats[target_col].get("mean",0)
        rev_mean    = stats[rev_col].get("mean",0)
        achievement = (rev_mean / target_mean * 100) if target_mean > 0 else 0

        if achievement < 80:
            insights.append(build_insight(
                title="Target Gap: {:.0f}% Achievement — {:.0f}pp Below Target".format(
                    achievement, 100-achievement),
                problem="Average {:.0f}% quota achievement — team missing targets significantly".format(achievement),
                cause="Targets may be unrealistic, pipeline quality poor, or sales process broken",
                evidence="Avg revenue={:.0f} vs avg target={:.0f}. Achievement={:.0f}%.".format(
                    rev_mean, target_mean, achievement),
                action="1. Review if targets are market-realistic (benchmark vs industry)  "
                       "2. Pipeline quality audit — identify qualification issues  "
                       "3. Sales process coaching for bottom quartile reps",
                impact="{:.0f}% achievement gap = {:.0f}% revenue shortfall from plan".format(
                    100-achievement, 100-achievement),
                severity="critical" if achievement<70 else "warning",
                category="target"
            ))
            risks.append("{:.0f}% target achievement — revenue significantly below plan".format(achievement))
        elif achievement >= 100:
            insights.append(build_insight(
                title="Targets Exceeded: {:.0f}% Achievement".format(achievement),
                problem="N/A — exceeding targets",
                cause="Strong sales execution and/or conservative target setting",
                evidence="Avg revenue={:.0f} vs avg target={:.0f}".format(rev_mean, target_mean),
                action="1. Review if targets were set too conservatively  "
                       "2. Capture learnings from over-performers and scale",
                impact="Consistent over-achievement suggests capacity for higher targets",
                severity="positive", category="target"
            ))
            opps.append("{:.0f}% target achievement — review upside potential for next period".format(achievement))

    # ── Regional Analysis ──────────────────────────────────
    if region_col and rev_col and rev_col in df.columns:
        reg_perf = df.groupby(region_col)[rev_col].agg(["mean","sum","count"])
        reg_perf = reg_perf[reg_perf["count"]>=3].sort_values("sum", ascending=False)
        if len(reg_perf)>=2:
            top_r    = reg_perf.index[0]
            bottom_r = reg_perf.index[-1]
            top_share= reg_perf.loc[top_r,"sum"]/reg_perf["sum"].sum()*100
            gap      = reg_perf.loc[top_r,"mean"]-reg_perf.loc[bottom_r,"mean"]
            gap_pct  = gap/max(reg_perf.loc[bottom_r,"mean"],1)*100

            insights.append(build_insight(
                title="Regional Gap: '{}' {:.0f}x Performance vs '{}'".format(
                    top_r, reg_perf.loc[top_r,"mean"]/max(reg_perf.loc[bottom_r,"mean"],1), bottom_r),
                problem="'{}' underperforming by {:.0f}% vs top region '{}'".format(
                    bottom_r, gap_pct, top_r),
                cause="Market maturity, team capability, competitive landscape, or resource allocation differ by region",
                evidence="'{}' avg={:.0f} vs '{}' avg={:.0f}. "
                         "Top region '{}' holds {:.0f}% of total revenue.".format(
                    top_r, reg_perf.loc[top_r,"mean"],
                    bottom_r, reg_perf.loc[bottom_r,"mean"],
                    top_r, top_share),
                action="1. Understand '{}' success factors — replicate in '{}'  "
                       "2. Resource allocation review — is '{}' under-resourced?  "
                       "3. Market potential analysis for '{}'".format(
                    top_r, bottom_r, bottom_r, bottom_r),
                impact="Closing '{}'s gap by 30% = significant total revenue uplift".format(bottom_r),
                severity="warning" if gap_pct<50 else "critical",
                category="regional"
            ))
            findings.append("Top region '{}' contributes {:.0f}% of total revenue — concentration risk".format(
                top_r, top_share) if top_share > 50 else
                "Revenue reasonably distributed across {} regions".format(len(reg_perf)))

    # ── Product/Category Performance ──────────────────────
    if product_col and rev_col and rev_col in df.columns:
        prod_perf = df.groupby(product_col)[rev_col].agg(["sum","count"])
        prod_perf = prod_perf[prod_perf["count"]>=3].sort_values("sum", ascending=False)
        if len(prod_perf)>=2:
            total_rev = prod_perf["sum"].sum()
            top_prod  = prod_perf.index[0]
            top_share = prod_perf.loc[top_prod,"sum"]/total_rev*100
            _top2_share= prod_perf.iloc[:2]["sum"].sum()/total_rev*100

            if top_share > 40:
                risks.append(
                    "'{}' product/category = {:.0f}% of total revenue — "
                    "dangerous concentration. Losing this = severe revenue impact.".format(
                        top_prod, top_share))
            opps.append(
                "Bottom 3 products/categories contribute only {:.0f}% of revenue — "
                "investigate if resources should be reallocated".format(
                    prod_perf.iloc[-3:]["sum"].sum()/total_rev*100))

    # ── Profit Margin ──────────────────────────────────────
    if profit_col and profit_col in stats:
        st         = stats[profit_col]
        mean_profit= st.get("mean",0)
        neg_n      = int((df[profit_col].dropna()<0).sum()) if profit_col in df.columns else 0
        neg_pct    = round(neg_n/len(df)*100,1)

        if neg_n > 0:
            insights.append(build_insight(
                title="{:,} Loss-Making Transactions ({:.0f}%) — Immediate Review".format(neg_n, neg_pct),
                problem="{:,} transactions ({:.0f}%) generating negative profit/margin".format(neg_n, neg_pct),
                cause="Below-cost pricing, excessive discounts, high returns, or incorrect cost allocation",
                evidence="{:,} negative profit transactions. Mean margin={:.2f}. "
                         "Loss transactions erode overall profitability.".format(neg_n, mean_profit),
                action="1. Identify all loss-making transactions this week  "
                       "2. Root cause: pricing error, returns, or discounts?  "
                       "3. Reprice or discontinue unprofitable products",
                impact="Eliminating {:.0f}% loss transactions = direct profitability improvement".format(neg_pct),
                severity="critical" if neg_pct>10 else "warning",
                category="profitability"
            ))
            risks.append("{:,} loss-making transactions ({:.0f}%) — eroding overall profitability".format(
                neg_n, neg_pct))

    actions.extend([
        "Weekly revenue vs target review — per rep and per region",
        "Identify top 3 deals at risk in pipeline — intervention strategy",
        "Replicate top performer playbook — what do they do differently?",
        "Revenue concentration audit — reduce dependency on single customer/product",
        "Quarterly pricing review — ensure margins are healthy per product category",
    ])

    return {"findings":findings, "risks":risks, "opportunities":opps,
            "actions":actions, "insights":insights}


# ══════════════════════════════════════════════════════════
#  GENERAL INSIGHTS
# ══════════════════════════════════════════════════════════


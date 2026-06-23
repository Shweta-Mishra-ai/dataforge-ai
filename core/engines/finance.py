"""
core/engines/finance.py — Finance & Accounting domain engine.
Single responsibility: P&L, margin, budget variance, and cost driver insights.
"""
from __future__ import annotations
import logging
from typing import Dict, List

import numpy as np
import pandas as pd

from core.engines.base import Insight, build_insight, col_stats

logger = logging.getLogger(__name__)


def _insights_finance(df: pd.DataFrame, stats: Dict, corrs: List) -> Dict:
    """
    Deep finance domain analysis.
    Detects P&L columns, computes margins, budget variance, cost drivers.
    All values from dataset only — no hardcoded external benchmarks.
    """
    findings, risks, opps, actions = [], [], [], []
    insights = []
    cols_lower = {c: c.lower() for c in df.columns}

    # ── Column Detection ──────────────────────────────────────────────────
    def _find(keywords, exclude=None):
        excl = exclude or []
        for c, cl in cols_lower.items():
            if any(k in cl for k in keywords) and not any(e in cl for e in excl):
                if c in df.columns:
                    return c
        return None

    rev_col    = _find(["revenue","total_revenue","net_revenue","income","turnover","sales_amount","gross_revenue"])
    cost_col   = _find(["cost","cogs","cost_of_goods","cost_of_sales","direct_cost"])
    opex_col   = _find(["opex","operating_expense","overhead","indirect_cost","operating_cost"])
    _profit_col = _find(["net_profit","net_income","profit_after","bottom_line"])
    gross_col  = _find(["gross_profit","gross_income"])
    budget_col = _find(["budget","plan","target","forecast"])
    actual_col = _find(["actual","actuals","realized","achieved"], exclude=["target","budget"])
    # If no dedicated actual col but there's a budget col, try revenue as actual
    if budget_col and not actual_col:
        actual_col = rev_col

    period_col = _find(["month","quarter","period","year","date","fiscal"])
    cat_col    = _find(["category","department","cost_center","account","segment","division"])
    amt_col    = _find(["amount","value","sum","total"], exclude=["revenue","budget"])
    expense_col= _find(["expense","spend","expenditure"])

    # Fallback: pick largest numeric cols as revenue/cost if nothing detected
    num_cols = df.select_dtypes(include="number").columns.tolist()
    if not rev_col and len(num_cols) >= 1:
        # Largest mean = likely revenue
        means = {c: df[c].mean() for c in num_cols if df[c].mean() > 0}
        if means:
            rev_col = max(means, key=means.get)
    if not cost_col and len(num_cols) >= 2 and rev_col:
        remaining = [c for c in num_cols if c != rev_col]
        if remaining:
            cost_col = remaining[0]

    # ── 1. GROSS MARGIN ANALYSIS ──────────────────────────────────────────
    if rev_col and (cost_col or gross_col):
        try:
            rev_vals = df[rev_col].dropna()
            if gross_col:
                gp_vals = df[gross_col].dropna()
                margin_series = (gp_vals / rev_vals.where(rev_vals != 0)).dropna() * 100
            else:
                cogs_vals  = df[cost_col].dropna()
                gp_vals    = rev_vals - cogs_vals
                margin_series = (gp_vals / rev_vals.where(rev_vals != 0)).dropna() * 100

            avg_margin  = float(margin_series.mean())
            min_margin  = float(margin_series.min())
            neg_periods = int((margin_series < 0).sum())
            margin_std  = float(margin_series.std())
            total_rev   = float(rev_vals.sum())
            total_cost  = float(df[cost_col].sum()) if cost_col else float(rev_vals.sum() - gp_vals.sum())

            if avg_margin < 0:
                sev = "critical"
                margin_msg = f"NEGATIVE gross margin ({avg_margin:.1f}%) — costs exceed revenue. Organisation is structurally loss-making."
            elif avg_margin < 15:
                sev = "critical"
                margin_msg = f"Gross margin {avg_margin:.1f}% is extremely thin — minimal buffer for operating expenses."
            elif avg_margin < 30:
                sev = "warning"
                margin_msg = f"Gross margin {avg_margin:.1f}% is below typical healthy levels. Investigate cost drivers."
            else:
                sev = "positive"
                margin_msg = f"Gross margin {avg_margin:.1f}% — healthy. Focus on protecting it as revenue scales."

            insights.append(build_insight(
                title    = f"Gross Margin: {avg_margin:.1f}% (Revenue: {total_rev:,.0f} | COGS: {total_cost:,.0f})",
                problem  = margin_msg,
                cause    = ("Gross margin is determined by the relationship between revenue pricing and "
                            "direct production/sourcing costs. Margin compression suggests either revenue "
                            "growth is lagging cost growth, or pricing is insufficient relative to cost base. "
                            "These are patterns in the data — root causes require operational investigation."),
                evidence = (f"Avg gross margin: {avg_margin:.1f}% | Min: {min_margin:.1f}% | "
                            f"Std dev: {margin_std:.1f}pp | "
                            f"Negative-margin periods: {neg_periods}"),
                action   = ("1. Identify which periods/categories have lowest margin — start there. "
                            "2. Compare cost growth rate vs revenue growth rate period-over-period. "
                            "3. Review pricing strategy for below-average-margin segments. "
                            "4. Cost reduction opportunities in highest-cost categories."),
                impact   = (f"Each 1pp improvement in gross margin adds {total_rev * 0.01:,.0f} "
                            f"to bottom line at current revenue scale."),
                severity = sev, category = "margin"
            ))
            if neg_periods > 0:
                risks.append(f"{neg_periods} periods with negative gross margin — structural cost problem")
                findings.append(f"Gross margin ranges from {min_margin:.1f}% to {float(margin_series.max()):.1f}% — high variability")

        except Exception:
            logger.debug("%s silent skip", exc_info=True)

    # ── 2. REVENUE TREND (period analysis) ───────────────────────────────
    if period_col and rev_col:
        try:
            period_rev = df.groupby(period_col)[rev_col].sum()
            if len(period_rev) >= 2:
                # Sort periods if possible
                try:
                    period_rev = period_rev.sort_index()
                except Exception:
                    logger.debug("%s silent skip", exc_info=True)

                first_half_mean  = float(period_rev.iloc[:len(period_rev)//2].mean())
                second_half_mean = float(period_rev.iloc[len(period_rev)//2:].mean())
                growth_pct       = ((second_half_mean - first_half_mean) / abs(first_half_mean) * 100
                                    if first_half_mean != 0 else 0)
                best_period      = period_rev.idxmax()
                worst_period     = period_rev.idxmin()
                peak_val         = float(period_rev.max())
                trough_val       = float(period_rev.min())
                variance_ratio   = (peak_val - trough_val) / abs(trough_val) * 100 if trough_val != 0 else 0

                if growth_pct > 10:
                    rev_trend_sev = "positive"
                    rev_trend_msg = f"Revenue growing — second half averaged {growth_pct:.1f}% above first half."
                elif growth_pct < -10:
                    rev_trend_sev = "critical"
                    rev_trend_msg = f"Revenue declining — second half averaged {abs(growth_pct):.1f}% below first half."
                else:
                    rev_trend_sev = "warning"
                    rev_trend_msg = f"Revenue relatively flat ({growth_pct:+.1f}% half-over-half change)."

                insights.append(build_insight(
                    title    = f"Revenue Trend: {growth_pct:+.1f}% Half-over-Half | Peak: {period_rev.idxmax()}",
                    problem  = rev_trend_msg,
                    cause    = ("Revenue trend patterns in this data may reflect seasonality, "
                                "business cycle effects, or structural changes. The dataset alone "
                                "cannot distinguish between these — operational context is needed."),
                    evidence = (f"Periods analysed: {len(period_rev)} | "
                                f"Best period: {best_period} ({peak_val:,.0f}) | "
                                f"Worst: {worst_period} ({trough_val:,.0f}) | "
                                f"Peak-to-trough range: {variance_ratio:.0f}%"),
                    action   = ("1. Investigate what drove peak performance in period: " + str(best_period) + ". "
                                "2. Identify what caused the trough in period: " + str(worst_period) + ". "
                                "3. If seasonal, plan resource allocation accordingly. "
                                "4. Set forward-looking targets based on trend, not single-period snapshots."),
                    impact   = (f"If growth trend continues, projected next-period revenue: "
                                f"{float(period_rev.iloc[-1]) * (1 + growth_pct/100/len(period_rev)):,.0f}. "
                                f"This is a simple projection — not a forecast."),
                    severity = rev_trend_sev, category = "revenue_trend"
                ))
                findings.append(f"Revenue across {len(period_rev)} periods: peak {peak_val:,.0f} in {best_period}, "
                                 f"trough {trough_val:,.0f} in {worst_period}")
        except Exception:
            logger.debug("%s silent skip", exc_info=True)

    # ── 3. BUDGET VS ACTUAL ───────────────────────────────────────────────
    if budget_col and actual_col and budget_col != actual_col:
        try:
            budget_vals = df[budget_col].dropna()
            actual_vals = df[actual_col].dropna()
            common_idx  = budget_vals.index.intersection(actual_vals.index)
            if len(common_idx) >= 3:
                bv = budget_vals.loc[common_idx]
                av = actual_vals.loc[common_idx]
                variance_pct    = float(((av - bv) / bv.where(bv != 0)).mean() * 100)
                over_budget     = int((av > bv).sum())
                under_budget    = int((av < bv).sum())
                worst_over_pct  = float(((av - bv) / bv.where(bv != 0)).max() * 100)

                if variance_pct > 10:
                    bv_sev = "warning"
                    bv_msg = f"Actual exceeds budget by avg {variance_pct:.1f}% — overspending or underforecasting."
                elif variance_pct < -10:
                    bv_sev = "warning"
                    bv_msg = f"Actual below budget by avg {abs(variance_pct):.1f}% — underspending or revenue shortfall."
                else:
                    bv_sev = "positive"
                    bv_msg = f"Budget vs actual variance within ±10% ({variance_pct:+.1f}%) — good forecast accuracy."

                insights.append(build_insight(
                    title    = f"Budget Accuracy: {variance_pct:+.1f}% Avg Variance ({over_budget} over, {under_budget} under)",
                    problem  = bv_msg,
                    cause    = ("Budget variance can reflect poor forecasting methodology, "
                                "unexpected cost events, or deliberate budget padding. "
                                "Consistent over-budget or under-budget patterns warrant a "
                                "review of the forecasting process itself."),
                    evidence = (f"Avg variance: {variance_pct:+.1f}% | "
                                f"Rows over budget: {over_budget} | "
                                f"Rows under budget: {under_budget} | "
                                f"Worst overage: {worst_over_pct:.1f}%"),
                    action   = ("1. Identify the rows with worst budget variance — what drove them? "
                                "2. Review forecasting methodology for largest-variance categories. "
                                "3. Set variance triggers: any item >15% variance gets a review. "
                                "4. Track rolling forecast accuracy as a KPI."),
                    impact   = ("Improving forecast accuracy by 5pp reduces planning uncertainty "
                                "and allows more precise resource allocation."),
                    severity = bv_sev, category = "budget_variance"
                ))
        except Exception:
            logger.debug("%s silent skip", exc_info=True)

    # ── 4. COST CATEGORY CONCENTRATION ───────────────────────────────────
    if cat_col and (cost_col or expense_col or amt_col):
        val_col = cost_col or expense_col or amt_col
        try:
            cat_totals = df.groupby(cat_col)[val_col].sum().sort_values(ascending=False)
            total_cost_cat = float(cat_totals.sum())
            if total_cost_cat > 0 and len(cat_totals) >= 2:
                top1_pct = float(cat_totals.iloc[0] / total_cost_cat * 100)
                top3_pct = float(cat_totals.iloc[:3].sum() / total_cost_cat * 100)
                top_cat  = str(cat_totals.index[0])
                top_val  = float(cat_totals.iloc[0])

                sev_conc = "critical" if top1_pct > 50 else "warning" if top1_pct > 35 else "info"
                insights.append(build_insight(
                    title    = f"Cost Concentration: '{top_cat}' = {top1_pct:.1f}% of Total. Top 3 = {top3_pct:.1f}%",
                    problem  = (f"'{top_cat}' accounts for {top1_pct:.1f}% of total cost ({top_val:,.0f}). "
                                f"{'Extreme concentration — single category dependency risk.' if top1_pct > 50 else 'High concentration — moderate dependency risk.'}"
                                ),
                    cause    = ("Cost concentration may be structural (e.g., headcount is always the largest cost) "
                                "or may indicate an unbalanced cost structure that merits review. "
                                "The pattern is from the data — context determines whether it is appropriate."),
                    evidence = " | ".join([f"{str(idx)[:20]}: {val/total_cost_cat*100:.1f}%"
                                           for idx,val in cat_totals.head(5).items()]),
                    action   = (f"1. Review if '{top_cat}' cost is optimised or contains inefficiencies. "
                                f"2. Benchmark '{top_cat}' costs against prior periods. "
                                f"3. Identify if any category is growing faster than revenue."),
                    impact   = (f"A 5% reduction in the top cost category ('{top_cat}') saves "
                                f"{top_val * 0.05:,.0f} at current levels."),
                    severity = sev_conc, category = "cost_concentration"
                ))
                findings.append(f"Top cost category '{top_cat}': {top1_pct:.1f}% of total | "
                                 f"Top 3 combined: {top3_pct:.1f}%")

                # Bottom categories — potential pruning or reinvestment
                if len(cat_totals) >= 5:
                    bottom_cats = cat_totals.tail(3)
                    bottom_pct  = float(bottom_cats.sum() / total_cost_cat * 100)
                    opps.append(
                        f"Bottom 3 cost categories ({bottom_pct:.1f}% of total) — "
                        f"review whether these activities generate sufficient return"
                    )
        except Exception:
            logger.debug("%s silent skip", exc_info=True)

    # ── 5. OPERATING EXPENSE RATIO ────────────────────────────────────────
    if opex_col and rev_col:
        try:
            opex_total = float(df[opex_col].sum())
            rev_total  = float(df[rev_col].sum())
            if rev_total > 0:
                opex_ratio = opex_total / rev_total * 100
                if opex_ratio > 60:
                    opex_sev = "critical"
                    opex_msg = f"OpEx ratio {opex_ratio:.1f}% — operating expenses consuming {opex_ratio:.1f}¢ of every revenue £/$/€."
                elif opex_ratio > 40:
                    opex_sev = "warning"
                    opex_msg = f"OpEx ratio {opex_ratio:.1f}% — elevated. Review for efficiency opportunities."
                else:
                    opex_sev = "positive"
                    opex_msg = f"OpEx ratio {opex_ratio:.1f}% — well-controlled operating expenses."

                insights.append(build_insight(
                    title    = f"OpEx Ratio: {opex_ratio:.1f}% (OpEx: {opex_total:,.0f} / Revenue: {rev_total:,.0f})",
                    problem  = opex_msg,
                    cause    = ("OpEx ratio reflects operational efficiency. High ratios indicate "
                                "overhead-heavy operations. Trend direction matters more than absolute level — "
                                "rising ratio suggests costs scaling faster than revenue."),
                    evidence = f"OpEx: {opex_total:,.0f} | Revenue: {rev_total:,.0f} | Ratio: {opex_ratio:.1f}%",
                    action   = ("1. Break down OpEx by sub-category to identify largest components. "
                                "2. Set a target OpEx ratio for the next period. "
                                "3. Identify fixed vs variable components — fixed costs create leverage risk."),
                    impact   = (f"Each 1pp reduction in OpEx ratio saves {rev_total * 0.01:,.0f} "
                                f"at current revenue scale."),
                    severity = opex_sev, category = "opex_ratio"
                ))
        except Exception:
            logger.debug("%s silent skip", exc_info=True)

    # ── 6. REVENUE CONCENTRATION (category) ──────────────────────────────
    if cat_col and rev_col:
        try:
            cat_rev = df.groupby(cat_col)[rev_col].sum().sort_values(ascending=False)
            total_r = float(cat_rev.sum())
            if total_r > 0 and len(cat_rev) >= 3:
                top1_rev_pct = float(cat_rev.iloc[0] / total_r * 100)
                top3_rev_pct = float(cat_rev.iloc[:3].sum() / total_r * 100)
                top_rev_cat  = str(cat_rev.index[0])

                if top1_rev_pct > 50:
                    risks.append(
                        f"Revenue concentration: '{top_rev_cat}' = {top1_rev_pct:.1f}% of total revenue — "
                        f"high dependency on single category"
                    )
                    opps.append(
                        f"Revenue diversification opportunity: reduce '{top_rev_cat}' dependency "
                        f"from {top1_rev_pct:.1f}% by developing second-tier categories"
                    )
                elif top3_rev_pct > 75:
                    risks.append(
                        f"Top 3 categories = {top3_rev_pct:.1f}% of revenue — moderate concentration risk"
                    )
        except Exception:
            logger.debug("%s silent skip", exc_info=True)

    # ── 7. PERIOD-OVER-PERIOD COST GROWTH ─────────────────────────────────
    if period_col and cost_col and rev_col:
        try:
            period_data = df.groupby(period_col)[[rev_col, cost_col]].sum()
            try:
                period_data = period_data.sort_index()
            except Exception:
                logger.debug("%s silent skip", exc_info=True)
            if len(period_data) >= 3:
                rev_growth  = period_data[rev_col].pct_change().mean() * 100
                cost_growth = period_data[cost_col].pct_change().mean() * 100
                if not pd.isna(rev_growth) and not pd.isna(cost_growth):
                    spread = cost_growth - rev_growth
                    if spread > 5:
                        risks.append(
                            f"Cost growing {cost_growth:.1f}%/period vs revenue {rev_growth:.1f}%/period "
                            f"— margin compression likely if trend continues"
                        )
                        actions.append(
                            "Cost growth exceeding revenue growth — conduct cost driver analysis immediately"
                        )
                    elif spread < -5:
                        opps.append(
                            f"Revenue growing {rev_growth:.1f}%/period faster than costs {cost_growth:.1f}%/period "
                            f"— positive operating leverage"
                        )
                    findings.append(
                        f"Avg period revenue growth: {rev_growth:.1f}% | "
                        f"Avg period cost growth: {cost_growth:.1f}%"
                    )
        except Exception:
            logger.debug("%s silent skip", exc_info=True)

    # ── Actions ───────────────────────────────────────────────────────────
    if not actions:
        actions = [
            "Build a rolling P&L dashboard — margin, revenue, cost tracked monthly",
            "Set KPI targets for gross margin, OpEx ratio, and budget variance",
            "Identify the top 3 cost drivers and set reduction targets for each",
            "Implement variance reporting: flag any line item >15% from budget",
        ]
    else:
        actions += [
            "Establish a monthly finance review: actual vs budget vs prior period",
            "Identify fixed vs variable costs — separate for scenario planning",
        ]

    return {
        "findings":      findings,
        "risks":         risks,
        "opportunities": opps,
        "actions":       actions,
        "insights":      insights,
    }




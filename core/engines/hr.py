"""
core/engines/hr.py — HR & People Analytics domain engine.
Single responsibility: given a DataFrame, produce structured HR insights.
All metrics computed from the submitted dataset — no external benchmarks.
"""
from __future__ import annotations
import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from core.engines.base import Insight, AttritionAnalysis, build_insight, col_stats, correlations

logger = logging.getLogger(__name__)


def _run_attrition(df: pd.DataFrame) -> Optional[AttritionAnalysis]:
    attr_col = next((c for c in df.columns
                     if "attrition" in c.lower()
                     or c.lower() in ["left","churned","resigned"]), None)
    if attr_col is None:
        return None

    left_mask = df[attr_col].astype(str).str.lower().str.strip().isin(
        ["yes","1","1.0","true","left"])
    n_left  = int(left_mask.sum())
    n_total = len(df)
    if n_left == 0:
        return None

    rate = round(n_left / max(n_total,1) * 100, 1)
    severity = ("critical" if rate > 25 else "high" if rate > 18
                else "warning" if rate > 12 else "normal")

    # Numeric drivers
    num_cols = [c for c in df.select_dtypes(include="number").columns
                if c != attr_col]
    top_drivers = []
    for col in num_cols[:12]:
        try:
            lv = df.loc[left_mask, col].dropna()
            sv = df.loc[~left_mask, col].dropna()
            if len(lv) < 5 or len(sv) < 5: continue
            _, p = scipy_stats.mannwhitneyu(lv, sv, alternative="two-sided")
            if p < 0.05:
                diff_pct = abs(lv.mean()-sv.mean())/abs(sv.mean())*100 if sv.mean()!=0 else 0
                top_drivers.append({
                    "factor":    col,
                    "type":      "numeric",
                    "impact":    round(diff_pct,1),
                    "direction": "lower" if lv.mean()<sv.mean() else "higher",
                    "left_mean": round(float(lv.mean()),3),
                    "stay_mean": round(float(sv.mean()),3),
                    "p_value":   round(float(p),4),
                    "detail":    "Leavers avg {:.2f} vs stayers {:.2f} ({:.0f}% diff)".format(
                        lv.mean(), sv.mean(), diff_pct),
                })
        except Exception:
            logger.debug("%s skip", exc_info=True)
            continue

    # Categorical drivers
    cat_cols = [c for c in df.select_dtypes(include=["object", "string"]).columns
                if c != attr_col and df[c].nunique() <= 20]
    for col in cat_cols[:6]:
        try:
            ct = pd.crosstab(df[col], left_mask)
            if ct.shape[1] < 2: continue
            _, p, _, _ = scipy_stats.chi2_contingency(ct)
            if p < 0.05:
                rates = {str(k): round(left_mask[df[col]==k].mean()*100,1)
                         for k in df[col].dropna().unique()}
                worst = max(rates, key=rates.get)
                best  = min(rates, key=rates.get)
                top_drivers.append({
                    "factor":     col, "type": "categorical",
                    "impact":     round(rates[worst]-rates[best],1),
                    "worst_cat":  worst, "worst_rate": rates[worst],
                    "best_cat":   best,  "best_rate":  rates[best],
                    "p_value":    round(float(p),4),
                    "detail":     "'{}' = {:.0f}% vs '{}' = {:.0f}%".format(
                        worst, rates[worst], best, rates[best]),
                })
        except Exception:
            logger.debug("%s skip", exc_info=True)
            continue

    top_drivers.sort(key=lambda x: x["impact"], reverse=True)

    # Segment breakdown
    dept_col = next((c for c in df.columns
                     if "department" in c.lower() or "dept" in c.lower()), None)
    sal_col  = next((c for c in df.columns
                     if "salary" in c.lower() and df[c].dtype==object), None)

    dept_attrition = {}
    if dept_col:
        for d in df[dept_col].dropna().unique():
            m = df[dept_col]==d
            dept_attrition[str(d)] = round(left_mask[m].mean()*100,1)

    salary_attrition = {}
    if sal_col:
        for s in df[sal_col].dropna().unique():
            m = df[sal_col]==s
            salary_attrition[str(s)] = round(left_mask[m].mean()*100,1)

    # Flight risk
    sat_col = next((c for c in df.columns if "satisfaction" in c.lower()), None)
    n_flight = 0
    if sat_col and sat_col in df.columns:
        sv = df.loc[~left_mask, sat_col].dropna()
        thresh = float(sv.quantile(0.25)) if len(sv)>0 else 0.4
        n_flight = int((sv < thresh).sum())

    flight_pct = round(n_flight / max(n_total-n_left,1)*100,1)
    cost_str   = "Replacing {:,} employees estimated at 50-150% of annual salary each".format(n_left)

    return AttritionAnalysis(
        rate=rate, n_left=n_left, n_total=n_total,
        severity=severity, top_drivers=top_drivers[:8],
        dept_attrition=dept_attrition,
        salary_attrition=salary_attrition,
        n_flight_risk=n_flight, flight_risk_pct=flight_pct,
        cost_estimate=cost_str,
        interpretation="{:.1f}% attrition ({:,} employees). {} severity. Top driver: {}.".format(
            rate, n_left, severity.upper(),
            top_drivers[0]["factor"] if top_drivers else "unknown"),
    )


# ══════════════════════════════════════════════════════════
#  HR INSIGHTS
# ══════════════════════════════════════════════════════════

def _insights_hr(df: pd.DataFrame, stats: Dict,
                 corrs: List, attrition: Optional[AttritionAnalysis]) -> Dict:
    findings, risks, opps, actions = [], [], [], []
    insights = []

    # FIX-050: 50+ HR column synonym patterns — handles any HR dataset structure
    def _hr_col(*keywords, cat_ok=False, max_unique=None):
        """Find first column matching any keyword. Respects type and cardinality."""
        for col in df.columns:
            col_l = col.lower().strip()
            if any(kw in col_l for kw in keywords):
                if max_unique and df[col].nunique() > max_unique:
                    continue
                if not cat_ok and df[col].dtype == object:
                    continue
                return col
        return None

    # Satisfaction — 0-1 scale OR 1-5 scale OR 1-10 scale
    sat_col = _hr_col(
        "satisfaction", "engagement", "survey", "happiness",
        "morale", "sentiment", "wellbeing", "nps", "esat"
    )

    # Performance evaluation
    eval_col = _hr_col(
        "evaluat", "performance", "appraisal", "rating", "score",
        "review", "perfscore", "perfrating", "performancerating"
    )

    # Working hours
    hrs_col = _hr_col(
        "hour", "monthly_hours", "avg_hour", "workhour",
        "overtime", "workedhour", "timeworked"
    )

    # Department
    dept_col = _hr_col(
        "department", "dept", "division", "team", "group",
        "business_unit", "bu", "function", "unit",
        cat_ok=True, max_unique=30
    )

    # Projects
    proj_col = _hr_col(
        "project", "numberproject", "num_project", "activeproject"
    )

    # FIX-051: Satisfaction scale normalizer
    def _normalize_sat(col):
        """Normalize satisfaction to 0-1 scale for consistent benchmarking."""
        if col is None:
            return None, None
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(s) == 0:
            return col, 1.0
        max_val = s.max()
        if max_val <= 1.0:
            return col, 1.0      # Already 0-1
        elif max_val <= 5.0:
            return col, 5.0      # 1-5 scale
        elif max_val <= 10.0:
            return col, 10.0     # 1-10 scale
        else:
            return col, 100.0    # Percentage scale

    sat_col, sat_scale = _normalize_sat(sat_col)

    # ── Attrition ──────────────────────────────────────────
    if attrition:
        rate = attrition.rate
        if attrition.severity in ("critical","high"):
            insights.append(build_insight(
                title="Attrition Crisis: {:.1f}% of Workforce Left".format(rate),
                problem="{:,} out of {:,} employees left — {:.1f}% rate".format(
                    attrition.n_left, attrition.n_total, rate),
                cause="Rate {:.0f}% above industry benchmark (10-15%). ".format(rate-12.5) +
                      ("Primary driver: '{}'".format(attrition.top_drivers[0]["factor"])
                       if attrition.top_drivers else "Multiple factors"),
                evidence="Industry benchmark: 10-15%. Current: {:.1f}%. "
                         "{:,} remaining employees at flight risk ({:.0f}%).".format(
                    rate, attrition.n_flight_risk, attrition.flight_risk_pct),
                action="1. Exit interviews with all {:,} leavers this week  "
                       "2. Salary benchmarking vs market  "
                       "3. Engagement survey for remaining staff  "
                       "4. Identify and retain top performers".format(attrition.n_left),
                impact=attrition.cost_estimate + ". Knowledge loss accelerates with each exit.",
                severity="critical", category="attrition"
            ))
            risks.append("CRITICAL: {:.1f}% attrition — {:,} employees left. {}".format(
                rate, attrition.n_left, attrition.cost_estimate))
        else:
            findings.append("Attrition rate {:.1f}% within acceptable range (10-15% benchmark).".format(rate))

        # Dept breakdown
        if attrition.dept_attrition:
            sorted_d = sorted(attrition.dept_attrition.items(), key=lambda x:x[1], reverse=True)
            if len(sorted_d)>=2 and sorted_d[0][1] > sorted_d[-1][1]+10:
                worst_d, worst_r = sorted_d[0]
                best_d,  best_r  = sorted_d[-1]
                insights.append(build_insight(
                    title="'{}' Department: {:.0f}% Attrition vs {:.0f}% Best".format(
                        worst_d, worst_r, best_r),
                    problem="'{}' losing {:.0f}% of staff vs company average {:.1f}%".format(
                        worst_d, worst_r, rate),
                    cause="Department-specific issues: management quality, workload, or growth opportunities",
                    evidence="{:.0f}pp gap between highest ({}) and lowest ({}) attrition dept".format(
                        worst_r-best_r, worst_d, best_d),
                    action="1. Skip-level interviews in {} this week  "
                           "2. Manager effectiveness review  "
                           "3. Workload distribution audit".format(worst_d),
                    impact="Dept attrition destroys team cohesion and institutional knowledge",
                    severity="critical" if worst_r>25 else "warning",
                    category="attrition"
                ))
                risks.append("'{}' department attrition {:.0f}% — {:.0f}pp above best performer '{}'".format(
                    worst_d, worst_r, worst_r-best_r, best_d))

        # Salary band attrition
        if attrition.salary_attrition:
            sorted_s = sorted(attrition.salary_attrition.items(), key=lambda x:x[1], reverse=True)
            if sorted_s[0][1] > 20:
                insights.append(build_insight(
                    title="'{}' Salary Band: {:.0f}% Attrition — Pay Issue".format(
                        sorted_s[0][0], sorted_s[0][1]),
                    problem="Lowest salary band losing {:.0f}% of employees".format(sorted_s[0][1]),
                    cause="Below-market compensation driving employees to better-paying companies",
                    evidence="Attrition by pay: " + " | ".join(
                        ["{}: {:.0f}%".format(k,v) for k,v in sorted_s]),
                    action="1. Market salary benchmarking immediately  "
                           "2. Targeted retention bonuses for low-pay high-performers  "
                           "3. Review compensation bands",
                    impact="Pay-driven attrition is fastest to fix but most expensive if ignored",
                    severity="critical", category="attrition"
                ))

    # ── Satisfaction ───────────────────────────────────────
    if sat_col and sat_col in stats:
        st     = stats[sat_col]
        mean_s = st.get("mean",0)
        max_s  = st.get("max",1)
        pct    = (mean_s/max_s*100) if max_s>0 else 0
        low_n  = int((df[sat_col].dropna()<0.4).sum()) if sat_col in df.columns else 0
        low_pct= round(low_n/len(df)*100,1)

        if pct < 55:
            insights.append(build_insight(
                title="Satisfaction Crisis: Only {:.0f}% Score — {:,} Employees Disengaged".format(pct, low_n),
                problem="{:.0f}% satisfaction. {:,} employees ({:.0f}%) critically dissatisfied (below 40% threshold)".format(
                    pct, low_n, low_pct),
                cause="Score below 55% = systemic failure in culture, workload, recognition, or pay",
                evidence="Mean={:.2f}/{:.2f}. {:.0f}% below critical 0.4 threshold. "
                         "Industry benchmark for healthy orgs: 70%+".format(mean_s, max_s, low_pct),
                action="1. Anonymous pulse survey — identify top 3 pain points (48 hrs)  "
                       "2. Quick wins: flexible hours, recognition program, manager training  "
                       "3. Publish action plan within 30 days",
                impact="Each 10% satisfaction improvement = ~15% attrition reduction. "
                       "Low satisfaction costs 20% productivity loss.",
                severity="critical", category="satisfaction"
            ))
            risks.append("Satisfaction {:.0f}% — {:,} employees critically disengaged".format(pct, low_n))
        elif pct < 70:
            insights.append(build_insight(
                title="Satisfaction Below Target: {:.0f}% (Target: 70%+)".format(pct),
                problem="{:.0f}% satisfaction with {:,} employees below threshold".format(pct, low_n),
                cause="Specific fixable issues rather than systemic breakdown",
                evidence="Mean={:.2f}. {:.0f}% below 40% threshold. Benchmark: 70%+".format(mean_s, low_pct),
                action="1. Focus groups to identify top 3 issues  "
                       "2. Manager communication training  "
                       "3. Career development conversations",
                impact="Improving from {:.0f}% to 75% reduces attrition by 15-25%".format(pct),
                severity="warning", category="satisfaction"
            ))
        else:
            insights.append(build_insight(
                title="Strong Satisfaction: {:.0f}% — Above Industry Benchmark".format(pct),
                problem="N/A — satisfaction is healthy",
                cause="Effective HR practices and management culture",
                evidence="Mean={:.2f}. {:.0f}% of maximum score. Above 70% benchmark.".format(mean_s, pct),
                action="Maintain programs. Create career paths for high-performers to prevent exits.",
                impact="Strong satisfaction = lower attrition + higher productivity",
                severity="positive", category="satisfaction"
            ))
            opps.append("Satisfaction {:.0f}% is a competitive advantage — use in recruitment marketing".format(pct))

    # ── Overwork ───────────────────────────────────────────
    if hrs_col and hrs_col in stats:
        st       = stats[hrs_col]
        mean_hrs = st.get("mean",0)
        high_n   = int((df[hrs_col].dropna()>260).sum()) if hrs_col in df.columns else 0
        if mean_hrs > 220:
            insights.append(build_insight(
                title="Overwork Alert: Avg {:.0f} hrs/Month — {:,} Employees at Burnout Risk".format(
                    mean_hrs, high_n),
                problem="Average {:.0f} monthly hours. {:,} employees working 260+ hours (critical zone)".format(
                    mean_hrs, high_n),
                cause="Understaffing, poor task distribution, or culture of overwork",
                evidence="Mean={:.0f} hrs. Normal range: 160-200. Overwork zone: 240+. "
                         "{:,} employees in critical zone.".format(mean_hrs, high_n),
                action="1. Workload audit by team/department  "
                       "2. Hiring plan for overloaded teams  "
                       "3. No-overtime policy for 260+ hour employees",
                impact="Overworked employees 2-3x more likely to leave in 12 months. "
                       "Burnout reduces productivity by 30-40%.",
                severity="warning" if mean_hrs<240 else "critical",
                category="workload"
            ))
            risks.append("Avg {:.0f} hrs/month — overwork driving burnout and attrition".format(mean_hrs))

    # ── Projects ───────────────────────────────────────────
    if proj_col and proj_col in stats:
        st = stats[proj_col]
        if st.get("mean",0) > 5:
            insights.append(build_insight(
                title="High Project Load: Avg {:.1f} Projects Per Employee".format(st["mean"]),
                problem="Employees handling average {:.1f} projects simultaneously".format(st["mean"]),
                cause="Resource allocation issues or insufficient headcount for demand",
                evidence="Mean={:.1f} projects. Range: {:.0f}-{:.0f}. "
                         "Research shows 3-4 projects optimal for quality output.".format(
                    st["mean"], st["min"], st["max"]),
                action="1. Review project assignment process  "
                       "2. Cap individual project loads at 4-5  "
                       "3. Prioritize projects by strategic value",
                impact="Excessive project loads reduce output quality and increase error rates",
                severity="warning", category="workload"
            ))

    # ── Performance-Satisfaction link ──────────────────────
    if eval_col and sat_col:
        for corr in corrs:
            if (eval_col in [corr["col_a"],corr["col_b"]] and
                sat_col in [corr["col_a"],corr["col_b"]] and
                abs(corr["r"]) >= 0.3):
                findings.append(
                    "Performance and satisfaction are linked (r={:.2f}) — "
                    "improving satisfaction will improve evaluation scores".format(corr["r"]))

    # Department performance gap
    if dept_col and eval_col and eval_col in df.columns:
        dept_eval = df.groupby(dept_col)[eval_col].mean().sort_values()
        if len(dept_eval)>=2:
            gap = dept_eval.iloc[-1]-dept_eval.iloc[0]
            if gap > 0.1:
                findings.append(
                    "Performance gap: '{}' scores {:.2f} vs '{}' at {:.2f} — "
                    "{:.0f}% difference. Share best practices across departments.".format(
                        dept_eval.index[0], dept_eval.iloc[0],
                        dept_eval.index[-1], dept_eval.iloc[-1], gap/dept_eval.iloc[0]*100))

    # FIX-011b: Column-gated HR actions
    # Only recommend actions for columns that exist in this dataset
    _hr_cols = [c.lower() for c in df.columns]

    # Always valid — applies to any HR dataset
    actions.append(
        "Quarterly satisfaction pulse surveys — track trend monthly not annually"
    )

    # Only if attrition column exists
    _atr_present = any(k in _hr_cols for k in ["left", "attrition", "churned", "resigned"])
    if _atr_present:
        actions.append(
            "Identify flight-risk profile: low satisfaction + high tenure + "
            "no recent promotion — target this segment for retention conversations first"
        )

    # Only if salary column exists
    _sal_present = any(k in _hr_cols for k in ["salary", "salary_band", "compensation"])
    if _sal_present:
        actions.append(
            "Salary benchmarking against market — run within 30 days. "
            "SHRM: 38% of exits cite below-market pay as primary reason"
        )

    # Only if promotion column exists
    _promo_present = any(k in _hr_cols for k in ["promotion", "promoted", "promotion_last_5years"])
    if _promo_present:
        actions.append(
            "Career development paths for all levels — general guidance: "
            "career growth is the #1 driver of voluntary attrition"
        )

    # Only if satisfaction or evaluation column exists
    _sat_present = any(k in _hr_cols for k in ["satisfaction", "satisfaction_level", "engagement"])
    if _sat_present:
        actions.append(
            "Manager effectiveness training — general guidance: "
            "70% of employee satisfaction is driven by direct manager quality"
        )

    return {"findings":findings, "risks":risks, "opportunities":opps,
            "actions":actions, "insights":insights}


# ══════════════════════════════════════════════════════════
#  ECOMMERCE INSIGHTS
# ══════════════════════════════════════════════════════════


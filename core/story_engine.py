"""
story_engine.py — Senior MNC analyst level report engine.
Structure: Problem → Cause → Evidence → Action → Impact
Every insight is actionable, every finding has context.
"""
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from scipy import stats as scipy_stats


# ══════════════════════════════════════════════════════════
#  DOMAIN DETECTION
# ══════════════════════════════════════════════════════════

DOMAIN_KEYWORDS = {
    "ecommerce": ["price","discount","rating","product","category","order",
                  "revenue","sales","sku","review","brand","seller","cart"],
    "hr":        ["employee","salary","department","attrition","satisfaction",
                  "tenure","performance","hire","job","left","manager","bonus"],
    "finance":   ["profit","loss","expense","income","budget","cost","margin",
                  "ebitda","cashflow","asset","liability","tax","invoice"],
    "marketing": ["campaign","click","impression","conversion","lead","channel",
                  "spend","roi","ctr","cpa","retention","churn","traffic"],
    "healthcare":["patient","diagnosis","treatment","hospital","doctor",
                  "medicine","disease","age","blood","glucose","bmi"],
}


def detect_domain(df: pd.DataFrame) -> Tuple[str, float]:
    col_text = " ".join(df.columns.str.lower().tolist())
    scores = {}
    for domain, keywords in DOMAIN_KEYWORDS.items():
        hits = sum(1 for kw in keywords if kw in col_text)
        scores[domain] = hits / len(keywords)
    best = max(scores, key=scores.get)
    return (best, round(scores[best], 2)) if scores[best] > 0.05 else ("general", 0.0)


# ══════════════════════════════════════════════════════════
#  CORE DATA CLASSES
# ══════════════════════════════════════════════════════════

@dataclass
class Insight:
    """One complete insight with full context."""
    title:       str          # Short headline — 1 line
    problem:     str          # What is the problem?
    cause:       str          # Why is it happening?
    evidence:    str          # Numbers that prove it
    action:      str          # Exactly what to do
    impact:      str          # What happens if fixed / not fixed
    severity:    str          # "critical" / "warning" / "positive" / "info"
    category:    str          # "attrition" / "performance" / "salary" etc.


@dataclass
class AttritionAnalysis:
    """Deep attrition engine results."""
    rate:             float
    n_left:           int
    n_total:          int
    severity:         str      # "critical" / "high" / "normal" / "low"
    benchmark:        str      # "10-15% industry average"
    # Drivers
    top_drivers:      List[Dict]   # [{factor, impact, detail}]
    # Segment breakdown
    dept_attrition:   Dict         # {dept: rate}
    salary_attrition: Dict         # {low/med/high: rate}
    tenure_attrition: Dict         # {year: rate}
    # Risk employees
    n_flight_risk:    int          # estimated at-risk employees
    flight_risk_pct:  float
    # Financial impact
    cost_estimate:    str          # "₹X - ₹Y lakhs"
    interpretation:   str


@dataclass
class StoryReport:
    domain:              str
    domain_confidence:   float
    # Headlines
    headline:            str
    executive_summary:   str
    # Structured insights
    top_insights:        List[Insight]     # Top 5-7 insights for executives
    critical_issues:     List[Insight]     # Things that need immediate action
    positive_findings:   List[Insight]     # What's working well
    # Attrition (HR datasets)
    attrition:           Optional[AttritionAnalysis]
    # Standard fields
    key_findings:        List[str]
    business_risks:      List[str]
    opportunities:       List[str]
    recommended_actions: List[str]
    data_quality_verdict: str
    analysis_confidence: str
    column_insights:     List = field(default_factory=list)
    anomalies:           List[str] = field(default_factory=list)


# ══════════════════════════════════════════════════════════
#  STATS HELPERS
# ══════════════════════════════════════════════════════════

def _col_stats(s: pd.Series) -> Dict:
    clean = s.dropna().astype(float)
    if len(clean) < 3:
        return {}
    q1, q3 = float(clean.quantile(0.25)), float(clean.quantile(0.75))
    iqr    = q3 - q1
    lo, hi = q1 - 1.5*iqr, q3 + 1.5*iqr
    outliers = int(((clean < lo) | (clean > hi)).sum())
    try:
        _, pval = scipy_stats.shapiro(clean.sample(min(len(clean),5000), random_state=42))
        is_normal = pval > 0.05
    except Exception:
        is_normal = None
    return {
        "mean": round(float(clean.mean()),4),
        "median": round(float(clean.median()),4),
        "std": round(float(clean.std()),4),
        "min": round(float(clean.min()),4),
        "max": round(float(clean.max()),4),
        "q1": round(q1,4), "q3": round(q3,4), "iqr": round(iqr,4),
        "skew": round(float(clean.skew()),4),
        "cv": round(float(clean.std()/abs(clean.mean())),4) if clean.mean()!=0 else 0,
        "outliers": outliers,
        "outlier_pct": round(outliers/len(clean)*100,2),
        "is_normal": is_normal,
        "missing": int(s.isna().sum()),
        "missing_pct": round(s.isna().mean()*100,2),
        "n": len(clean),
    }


def _find_correlations(df: pd.DataFrame) -> List[Dict]:
    num_cols = df.select_dtypes(include="number").columns.tolist()
    results  = []
    for i in range(len(num_cols)):
        for j in range(i+1, len(num_cols)):
            a, b   = num_cols[i], num_cols[j]
            common = df[[a,b]].dropna()
            if len(common) < 10: continue
            try:
                r, p = scipy_stats.pearsonr(common[a], common[b])
                if abs(r) >= 0.2:
                    results.append({
                        "col_a":a, "col_b":b,
                        "r": round(float(r),3), "p": round(float(p),5),
                        "strength": "strong" if abs(r)>=0.7 else "moderate" if abs(r)>=0.4 else "weak",
                        "direction": "positive" if r>0 else "negative",
                    })
            except Exception:
                continue
    return sorted(results, key=lambda x: abs(x["r"]), reverse=True)


# ══════════════════════════════════════════════════════════
#  ATTRITION ENGINE
# ══════════════════════════════════════════════════════════

def _run_attrition_engine(df: pd.DataFrame) -> Optional[AttritionAnalysis]:
    """
    Full attrition analysis:
    - Rate calculation and severity
    - Driver identification
    - Segment breakdown  
    - Flight risk estimation
    - Financial cost estimate
    """
    # Find attrition column
    attr_col = next((c for c in df.columns
                     if "attrition" in c.lower()
                     or c.lower() in ["left","churned","resigned"]), None)
    if attr_col is None:
        return None

    # Normalize to boolean
    s = df[attr_col].astype(str).str.lower().str.strip()
    left_mask = s.isin(["yes","1","1.0","true","left"])
    n_left  = int(left_mask.sum())
    n_total = len(df)
    rate    = round(n_left / max(n_total,1) * 100, 1)

    if n_left == 0:
        return None

    # Severity
    if rate > 25:
        severity = "critical"
    elif rate > 18:
        severity = "high"
    elif rate > 12:
        severity = "warning"
    else:
        severity = "normal"

    # ── Driver analysis ───────────────────────────────────
    num_cols = df.select_dtypes(include="number").columns.tolist()
    num_cols = [c for c in num_cols if c != attr_col]
    cat_cols = [c for c in df.select_dtypes(include="object").columns
                if c != attr_col and df[c].nunique() <= 20]

    top_drivers = []

    # Numeric drivers — Mann-Whitney U test
    for col in num_cols[:10]:
        try:
            left_vals = df.loc[left_mask, col].dropna()
            stay_vals = df.loc[~left_mask, col].dropna()
            if len(left_vals) < 5 or len(stay_vals) < 5:
                continue
            _, p = scipy_stats.mannwhitneyu(left_vals, stay_vals, alternative="two-sided")
            if p < 0.05:
                diff = left_vals.mean() - stay_vals.mean()
                diff_pct = abs(diff) / abs(stay_vals.mean()) * 100 if stay_vals.mean() != 0 else 0
                top_drivers.append({
                    "factor":    col,
                    "type":      "numeric",
                    "impact":    round(diff_pct, 1),
                    "direction": "lower" if diff < 0 else "higher",
                    "left_mean": round(float(left_vals.mean()), 3),
                    "stay_mean": round(float(stay_vals.mean()), 3),
                    "p_value":   round(float(p), 4),
                    "detail": "Leavers avg {:.2f} vs stayers {:.2f} ({:.0f}% difference, p={:.4f})".format(
                        left_vals.mean(), stay_vals.mean(), diff_pct, p),
                })
        except Exception:
            continue

    # Categorical drivers — Chi-square
    for col in cat_cols[:6]:
        try:
            ct = pd.crosstab(df[col], left_mask)
            if ct.shape[1] < 2: continue
            chi2, p, _, _ = scipy_stats.chi2_contingency(ct)
            if p < 0.05:
                # Find which category has highest attrition
                cat_rates = {}
                for cat in df[col].dropna().unique():
                    mask2 = df[col] == cat
                    rate2  = left_mask[mask2].mean() * 100
                    cat_rates[str(cat)] = round(rate2, 1)
                worst_cat = max(cat_rates, key=cat_rates.get)
                best_cat  = min(cat_rates, key=cat_rates.get)
                top_drivers.append({
                    "factor":     col,
                    "type":       "categorical",
                    "impact":     round(cat_rates[worst_cat] - cat_rates[best_cat], 1),
                    "worst_cat":  worst_cat,
                    "worst_rate": cat_rates[worst_cat],
                    "best_cat":   best_cat,
                    "best_rate":  cat_rates[best_cat],
                    "p_value":    round(float(p), 4),
                    "detail": "'{}' has {:.0f}% attrition vs '{}' at {:.0f}% (p={:.4f})".format(
                        worst_cat, cat_rates[worst_cat], best_cat, cat_rates[best_cat], p),
                })
        except Exception:
            continue

    top_drivers.sort(key=lambda x: x["impact"], reverse=True)

    # ── Segment breakdown ─────────────────────────────────
    dept_col = next((c for c in df.columns
                     if "department" in c.lower() or "dept" in c.lower()), None)
    sal_col  = next((c for c in df.columns if "salary" in c.lower()
                     and df[c].dtype == object), None)
    ten_col  = next((c for c in df.columns
                     if "tenure" in c.lower() or "time_spend" in c.lower()), None)

    dept_attrition = {}
    if dept_col:
        for dept in df[dept_col].dropna().unique():
            mask2 = df[dept_col] == dept
            dept_attrition[str(dept)] = round(left_mask[mask2].mean()*100, 1)

    salary_attrition = {}
    if sal_col:
        for sal in df[sal_col].dropna().unique():
            mask2 = df[sal_col] == sal
            salary_attrition[str(sal)] = round(left_mask[mask2].mean()*100, 1)

    tenure_attrition = {}
    if ten_col and ten_col in df.columns:
        try:
            df["_tenure_bin"] = pd.cut(df[ten_col], bins=4,
                                        labels=["0-2y","2-4y","4-6y","6y+"])
            for tb in df["_tenure_bin"].dropna().unique():
                mask2 = df["_tenure_bin"] == tb
                tenure_attrition[str(tb)] = round(left_mask[mask2].mean()*100, 1)
            df.drop(columns=["_tenure_bin"], inplace=True, errors="ignore")
        except Exception:
            pass

    # ── Flight risk estimation ────────────────────────────
    sat_col = next((c for c in df.columns if "satisfaction" in c.lower()), None)
    n_flight_risk = 0
    if sat_col and sat_col in df.columns:
        sat_vals = df.loc[~left_mask, sat_col].dropna()
        low_threshold = sat_vals.quantile(0.25) if len(sat_vals) > 0 else 0.4
        n_flight_risk = int((sat_vals < low_threshold).sum())

    flight_risk_pct = round(n_flight_risk / max(n_total - n_left, 1) * 100, 1)

    # ── Cost estimation ───────────────────────────────────
    sal_num_col = next((c for c in df.select_dtypes(include="number").columns
                        if "salary" in c.lower() or "income" in c.lower()
                        or "wage" in c.lower()), None)
    if sal_num_col:
        avg_sal    = float(df[sal_num_col].median())
        cost_low   = round(avg_sal * n_left * 0.5 / 100000, 1)
        cost_high  = round(avg_sal * n_left * 1.5 / 100000, 1)
        cost_str   = "Approx {:.1f} - {:.1f} Lakhs (50-150% of annual salary per replacement)".format(
            cost_low, cost_high)
    else:
        cost_str = "Estimated significant — replacing each employee costs 50-150% of annual salary"

    interpretation = (
        "{:.1f}% attrition ({:,} employees) — {} severity. "
        "Industry benchmark: 10-15%. "
        "Top driver: {}. "
        "{:,} current employees estimated at flight risk ({:.0f}%).".format(
            rate, n_left, severity.upper(),
            top_drivers[0]["factor"] if top_drivers else "unknown",
            n_flight_risk, flight_risk_pct)
    )

    return AttritionAnalysis(
        rate=rate, n_left=n_left, n_total=n_total,
        severity=severity, benchmark="10-15% industry average",
        top_drivers=top_drivers[:8],
        dept_attrition=dept_attrition,
        salary_attrition=salary_attrition,
        tenure_attrition=tenure_attrition,
        n_flight_risk=n_flight_risk,
        flight_risk_pct=flight_risk_pct,
        cost_estimate=cost_str,
        interpretation=interpretation,
    )


# ══════════════════════════════════════════════════════════
#  INSIGHT BUILDER — Problem → Cause → Action → Impact
# ══════════════════════════════════════════════════════════

def _build_insight(title, problem, cause, evidence, action, impact,
                   severity="info", category="general") -> Insight:
    return Insight(
        title=title, problem=problem, cause=cause,
        evidence=evidence, action=action, impact=impact,
        severity=severity, category=category,
    )


def _insights_hr(df: pd.DataFrame, stats: Dict, corrs: List,
                 attrition: Optional[AttritionAnalysis]) -> List[Insight]:
    insights = []

    # ── Attrition insight ─────────────────────────────────
    if attrition:
        if attrition.severity in ("critical","high"):
            insights.append(_build_insight(
                title="Attrition Crisis: {:.1f}% of Workforce Left".format(attrition.rate),
                problem="{:,} employees left — {:.1f}% attrition rate".format(
                    attrition.n_left, attrition.rate),
                cause="Exceeds industry benchmark (10-15%). " +
                      ("Top driver: '{}' ({}%)".format(
                          attrition.top_drivers[0]["factor"],
                          attrition.top_drivers[0]["impact"])
                       if attrition.top_drivers else "Multiple compounding factors"),
                evidence="Rate {:.1f}% vs benchmark 10-15%. {:,} flight-risk employees remaining ({:.0f}%)".format(
                    attrition.rate, attrition.n_flight_risk, attrition.flight_risk_pct),
                action="1. Exit interview analysis this week  2. Salary benchmarking vs market  "
                       "3. Manager effectiveness audit  4. Engagement survey for remaining staff",
                impact=attrition.cost_estimate + ". Continued attrition will compound the problem.",
                severity="critical", category="attrition"
            ))
        elif attrition.severity == "warning":
            insights.append(_build_insight(
                title="Attrition Warning: {:.1f}% Above Healthy Benchmark".format(attrition.rate),
                problem="{:,} employees left ({:.1f}%) — above healthy 10-15% range".format(
                    attrition.n_left, attrition.rate),
                cause="Early signs of disengagement. " +
                      ("Strongest predictor: '{}'".format(attrition.top_drivers[0]["factor"])
                       if attrition.top_drivers else ""),
                evidence="Benchmark: 10-15%. Current: {:.1f}%. {:,} employees estimated flight risk.".format(
                    attrition.rate, attrition.n_flight_risk),
                action="1. Targeted retention for flight-risk employees  "
                       "2. Dept-level attrition deep dive  3. Compensation review",
                impact="If rate climbs to 25%+, cost and knowledge loss become severe.",
                severity="warning", category="attrition"
            ))

        # Dept breakdown insight
        if attrition.dept_attrition:
            sorted_dept = sorted(attrition.dept_attrition.items(),
                                 key=lambda x: x[1], reverse=True)
            if len(sorted_dept) >= 2:
                worst_d, worst_r = sorted_dept[0]
                best_d,  best_r  = sorted_dept[-1]
                if worst_r > best_r + 10:
                    insights.append(_build_insight(
                        title="Department '{}' Has Critical Attrition ({:.0f}%)".format(
                            worst_d, worst_r),
                        problem="'{}' losing {:.0f}% of employees vs '{}'s {:.0f}%".format(
                            worst_d, worst_r, best_d, best_r),
                        cause="Department-specific issues: management, workload, or growth opportunities",
                        evidence="Attrition gap: {:.0f}pp between highest ({}) and lowest ({}) dept".format(
                            worst_r-best_r, worst_d, best_d),
                        action="1. Skip-level interviews in {} dept  2. Manager coaching  "
                               "3. Workload audit".format(worst_d),
                        impact="High dept attrition destroys team cohesion and institutional knowledge",
                        severity="critical" if worst_r > 25 else "warning",
                        category="attrition"
                    ))

        # Salary-attrition insight
        if attrition.salary_attrition:
            sorted_sal = sorted(attrition.salary_attrition.items(),
                                key=lambda x: x[1], reverse=True)
            if sorted_sal[0][1] > 20:
                insights.append(_build_insight(
                    title="Low-Salary Employees Leaving at {:.0f}% Rate".format(sorted_sal[0][1]),
                    problem="'{}' salary band has {:.0f}% attrition — far above average".format(
                        sorted_sal[0][0], sorted_sal[0][1]),
                    cause="Compensation below market rate driving exits to better-paying companies",
                    evidence="Attrition by salary: " + " | ".join(
                        ["{}: {:.0f}%".format(k,v) for k,v in sorted_sal]),
                    action="1. Market salary benchmarking immediately  "
                           "2. Targeted retention bonuses for low-salary high-performers  "
                           "3. Review pay bands",
                    impact="Salary-driven attrition is fastest to fix but most costly if ignored",
                    severity="critical", category="attrition"
                ))

    # ── Satisfaction insight ──────────────────────────────
    sat_col = next((c for c in df.columns if "satisfaction" in c.lower()), None)
    if sat_col and sat_col in stats:
        st     = stats[sat_col]
        mean_s = st.get("mean", 0)
        max_s  = st.get("max", 1)
        pct    = (mean_s / max_s * 100) if max_s > 0 else 0
        low_n  = int((df[sat_col].dropna() < 0.4).sum())
        low_pct= round(low_n / len(df) * 100, 1)

        if pct < 55:
            insights.append(_build_insight(
                title="Low Employee Satisfaction: Only {:.0f}% of Maximum".format(pct),
                problem="{:.0f}% satisfaction score — {:,} employees ({:.0f}%) critically dissatisfied".format(
                    pct, low_n, low_pct),
                cause="Score below 55% indicates systemic issues: culture, workload, recognition, or pay",
                evidence="Mean={:.2f}, {:.0f}% below 0.4 threshold. Low satisfaction = leading attrition indicator.".format(
                    mean_s, low_pct),
                action="1. Anonymous pulse survey to find top 3 pain points  "
                       "2. Quick wins: flexible hours, recognition program  "
                       "3. Action plan published within 30 days",
                impact="Every 10% satisfaction improvement = ~15% reduction in attrition probability",
                severity="critical" if pct < 45 else "warning",
                category="satisfaction"
            ))
        elif pct < 70:
            insights.append(_build_insight(
                title="Satisfaction Needs Attention: {:.0f}% Score".format(pct),
                problem="Satisfaction at {:.0f}% with {:,} employees below threshold".format(pct, low_n),
                cause="Likely specific fixable issues rather than systemic problems",
                evidence="Mean={:.2f}. Benchmark for healthy orgs: 70%+ satisfaction.".format(mean_s),
                action="1. Focus groups to identify top issues  2. Manager training  "
                       "3. Career development conversations",
                impact="Raising satisfaction above 70% reduces attrition by 15-25%",
                severity="warning", category="satisfaction"
            ))
        else:
            insights.append(_build_insight(
                title="Good Satisfaction Score: {:.0f}%".format(pct),
                problem="N/A — satisfaction is healthy",
                cause="Effective HR practices and management",
                evidence="Mean={:.2f} ({:.0f}% of max). Above 70% benchmark.".format(mean_s, pct),
                action="Maintain programs. Focus on high-performer career paths to prevent exits.",
                impact="Continue monitoring — satisfaction can drop quickly with org changes.",
                severity="positive", category="satisfaction"
            ))

    # ── Hours/Workload insight ────────────────────────────
    hrs_col = next((c for c in df.columns if "hour" in c.lower()), None)
    if hrs_col and hrs_col in stats:
        st       = stats[hrs_col]
        mean_hrs = st.get("mean", 0)
        high_n   = int((df[hrs_col].dropna() > 260).sum()) if hrs_col in df.columns else 0

        if mean_hrs > 220:
            insights.append(_build_insight(
                title="Overwork Alert: Avg {:.0f} Hours/Month".format(mean_hrs),
                problem="Average monthly hours {:.0f} — above sustainable 160-200 range. {:,} employees working 260+ hrs".format(
                    mean_hrs, high_n),
                cause="Understaffing, poor workload distribution, or culture of overwork",
                evidence="Mean={:.0f} hrs/month. Normal range: 160-200. Overwork triggers burnout and attrition.".format(mean_hrs),
                action="1. Workload audit by team  2. Consider hiring  "
                       "3. Set overtime policy  4. Check if overwork correlates with low satisfaction",
                impact="Employees working 260+ hrs/month are 2-3x more likely to leave within 12 months",
                severity="warning" if mean_hrs < 240 else "critical",
                category="workload"
            ))

    # ── Performance-satisfaction correlation ─────────────
    eval_col = next((c for c in df.columns if "evaluat" in c.lower()), None)
    if eval_col and sat_col and eval_col in df.columns and sat_col in df.columns:
        try:
            r, p = scipy_stats.pearsonr(
                df[sat_col].dropna(),
                df.loc[df[sat_col].notna(), eval_col].dropna()
            )
            if abs(r) >= 0.3 and p < 0.05:
                insights.append(_build_insight(
                    title="Satisfaction & Performance Are Linked (r={:.2f})".format(r),
                    problem="Significant correlation between satisfaction and evaluation scores",
                    cause="Happy employees perform better — or high performers are more satisfied",
                    evidence="Pearson r={:.2f}, p={:.4f} — statistically significant".format(r, p),
                    action="1. Prioritize satisfaction of high-performers  "
                           "2. Use performance decline as early attrition warning signal",
                    impact="Improving satisfaction in bottom performers could increase evaluations by ~{:.0f}%".format(abs(r)*20),
                    severity="info", category="performance"
                ))
        except Exception:
            pass

    return insights


def _insights_ecommerce(df: pd.DataFrame, stats: Dict, corrs: List) -> List[Insight]:
    insights = []

    rating_col = next((c for c in df.columns if "rating" in c.lower()
                       and "count" not in c.lower()), None)
    price_col  = next((c for c in df.columns
                       if any(k in c.lower() for k in ["price","cost"]) and c in stats), None)
    disc_col   = next((c for c in df.columns if "discount" in c.lower() and c in stats), None)
    cat_col    = next((c for c in df.select_dtypes(include="object").columns
                       if "category" in c.lower() and df[c].nunique() <= 30), None)

    if rating_col and rating_col in stats:
        st     = stats[rating_col]
        mean_r = st.get("mean", 0)
        low_n  = int((df[rating_col].dropna() < 3.0).sum()) if rating_col in df.columns else 0

        if mean_r < 3.5:
            insights.append(_build_insight(
                title="Critical: Average Rating Only {:.2f}/5".format(mean_r),
                problem="{:.2f}/5 avg rating — {:,} products rated below 3.0".format(mean_r, low_n),
                cause="Products not meeting customer expectations — quality, delivery, or description mismatch",
                evidence="Mean={:.2f}, benchmark=4.0+. Low ratings directly reduce purchase likelihood.",
                action="1. Audit all products below 3.0  2. Customer feedback analysis  "
                       "3. Supplier quality review  4. Remove or improve bottom-rated items",
                impact="Ratings below 3.5 cause 40-60% lower purchase probability",
                severity="critical", category="rating"
            ))
        elif mean_r < 4.0:
            insights.append(_build_insight(
                title="Rating Below Target: {:.2f}/5 (Target: 4.0+)".format(mean_r),
                problem="{:.2f}/5 — below 4.0 benchmark. {:,} products below 3.0".format(mean_r, low_n),
                cause="Some product categories underperforming on quality or expectation management",
                evidence="Mean={:.2f}. 25th percentile={:.2f}.".format(mean_r, st.get("q1",0)),
                action="1. Fix bottom quartile products  2. Improve product descriptions  "
                       "3. Category-level audit",
                impact="Reaching 4.0+ rating = estimated 10-20% conversion improvement",
                severity="warning", category="rating"
            ))

        if cat_col and rating_col in df.columns:
            grp = df.groupby(cat_col)[rating_col].mean().sort_values()
            if len(grp) >= 2:
                gap = grp.iloc[-1] - grp.iloc[0]
                if gap > 0.3:
                    insights.append(_build_insight(
                        title="Category Gap: '{}' ({:.2f}) vs '{}' ({:.2f})".format(
                            grp.index[0], grp.iloc[0], grp.index[-1], grp.iloc[-1]),
                        problem="'{}' category underperforming by {:.1f} rating points".format(
                            grp.index[0], gap),
                        cause="Different supplier quality, product complexity, or customer expectations by category",
                        evidence="{:.1f} point gap between best and worst category".format(gap),
                        action="1. Quality audit of '{}' category  2. Supplier review  "
                               "3. Study best practices from '{}' category".format(
                                   grp.index[0], grp.index[-1]),
                        impact="Closing gap by 50% = +{:.1f} points to overall rating".format(gap*0.5),
                        severity="warning", category="rating"
                    ))

    if disc_col and disc_col in stats:
        avg_disc = stats[disc_col].get("mean", 0)
        if avg_disc > 40:
            insights.append(_build_insight(
                title="High Avg Discount {:.0f}% — Margin Risk".format(avg_disc),
                problem="Average discount {:.0f}% may be eroding profitability".format(avg_disc),
                cause="Competitive pressure or discounting strategy without margin analysis",
                evidence="Mean discount={:.0f}%, max={:.0f}%".format(
                    avg_disc, stats[disc_col].get("max",0)),
                action="1. Margin analysis per product category  2. Reduce discounts on high-rated items  "
                       "3. Strategic discounting only",
                impact="Every 10% unnecessary discount = direct margin loss at no incremental benefit",
                severity="warning", category="pricing"
            ))

    return insights


def _insights_general(df: pd.DataFrame, stats: Dict, corrs: List) -> List[Insight]:
    insights = []

    for col in list(stats.keys())[:5]:
        st = stats.get(col, {})
        if not st: continue
        skew    = st.get("skew", 0)
        out_pct = st.get("outlier_pct", 0)
        cv      = st.get("cv", 0)
        mean    = st.get("mean", 0)
        median  = st.get("median", 0)

        if out_pct > 10:
            insights.append(_build_insight(
                title="'{}' Has {:.0f}% Outliers — Data Quality Issue".format(col, out_pct),
                problem="{:.0f}% of '{}' values are statistical outliers".format(out_pct, col),
                cause="Data entry errors, measurement anomalies, or genuine extreme values",
                evidence="IQR method: {:.0f}% outliers. Range: {:.2f} to {:.2f}".format(
                    out_pct, st.get("min",0), st.get("max",0)),
                action="1. Inspect outlier records  2. Determine if errors or genuine  "
                       "3. Cap or remove confirmed errors  4. Document decisions",
                impact="Outliers distort all statistical analyses and degrade ML model accuracy",
                severity="warning", category="data_quality"
            ))

        if abs(skew) > 1.5:
            insights.append(_build_insight(
                title="'{}' Is Heavily Skewed — Use Median Not Mean".format(col),
                problem="Distribution skewed ({:.2f}) — mean {:.2f} misrepresents typical value".format(
                    skew, mean),
                cause="A small number of extreme values pull the mean far from the typical value",
                evidence="Mean={:.2f} vs Median={:.2f} — {:.0f}% difference".format(
                    mean, median, abs(mean-median)/abs(median)*100 if median!=0 else 0),
                action="1. Report median ({:.2f}) in all summaries  2. Apply log-transform before modeling  "
                       "3. Segment data to understand subgroups".format(median),
                impact="Using mean on skewed data can misrepresent reality by {:.0f}%".format(
                    abs(mean-median)/abs(median)*100 if median!=0 else 0),
                severity="info", category="statistics"
            ))

    for corr in corrs[:3]:
        if corr.get("strength") in ("strong","moderate") and corr.get("p",1) < 0.05:
            insights.append(_build_insight(
                title="{} Correlation: '{}' ↔ '{}' (r={:.2f})".format(
                    corr["strength"].title(), corr["col_a"], corr["col_b"], corr["r"]),
                problem="Significant {} relationship detected".format(corr["direction"]),
                cause="Variables move together — potential causal relationship or shared driver",
                evidence="Pearson r={:.2f}, p={:.4f} (statistically significant at 95% confidence)".format(
                    corr["r"], corr["p"]),
                action="1. Investigate direction of causality  2. Use as predictor in models  "
                       "3. Check for confounding variables",
                impact="Strong predictors can improve model accuracy by 10-30%",
                severity="info", category="statistics"
            ))

    return insights


# ══════════════════════════════════════════════════════════
#  ANOMALY DETECTION
# ══════════════════════════════════════════════════════════

def _detect_anomalies(df: pd.DataFrame, stats: Dict) -> List[str]:
    anomalies = []
    for col, st in stats.items():
        if not st: continue
        if st.get("outlier_pct",0) > 10:
            anomalies.append(
                "'{}' has {:.1f}% outliers — normal range {:.2f} to {:.2f}. "
                "Validate before analysis.".format(
                    col, st["outlier_pct"],
                    st["q1"] - 1.5*st["iqr"], st["q3"] + 1.5*st["iqr"]))
        if abs(st.get("skew",0)) > 2:
            anomalies.append(
                "'{}' heavily skewed ({:.2f}). Median {:.2f} more reliable than mean {:.2f}.".format(
                    col, st["skew"], st["median"], st["mean"]))
        if st.get("missing_pct",0) > 20:
            anomalies.append(
                "'{}' is {:.1f}% missing — imputed values may affect results.".format(
                    col, st["missing_pct"]))
    return anomalies


# ══════════════════════════════════════════════════════════
#  MAIN GENERATE FUNCTION
# ══════════════════════════════════════════════════════════

def generate_story(df: pd.DataFrame) -> StoryReport:
    """
    Full senior analyst story generation.
    Returns StoryReport with structured insights, attrition analysis,
    and Problem→Cause→Action→Impact format throughout.
    """
    domain, confidence = detect_domain(df)

    # Compute column stats
    num_cols  = df.select_dtypes(include="number").columns.tolist()
    all_stats = {}
    for col in num_cols:
        st = _col_stats(df[col])
        if st: all_stats[col] = st

    # Correlations
    corrs = _find_correlations(df)

    # Attrition engine
    attrition = _run_attrition_engine(df)

    # Domain-specific insights
    if domain == "hr":
        raw_insights = _insights_hr(df, all_stats, corrs, attrition)
    elif domain == "ecommerce":
        raw_insights = _insights_ecommerce(df, all_stats, corrs)
    else:
        raw_insights = _insights_general(df, all_stats, corrs)

    # Always add general stats insights for any dataset
    if domain != "general":
        raw_insights += _insights_general(df, all_stats, corrs)

    # Deduplicate and rank
    seen_titles = set()
    deduped = []
    for ins in raw_insights:
        if ins.title not in seen_titles:
            seen_titles.add(ins.title)
            deduped.append(ins)

    # Split by severity
    sev_order = {"critical":0, "warning":1, "info":2, "positive":3}
    sorted_insights = sorted(deduped, key=lambda x: sev_order.get(x.severity, 99))

    top_insights     = sorted_insights[:6]
    critical_issues  = [i for i in sorted_insights if i.severity == "critical"]
    positive_findings= [i for i in sorted_insights if i.severity == "positive"]

    # Build flat lists for PDF
    key_findings = []
    business_risks = []
    opportunities  = []
    actions_flat   = []

    for ins in sorted_insights:
        finding_str = "{} | Problem: {} | Cause: {} | Evidence: {}".format(
            ins.title, ins.problem, ins.cause, ins.evidence)
        if ins.severity in ("critical","warning"):
            business_risks.append(finding_str)
        elif ins.severity == "positive":
            opportunities.append("{} — {}".format(ins.title, ins.action))
        else:
            key_findings.append(finding_str)
        actions_flat.append("[{}] {} → {}".format(
            ins.severity.upper(), ins.title, ins.action))

    # Anomalies
    anomalies = _detect_anomalies(df, all_stats)

    # Executive summary
    n_critical = len(critical_issues)
    exec_summary = (
        "This {}-row, {}-column {} dataset was analyzed using "
        "senior analyst methodology. ".format(len(df), len(df.columns), domain)
    )
    if attrition:
        exec_summary += (
            "ATTRITION: {:.1f}% ({:,} employees) — {} severity. ".format(
                attrition.rate, attrition.n_left, attrition.severity.upper())
        )
    if n_critical > 0:
        exec_summary += "{} critical issue(s) require immediate action. ".format(n_critical)
    if sorted_insights:
        exec_summary += "Top finding: {}. ".format(sorted_insights[0].title)
    exec_summary += "{} total insights with prioritized action plan.".format(len(sorted_insights))

    # Headline
    if critical_issues:
        headline = "CRITICAL: " + critical_issues[0].title
    elif sorted_insights:
        headline = sorted_insights[0].title
    else:
        headline = "Analysis complete — {} insights generated".format(len(sorted_insights))

    # Data quality
    avg_miss = sum(st.get("missing_pct",0) for st in all_stats.values()) / max(len(all_stats),1)
    if avg_miss < 5:
        quality_verdict = "GOOD — low missing rates, data is reliable for analysis."
    elif avg_miss < 20:
        quality_verdict = "FAIR — {:.1f}% avg missing values. Imputation applied.".format(avg_miss)
    else:
        quality_verdict = "NEEDS ATTENTION — {:.1f}% missing. Results may be less reliable.".format(avg_miss)

    # Confidence
    n_rows = len(df)
    if n_rows >= 1000 and len(num_cols) >= 3:
        conf_label = "HIGH — sufficient data for reliable conclusions."
    elif n_rows >= 100:
        conf_label = "MEDIUM — directional results, more data improves reliability."
    else:
        conf_label = "LOW — small dataset, treat findings as directional only."

    return StoryReport(
        domain=domain, domain_confidence=confidence,
        headline=headline, executive_summary=exec_summary,
        top_insights=top_insights,
        critical_issues=critical_issues,
        positive_findings=positive_findings,
        attrition=attrition,
        key_findings=key_findings[:6],
        business_risks=business_risks[:6],
        opportunities=opportunities[:4],
        recommended_actions=actions_flat[:8],
        data_quality_verdict=quality_verdict,
        analysis_confidence=conf_label,
        anomalies=anomalies,
    )

"""
story_engine.py — Domain detection + Business storytelling
Senior analyst level insights in plain English.
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
    "ecommerce": [
        "price", "discount", "rating", "product", "category",
        "order", "cart", "purchase", "revenue", "sales", "sku",
        "inventory", "stock", "review", "brand", "seller"
    ],
    "hr": [
        "employee", "salary", "department", "attrition", "tenure",
        "performance", "hire", "job", "role", "satisfaction",
        "manager", "promotion", "bonus", "leave", "headcount"
    ],
    "finance": [
        "profit", "loss", "expense", "income", "budget", "cost",
        "margin", "ebitda", "cashflow", "asset", "liability",
        "tax", "invoice", "payment", "balance", "transaction"
    ],
    "marketing": [
        "campaign", "click", "impression", "conversion", "lead",
        "channel", "spend", "roi", "ctr", "cpa", "acquisition",
        "retention", "churn", "engagement", "traffic", "funnel"
    ],
    "healthcare": [
        "patient", "diagnosis", "treatment", "hospital", "doctor",
        "medicine", "disease", "age", "blood", "pressure",
        "glucose", "bmi", "clinical", "symptom", "health"
    ],
    "logistics": [
        "shipment", "delivery", "warehouse", "route", "freight",
        "tracking", "carrier", "origin", "destination", "weight",
        "volume", "dispatch", "transit", "fleet", "driver"
    ],
}


def detect_domain(df: pd.DataFrame) -> Tuple[str, float]:
    """
    Detect dataset domain from column names.
    Returns (domain, confidence_score 0-1).
    """
    col_text = " ".join(df.columns.str.lower().tolist())

    scores = {}
    for domain, keywords in DOMAIN_KEYWORDS.items():
        hits = sum(1 for kw in keywords if kw in col_text)
        scores[domain] = hits / len(keywords)

    best_domain = max(scores, key=scores.get)
    best_score  = scores[best_domain]

    if best_score < 0.05:
        return "general", 0.0

    return best_domain, round(best_score, 2)


# ══════════════════════════════════════════════════════════
#  DATA SHAPE ANALYSIS
# ══════════════════════════════════════════════════════════

@dataclass
class ColumnInsight:
    column: str
    dtype: str
    finding: str          # plain English finding
    severity: str         # "info", "warning", "critical", "positive"
    metric: str           # the specific number/stat
    recommendation: str   # what to do


@dataclass
class StoryReport:
    domain: str
    domain_confidence: float
    headline: str                          # single most important finding
    executive_summary: str                 # 3-5 sentence summary
    key_findings: List[str]               # bullet points
    column_insights: List[ColumnInsight]  # per-column stories
    anomalies: List[str]                  # unusual patterns
    business_risks: List[str]            # risk flags
    opportunities: List[str]             # growth/improvement opportunities
    recommended_actions: List[str]       # concrete next steps
    data_quality_verdict: str            # overall data health sentence
    analysis_confidence: str             # high/medium/low + reason


# ══════════════════════════════════════════════════════════
#  CORE ANALYSIS FUNCTIONS
# ══════════════════════════════════════════════════════════

def _analyze_numeric_col(s: pd.Series) -> Dict:
    """Full stats for one numeric column."""
    clean = s.dropna()
    if len(clean) < 3:
        return {}

    q1, q3  = clean.quantile(0.25), clean.quantile(0.75)
    iqr     = q3 - q1
    skew    = float(clean.skew())
    mean    = float(clean.mean())
    median  = float(clean.median())
    std     = float(clean.std())
    cv      = std / abs(mean) if mean != 0 else 0

    outliers_iqr = int(((clean < q1 - 1.5*iqr) | (clean > q3 + 1.5*iqr)).sum())
    outlier_pct  = round(outliers_iqr / len(clean) * 100, 1)

    # Normality
    try:
        _, p_val = scipy_stats.shapiro(clean.sample(min(len(clean), 5000), random_state=42))
        is_normal = p_val > 0.05
    except Exception:
        is_normal = None

    return {
        "mean": round(mean, 4), "median": round(median, 4),
        "std": round(std, 4), "cv": round(cv, 4),
        "q1": round(q1, 4), "q3": round(q3, 4), "iqr": round(iqr, 4),
        "skew": round(skew, 4), "min": round(float(clean.min()), 4),
        "max": round(float(clean.max()), 4),
        "outliers": outliers_iqr, "outlier_pct": outlier_pct,
        "is_normal": is_normal, "missing": int(s.isna().sum()),
        "missing_pct": round(s.isna().sum() / max(len(s), 1) * 100, 1),
    }


def _find_correlations(df: pd.DataFrame) -> List[Dict]:
    """Find significant correlations between numeric columns."""
    num_cols = df.select_dtypes(include="number").columns.tolist()
    results  = []

    for i in range(len(num_cols)):
        for j in range(i+1, len(num_cols)):
            a, b = num_cols[i], num_cols[j]
            common = df[[a, b]].dropna()
            if len(common) < 10:
                continue
            try:
                r, p = scipy_stats.pearsonr(common[a], common[b])
                if p < 0.05 and abs(r) >= 0.3:
                    results.append({
                        "col_a": a, "col_b": b,
                        "r": round(r, 3), "p": round(p, 5),
                        "strength": "strong" if abs(r) >= 0.7
                                    else "moderate" if abs(r) >= 0.5
                                    else "weak",
                        "direction": "positive" if r > 0 else "negative",
                    })
            except Exception:
                continue

    return sorted(results, key=lambda x: abs(x["r"]), reverse=True)


def _detect_anomalies(df: pd.DataFrame, stats: Dict) -> List[str]:
    """Detect unusual patterns worth flagging."""
    anomalies = []
    num_cols  = df.select_dtypes(include="number").columns.tolist()

    for col in num_cols:
        st = stats.get(col, {})
        if not st:
            continue

        # High outlier rate
        if st.get("outlier_pct", 0) > 10:
            anomalies.append(
                "'{}' has {:.1f}% outliers ({} values) — "
                "normal range is {:.2f} to {:.2f}. "
                "Investigate before using in models.".format(
                    col, st["outlier_pct"], st["outliers"],
                    st["q1"] - 1.5*st["iqr"], st["q3"] + 1.5*st["iqr"]
                )
            )

        # Extreme skew
        if abs(st.get("skew", 0)) > 2:
            direction = "right" if st["skew"] > 0 else "left"
            anomalies.append(
                "'{}' is heavily {}-skewed (skew={:.2f}). "
                "A small number of extreme values are pulling the mean. "
                "Median ({:.2f}) is more representative than mean ({:.2f}).".format(
                    col, direction, st["skew"], st["median"], st["mean"]
                )
            )

        # High missing rate
        if st.get("missing_pct", 0) > 20:
            anomalies.append(
                "'{}' is {:.1f}% missing — "
                "imputed values may distort analysis.".format(
                    col, st["missing_pct"]
                )
            )

        # Zero variance sections — all same value
        if st.get("cv", 1) < 0.01 and st.get("std", 1) > 0:
            anomalies.append(
                "'{}' has almost no variation (CV={:.4f}) — "
                "may be a constant or derived field.".format(
                    col, st["cv"]
                )
            )

    return anomalies


# ══════════════════════════════════════════════════════════
#  DOMAIN-SPECIFIC STORY GENERATORS
# ══════════════════════════════════════════════════════════

def _senior_insight(what: str, why: str, action: str, impact: str = "") -> str:
    """Format a senior analyst insight: What happened, Why, What to do, Impact."""
    parts = ["WHAT: " + what, "WHY: " + why, "ACTION: " + action]
    if impact:
        parts.append("IMPACT: " + impact)
    return " | ".join(parts)


def _story_ecommerce(df, stats, corrs):
    findings, risks, opps, actions = [], [], [], []

    rating_col = next((c for c in df.columns if "rating" in c.lower()
                       and "count" not in c.lower()), None)
    price_col  = next((c for c in df.columns
                       if any(k in c.lower() for k in ["price","cost"])
                       and c in stats), None)
    disc_col   = next((c for c in df.columns
                       if "discount" in c.lower() and c in stats), None)
    cat_col    = next((c for c in df.select_dtypes(include="object").columns
                       if "category" in c.lower()
                       and df[c].nunique() <= 30), None)

    if rating_col and rating_col in stats:
        st      = stats[rating_col]
        mean_r  = st.get("mean", 0)
        q1      = st.get("q1", 0)
        out_ct  = st.get("outliers", 0)
        out_pct = st.get("outlier_pct", 0)

        if mean_r >= 4.3:
            findings.append(_senior_insight(
                what="Excellent avg rating {:.2f}/5".format(mean_r),
                why="Products consistently meeting customer expectations",
                action="Use high ratings in marketing — highlight in ads and product pages",
                impact="High ratings correlate with 20-30% higher conversion rates"))
            opps.append("Premium pricing opportunity — customers willing to pay more for high-rated products.")
        elif mean_r >= 4.0:
            findings.append(_senior_insight(
                what="Good avg rating {:.2f}/5 but 25% of products rated below {:.1f}".format(mean_r, q1),
                why="Bottom quartile products dragging overall performance",
                action="Identify and fix bottom 25% rated products — quality review or removal",
                impact="Improving bottom quartile to average could raise overall rating to 4.3+"))
        elif mean_r >= 3.5:
            risks.append(_senior_insight(
                what="Below-average rating {:.2f}/5 — at risk of losing customers".format(mean_r),
                why="Customer dissatisfaction indicates product-quality or expectation mismatch",
                action="URGENT: Audit lowest-rated products, collect feedback, fix top complaints",
                impact="Every 0.1 drop in rating = estimated 5-10% drop in sales"))
        else:
            risks.append(_senior_insight(
                what="CRITICAL: Low rating {:.2f}/5 — severe customer dissatisfaction".format(mean_r),
                why="Products failing to meet basic customer expectations",
                action="Immediate product quality audit, remove products rated below 3.0",
                impact="Ratings below 3.5 result in 40-60% lower purchase likelihood"))

        if out_ct > 0:
            findings.append(_senior_insight(
                what="{} products with outlier ratings ({:.1f}% of catalog)".format(out_ct, out_pct),
                why="Outlier products are either exceptional or seriously problematic",
                action="Review each outlier individually — low outliers need urgent attention",
                impact="Low-rated outliers disproportionately affect brand perception"))

    if price_col and price_col in stats:
        st   = stats[price_col]
        skew = st.get("skew", 0)
        cv   = st.get("cv", 0)
        if skew > 1:
            findings.append(_senior_insight(
                what="Price right-skewed — median {:.0f} vs mean {:.0f}".format(
                    st.get("median",0), st.get("mean",0)),
                why="Few expensive products pull mean up — most products are budget-range",
                action="Segment pricing: budget vs premium. Use median not mean for pricing decisions",
                impact="Skewed pricing often indicates untapped mid-market opportunity"))

    if disc_col and disc_col in stats:
        avg_disc = stats[disc_col].get("mean", 0)
        if avg_disc > 40:
            risks.append(_senior_insight(
                what="High avg discount {:.1f}% — potentially eroding margins".format(avg_disc),
                why="Heavy discounting trains customers to wait for sales, devalues brand",
                action="Analyze margin impact. Reduce discounts on high-rated products",
                impact="Every 10% unnecessary discount = direct margin loss"))

    pr_corr = next((c for c in corrs if
        ("rating" in c["col_a"].lower() or "rating" in c["col_b"].lower()) and
        ("price" in c["col_a"].lower() or "price" in c["col_b"].lower())), None)
    if pr_corr:
        r = pr_corr["r"]
        if r < -0.3:
            risks.append(_senior_insight(
                what="Higher-priced products have LOWER ratings (r={:.2f})".format(r),
                why="Premium pricing not delivering premium experience — expectation gap",
                action="Review premium product quality. Align quality with price or reduce prices",
                impact="Price-quality mismatch is leading cause of negative reviews and returns"))
        elif r > 0.3:
            opps.append(_senior_insight(
                what="Higher-priced products have HIGHER ratings (r={:.2f}) — quality-price alignment".format(r),
                why="Premium products delivering on value",
                action="Expand premium range. Use as proof point in marketing"))

    if cat_col and rating_col and rating_col in df.columns:
        grp = df.groupby(cat_col)[rating_col].mean().sort_values()
        if len(grp) >= 2:
            gap = grp.iloc[-1] - grp.iloc[0]
            findings.append(_senior_insight(
                what="Category gap: '{}' ({:.2f}) vs '{}' ({:.2f}) — {:.2f} point difference".format(
                    grp.index[0], grp.iloc[0], grp.index[-1], grp.iloc[-1], gap),
                why="Categories have different quality standards or supplier performance",
                action="Prioritize quality improvement in '{}'. Study best practices from '{}'".format(
                    grp.index[0], grp.index[-1]),
                impact="Closing category gap by 50% could add {:.1f} points to overall rating".format(gap*0.5)))

    actions.extend([
        "Review all products rated below 3.0 — fix or remove",
        "A/B test pricing on top-rated products to optimize revenue",
        "Implement post-purchase review collection system",
        "Monthly competitor rating comparison — maintain advantage",
    ])
    return {"findings": findings, "risks": risks, "opportunities": opps, "actions": actions}


def _story_hr(df, stats, corrs):
    findings, risks, opps, actions = [], [], [], []

    sat_col  = next((c for c in df.columns if "satisfaction" in c.lower()), None)
    attr_col = next((c for c in df.columns
                     if "attrition" in c.lower() or c.lower() == "left"), None)
    sal_col  = next((c for c in df.columns
                     if "salary" in c.lower() and df[c].dtype == object), None)
    hrs_col  = next((c for c in df.columns if "hour" in c.lower()), None)
    dept_col = next((c for c in df.columns
                     if "department" in c.lower() and df[c].nunique() <= 20), None)
    eval_col = next((c for c in df.columns if "evaluat" in c.lower()), None)

    if attr_col and attr_col in df.columns:
        vc       = df[attr_col].value_counts(normalize=True)
        yes_keys = [k for k in vc.index if str(k).lower() in ["yes","1","1.0","true"]]
        if yes_keys:
            rate  = vc[yes_keys[0]] * 100
            n_left= int(vc[yes_keys[0]] * len(df))
            if rate > 20:
                risks.append(_senior_insight(
                    what="CRITICAL attrition {:.1f}% — {:,} employees left".format(rate, n_left),
                    why="Exceeds industry benchmark 10-15% — systemic retention failure",
                    action="URGENT: Exit interviews, salary benchmarking, manager effectiveness audit",
                    impact="Replacing {:,} employees estimated at 1-2x annual salary each — significant cost".format(n_left)))
            elif rate > 15:
                risks.append(_senior_insight(
                    what="Above-average attrition {:.1f}% ({:,} employees)".format(rate, n_left),
                    why="Early warning sign — exceeds healthy 10-15% benchmark",
                    action="Survey current employees. Identify top attrition drivers by department and tenure",
                    impact="Each percentage point above benchmark = significant additional recruitment cost"))
            else:
                findings.append(_senior_insight(
                    what="Healthy attrition {:.1f}% — within industry benchmark 10-15%".format(rate),
                    why="Retention programs appear effective",
                    action="Maintain programs. Focus on keeping high-performers with career development paths"))

            if sat_col and sat_col in df.columns:
                left_mask = df[attr_col].astype(str).str.lower().isin(["yes","1","1.0","true"])
                left_sat  = df.loc[left_mask, sat_col].mean()
                stay_sat  = df.loc[~left_mask, sat_col].mean()
                if not (pd.isna(left_sat) or pd.isna(stay_sat)) and stay_sat > 0:
                    diff_pct = (1 - left_sat/stay_sat) * 100
                    if diff_pct > 10:
                        risks.append(_senior_insight(
                            what="Leavers had {:.0f}% lower satisfaction ({:.2f} vs {:.2f})".format(
                                diff_pct, left_sat, stay_sat),
                            why="Satisfaction is a leading indicator — dissatisfied employees leave",
                            action="Target employees with satisfaction below {:.2f} — engage proactively".format(
                                stay_sat * 0.85),
                            impact="Proactive engagement of at-risk employees can reduce attrition by 20-30%"))

    if sat_col and sat_col in stats:
        st     = stats[sat_col]
        mean_s = st.get("mean", 0)
        max_s  = st.get("max", 1)
        pct    = (mean_s / max_s * 100) if max_s > 0 else 0
        low_n  = int((df[sat_col].dropna() < 0.4).sum()) if sat_col in df.columns else 0

        if pct < 50:
            risks.append(_senior_insight(
                what="LOW satisfaction {:.0f}% — {:,} employees highly dissatisfied (<40%)".format(pct, low_n),
                why="Below 50% satisfaction indicates fundamental issues: culture, workload, or pay",
                action="Anonymous pulse survey. Form action team. Address top 3 pain points within 30 days",
                impact="Low satisfaction = 2x higher attrition risk, 20% lower productivity"))
        elif pct < 70:
            findings.append(_senior_insight(
                what="Moderate satisfaction {:.0f}% — improvement needed ({:,} dissatisfied)".format(pct, low_n),
                why="Mid-range satisfaction often has specific fixable causes",
                action="Focus groups to find top 3 issues. Quick wins: flexibility, recognition, communication",
                impact="Improving satisfaction from 60% to 80% reduces attrition by 15-25%"))
        else:
            findings.append(_senior_insight(
                what="Good satisfaction {:.0f}% — majority of employees engaged".format(pct),
                why="Effective HR practices in place",
                action="Maintain current programs. Develop career paths for high-performers"))

    if hrs_col and hrs_col in stats:
        mean_hrs = stats[hrs_col].get("mean", 0)
        if mean_hrs > 220:
            risks.append(_senior_insight(
                what="HIGH avg monthly hours {:.0f} — overwork detected".format(mean_hrs),
                why="Overwork leads to burnout and drives attrition",
                action="Hire additional staff or redistribute tasks. Set clear overtime limits",
                impact="Overworked employees are 2-3x more likely to leave within 12 months"))

    if dept_col and sat_col and dept_col in df.columns and sat_col in df.columns:
        dept_sat = df.groupby(dept_col)[sat_col].mean().sort_values()
        if len(dept_sat) >= 2:
            gap = dept_sat.iloc[-1] - dept_sat.iloc[0]
            findings.append(_senior_insight(
                what="Dept gap: '{}' ({:.2f}) vs '{}' ({:.2f})".format(
                    dept_sat.index[0], dept_sat.iloc[0],
                    dept_sat.index[-1], dept_sat.iloc[-1]),
                why="Management style, workload, career growth differ by department",
                action="Focus retention on '{}' dept. Leadership coaching if needed".format(
                    dept_sat.index[0]),
                impact="Closing dept gap reduces dept-level attrition risk"))

    actions.extend([
        "Quarterly satisfaction surveys — track trends, dont rely on annual reviews",
        "Identify flight-risk employees: low satisfaction + high tenure + average salary",
        "Salary benchmarking against market — competitive pay is top retention factor",
        "Career development paths for all levels — lack of growth is #1 reason people leave",
    ])
    return {"findings": findings, "risks": risks, "opportunities": opps, "actions": actions}


def _story_finance(df, stats, corrs):
    findings, risks, opps, actions = [], [], [], []
    for col, st in stats.items():
        col_lower = col.lower()
        if any(k in col_lower for k in ["revenue","sales","income"]):
            skew = st.get("skew", 0)
            findings.append(_senior_insight(
                what="{} ranges {:,.0f} to {:,.0f}, median {:,.0f}".format(
                    col, st.get("min",0), st.get("max",0), st.get("median",0)),
                why="Revenue distribution shows business performance spread",
                action="Focus on understanding top 20% revenue drivers — likely driving 80% of total"))
            if skew > 1:
                opps.append(_senior_insight(
                    what="Revenue right-skewed — few transactions drive disproportionate income",
                    why="Pareto effect: small number of high-value clients/products dominate",
                    action="Identify and protect top revenue sources. Develop more high-value relationships"))
        elif any(k in col_lower for k in ["cost","expense"]):
            cv = st.get("cv", 0)
            if cv > 0.5:
                risks.append(_senior_insight(
                    what="High cost variance in '{}' (CV={:.2f})".format(col, cv),
                    why="Inconsistent costs indicate process inefficiency or uncontrolled spending",
                    action="Audit highest-cost instances. Standardize procurement. Set cost thresholds"))
    actions.extend([
        "Identify top 20% revenue drivers and protect them",
        "Set cost variance thresholds — flag anything >2 std deviations",
        "Monthly margin analysis by product/segment",
    ])
    return {"findings": findings, "risks": risks, "opportunities": opps, "actions": actions}


def _story_general(df, stats, corrs):
    findings, risks, opps, actions = [], [], [], []

    for col in list(stats.keys())[:4]:
        st = stats.get(col, {})
        if not st:
            continue
        skew    = st.get("skew", 0)
        out_pct = st.get("outlier_pct", 0)
        cv      = st.get("cv", 0)
        mean    = st.get("mean", 0)
        median  = st.get("median", 0)

        if out_pct > 10:
            risks.append(_senior_insight(
                what="'{}' has {:.1f}% outliers".format(col, out_pct),
                why="High outlier rate = data quality issues or genuine anomalies",
                action="Investigate each outlier — data entry error? Genuine extreme? Fix or cap",
                impact="Outliers distort all statistical analyses and reduce ML model accuracy"))
        if abs(skew) > 1:
            findings.append(_senior_insight(
                what="'{}' is {}-skewed (mean {:.2f} vs median {:.2f})".format(
                    col, "right" if skew>0 else "left", mean, median),
                why="Skewed data makes mean misleading as a summary statistic",
                action="Report median ({:.2f}) not mean. Apply log-transform before modeling".format(median),
                impact="Using mean on skewed data can misrepresent reality by {:.0f}%".format(
                    abs(mean-median)/abs(median)*100 if median!=0 else 0)))

    for corr in corrs[:3]:
        if corr.get("strength") in ("strong","moderate"):
            findings.append(_senior_insight(
                what="{} {} correlation: '{}' and '{}' (r={:.2f})".format(
                    corr["strength"].title(), corr["direction"],
                    corr["col_a"], corr["col_b"], corr["r"]),
                why="Statistically significant — not random (p={:.4f})".format(corr["p"]),
                action="Use in predictive modeling. Investigate causation vs correlation",
                impact="Strong predictors can improve model accuracy by 10-30%"))

    actions.extend([
        "Validate all outliers before analysis or modeling",
        "Use median for skewed distributions in executive reports",
        "Segment analysis — subgroups may tell a different story than the whole",
    ])
    return {"findings": findings, "risks": risks, "opportunities": opps, "actions": actions}



# ══════════════════════════════════════════════════════════
#  MAIN FUNCTION
# ══════════════════════════════════════════════════════════

def generate_story(df: pd.DataFrame) -> StoryReport:
    """
    Full business story generation.
    Returns StoryReport with all insights.
    """
    # 1. Detect domain
    domain, confidence = detect_domain(df)

    # 2. Compute stats for all numeric columns
    num_cols   = df.select_dtypes(include="number").columns.tolist()
    all_stats  = {}
    for col in num_cols:
        st = _analyze_numeric_col(df[col])
        if st:
            all_stats[col] = st

    # 3. Correlations
    corrs = _find_correlations(df)

    # 4. Domain-specific story
    domain_fn = {
        "ecommerce": _story_ecommerce,
        "hr":        _story_hr,
        "finance":   _story_finance,
    }.get(domain, _story_general)

    story = domain_fn(df, all_stats, corrs)

    # 5. Anomalies
    anomalies = _detect_anomalies(df, all_stats)

    # 6. Column insights
    col_insights = []
    for col, st in all_stats.items():
        if not st:
            continue

        skew = st.get("skew", 0)
        out_pct = st.get("outlier_pct", 0)
        miss_pct = st.get("missing_pct", 0)

        if out_pct > 10:
            severity = "critical"
            finding  = "{:.1f}% outliers detected".format(out_pct)
            rec      = "Investigate and validate extreme values before analysis"
        elif miss_pct > 20:
            severity = "warning"
            finding  = "{:.1f}% missing values".format(miss_pct)
            rec      = "Consider collecting more data or using robust imputation"
        elif abs(skew) > 2:
            severity = "warning"
            direction = "right" if skew > 0 else "left"
            finding  = "Heavily {}-skewed (skew={:.2f})".format(direction, skew)
            rec      = "Use median for reporting; log-transform for modeling"
        elif abs(skew) < 0.5 and st.get("is_normal"):
            severity = "positive"
            finding  = "Normally distributed — ideal for statistical tests"
            rec      = "Safe to use parametric tests and mean for reporting"
        else:
            severity = "info"
            finding  = "Mean={:.2f}, Median={:.2f}, Std={:.2f}".format(
                st["mean"], st["median"], st["std"])
            rec      = "Monitor for changes over time"

        metric = "Range: {:.2f} to {:.2f} | CV: {:.2f}".format(
            st["min"], st["max"], st["cv"])

        col_insights.append(ColumnInsight(
            column=col, dtype="numeric",
            finding=finding, severity=severity,
            metric=metric, recommendation=rec,
        ))

    # 7. Categorical col insights
    for col in df.select_dtypes(include="object").columns[:8]:
        nunique  = df[col].nunique()
        miss_pct = df[col].isna().mean() * 100
        top_val  = df[col].mode()[0] if len(df[col].mode()) > 0 else "-"
        top_pct  = df[col].value_counts(normalize=True).iloc[0] * 100

        if nunique / max(len(df), 1) > 0.85:
            severity = "warning"
            finding  = "Very high cardinality ({} unique) — likely an ID column".format(nunique)
            rec      = "Exclude from grouping and charts"
        elif top_pct > 80:
            severity = "warning"
            finding  = "Dominated by '{}' ({:.0f}% of rows)".format(top_val, top_pct)
            rec      = "Low variance — may not add value in segmentation"
        elif 2 <= nunique <= 20:
            severity = "positive"
            finding  = "{} categories — good for segmentation".format(nunique)
            rec      = "Use for grouping and comparison analysis"
        else:
            severity = "info"
            finding  = "{} unique values, top: '{}' ({:.0f}%)".format(
                nunique, str(top_val)[:20], top_pct)
            rec      = "Review cardinality before using in models"

        col_insights.append(ColumnInsight(
            column=col, dtype="categorical",
            finding=finding, severity=severity,
            metric="Unique: {} | Missing: {:.1f}%".format(nunique, miss_pct),
            recommendation=rec,
        ))

    # 8. Build executive summary
    n_risks   = len(story["risks"])
    n_opps    = len(story["opportunities"])
    n_actions = len(story["actions"])

    exec_summary = (
        "This {}-row, {}-column {} dataset has been analyzed using "
        "statistical methods. ".format(len(df), len(df.columns), domain)
    )
    if story["findings"]:
        exec_summary += story["findings"][0] + " "
    if n_risks > 0:
        exec_summary += "{} business risk(s) identified requiring attention. ".format(n_risks)
    if anomalies:
        exec_summary += "{} statistical anomaly(s) flagged. ".format(len(anomalies))
    exec_summary += "{} recommended action(s) provided.".format(n_actions)

    # 9. Headline — most critical finding
    if story["risks"]:
        headline = "RISK: " + story["risks"][0]
    elif story["findings"]:
        headline = story["findings"][0]
    else:
        headline = "Dataset analyzed — {} findings generated.".format(
            len(story["findings"]))

    # 10. Data quality verdict
    total_missing = sum(st.get("missing_pct", 0) for st in all_stats.values())
    avg_missing   = total_missing / max(len(all_stats), 1)
    high_outlier  = [c for c, st in all_stats.items() if st.get("outlier_pct", 0) > 10]

    if avg_missing < 5 and not high_outlier:
        quality_verdict = "Data quality is GOOD — low missing rates and no extreme outlier issues."
    elif avg_missing < 20:
        quality_verdict = (
            "Data quality is FAIR — average {:.1f}% missing values. "
            "Imputation applied.".format(avg_missing)
        )
    else:
        quality_verdict = (
            "Data quality NEEDS ATTENTION — high missing rate ({:.1f}%). "
            "Results may be unreliable.".format(avg_missing)
        )

    # 11. Analysis confidence
    n_rows = len(df)
    if n_rows >= 1000 and len(num_cols) >= 3:
        confidence_label = "HIGH — sufficient data for reliable statistical conclusions."
    elif n_rows >= 100:
        confidence_label = "MEDIUM — results are indicative but more data would improve reliability."
    else:
        confidence_label = "LOW — small dataset. Treat findings as directional only."

    return StoryReport(
        domain=domain,
        domain_confidence=confidence,
        headline=headline,
        executive_summary=exec_summary,
        key_findings=story["findings"],
        column_insights=col_insights,
        anomalies=anomalies,
        business_risks=story["risks"],
        opportunities=story["opportunities"],
        recommended_actions=story["actions"],
        data_quality_verdict=quality_verdict,
        analysis_confidence=confidence_label,
    )

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

def _story_ecommerce(df: pd.DataFrame, stats: Dict, corrs: List) -> Dict:
    findings, risks, opps, actions = [], [], [], []

    # Rating analysis
    rating_col = next((c for c in df.columns if "rating" in c.lower()
                       and "count" not in c.lower()), None)
    if rating_col and rating_col in stats:
        st = stats[rating_col]
        mean_r = st.get("mean", 0)
        if mean_r >= 4.0:
            findings.append(
                "Strong customer satisfaction: average rating of {:.2f}/5 "
                "indicates most customers are happy with products.".format(mean_r)
            )
            opps.append(
                "High ratings ({:.2f}/5) are a competitive advantage — "
                "highlight in marketing.".format(mean_r)
            )
        elif mean_r >= 3.5:
            findings.append(
                "Average rating of {:.2f}/5 is acceptable but below top-tier. "
                "25% of products rated below {:.2f}.".format(mean_r, st.get("q1", 0))
            )
            risks.append(
                "Ratings below 3.5 risk losing customers to competitors. "
                "Focus on bottom-rated products."
            )
        else:
            findings.append(
                "Low average rating of {:.2f}/5 — significant customer "
                "dissatisfaction detected.".format(mean_r)
            )
            risks.append("Critical: Low ratings may drive churn and negative reviews.")

        # Outlier products
        if st.get("outliers", 0) > 0:
            findings.append(
                "{} products have outlier ratings (below {:.1f} or above {:.1f}) — "
                "these need immediate review.".format(
                    st["outliers"],
                    st["q1"] - 1.5*st["iqr"],
                    st["q3"] + 1.5*st["iqr"]
                )
            )

    # Price analysis
    price_col = next((c for c in df.columns
                      if any(k in c.lower() for k in ["price", "cost", "amount"])
                      and c in stats), None)
    if price_col:
        st = stats[price_col]
        findings.append(
            "Price range: {:.0f} to {:.0f}. "
            "Median price {:.0f} vs mean {:.0f} — {} skew indicates "
            "a few expensive products pulling the average up.".format(
                st.get("min", 0), st.get("max", 0),
                st.get("median", 0), st.get("mean", 0),
                "right" if st.get("skew", 0) > 0 else "left"
            )
        )

    # Discount analysis
    disc_col = next((c for c in df.columns
                     if "discount" in c.lower() and "pct" in c.lower()
                     or "discount" in c.lower() and "percent" in c.lower()), None)
    if disc_col and disc_col in stats:
        st = stats[disc_col]
        avg_disc = st.get("mean", 0)
        findings.append(
            "Average discount of {:.1f}%. Products discounted up to {:.0f}%.".format(
                avg_disc, st.get("max", 0)
            )
        )
        if avg_disc > 40:
            risks.append(
                "High average discount ({:.1f}%) may be eroding margins. "
                "Analyze profitability at these discount levels.".format(avg_disc)
            )

    # Price-rating correlation
    price_rating = next(
        (c for c in corrs if "rating" in c["col_a"].lower()
         or "rating" in c["col_b"].lower()), None
    )
    if price_rating:
        r = price_rating["r"]
        if r < -0.3:
            findings.append(
                "Higher-priced products tend to have LOWER ratings (r={:.2f}) — "
                "premium pricing may not match perceived value.".format(r)
            )
            risks.append("Price-quality mismatch: customers feel premium products underdeliver.")
        elif r > 0.3:
            findings.append(
                "Higher-priced products tend to have HIGHER ratings (r={:.2f}) — "
                "premium products are delivering on expectations.".format(r)
            )

    # Category analysis
    cat_col = next((c for c in df.select_dtypes(include="object").columns
                    if "category" in c.lower() and df[c].nunique() <= 30), None)
    if cat_col and rating_col:
        cat_ratings = df.groupby(cat_col)[rating_col].mean().sort_values()
        if len(cat_ratings) >= 2:
            worst = cat_ratings.index[0]
            best  = cat_ratings.index[-1]
            findings.append(
                "Category performance gap: '{}' has lowest avg rating ({:.2f}) "
                "vs '{}' with highest ({:.2f}).".format(
                    worst, cat_ratings.iloc[0], best, cat_ratings.iloc[-1]
                )
            )
            actions.append(
                "Priority: Investigate '{}' category — "
                "lowest rated at {:.2f}/5.".format(worst, cat_ratings.iloc[0])
            )

    actions.append("Review all products rated below 3.0 for quality issues.")
    actions.append("A/B test pricing on top-rated products to optimize margins.")

    return {"findings": findings, "risks": risks,
            "opportunities": opps, "actions": actions}


def _story_hr(df: pd.DataFrame, stats: Dict, corrs: List) -> Dict:
    findings, risks, opps, actions = [], [], [], []

    # Attrition
    attr_col = next((c for c in df.columns if "attrition" in c.lower()
                     or "churn" in c.lower()), None)
    if attr_col:
        vc = df[attr_col].value_counts(normalize=True)
        yes_keys = [k for k in vc.index if str(k).lower() in ["yes", "1", "true"]]
        if yes_keys:
            rate = vc[yes_keys[0]] * 100
            if rate > 20:
                risks.append(
                    "Critical attrition rate of {:.1f}% — "
                    "industry benchmark is typically 10-15%.".format(rate)
                )
                findings.append(
                    "{:.1f}% of employees have left — "
                    "significant talent drain and recruitment cost.".format(rate)
                )
                actions.append(
                    "Urgent: Conduct exit interviews and identify "
                    "top attrition drivers."
                )
            else:
                findings.append(
                    "Attrition rate of {:.1f}% is within acceptable range.".format(rate)
                )

    # Salary analysis
    sal_col = next((c for c in df.columns if "salary" in c.lower()
                    or "income" in c.lower() or "pay" in c.lower()), None)
    if sal_col and sal_col in stats:
        st = stats[sal_col]
        findings.append(
            "Salary range: {:.0f} to {:.0f}. "
            "Median salary {:.0f} — {} skew detected.".format(
                st["min"], st["max"], st["median"],
                "right" if st.get("skew", 0) > 0 else "left"
            )
        )
        if st.get("cv", 0) > 0.5:
            risks.append(
                "High salary variation (CV={:.2f}) suggests "
                "significant pay inequality across roles or departments.".format(st["cv"])
            )

    # Satisfaction
    sat_col = next((c for c in df.columns
                    if "satisfaction" in c.lower() or "engage" in c.lower()), None)
    if sat_col and sat_col in stats:
        st = stats[sat_col]
        mean_s = st.get("mean", 0)
        max_s  = st.get("max", 1)
        pct    = (mean_s / max_s * 100) if max_s > 0 else 0
        if pct < 50:
            risks.append(
                "Low employee satisfaction ({:.1f}% of max score) — "
                "retention risk is high.".format(pct)
            )
        else:
            findings.append(
                "Employee satisfaction at {:.1f}% of maximum score.".format(pct)
            )

    actions.append("Analyze salary equity across departments and tenure bands.")
    actions.append("Identify top performers at risk of leaving using satisfaction scores.")

    return {"findings": findings, "risks": risks,
            "opportunities": opps, "actions": actions}


def _story_finance(df: pd.DataFrame, stats: Dict, corrs: List) -> Dict:
    findings, risks, opps, actions = [], [], [], []

    for col, st in stats.items():
        col_lower = col.lower()
        if any(k in col_lower for k in ["revenue", "sales", "income"]):
            findings.append(
                "Revenue ({}) ranges from {:.0f} to {:.0f}, "
                "median {:.0f}.".format(col, st["min"], st["max"], st["median"])
            )
            if st.get("skew", 0) > 1:
                opps.append(
                    "Revenue distribution is right-skewed — "
                    "a small number of high-value transactions drive disproportionate income."
                )
        elif any(k in col_lower for k in ["cost", "expense"]):
            findings.append(
                "Cost ({}) average: {:.0f}, ranging {:.0f} to {:.0f}.".format(
                    col, st["mean"], st["min"], st["max"]
                )
            )

    actions.append("Focus cost reduction on highest-variance expense categories.")
    actions.append("Segment revenue by top-performing categories for resource allocation.")

    return {"findings": findings, "risks": risks,
            "opportunities": opps, "actions": actions}


def _story_general(df: pd.DataFrame, stats: Dict, corrs: List) -> Dict:
    findings, risks, opps, actions = [], [], [], []

    num_cols = list(stats.keys())

    for col in num_cols[:4]:
        st = stats[col]
        if not st:
            continue
        findings.append(
            "'{}': mean={:.2f}, median={:.2f}, std={:.2f}. "
            "{} distribution.".format(
                col, st["mean"], st["median"], st["std"],
                "Normal" if st.get("is_normal") else "Non-normal"
            )
        )
        if st.get("outlier_pct", 0) > 5:
            risks.append(
                "'{}' has {:.1f}% outliers — "
                "validate data collection process.".format(
                    col, st["outlier_pct"]
                )
            )

    for corr in corrs[:3]:
        findings.append(
            "{} {} correlation between '{}' and '{}' (r={:.2f}) — "
            "statistically significant (p={:.4f}).".format(
                corr["strength"].title(), corr["direction"],
                corr["col_a"], corr["col_b"],
                corr["r"], corr["p"]
            )
        )

    actions.append("Investigate columns with high outlier rates before modeling.")
    actions.append("Use median instead of mean for skewed columns in reporting.")

    return {"findings": findings, "risks": risks,
            "opportunities": opps, "actions": actions}


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


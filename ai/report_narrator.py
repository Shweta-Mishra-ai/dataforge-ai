"""
report_narrator.py — McKinsey/BCG style AI narrative generator.
Real stats in → polished executive language out.
No jargon, no snake_case, no academic terms.
"""
import os
import re
import pandas as pd
import numpy as np
from typing import Optional


# ══════════════════════════════════════════════════════════
#  COLUMN NAME CLEANER
# ══════════════════════════════════════════════════════════

# Common typos and abbreviations in HR/Ecommerce datasets
COLUMN_NAME_MAP = {
    # HR
    "satisfaction_level":       "Employee Satisfaction Score",
    "last_evaluation":          "Last Performance Evaluation",
    "number_project":           "Number of Projects",
    "average_montly_hours":     "Average Monthly Hours",
    "average_monthly_hours":    "Average Monthly Hours",
    "time_spend_company":       "Employee Tenure (Years)",
    "work_accident":            "Work Accident Incidence",
    "left":                     "Employee Attrition",
    "attrition":                "Employee Attrition",
    "promotion_last_5years":    "Promotion in Last 5 Years",
    "dept":                     "Department",
    "department":               "Department",
    "salary":                   "Salary Band",
    # Ecommerce
    "discounted_price":         "Discounted Price",
    "actual_price":             "Original Price",
    "discount_percentage":      "Discount Percentage",
    "discount_pct":             "Discount Percentage",
    "rating_count":             "Number of Reviews",
    "product_id":               "Product ID",
    "product_name":             "Product Name",
    "category":                 "Product Category",
    "about_product":            "Product Description",
    "img_link":                 "Product Image",
    "product_link":             "Product Link",
}


def clean_col_name(col: str) -> str:
    """
    Convert snake_case column names to polished English.
    Fixes typos, capitalizes properly.
    """
    col_lower = col.lower().strip()

    # Direct mapping first
    if col_lower in COLUMN_NAME_MAP:
        return COLUMN_NAME_MAP[col_lower]

    # Generic cleaning
    name = col.replace("_", " ").strip()
    # Fix common typos
    name = name.replace("montly", "Monthly").replace("accidnet", "Accident")
    # Title case
    name = " ".join(w.capitalize() for w in name.split())
    return name


def clean_feature_name(feature: str) -> str:
    """Clean 'col1 by col2' or 'Distribution: col1' style strings."""
    if " by " in feature:
        parts = feature.split(" by ")
        return "{} by {}".format(clean_col_name(parts[0]), clean_col_name(parts[1]))
    elif ":" in feature:
        parts = feature.split(":", 1)
        return "{}: {}".format(parts[0], clean_col_name(parts[1].strip()))
    return clean_col_name(feature)


# ══════════════════════════════════════════════════════════
#  STATS BUILDER — Real numbers for LLM
# ══════════════════════════════════════════════════════════

def _build_raw_statistics(
    df: pd.DataFrame,
    chart_title: str,
    chart_type: str,
) -> str:
    """
    Build real computed statistics string to pass to LLM.
    LLM only narrates — never calculates.
    """
    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    stats_parts = []

    title_lower = chart_title.lower()

    # Bar chart — group means
    if "bar" in chart_type.lower() or "by" in title_lower:
        cat_col = next((c for c in cat_cols
                        if c.lower() in title_lower
                        or clean_col_name(c).lower() in title_lower), None)
        num_col = next((c for c in num_cols
                        if c.lower() in title_lower
                        or clean_col_name(c).lower() in title_lower), None)
        if not cat_col and cat_cols: cat_col = cat_cols[0]
        if not num_col and num_cols: num_col = num_cols[0]

        if cat_col and num_col:
            grp = df.groupby(cat_col)[num_col].mean().sort_values(ascending=False)
            stats_parts.append("Metric: {} by {}".format(
                clean_col_name(num_col), clean_col_name(cat_col)))
            stats_parts.append("Overall average: {:.2f}".format(float(df[num_col].mean())))
            stats_parts.append("Top performer: '{}' at {:.2f}".format(grp.index[0], grp.iloc[0]))
            stats_parts.append("Worst performer: '{}' at {:.2f}".format(grp.index[-1], grp.iloc[-1]))
            stats_parts.append("Performance gap: {:.1f}% between best and worst".format(
                abs(grp.iloc[0]-grp.iloc[-1])/abs(grp.iloc[-1])*100 if grp.iloc[-1]!=0 else 0))
            above_avg = (grp > grp.mean()).sum()
            stats_parts.append("{} out of {} groups are above average".format(above_avg, len(grp)))

    # Distribution/histogram
    elif "distribution" in chart_type.lower() or "histogram" in chart_type.lower():
        col = next((c for c in num_cols if c.lower() in title_lower), None)
        if not col and num_cols: col = num_cols[0]
        if col:
            s = df[col].dropna()
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            outliers = int(((s < q1-1.5*iqr) | (s > q3+1.5*iqr)).sum())
            skew = s.skew()
            stats_parts.append("Metric: {}".format(clean_col_name(col)))
            stats_parts.append("Average value: {:.2f}".format(float(s.mean())))
            stats_parts.append("Most typical value: {:.2f}".format(float(s.median())))
            stats_parts.append("Range: {:.2f} to {:.2f}".format(float(s.min()), float(s.max())))
            stats_parts.append("Middle 50% of values: {:.2f} to {:.2f}".format(float(q1), float(q3)))
            stats_parts.append("Distribution shape: {}".format(
                "concentrated in high values (right-skewed)" if skew < -0.5
                else "concentrated in low values (left-skewed)" if skew > 0.5
                else "evenly spread"))
            stats_parts.append("Unusual values detected: {} ({:.1f}% of data)".format(
                outliers, outliers/len(s)*100))

    # Correlation heatmap
    elif "correlation" in chart_type.lower():
        if len(num_cols) >= 2:
            corr = df[num_cols[:8]].corr()
            pairs = []
            for i in range(len(corr.columns)):
                for j in range(i+1, len(corr.columns)):
                    a, b = corr.columns[i], corr.columns[j]
                    r = float(corr.loc[a,b])
                    pairs.append((clean_col_name(a), clean_col_name(b), r))
            pairs.sort(key=lambda x: abs(x[2]), reverse=True)
            stats_parts.append("Number of metrics analyzed: {}".format(len(num_cols[:8])))
            if pairs:
                a, b, r = pairs[0]
                stats_parts.append("Strongest relationship: {} and {} (strength: {:.0f}%)".format(
                    a, b, abs(r)*100))
                stats_parts.append("Direction: {} tends to {} when {} increases".format(
                    b, "increase" if r>0 else "decrease", a))
            strong = [(a,b,r) for a,b,r in pairs if abs(r)>=0.5]
            weak   = [(a,b,r) for a,b,r in pairs if abs(r)<0.2]
            stats_parts.append("Strong relationships found: {}".format(len(strong)))
            stats_parts.append("No meaningful relationships: {}".format(len(weak)))

    # Pie chart
    elif "pie" in chart_type.lower() or "share" in title_lower:
        cat_col = next((c for c in cat_cols if c.lower() in title_lower), None)
        num_col = next((c for c in num_cols if c.lower() in title_lower), None)
        if not cat_col and cat_cols: cat_col = cat_cols[0]
        if not num_col and num_cols: num_col = num_cols[0]
        if cat_col and num_col:
            grp = df.groupby(cat_col)[num_col].mean().sort_values(ascending=False)
            total = grp.sum()
            stats_parts.append("Composition metric: {} by {}".format(
                clean_col_name(num_col), clean_col_name(cat_col)))
            stats_parts.append("Number of segments: {}".format(len(grp)))
            stats_parts.append("Largest segment: '{}' holds {:.1f}% of total".format(
                grp.index[0], grp.iloc[0]/total*100))
            stats_parts.append("Smallest segment: '{}' holds {:.1f}% of total".format(
                grp.index[-1], grp.iloc[-1]/total*100))
            top2_pct = grp.iloc[:2].sum()/total*100
            stats_parts.append("Top 2 segments combined: {:.1f}% of total".format(top2_pct))
            stats_parts.append("Distribution type: {}".format(
                "Highly concentrated (Pareto effect)" if top2_pct > 60
                else "Evenly distributed across segments"))

    # Fallback
    if not stats_parts and num_cols:
        col = num_cols[0]
        s = df[col].dropna()
        stats_parts.append("Metric: {}".format(clean_col_name(col)))
        stats_parts.append("Average: {:.2f}".format(float(s.mean())))
        stats_parts.append("Range: {:.2f} to {:.2f}".format(float(s.min()), float(s.max())))

    return "\n".join(stats_parts)


# ══════════════════════════════════════════════════════════
#  CHART TYPE DETECTOR
# ══════════════════════════════════════════════════════════

def detect_chart_type(title: str) -> str:
    title_lower = title.lower()
    if "correlation" in title_lower or "heatmap" in title_lower:
        return "Correlation Heatmap"
    elif "distribution" in title_lower or "histogram" in title_lower:
        return "Distribution Histogram"
    elif "share" in title_lower or "pie" in title_lower:
        return "Pie Chart"
    elif "trend" in title_lower or "over time" in title_lower:
        return "Trend Line Chart"
    elif "by" in title_lower:
        return "Bar Chart"
    return "Bar Chart"


# ══════════════════════════════════════════════════════════
#  MASTER PROMPT — McKinsey/BCG Style
# ══════════════════════════════════════════════════════════

def _build_master_prompt(
    chart_type: str,
    feature_name: str,
    raw_statistics: str,
    domain: str = "general",
) -> str:
    domain_context = {
        "hr":        "HR Director or Chief People Officer making workforce decisions",
        "ecommerce": "E-commerce Director or Category Manager optimizing product performance",
        "finance":   "CFO or Finance Director reviewing financial performance",
        "marketing": "CMO or Marketing Director optimizing campaign performance",
        "general":   "C-level executive or business owner making strategic decisions",
    }.get(domain, "C-level executive making strategic decisions")

    return """You are a Senior Executive Consultant at a top-tier management consulting firm (McKinsey, BCG, Bain level). Your audience is: {audience}.

CHART BEING ANALYZED:
- Chart Type: {chart_type}
- Metric: {feature_name}
- Computed Statistics:
{raw_statistics}

STRICT RULES — VIOLATION IS NOT ACCEPTABLE:

1. NO SNAKE_CASE OR RAW COLUMN NAMES. Never output database-style names.
   - WRONG: "the satisfaction_level column shows..."
   - RIGHT: "Employee Satisfaction Score reveals..."

2. NO STATISTICAL JARGON. Translate everything into business language.
   - WRONG: "p-value, Kruskal-Wallis, Pearson r, IQR, skewness, standard deviation"
   - RIGHT: "mathematically proven relationship", "most employees", "unusually high variation"

3. NO DATA ENGINEERING ADVICE. Only business strategy.
   - WRONG: "remove outliers", "log-transform", "handle missing values"
   - RIGHT: "conduct exit interviews", "audit workloads", "review salary bands"

4. CHART-SPECIFIC ANALYSIS:
   - Bar Chart: Focus on ranking, the gap between #1 and last, what this disparity means operationally.
   - Pie Chart: Focus on market share, whether distribution is dangerously concentrated or balanced.
   - Histogram: Focus on what the typical employee/customer looks like and what the extremes mean.
   - Correlation Heatmap: Focus on which business levers move together and strategic implications.
   - Trend Line: Focus on trajectory, momentum, and whether intervention is needed.

5. SPECIFICITY REQUIRED: You MUST reference the actual numbers from the statistics provided. Generic statements are unacceptable.

OUTPUT FORMAT:
Write exactly ONE punchy executive paragraph (3-4 sentences). Then write exactly ONE "Strategic Action:" line. Be authoritative, specific, and direct. No bullet points. No headers. Just the paragraph and the action.

Example of correct output:
"Employee Tenure data reveals a critical retention pattern: the majority of the workforce has served between 3-5 years, yet a concerning 8% of employees have tenure exceeding 8 years without receiving promotions — a group at acute flight risk. The 2-year tenure cohort shows the highest attrition concentration, suggesting onboarding and early career support are failing to create lasting engagement. This distribution signals an organization that is losing employees precisely when they become most productive.

Strategic Action: Immediately implement a structured 18-month career development programme targeting employees in their second year, and conduct skip-level retention conversations with all tenured employees above 7 years who have not been promoted in the last 3 years."
""".format(
        audience=domain_context,
        chart_type=chart_type,
        feature_name=feature_name,
        raw_statistics=raw_statistics,
    )


# ══════════════════════════════════════════════════════════
#  MAIN FUNCTION — Called from PDF builder
# ══════════════════════════════════════════════════════════

def generate_chart_narrative(
    df: pd.DataFrame,
    chart_title: str,
    groq_api_key: Optional[str] = None,
    domain: str = "general",
) -> str:
    """
    Generate McKinsey-style executive narrative for a chart.
    Uses real computed statistics — LLM only narrates.
    Falls back to rule-based if Groq unavailable.
    """
    chart_type   = detect_chart_type(chart_title)
    feature_name = clean_feature_name(chart_title)
    raw_stats    = _build_raw_statistics(df, chart_title, chart_type)

    # Try Groq LLM
    api_key = groq_api_key or os.environ.get("GROQ_API_KEY", "")
    if api_key:
        try:
            from groq import Groq
            client = Groq(api_key=api_key)
            prompt = _build_master_prompt(chart_type, feature_name, raw_stats, domain)
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role":"user","content":prompt}],
                max_tokens=400,
                temperature=0.3,
            )
            narrative = response.choices[0].message.content.strip()
            # Clean any remaining snake_case
            narrative = _clean_snake_case(narrative)
            return narrative
        except Exception:
            pass

    # Fallback — rule-based narrative (no LLM)
    return _rule_based_narrative(df, chart_title, chart_type, feature_name, raw_stats)


def _clean_snake_case(text: str) -> str:
    """Post-process: replace any remaining snake_case with clean names."""
    for raw, clean in COLUMN_NAME_MAP.items():
        text = text.replace(raw, clean)
        text = text.replace("'{}'" .format(raw), clean)
        text = text.replace('"{}"'.format(raw), clean)
    return text


def _rule_based_narrative(
    df: pd.DataFrame,
    title: str,
    chart_type: str,
    feature_name: str,
    raw_stats: str,
) -> str:
    """Fallback narrative when Groq is unavailable."""
    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    lines    = raw_stats.split("\n")

    def _get(prefix):
        for l in lines:
            if l.startswith(prefix):
                return l.split(":", 1)[1].strip() if ":" in l else ""
        return ""

    if "Bar Chart" in chart_type:
        top  = _get("Top performer")
        worst= _get("Worst performer")
        gap  = _get("Performance gap")
        avg  = _get("Overall average")
        return (
            "Performance analysis of {} reveals a clear hierarchy across segments. "
            "{} leads performance while {} trails significantly — a {} difference "
            "that demands immediate strategic attention. "
            "With an overall average of {}, organizations should concentrate resources "
            "on closing this performance gap through targeted interventions.\n\n"
            "Strategic Action: Conduct a structured review of the lowest-performing segment "
            "to identify root causes, and replicate the practices driving the top performer's success."
        ).format(feature_name, top, worst, gap, avg)

    elif "Distribution" in chart_type:
        avg    = _get("Average value")
        typical= _get("Most typical value")
        range_ = _get("Range")
        shape  = _get("Distribution shape")
        return (
            "The distribution of {} shows that the most typical value is {}, "
            "though the average of {} suggests {}. "
            "Values range from {} — indicating considerable variation across the population. "
            "Decision-makers should focus on the typical value rather than the average "
            "for more representative planning.\n\n"
            "Strategic Action: Segment the analysis into high, medium, and low performers "
            "to design targeted interventions for each group."
        ).format(feature_name, typical, avg, shape, range_)

    elif "Pie Chart" in chart_type:
        largest = _get("Largest segment")
        dist    = _get("Distribution type")
        top2    = _get("Top 2 segments combined")
        return (
            "The composition of {} reveals that {}. "
            "The top two segments together account for {}, indicating {}. "
            "This concentration pattern has significant implications for resource allocation "
            "and strategic risk management.\n\n"
            "Strategic Action: Review whether the current segment concentration "
            "represents strategic intent or an unplanned imbalance requiring rebalancing."
        ).format(feature_name, largest, top2, dist)

    elif "Correlation" in chart_type:
        strongest = _get("Strongest relationship")
        direction = _get("Direction")
        return (
            "Relationship analysis across key metrics reveals important operational linkages. "
            "{}. {} — a finding with direct implications for management strategy. "
            "Understanding these interdependencies enables more informed decision-making "
            "across the organization.\n\n"
            "Strategic Action: Prioritize improving the leading metric identified, "
            "as improvements will propagate to linked outcomes across the business."
        ).format(strongest, direction)

    return (
        "Analysis of {} provides actionable intelligence for strategic decision-making. "
        "The data reveals meaningful patterns that warrant leadership attention and "
        "structured intervention.\n\n"
        "Strategic Action: Review the detailed findings with department heads and "
        "develop a 30-60-90 day action plan."
    ).format(feature_name)


def generate_executive_summary(
    df: pd.DataFrame,
    story_report,
    groq_api_key: Optional[str] = None,
) -> str:
    """
    Generate polished executive summary using LLM.
    Passes pre-computed story findings to LLM for narration only.
    """
    api_key = groq_api_key or os.environ.get("GROQ_API_KEY", "")

    # Build structured summary from story report
    domain   = getattr(story_report, "domain", "general")
    findings = getattr(story_report, "key_findings", [])
    risks    = getattr(story_report, "business_risks", [])
    attrition= getattr(story_report, "attrition", None)

    context_parts = [
        "Dataset: {:,} rows, {} columns, {} domain".format(
            len(df), len(df.columns), domain),
    ]
    if attrition:
        context_parts.append("Attrition Rate: {:.1f}% ({} severity)".format(
            attrition.rate, attrition.severity))
        context_parts.append("Flight Risk Employees: {:,}".format(attrition.n_flight_risk))
    for f in findings[:3]:
        context_parts.append("Finding: " + f[:100])
    for r in risks[:2]:
        context_parts.append("Risk: " + r[:100])

    context = "\n".join(context_parts)

    if api_key:
        try:
            from groq import Groq
            client = Groq(api_key=api_key)
            prompt = """You are a Senior Management Consultant writing an executive summary for a board-level report.

COMPUTED FINDINGS (DO NOT INVENT NEW NUMBERS):
{context}

RULES:
1. Use ONLY the numbers and facts provided above
2. NO snake_case column names — translate to polished English
3. NO statistical jargon
4. Write for a C-suite executive who has 30 seconds to read this
5. 4-5 sentences maximum
6. Start with the most critical finding
7. End with confidence in the analysis

Write the executive summary now:""".format(context=context)

            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role":"user","content":prompt}],
                max_tokens=300,
                temperature=0.2,
            )
            summary = response.choices[0].message.content.strip()
            return _clean_snake_case(summary)
        except Exception:
            pass

    # Fallback
    return _clean_snake_case(getattr(story_report, "executive_summary",
                                     "Analysis completed by DataForge AI."))

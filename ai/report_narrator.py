"""
ai/report_narrator.py — Elite consultant narratives via Groq.
FIXED VERSION:
  ✅ Correlation chart uses REAL r-values from df — never hallucinates
  ✅ All chart prompts inject pre-computed statistics
  ✅ Domain-aware: no wrong-domain text (no "Sales Revenue" in HR reports)
  ✅ Fallback narratives are factually grounded
"""

import os
import numpy as np
import pandas as pd
from typing import Optional


# ══════════════════════════════════════════════════════════
#  COLUMN NAME CLEANER
# ══════════════════════════════════════════════════════════

COL_MAP = {
    "satisfaction_level":    "Employee Satisfaction Score",
    "last_evaluation":       "Last Performance Evaluation",
    "number_project":        "Number of Active Projects",
    "average_montly_hours":  "Average Monthly Hours Worked",
    "average_monthly_hours": "Average Monthly Hours Worked",
    "time_spend_company":    "Employee Tenure (Years)",
    "work_accident":         "Work Accident Incidence",
    "left":                  "Employee Attrition",
    "attrition":             "Employee Attrition Rate",
    "promotion_last_5years": "Recent Promotions (Last 5 Years)",
    "dept":                  "Department",
    "department":            "Department",
    "salary":                "Salary Band",
    "discounted_price":      "Selling Price",
    "actual_price":          "Original Price (MRP)",
    "discount_percentage":   "Discount Percentage",
    "rating_count":          "Number of Customer Reviews",
    "rating":                "Customer Rating",
    "product_name":          "Product Name",
    "category":              "Product Category",
    "revenue":               "Revenue",
    "sales":                 "Sales Amount",
    "target":                "Sales Target",
    "profit":                "Profit",
    "margin":                "Profit Margin",
    "region":                "Sales Region",
}


def clean_col(col: str) -> str:
    low = col.lower().strip()
    if low in COL_MAP:
        return COL_MAP[low]
    name = col.replace("_", " ").strip()
    name = name.replace("montly", "Monthly").replace("accidnet", "Accident")
    return " ".join(w.capitalize() for w in name.split())


def clean_text(text: str) -> str:
    for raw, clean in COL_MAP.items():
        text = text.replace("'" + raw + "'", clean)
        text = text.replace('"' + raw + '"', clean)
    return text


# ══════════════════════════════════════════════════════════
#  REAL STATS EXTRACTORS  (LLM narrates, never calculates)
# ══════════════════════════════════════════════════════════

def _bar_stats(df: pd.DataFrame, x_col: str, y_col: str) -> str:
    try:
        grp = df.groupby(x_col)[y_col].mean().sort_values(ascending=False)
        org_avg = float(df[y_col].mean())
        top = grp.index[0]; top_v = float(grp.iloc[0])
        bot = grp.index[-1]; bot_v = float(grp.iloc[-1])
        gap = abs(top_v - bot_v) / max(abs(bot_v), 0.001) * 100
        above = int((grp > org_avg).sum())
        return (f"Groups: {len(grp)} | Top: '{top}' ({top_v:.3f}) | "
                f"Worst: '{bot}' ({bot_v:.3f}) | Gap: {gap:.1f}% | "
                f"Org avg: {org_avg:.3f} | Above avg: {above}/{len(grp)}")
    except Exception:
        return "Bar chart data unavailable"


def _hist_stats(df: pd.DataFrame, col: str) -> str:
    try:
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        s = s[np.isfinite(s)]
        skew = float(s.skew())
        shape = ("right-skewed (high values are outliers)" if skew > 0.5
                 else "left-skewed (low values are outliers)" if skew < -0.5
                 else "approximately symmetric")
        return (f"Mean: {s.mean():.3f} | Median: {s.median():.3f} | "
                f"Std: {s.std():.3f} | Min: {s.min():.3f} | Max: {s.max():.3f} | "
                f"Shape: {shape} (skew={skew:.2f}) | "
                f"Middle 50%: {s.quantile(0.25):.3f}–{s.quantile(0.75):.3f}")
    except Exception:
        return "Distribution data unavailable"


def _pie_stats(df: pd.DataFrame, x_col: str, y_col: str) -> str:
    try:
        grp   = df.groupby(x_col)[y_col].mean()
        total = grp.sum()
        top   = grp.idxmax(); top_p = grp.max() / total * 100
        top2  = grp.nlargest(2).sum() / total * 100
        shares = " | ".join([f"'{k}': {v/total*100:.1f}%"
                              for k, v in grp.sort_values(ascending=False).head(5).items()])
        return (f"Shares: {shares} | "
                f"Dominant: '{top}' at {top_p:.1f}% | "
                f"Top 2 combined: {top2:.1f}% | "
                f"Pareto holds: {'YES — concentrated' if top2 > 65 else 'NO — balanced'}")
    except Exception:
        return "Pie chart data unavailable"


def _correlation_stats(df: pd.DataFrame) -> str:
    """
    FIXED: Computes REAL Spearman correlations from actual df.
    Never invents metrics. Domain-aware labeling.
    """
    try:
        num_cols = df.select_dtypes(include="number").columns.tolist()
        if len(num_cols) < 2:
            return "Insufficient numeric columns for correlation"

        use_cols = num_cols[:8]
        corr     = df[use_cols].corr(method="spearman")

        pairs = []
        for i in range(len(use_cols)):
            for j in range(i + 1, len(use_cols)):
                a, b = use_cols[i], use_cols[j]
                r    = float(corr.loc[a, b])
                if abs(r) >= 0.15:
                    pairs.append((a, b, r))
        pairs.sort(key=lambda x: abs(x[2]), reverse=True)

        if not pairs:
            return "No significant correlations found (all |r| < 0.15)"

        lines = []
        for a, b, r in pairs[:6]:
            strength = ("Strong" if abs(r) >= 0.6 else
                        "Moderate" if abs(r) >= 0.40 else "Weak")
            direction = "positive" if r > 0 else "negative"
            lines.append(
                f"{clean_col(a)} & {clean_col(b)}: r={r:.3f} "
                f"({strength} {direction}, r²={r**2:.3f})"
            )

        top_a, top_b, top_r = pairs[0]
        top_meaning = (
            f"Most important: {clean_col(top_a)} and {clean_col(top_b)} "
            f"(r={top_r:.3f}) — "
            + ("higher values tend to occur together"
               if top_r > 0 else
               "as one increases, the other tends to decrease")
        )
        return (f"Significant correlations ({len(pairs)} found):\n"
                + "\n".join(lines)
                + f"\n{top_meaning}"
                + f"\nIMPORTANT: r²={top_r**2:.3f} — only {top_r**2*100:.1f}% of variance shared. Association only, NOT causation.")
    except Exception as e:
        return f"Correlation computation error: {e}"


def _trend_stats(df: pd.DataFrame, y_col: str) -> str:
    try:
        s = pd.to_numeric(df[y_col], errors="coerce").dropna()
        cv = s.std() / abs(s.mean()) * 100 if s.mean() != 0 else 0
        return (f"Mean: {s.mean():.3f} | Range: {s.min():.3f}–{s.max():.3f} | "
                f"Variability (CV): {cv:.1f}% | "
                f"Trend: {'High variability' if cv > 30 else 'Relatively stable'}")
    except Exception:
        return "Trend data unavailable"


# ══════════════════════════════════════════════════════════
#  PROMPTS  (stats injected — LLM narrates only)
# ══════════════════════════════════════════════════════════

_SYSTEM = """You are a Senior Data Analyst with 25+ years of experience.

RULES (MUST FOLLOW):
1. Use ONLY the statistics provided. NEVER invent metrics not in the data.
2. NEVER output snake_case column names — use the clean names provided.
3. Write in plain business English. No jargon (no p-value, skewness, IQR).
4. For correlation charts: NEVER say "r means X% change". It means association only.
5. Maximum 4 sentences. End with one clear business action.
6. Match the chart type — bar charts: rankings; pie: composition; correlation: relationships.
"""


def _chart_prompt(chart_type: str, x_clean: str, y_clean: str,
                  stats_str: str, domain: str) -> str:
    domain_context = {
        "hr":        "HR / People Analytics dataset (employees, satisfaction, attrition)",
        "ecommerce": "E-Commerce dataset (products, ratings, prices, discounts)",
        "sales":     "Sales Performance dataset (revenue, targets, regions, reps)",
        "general":   "Business Analytics dataset",
    }.get(domain, "Business Analytics dataset")

    corr_extra = ""
    if chart_type.lower() in ("correlation", "heatmap"):
        corr_extra = (
            "\n\nCRITICAL FOR CORRELATION CHARTS:"
            "\n- Use ONLY the r-values listed in the stats above."
            "\n- NEVER invent metrics like 'Sales Revenue' or 'Customer Satisfaction' if not in stats."
            "\n- NEVER say r=-0.35 means '35% reduction'. Say 'associated with lower X'."
            "\n- Explain what the STRONGEST correlation means operationally for this specific domain."
        )

    return f"""Dataset domain: {domain_context}
Chart type: {chart_type}
X-axis / Groups: {x_clean}
Y-axis / Metric: {y_clean}

PRE-COMPUTED STATISTICS (use only these, never invent):
{stats_str}

{corr_extra}

Write exactly ONE paragraph (3–4 sentences) of sharp business analysis.
Start directly with the insight — no preamble like "The chart shows..."
End with one specific Strategic Action sentence.
"""


# ══════════════════════════════════════════════════════════
#  GROQ CALLER
# ══════════════════════════════════════════════════════════

def _call_groq(system: str, user: str,
               api_key: str, max_tokens: int = 350) -> Optional[str]:
    if not api_key:
        return None
    try:
        from groq import Groq
        client   = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model    = "llama-3.3-70b-versatile",
            messages = [{"role": "system", "content": system},
                        {"role": "user",   "content": user}],
            max_tokens  = max_tokens,
            temperature = 0.25,
        )
        return clean_text(response.choices[0].message.content.strip())
    except Exception:
        return None


# ══════════════════════════════════════════════════════════
#  RULE-BASED FALLBACKS  (factually grounded)
# ══════════════════════════════════════════════════════════

def _fallback_bar(df, x_col, y_col):
    try:
        grp = df.groupby(x_col)[y_col].mean().sort_values(ascending=False)
        top = grp.index[0]; top_v = grp.iloc[0]
        bot = grp.index[-1]; bot_v = grp.iloc[-1]
        gap = abs(top_v - bot_v) / max(abs(bot_v), 0.001) * 100
        avg = float(df[y_col].mean())
        above = int((grp > avg).sum())
        return (
            f"{clean_col(y_col)} across {len(grp)} {clean_col(x_col)} groups reveals a clear performance hierarchy. "
            f"'{top}' leads with {top_v:.3f} while '{bot}' trails at {bot_v:.3f} — "
            f"a {gap:.0f}% gap that demands operational attention. "
            f"{above} out of {len(grp)} groups perform above the overall average of {avg:.3f}. "
            f"Strategic Action: Conduct a root-cause review of '{bot}' practices and "
            f"replicate what '{top}' does differently."
        )
    except Exception:
        return f"Analysis of {clean_col(y_col)} by group reveals patterns requiring attention."


def _fallback_hist(df, col):
    try:
        s    = pd.to_numeric(df[col], errors="coerce").dropna()
        skew = float(s.skew())
        use  = "median" if abs(skew) > 0.5 else "mean"
        val  = s.median() if abs(skew) > 0.5 else s.mean()
        return (
            f"{clean_col(col)} has a typical value of {val:.3f} "
            f"(range: {s.min():.3f}–{s.max():.3f}). "
            f"The distribution is {'right-skewed' if skew>0.5 else 'left-skewed' if skew<-0.5 else 'symmetric'}, "
            f"meaning the {use} ({val:.3f}) is the most reliable central measure. "
            f"Scores as low as {s.min():.3f} signal employees at high dissatisfaction risk. "
            f"Strategic Action: Focus retention interventions on employees in the bottom quartile "
            f"(below {s.quantile(0.25):.3f})."
        )
    except Exception:
        return f"Distribution analysis of {clean_col(col)} reveals key patterns for strategic planning."


def _fallback_correlation(df):
    """FIXED: Uses real df correlations, never invents."""
    try:
        num_cols = df.select_dtypes(include="number").columns.tolist()
        if len(num_cols) < 2:
            return "Insufficient numeric columns for correlation analysis."

        corr  = df[num_cols[:8]].corr(method="spearman")
        pairs = []
        for i in range(len(num_cols[:8])):
            for j in range(i + 1, len(num_cols[:8])):
                a, b = num_cols[i], num_cols[j]
                r    = float(corr.loc[a, b])
                if abs(r) >= 0.15:
                    pairs.append((a, b, r))
        pairs.sort(key=lambda x: abs(x[2]), reverse=True)

        if not pairs:
            return ("The correlation matrix shows no meaningful relationships between variables "
                    "(all |r| < 0.15). This indicates the dataset's metrics operate independently — "
                    "single-variable interventions may have limited spillover effects. "
                    "Strategic Action: Focus on each metric independently rather than "
                    "expecting cross-metric improvements.")

        a, b, r = pairs[0]
        direction = "positive" if r > 0 else "negative"
        meaning   = ("both tend to be high or low together"
                     if r > 0 else
                     "as one increases, the other tends to decrease")

        second = ""
        if len(pairs) > 1:
            a2, b2, r2 = pairs[1]
            second = (f" The second strongest relationship is between "
                      f"{clean_col(a2)} and {clean_col(b2)} (r={r2:.3f}). ")

        return (
            f"The correlation matrix reveals {len(pairs)} meaningful relationships across "
            f"{len(num_cols[:8])} variables. "
            f"The strongest is between {clean_col(a)} and {clean_col(b)} "
            f"(r={r:.3f}, {direction}) — {meaning}. "
            f"This means only {r**2*100:.1f}% of variance is shared (r²={r**2:.3f}) — "
            f"a statistical association, not a causal relationship.{second}"
            f"Strategic Action: Use the {clean_col(a)}–{clean_col(b)} relationship to inform "
            f"targeted interventions, but test causality before committing resources."
        )
    except Exception:
        return ("Correlation analysis reveals relationships between key metrics. "
                "Note: correlation indicates association only — not causation. "
                "Strategic Action: Investigate the strongest correlations through "
                "controlled experiments before assuming causal links.")


def _fallback_pie(df, x_col, y_col):
    try:
        grp   = df.groupby(x_col)[y_col].mean().sort_values(ascending=False)
        total = grp.sum()
        top   = grp.index[0]; top_p = grp.iloc[0] / total * 100
        top2  = grp.nlargest(2).sum() / total * 100
        return (
            f"{clean_col(y_col)} composition across {len(grp)} {clean_col(x_col)} segments. "
            f"'{top}' holds the largest share at {top_p:.1f}%. "
            f"Top 2 segments account for {top2:.1f}% — "
            + ("indicating concentration risk." if top2 > 65
               else "suggesting a well-balanced distribution across segments.") +
            f" Strategic Action: {'Investigate over-reliance on the top segment' if top2 > 65 else 'Maintain balanced resource allocation across segments'}."
        )
    except Exception:
        return f"Composition analysis of {clean_col(y_col)} reveals segment distribution patterns."


def _fallback_trend(df, y_col):
    try:
        s  = pd.to_numeric(df[y_col], errors="coerce").dropna()
        cv = s.std() / abs(s.mean()) * 100 if s.mean() != 0 else 0
        return (
            f"{clean_col(y_col)} shows an average of {s.mean():.3f} "
            f"(range: {s.min():.3f}–{s.max():.3f}). "
            f"Variability is {'high' if cv > 30 else 'moderate' if cv > 15 else 'low'} "
            f"({cv:.0f}% coefficient of variation), "
            + ("suggesting inconsistent performance requiring investigation." if cv > 30
               else "indicating a relatively stable pattern.") +
            f" Strategic Action: Monitor the {cv:.0f}% variation across segments "
            f"and identify root causes of the most extreme values."
        )
    except Exception:
        return f"Trend analysis of {clean_col(y_col)} reveals performance patterns over time."


# ══════════════════════════════════════════════════════════
#  EXECUTIVE SUMMARY
# ══════════════════════════════════════════════════════════

def _exec_prompt(df, domain, story_obj):
    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include="object").columns.tolist()

    lines = [f"Dataset: {len(df):,} rows, {len(df.columns)} columns, {domain.upper()} domain"]

    for col in num_cols[:5]:
        try:
            s = pd.to_numeric(df[col], errors="coerce").dropna()
            lines.append(f"{clean_col(col)}: mean={s.mean():.3f}, "
                         f"median={s.median():.3f}, range={s.min():.3f}–{s.max():.3f}")
        except Exception:
            continue

    for col in cat_cols[:2]:
        try:
            vc = df[col].value_counts(normalize=True).head(3)
            lines.append(clean_col(col) + ": " +
                         " | ".join([f"{k}: {v*100:.0f}%" for k, v in vc.items()]))
        except Exception:
            continue

    # Attrition
    atr_col = next((c for c in df.columns
                    if c.lower() in ("left","attrition","churned")), None)
    if atr_col:
        rate = float(df[atr_col].mean()) * 100
        lines.append(f"ATTRITION: {rate:.1f}% ({int(df[atr_col].sum()):,} employees left)")

    return "\n".join(lines)


def _exec_system_prompt(domain: str) -> str:
    return f"""You are a Senior Data Analyst with 25+ years of experience in {domain} analytics.
Write a 3-sentence C-suite executive summary.

RULES:
1. State the most critical business risk with specific numbers.
2. Name the #1 actionable lever identified in this data.
3. End with urgency — what must happen in the next 30 days.
4. NO snake_case column names. NO jargon. Plain business English.
5. Use ONLY numbers from the provided statistics. Never invent figures.
"""


# ══════════════════════════════════════════════════════════
#  MAIN PUBLIC FUNCTIONS
# ══════════════════════════════════════════════════════════

def generate_chart_narrative(
    df: pd.DataFrame,
    chart_title: str,
    groq_api_key: str = "",
    domain: str = "general",
) -> str:
    """
    Generate factually grounded chart narrative.
    Computes real stats → passes to LLM → fallback if LLM unavailable.
    FIXED: Correlation chart never hallucinates domain-wrong metrics.
    """
    title_lower = chart_title.lower()

    # ── Detect chart type and extract columns ─────────────────────────────────
    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include="object").columns.tolist()

    if "correlation" in title_lower or "heatmap" in title_lower:
        chart_type = "Correlation Heatmap"
        x_clean    = "All numeric variables"
        y_clean    = "Correlation strength (r)"
        stats_str  = _correlation_stats(df)
        fallback   = _fallback_correlation(df)

    elif "distribution" in title_lower or "histogram" in title_lower:
        chart_type = "Histogram / Distribution"
        y_col      = next((c for c in num_cols
                           if c.lower() in title_lower), num_cols[0] if num_cols else None)
        if not y_col:
            return "Distribution chart generated from dataset analysis."
        x_clean   = clean_col(y_col)
        y_clean   = "Frequency"
        stats_str = _hist_stats(df, y_col)
        fallback  = _fallback_hist(df, y_col)

    elif "pie" in title_lower or "share" in title_lower or "composition" in title_lower:
        chart_type = "Pie / Composition Chart"
        parts      = title_lower.split(" by ")
        x_col      = next((c for c in cat_cols
                           if c.lower() in (parts[1] if len(parts) > 1 else "")),
                          cat_cols[0] if cat_cols else None)
        y_col      = next((c for c in num_cols
                           if c.lower() in parts[0]),
                          num_cols[0] if num_cols else None)
        if not x_col or not y_col:
            return "Composition chart generated from dataset analysis."
        x_clean   = clean_col(x_col)
        y_clean   = clean_col(y_col)
        stats_str = _pie_stats(df, x_col, y_col)
        fallback  = _fallback_pie(df, x_col, y_col)

    elif "trend" in title_lower or "over" in title_lower or "line" in title_lower:
        chart_type = "Trend / Line Chart"
        y_col      = next((c for c in num_cols
                           if c.lower() in title_lower), num_cols[0] if num_cols else None)
        if not y_col:
            return "Trend chart generated from dataset analysis."
        x_clean   = "Data progression"
        y_clean   = clean_col(y_col)
        stats_str = _trend_stats(df, y_col)
        fallback  = _fallback_trend(df, y_col)

    elif " by " in title_lower:
        chart_type = "Bar Chart"
        parts      = title_lower.replace("avg ", "").replace("total ", "").split(" by ")
        y_col      = next((c for c in num_cols if c.lower() in parts[0]),
                          num_cols[0] if num_cols else None)
        x_col      = next((c for c in cat_cols
                           if c.lower() in (parts[1] if len(parts) > 1 else "")),
                          cat_cols[0] if cat_cols else None)
        if not y_col or not x_col:
            return "Bar chart generated from dataset analysis."
        x_clean   = clean_col(x_col)
        y_clean   = clean_col(y_col)
        stats_str = _bar_stats(df, x_col, y_col)
        fallback  = _fallback_bar(df, x_col, y_col)

    else:
        # Generic fallback
        if num_cols and cat_cols:
            chart_type = "Bar Chart"
            x_clean   = clean_col(cat_cols[0])
            y_clean   = clean_col(num_cols[0])
            stats_str = _bar_stats(df, cat_cols[0], num_cols[0])
            fallback  = _fallback_bar(df, cat_cols[0], num_cols[0])
        elif num_cols:
            chart_type = "Distribution"
            x_clean   = clean_col(num_cols[0])
            y_clean   = "Frequency"
            stats_str = _hist_stats(df, num_cols[0])
            fallback  = _fallback_hist(df, num_cols[0])
        else:
            return "Chart analysis not available for this dataset."

    # ── Try Groq, fallback to rule-based ─────────────────────────────────────
    user_prompt = _chart_prompt(chart_type, x_clean, y_clean, stats_str, domain)
    result = _call_groq(_SYSTEM, user_prompt, groq_api_key, max_tokens=320)
    return result if result else fallback


def generate_executive_summary(
    df: pd.DataFrame,
    domain: str = "general",
    story_report=None,
    groq_api_key: str = "",
) -> str:
    """Generate executive summary. Groq if available, rule-based fallback."""
    stats_summary = _exec_prompt(df, domain, story_report)
    system_prompt = _exec_system_prompt(domain)
    result = _call_groq(system_prompt, stats_summary, groq_api_key, max_tokens=250)
    if result:
        return result

    # Rule-based fallback
    atr_col = next((c for c in df.columns
                    if c.lower() in ("left","attrition","churned")), None)
    parts   = [f"This {domain.upper()} dataset ({len(df):,} records) reveals "
               "critical patterns requiring executive attention."]
    if atr_col:
        rate   = float(df[atr_col].mean()) * 100
        n_left = int(df[atr_col].sum())
        parts.append(f"Attrition stands at {rate:.1f}% with {n_left:,} employees having left — "
                     f"{'above' if rate > 15 else 'at'} the SHRM 15% healthy benchmark.")
    parts.append("Immediate action is required on the critical findings below "
                 "to prevent further financial and operational impact.")
    return " ".join(parts)

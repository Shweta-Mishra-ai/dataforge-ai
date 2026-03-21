"""
report_narrator.py — Elite consultant narratives via Groq.
Two prompts: Executive Summary + Chart Analysis.
Real stats in → polished business English out.
No jargon, no snake_case, no academic terms.
"""
import os
import pandas as pd
import numpy as np
from typing import Optional, Dict


# ══════════════════════════════════════════════════════════
#  COLUMN NAME CLEANER
# ══════════════════════════════════════════════════════════

COL_MAP = {
    # HR
    "satisfaction_level":      "Employee Satisfaction Score",
    "last_evaluation":         "Last Performance Evaluation",
    "number_project":          "Number of Active Projects",
    "average_montly_hours":    "Average Monthly Hours Worked",
    "average_monthly_hours":   "Average Monthly Hours Worked",
    "time_spend_company":      "Employee Tenure (Years)",
    "work_accident":           "Work Accident Incidence",
    "left":                    "Employee Attrition",
    "attrition":               "Employee Attrition Rate",
    "promotion_last_5years":   "Recent Promotions (Last 5 Years)",
    "dept":                    "Department",
    "department":              "Department",
    "salary":                  "Salary Band",
    # Ecommerce
    "discounted_price":        "Selling Price",
    "actual_price":            "Original Price (MRP)",
    "discount_percentage":     "Discount Percentage",
    "discount_pct":            "Discount Percentage",
    "rating_count":            "Number of Customer Reviews",
    "rating":                  "Customer Rating",
    "product_id":              "Product ID",
    "product_name":            "Product Name",
    "category":                "Product Category",
    # Sales
    "revenue":                 "Revenue",
    "sales":                   "Sales Amount",
    "target":                  "Sales Target",
    "quota":                   "Sales Quota",
    "profit":                  "Profit",
    "margin":                  "Profit Margin",
    "region":                  "Sales Region",
    "territory":               "Sales Territory",
    "rep":                     "Sales Representative",
}


def clean_col(col: str) -> str:
    low = col.lower().strip()
    if low in COL_MAP:
        return COL_MAP[low]
    # Generic: snake_case → Title Case, fix typos
    name = col.replace("_", " ").strip()
    name = name.replace("montly", "Monthly").replace("accidnet", "Accident")
    return " ".join(w.capitalize() for w in name.split())


def clean_feature(feature: str) -> str:
    if " by " in feature:
        parts = feature.split(" by ")
        return "{} by {}".format(clean_col(parts[0]), clean_col(parts[1]))
    if ":" in feature:
        parts = feature.split(":", 1)
        return "{}: {}".format(parts[0], clean_col(parts[1].strip()))
    return clean_col(feature)


def clean_text(text: str) -> str:
    """Post-process: remove any remaining snake_case."""
    for raw, clean in COL_MAP.items():
        text = text.replace("'"+raw+"'", clean)
        text = text.replace('"'+raw+'"', clean)
        text = text.replace(raw.replace("_"," "), clean)
    return text


# ══════════════════════════════════════════════════════════
#  REAL STATS BUILDER
# ══════════════════════════════════════════════════════════

def _build_raw_data_summary(
    df: pd.DataFrame,
    domain: str = "general",
    story_report=None,
) -> str:
    """
    Build structured data summary for executive prompt.
    LLM only narrates — never calculates.
    """
    parts = []
    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include="object").columns.tolist()

    parts.append("Dataset: {:,} rows, {} columns, {} domain".format(
        len(df), len(df.columns), domain.upper()))

    # Key numeric stats
    for col in num_cols[:6]:
        try:
            arr = pd.to_numeric(df[col], errors="coerce").dropna().values
            arr = arr[np.isfinite(arr)]
            if len(arr) < 3: continue
            q1  = float(np.percentile(arr, 25))
            q3  = float(np.percentile(arr, 75))
            parts.append("{}: average={:.2f}, typical={:.2f}, range={:.2f}-{:.2f}".format(
                clean_col(col), float(np.mean(arr)), float(np.median(arr)),
                float(np.min(arr)), float(np.max(arr))))
        except Exception:
            continue

    # Categorical breakdowns
    for col in cat_cols[:3]:
        try:
            vc = df[col].value_counts(normalize=True).head(4)
            breakdown = " | ".join(["{}: {:.0f}%".format(k, v*100) for k,v in vc.items()])
            parts.append("{}: {}".format(clean_col(col), breakdown))
        except Exception:
            continue

    # Story report findings if available
    if story_report:
        attrition = getattr(story_report, "attrition", None)
        if attrition:
            parts.append("CRITICAL — Attrition Rate: {:.1f}% (benchmark: 10-15%). "
                         "{:,} employees left. Flight risk: {:,} remaining.".format(
                attrition.rate, attrition.n_left, attrition.n_flight_risk))
        for risk in getattr(story_report, "business_risks", [])[:3]:
            parts.append("Risk: " + risk[:100])

    return "\n".join(parts)


def _build_chart_data(
    df: pd.DataFrame,
    chart_type: str,
    x_axis: str,
    y_axis: str,
) -> str:
    """Build chart data dict as string for LLM."""
    try:
        if chart_type.lower() in ("bar chart", "bar"):
            grp = df.groupby(x_axis)[y_axis].mean().sort_values(ascending=False)
            data = {str(k): round(float(v), 3) for k,v in grp.items()}
            overall = float(df[y_axis].mean())
            return "Data: {} | Overall average: {:.3f}".format(str(data), overall)

        elif chart_type.lower() in ("pie chart", "pie"):
            grp   = df.groupby(x_axis)[y_axis].mean()
            total = grp.sum()
            data  = {str(k): "{:.1f}%".format(v/total*100) for k,v in grp.items()}
            top   = max(grp.items(), key=lambda x: x[1])
            return "Shares: {} | Largest: '{}' at {:.1f}%".format(
                str(data), top[0], top[1]/total*100)

        elif chart_type.lower() in ("histogram", "distribution"):
            arr    = pd.to_numeric(df[y_axis], errors="coerce").dropna().values
            arr    = arr[np.isfinite(arr)]
            return ("Distribution: avg={:.2f}, typical={:.2f}, "
                    "range={:.2f}-{:.2f}, shape={}".format(
                float(np.mean(arr)), float(np.median(arr)),
                float(np.min(arr)), float(np.max(arr)),
                "concentrated in high values" if float(pd.Series(arr).skew())<-0.5
                else "concentrated in low values" if float(pd.Series(arr).skew())>0.5
                else "evenly spread"))

        elif chart_type.lower() in ("correlation", "heatmap"):
            num_cols = df.select_dtypes(include="number").columns.tolist()
            corr     = df[num_cols[:6]].corr()
            pairs    = []
            for i in range(len(corr.columns)):
                for j in range(i+1, len(corr.columns)):
                    a, b = corr.columns[i], corr.columns[j]
                    r    = float(corr.loc[a,b])
                    if abs(r) >= 0.3:
                        pairs.append("{} & {}: {:.2f}".format(
                            clean_col(a), clean_col(b), r))
            return "Significant relationships: " + " | ".join(pairs[:5]) if pairs else "No strong relationships"

    except Exception:
        pass

    return "Chart data unavailable"


# ══════════════════════════════════════════════════════════
#  EXECUTIVE SUMMARY PROMPT
# ══════════════════════════════════════════════════════════

def _executive_prompt(raw_data_summary: str) -> str:
    return """You are an Elite Business Consultant writing an Executive Summary and Action Plan for a company Director.

RAW DATA SUMMARY:
{raw_data_summary}

Your task is to analyze this data and provide 3 major business risks and 3 strategic business actions.

CRITICAL RULES (YOU WILL BE PENALIZED FOR VIOLATING THESE):
1. INFER THE DOMAIN: Look at the variable names. Are they HR, E-Commerce, or Finance? Adopt the persona of a consultant in that specific industry.
2. TRANSLATE RAW VARIABLES: You are STRICTLY FORBIDDEN from outputting raw database column names. You MUST convert snake_case to polished English.
   - Example: Change 'time_spend_company' to 'Employee Tenure'.
   - Example: Change 'promotion_last_5years' to 'Recent Promotions'.
3. NO ACADEMIC JARGON: You are STRICTLY FORBIDDEN from using terms like: Kruskal-Wallis, p-value, Pearson, skewness, median, IQR, or standard deviation. Translate math into business English (e.g., "There is a mathematically proven performance gap").
4. NO DATA ENGINEERING ADVICE: Do NOT suggest log-transforms, machine learning models, handling missing values, or outlier removal. Recommend pure business actions like "Conduct exit interviews" or "Audit workloads".

OUTPUT FORMAT:
Write ONE powerful opening sentence summarizing the business situation.
Then provide exactly 3 bullet points of business risks (start each with RISK:).
Then provide exactly 3 bullet points of immediate strategic actions (start each with ACTION:).
Be direct, authoritative, and specific with numbers from the data.""".format(
        raw_data_summary=raw_data_summary)


# ══════════════════════════════════════════════════════════
#  CHART ANALYSIS PROMPT
# ══════════════════════════════════════════════════════════

def _chart_prompt(chart_type: str, x_axis: str, y_axis: str,
                  chart_data_str: str) -> str:
    x_clean = clean_col(x_axis)
    y_clean = clean_col(y_axis)

    return """You are a Data Storyteller presenting a {chart_type} to a client.
The chart compares {x_clean} against {y_clean}.
Chart Data: {chart_data}

CRITICAL RULES:
1. TRANSLATE VARIABLES: Never use raw snake_case names. '{x_axis}' = '{x_clean}', '{y_axis}' = '{y_clean}'.
2. NO ACADEMIC JARGON: Do not use terms like Kruskal-Wallis, p-value, or standard deviation.

STRICT CHART-SPECIFIC INSTRUCTIONS:
- IF THIS IS A BAR CHART: Focus strictly on rankings and disparities. Identify highest and lowest performers. Explain what this operational gap means for the business.
- IF THIS IS A PIE CHART: Do NOT discuss performance gaps. Focus strictly on market share and composition. Identify which category holds the largest slice and explain if the business is dangerously concentrated or well-balanced.
- IF THIS IS A TREND/LINE CHART: Focus strictly on momentum and trajectory. Is the situation improving or degrading?
- IF THIS IS A HISTOGRAM/DISTRIBUTION: Focus on what the typical value means for business and what the extremes indicate about risk.
- IF THIS IS A CORRELATION/HEATMAP: Focus on which business levers move together and what this means operationally.

OUTPUT FORMAT:
Write exactly ONE paragraph (3-4 sentences maximum) of high-impact business analysis. Do not repeat general dataset averages. Be specific with numbers from the chart data. End with one clear business implication.""".format(
        chart_type=chart_type,
        x_clean=x_clean, y_clean=y_clean,
        chart_data=chart_data_str,
        x_axis=x_axis, y_axis=y_axis)


# ══════════════════════════════════════════════════════════
#  GROQ CALLER
# ══════════════════════════════════════════════════════════

def _call_groq(prompt: str, api_key: str, max_tokens: int = 400) -> Optional[str]:
    """Call Groq API. Returns None if unavailable."""
    if not api_key:
        return None
    try:
        from groq import Groq
        client   = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role":"user","content":prompt}],
            max_tokens=max_tokens,
            temperature=0.3,
        )
        return clean_text(response.choices[0].message.content.strip())
    except Exception:
        return None


# ══════════════════════════════════════════════════════════
#  RULE-BASED FALLBACKS
# ══════════════════════════════════════════════════════════

def _fallback_executive(df: pd.DataFrame, domain: str,
                        story_report=None) -> str:
    """Rule-based executive summary when Groq unavailable."""
    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include="object").columns.tolist()

    attrition = getattr(story_report, "attrition", None) if story_report else None
    risks = getattr(story_report, "business_risks", []) if story_report else []
    actions = getattr(story_report, "recommended_actions", []) if story_report else []

    parts = []

    if domain == "hr" and attrition:
        parts.append(
            "This workforce dataset reveals a significant retention challenge "
            "requiring immediate leadership attention.\n\n"
            "RISK: Employee attrition at {:.1f}% exceeds the healthy 10-15% industry benchmark, "
            "with {:,} employees having departed.".format(attrition.rate, attrition.n_left))
        if attrition.dept_attrition:
            worst = max(attrition.dept_attrition.items(), key=lambda x:x[1])
            parts.append("RISK: {} department shows {:.0f}% attrition — the highest in the organization.".format(
                worst[0], worst[1]))
        parts.append("RISK: {:,} remaining employees estimated at flight risk ({:.0f}% of workforce).".format(
            attrition.n_flight_risk, attrition.flight_risk_pct))
        parts.append("\nACTION: Conduct exit interviews with all departed employees within 2 weeks to identify root causes.")
        parts.append("ACTION: Benchmark salaries against market rates — compensation is the #1 attrition driver.")
        parts.append("ACTION: Launch quarterly employee pulse surveys to monitor satisfaction in real time.")

    elif domain == "ecommerce":
        rating_col = next((c for c in num_cols if "rating" in c.lower()
                           and "count" not in c.lower()), None)
        parts.append("This e-commerce dataset reveals product performance patterns requiring strategic attention.\n")
        if rating_col:
            mean_r = float(df[rating_col].mean())
            low_n  = int((df[rating_col]<3.0).sum())
            parts.append("RISK: Average customer rating of {:.2f}/5 {'is below the 4.0 industry benchmark' if mean_r<4.0 else 'is competitive but requires maintenance'}, with {:,} products rated critically low.".format(mean_r, low_n))
        if risks:
            for r in risks[:2]:
                parts.append("RISK: " + r[:100])
        parts.append("\nACTION: Audit all products rated below 3.0 and either improve quality or remove from catalog.")
        parts.append("ACTION: A/B test a 5% price increase on all products rated above 4.3.")
        parts.append("ACTION: Implement post-purchase customer surveys to identify quality gaps before reviews are posted.")

    elif domain == "sales":
        rev_col = next((c for c in num_cols
                        if any(k in c.lower() for k in ["revenue","sales","amount"])), None)
        parts.append("This sales dataset reveals performance patterns with significant revenue implications.\n")
        if rev_col:
            mean_r = float(df[rev_col].mean())
            parts.append("RISK: Revenue distribution shows high variability — average {:.0f} may mask significant underperformers.".format(mean_r))
        if risks:
            for r in risks[:2]:
                parts.append("RISK: " + r[:100])
        parts.append("\nACTION: Weekly revenue vs target review by region and representative.")
        parts.append("ACTION: Identify top 20% revenue drivers and replicate their approach across the team.")
        parts.append("ACTION: Conduct pipeline quality audit to identify stalled deals and conversion blockers.")

    else:
        parts.append("Dataset analysis complete — key patterns identified for strategic action.\n")
        for r in risks[:3]:
            parts.append("RISK: " + r[:100])
        for a in actions[:3]:
            parts.append("ACTION: " + a.replace("[CRITICAL] ","").replace("[SHORT TERM] ","")
                         .replace("[LONG TERM] ","")[:100])

    return "\n".join(parts)


def _fallback_chart(df: pd.DataFrame, chart_type: str,
                    x_axis: str, y_axis: str) -> str:
    """Rule-based chart narrative when Groq unavailable."""
    x_clean = clean_col(x_axis)
    y_clean = clean_col(y_axis)

    try:
        if chart_type.lower() in ("bar chart","bar") and x_axis in df.columns and y_axis in df.columns:
            grp   = df.groupby(x_axis)[y_axis].mean().sort_values(ascending=False)
            top   = grp.index[0];   top_v  = grp.iloc[0]
            bot   = grp.index[-1];  bot_v  = grp.iloc[-1]
            avg   = float(df[y_axis].mean())
            gap   = abs(top_v-bot_v)/max(bot_v,0.001)*100
            above = (grp>avg).sum()
            return (
                "Analysis of {} across {} categories reveals a clear performance hierarchy. "
                "'{}' leads with {:.2f} while '{}' trails at {:.2f} — a {:.0f}% performance gap that "
                "demands immediate operational attention. "
                "{} out of {} categories perform above the overall average of {:.2f}. "
                "Closing this gap represents a significant untapped performance opportunity.".format(
                    y_clean, len(grp), top, top_v, bot, bot_v, gap,
                    above, len(grp), avg))

        elif chart_type.lower() in ("pie chart","pie") and x_axis in df.columns and y_axis in df.columns:
            grp   = df.groupby(x_axis)[y_axis].mean()
            total = grp.sum()
            top   = grp.idxmax()
            top_p = grp.max()/total*100
            top2  = grp.nlargest(2).sum()/total*100
            return (
                "The composition of {} across {} segments reveals {}. "
                "'{}' dominates with {:.1f}% share. "
                "The top 2 segments combined hold {:.1f}% — indicating {}. "
                "This concentration pattern has significant implications for risk and resource allocation.".format(
                    y_clean, len(grp),
                    "a highly concentrated distribution" if top_p>40 else "a balanced distribution",
                    top, top_p, top2,
                    "dangerous dependency" if top2>70 else "moderate balance"))

        elif y_axis in df.columns:
            arr = pd.to_numeric(df[y_axis], errors="coerce").dropna().values
            arr = arr[np.isfinite(arr)]
            if len(arr)>3:
                skew = float(pd.Series(arr).skew())
                return (
                    "{} shows an average of {:.2f} with a typical value of {:.2f}. "
                    "Values range from {:.2f} to {:.2f}, indicating {}. "
                    "The {} distribution suggests decision-makers should focus on {} for accurate planning.".format(
                        y_clean, float(np.mean(arr)), float(np.median(arr)),
                        float(np.min(arr)), float(np.max(arr)),
                        "high variability requiring segmented analysis" if float(np.std(arr))/abs(float(np.mean(arr)))>0.5 else "consistent performance",
                        "skewed" if abs(skew)>0.5 else "symmetric",
                        "the typical value ({:.2f})".format(float(np.median(arr))) if abs(skew)>0.5 else "the average ({:.2f})".format(float(np.mean(arr)))))
    except Exception:
        pass

    return ("Analysis of {} reveals patterns with strategic business implications. "
            "Review the chart for specific values and consult with department heads "
            "to develop targeted action plans.".format(y_clean))


# ══════════════════════════════════════════════════════════
#  MAIN PUBLIC FUNCTIONS
# ══════════════════════════════════════════════════════════

def generate_executive_summary(
    df: pd.DataFrame,
    domain: str = "general",
    story_report=None,
    groq_api_key: str = "",
) -> str:
    """
    Generate elite consultant executive summary.
    Groq LLM if available, rule-based fallback otherwise.
    """
    raw_summary = _build_raw_data_summary(df, domain, story_report)
    prompt      = _executive_prompt(raw_summary)

    result = _call_groq(prompt, groq_api_key, max_tokens=500)
    if result:
        return result

    return _fallback_executive(df, domain, story_report)


def generate_chart_narrative(
    df: pd.DataFrame,
    chart_title: str,
    groq_api_key: str = "",
    domain: str = "general",
) -> str:
    """
    Generate elite data storyteller narrative for a chart.
    Auto-detects chart type from title.
    """
    # Detect chart type
    title_lower = chart_title.lower()
    if "correlation" in title_lower or "heatmap" in title_lower:
        chart_type = "Correlation Heatmap"
        x_axis     = "metrics"
        y_axis     = "metrics"
    elif "distribution" in title_lower or "histogram" in title_lower:
        chart_type = "Histogram"
        # Extract column from title
        num_cols   = df.select_dtypes(include="number").columns.tolist()
        y_axis     = next((c for c in num_cols if c.lower() in title_lower), num_cols[0] if num_cols else "value")
        x_axis     = y_axis
    elif "pie" in title_lower or "share" in title_lower:
        chart_type = "Pie Chart"
        cat_cols   = df.select_dtypes(include="object").columns.tolist()
        num_cols   = df.select_dtypes(include="number").columns.tolist()
        x_axis     = next((c for c in cat_cols if c.lower() in title_lower), cat_cols[0] if cat_cols else "category")
        y_axis     = next((c for c in num_cols if c.lower() in title_lower), num_cols[0] if num_cols else "value")
    elif "by" in title_lower:
        chart_type = "Bar Chart"
        parts      = chart_title.lower().replace("avg ","").replace("total ","").split(" by ")
        num_cols   = df.select_dtypes(include="number").columns.tolist()
        cat_cols   = df.select_dtypes(include="object").columns.tolist()
        y_axis     = next((c for c in num_cols if c.lower() in parts[0]), num_cols[0] if num_cols else "value")
        x_axis     = next((c for c in cat_cols if c.lower() in (parts[1] if len(parts)>1 else "")),
                          cat_cols[0] if cat_cols else "category")
    else:
        chart_type = "Bar Chart"
        num_cols   = df.select_dtypes(include="number").columns.tolist()
        cat_cols   = df.select_dtypes(include="object").columns.tolist()
        y_axis     = num_cols[0] if num_cols else "value"
        x_axis     = cat_cols[0] if cat_cols else "category"

    chart_data_str = _build_chart_data(df, chart_type, x_axis, y_axis)
    prompt         = _chart_prompt(chart_type, x_axis, y_axis, chart_data_str)

    result = _call_groq(prompt, groq_api_key, max_tokens=300)
    if result:
        return result

    return _fallback_chart(df, chart_type, x_axis, y_axis)

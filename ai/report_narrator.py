import pandas as pd
from ai.llm_client import LLMClient
from core.data_profiler import DatasetProfile


def generate_executive_summary(
    df: pd.DataFrame,
    profile: DatasetProfile,
    client: LLMClient
) -> str:
    """Generate 5 key findings for executive summary page."""

    num_cols  = profile.numeric_cols[:3]
    cat_cols  = profile.categorical_cols[:2]

    prompt = f"""You are a senior data analyst writing a professional report.

DATASET:
- Rows: {profile.rows:,}
- Columns: {profile.cols}
- Quality Score: {profile.overall_quality_score}/100
- Missing cells: {profile.missing_pct}%
- Duplicate rows: {profile.duplicate_rows}
- Numeric columns: {num_cols}
- Category columns: {cat_cols}

NUMERIC STATS:
{df[num_cols].describe().round(2).to_string() if num_cols else "None"}

Write exactly 5 key findings for an executive summary.
Use simple professional English. Be specific with numbers.
Format as JSON array:
[
  {{"finding": "...", "type": "positive|negative|neutral"}},
  ...
]
Return ONLY JSON. No text before or after."""

    raw = client.chat_safe(
        messages=[{"role": "user", "content": prompt}],
        fallback='[{"finding": "Dataset analysis complete.", "type": "neutral"}]'
    )

    import json, re
    try:
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if match:
            return json.loads(match.group(0))
    except Exception:
        pass

    return [{"finding": "Dataset analysis complete.", "type": "neutral"}]


def generate_chart_narrative(
    chart_title: str,
    chart_type: str,
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    client: LLMClient
) -> str:
    """Generate 2-3 sentence analysis paragraph for each chart."""

    try:
        stats = df[[x_col, y_col]].describe().round(2).to_string() if y_col in df.columns else ""
    except Exception:
        stats = ""

    prompt = f"""You are a senior data analyst.

Chart: {chart_title}
Type: {chart_type}
X axis: {x_col}
Y axis: {y_col}
Stats:
{stats}

Write 2-3 sentences of professional analysis for this chart.
Mention specific numbers. Plain English. No jargon.
Return only the paragraph text."""

    return client.chat_safe(
        messages=[{"role": "user", "content": prompt}],
        fallback=f"The {chart_title} shows the distribution of {y_col} across {x_col}."
    )


def generate_recommendations(
    profile: DatasetProfile,
    df: pd.DataFrame,
    client: LLMClient
) -> dict:
    """Generate actionable recommendations in 3 timeframes."""

    prompt = f"""You are a senior data analyst.

Dataset quality score: {profile.overall_quality_score}/100
Missing values: {profile.missing_pct}%
Duplicates: {profile.duplicate_rows}
Recommendations from profiling:
{chr(10).join(profile.recommendations[:5])}

Generate professional recommendations in 3 timeframes.
Return ONLY this JSON:
{{
  "immediate": ["action 1", "action 2", "action 3"],
  "short_term": ["action 1", "action 2", "action 3"],
  "long_term":  ["action 1", "action 2", "action 3"]
}}"""

    import json, re
    raw = client.chat_safe(
        messages=[{"role": "user", "content": prompt}],
        fallback='{{"immediate":[],"short_term":[],"long_term":[]}}'
    )

    try:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            return json.loads(match.group(0))
    except Exception:
        pass

    return {"immediate": [], "short_term": [], "long_term": []}

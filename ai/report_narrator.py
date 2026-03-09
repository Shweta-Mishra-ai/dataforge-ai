import json
import re
import pandas as pd
from ai.llm_client import LLMClient
from core.data_profiler import DatasetProfile


def generate_executive_summary(
    df: pd.DataFrame,
    profile: DatasetProfile,
    client: LLMClient
) -> list:
    """Generate 5 key findings for executive summary page."""

    num_cols = profile.numeric_cols[:4]
    cat_cols = profile.categorical_cols[:2]

    stats_str = ""
    if num_cols:
        try:
            stats_str = df[num_cols].describe().round(2).to_string()
        except Exception:
            stats_str = ""

    cat_info = ""
    for col in cat_cols:
        try:
            top = df[col].value_counts().head(3)
            cat_info += f"\n{col} top values: {top.to_dict()}"
        except Exception:
            pass

    prompt = f"""You are a senior data analyst writing an executive summary.

DATASET OVERVIEW:
- Total rows: {profile.rows:,}
- Total columns: {profile.cols}
- Quality Score: {profile.overall_quality_score}/100
- Missing cells: {profile.missing_pct}%
- Duplicate rows: {profile.duplicate_rows}
- Numeric columns: {num_cols}
- Category columns: {cat_cols}

NUMERIC STATISTICS:
{stats_str}

CATEGORICAL BREAKDOWN:
{cat_info}

Write exactly 5 key findings. Each finding must:
- Be specific to THIS dataset with real numbers
- Be 1-2 sentences max
- Be written in plain professional English
- Classify as positive (good news), negative (concern), or neutral (observation)

Return ONLY a JSON array, no other text:
[
  {{"finding": "specific finding with numbers here", "type": "positive"}},
  {{"finding": "specific finding with numbers here", "type": "negative"}},
  {{"finding": "specific finding with numbers here", "type": "neutral"}},
  {{"finding": "specific finding with numbers here", "type": "positive"}},
  {{"finding": "specific finding with numbers here", "type": "neutral"}}
]"""

    raw = client.chat_safe(
        messages=[{"role": "user", "content": prompt}],
        fallback='[{"finding": "Dataset analysis complete with ' + str(profile.rows) + ' rows.", "type": "neutral"}]'
    )

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
    """Generate 2-3 sentence chart-specific analysis paragraph."""

    # Build chart-specific stats
    stats_lines = []

    try:
        if y_col and y_col in df.columns and pd.api.types.is_numeric_dtype(df[y_col]):
            s = df[y_col].dropna()
            stats_lines.append(f"{y_col}: mean={s.mean():.2f}, min={s.min():.2f}, max={s.max():.2f}, std={s.std():.2f}")

        if x_col and x_col in df.columns:
            if pd.api.types.is_numeric_dtype(df[x_col]):
                s = df[x_col].dropna()
                stats_lines.append(f"{x_col}: mean={s.mean():.2f}, min={s.min():.2f}, max={s.max():.2f}")
            else:
                top = df[x_col].value_counts().head(3)
                stats_lines.append(f"{x_col} top categories: {top.to_dict()}")

        # For bar/pie — add grouped stats
        if x_col in df.columns and y_col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[x_col]):
                grp = df.groupby(x_col)[y_col].sum().sort_values(ascending=False).head(3)
                stats_lines.append(f"Top groups by {y_col}: {grp.to_dict()}")

    except Exception:
        pass

    stats_str = "\n".join(stats_lines) if stats_lines else "No stats available"

    # Chart-type specific instructions
    chart_instructions = {
        "bar":         "Focus on which categories are highest/lowest and the gap between them.",
        "Bar":         "Focus on which categories are highest/lowest and the gap between them.",
        "line":        "Focus on the trend direction, any peaks or dips, and overall pattern.",
        "Line":        "Focus on the trend direction, any peaks or dips, and overall pattern.",
        "histogram":   "Focus on the distribution shape, where most values cluster, and any skew.",
        "Distribution":"Focus on the distribution shape, where most values cluster, and any skew.",
        "heatmap":     "Focus on the strongest positive and negative correlations found.",
        "Correlation": "Focus on the strongest positive and negative correlations found.",
        "pie":         "Focus on which categories dominate and their relative proportions.",
        "Share":       "Focus on which categories dominate and their relative proportions.",
    }

    instruction = "Analyze the key patterns visible in this chart."
    for key, val in chart_instructions.items():
        if key.lower() in chart_title.lower() or key.lower() in chart_type.lower():
            instruction = val
            break

    prompt = f"""You are a senior data analyst writing a chart analysis for a client report.

Chart title: {chart_title}
X axis: {x_col}
Y axis: {y_col}

Key statistics:
{stats_str}

Task: {instruction}

Write exactly 2-3 sentences of professional analysis.
Requirements:
- Use specific numbers from the statistics above
- Do NOT just repeat the column names or chart title
- Do NOT start with "The chart shows" or "This chart"
- Write as if explaining insight to a business executive
- Plain English, no technical jargon

Return only the paragraph text, nothing else."""

    return client.chat_safe(
        messages=[{"role": "user", "content": prompt}],
        fallback=f"Analysis of {chart_title} reveals key patterns in the data."
    )


def generate_recommendations(
    profile: DatasetProfile,
    df: pd.DataFrame,
    client: LLMClient
) -> dict:
    """Generate actionable recommendations in 3 timeframes."""

    # Build column-specific context
    col_issues = []
    for p in profile.column_profiles:
        if p.missing_pct > 5:
            col_issues.append(f"'{p.name}' has {p.missing_pct}% missing values")
        if p.has_outliers:
            col_issues.append(f"'{p.name}' has {p.outlier_count} outliers")

    issues_str = "\n".join(col_issues[:6]) if col_issues else "No major issues found"

    num_cols = profile.numeric_cols[:3]
    stats_str = ""
    if num_cols:
        try:
            stats_str = df[num_cols].describe().round(2).to_string()
        except Exception:
            pass

    prompt = f"""You are a senior data engineer giving recommendations after a data quality audit.

DATASET PROFILE:
- Rows: {profile.rows:,}
- Quality score: {profile.overall_quality_score}/100
- Missing values: {profile.missing_pct}%
- Duplicate rows: {profile.duplicate_rows}
- Numeric columns: {profile.numeric_cols}
- Categorical columns: {profile.categorical_cols}

IDENTIFIED ISSUES:
{issues_str}

STATS SNAPSHOT:
{stats_str}

Generate specific, actionable recommendations in 3 timeframes.
Each action must be concrete and relevant to this specific dataset.
Return ONLY this JSON, no other text:
{{
  "immediate": [
    "action specific to this dataset",
    "action specific to this dataset",
    "action specific to this dataset"
  ],
  "short_term": [
    "action specific to this dataset",
    "action specific to this dataset",
    "action specific to this dataset"
  ],
  "long_term": [
    "action specific to this dataset",
    "action specific to this dataset",
    "action specific to this dataset"
  ]
}}"""

    raw = client.chat_safe(
        messages=[{"role": "user", "content": prompt}],
        fallback='{"immediate":["Review data quality"],"short_term":["Address missing values"],"long_term":["Implement data governance"]}'
    )

    try:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            return json.loads(match.group(0))
    except Exception:
        pass

    return {
        "immediate": ["Review data quality issues identified in this report"],
        "short_term": ["Address missing values and outliers"],
        "long_term":  ["Implement automated data quality monitoring"]
    }

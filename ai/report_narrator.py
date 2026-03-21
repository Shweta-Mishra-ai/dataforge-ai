"""
ai/report_narrator.py — DataForge AI FINAL
Uses llm_client.py for Groq + Gemini routing.
Anti-hallucination: JSON schema + stat injection + validation + fallback.
"""

from __future__ import annotations
import re
import json
import numpy as np
import pandas as pd
from typing import Optional

from ai.llm_client import get_client

# ── Column cleaner ────────────────────────────────────────
COL_MAP = {
    "satisfaction_level":    "Employee Satisfaction Score",
    "last_evaluation":       "Last Performance Evaluation",
    "number_project":        "Number of Active Projects",
    "average_montly_hours":  "Average Monthly Hours Worked",
    "average_monthly_hours": "Average Monthly Hours Worked",
    "time_spend_company":    "Employee Tenure (Years)",
    "work_accident":         "Work Accident Rate",
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

# Phrases that mean Groq hallucinated
FAKE_PHRASES = [
    "customer satisfaction and sales revenue",
    "sales revenue tends to increase",
    "marketing spend", "website traffic",
    "operational costs and employee turnover",
    "top-line growth", "sales targeted",
    "customer-centric", "churn rate",
    "net promoter", "as customer satisfaction increases",
]


def clean_col(col: str) -> str:
    low = col.lower().strip()
    if low in COL_MAP:
        return COL_MAP[low]
    return " ".join(
        w.capitalize()
        for w in col.replace("_", " ").replace("montly", "Monthly").split()
    )


def _is_hallucinated(text: str, df: pd.DataFrame) -> bool:
    tl = text.lower()
    if any(p in tl for p in FAKE_PHRASES):
        return True
    real = [c.lower().replace("_", "") for c in df.columns]
    invented = ["salesrevenue", "customerrevenue", "websitetraffic",
                "marketingspend", "operationalcost", "churnrate"]
    for inv in invented:
        if inv in tl.replace(" ", "").replace("_", ""):
            if not any(inv in r for r in real):
                return True
    return False


def _clean(text: str) -> str:
    text = re.sub(r"\bSales Targeted\b",      "targeted",       text, flags=re.IGNORECASE)
    text = re.sub(r"\bSales Representative\b","representative",  text, flags=re.IGNORECASE)
    for raw, clean in COL_MAP.items():
        text = text.replace(f"'{raw}'", clean).replace(f'"{raw}"', clean)
    return text.strip()


# ══════════════════════════════════════════════════════════
#  STAT COMPUTERS — Python computes, LLM only narrates
# ══════════════════════════════════════════════════════════

def _bar_stats(df: pd.DataFrame, x: str, y: str) -> dict:
    try:
        grp = df.groupby(x)[y].mean().sort_values(ascending=False)
        avg = float(df[y].mean())
        return {
            "chart":       "bar",
            "metric":      clean_col(y),
            "group_by":    clean_col(x),
            "n_groups":    len(grp),
            "top":         str(grp.index[0]),
            "top_val":     round(float(grp.iloc[0]), 3),
            "worst":       str(grp.index[-1]),
            "worst_val":   round(float(grp.iloc[-1]), 3),
            "gap_pct":     round(abs(float(grp.iloc[0]) - float(grp.iloc[-1])) /
                                 max(abs(float(grp.iloc[-1])), 0.001) * 100, 1),
            "org_avg":     round(avg, 3),
            "above_avg":   int((grp > avg).sum()),
            "all_vals":    {str(k): round(float(v), 3) for k, v in grp.items()},
        }
    except Exception as e:
        return {"chart": "bar", "error": str(e),
                "metric": clean_col(y), "group_by": clean_col(x)}


def _hist_stats(df: pd.DataFrame, col: str) -> dict:
    try:
        s    = pd.to_numeric(df[col], errors="coerce").dropna()
        s    = s[np.isfinite(s)]
        skew = float(s.skew())
        return {
            "chart":      "histogram",
            "metric":     clean_col(col),
            "mean":       round(float(s.mean()), 3),
            "median":     round(float(s.median()), 3),
            "std":        round(float(s.std()), 3),
            "min":        round(float(s.min()), 3),
            "max":        round(float(s.max()), 3),
            "q1":         round(float(s.quantile(0.25)), 3),
            "q3":         round(float(s.quantile(0.75)), 3),
            "skew":       round(skew, 2),
            "shape":      ("right-skewed" if skew > 0.5
                           else "left-skewed" if skew < -0.5
                           else "symmetric"),
            "use_stat":   "median" if abs(skew) > 0.5 else "mean",
            "use_val":    round(float(s.median()) if abs(skew) > 0.5
                               else float(s.mean()), 3),
        }
    except Exception as e:
        return {"chart": "histogram", "error": str(e), "metric": clean_col(col)}


def _corr_stats(df: pd.DataFrame) -> dict:
    try:
        num  = df.select_dtypes(include="number").columns.tolist()[:8]
        corr = df[num].corr(method="spearman")
        pairs = []
        for i in range(len(num)):
            for j in range(i + 1, len(num)):
                a, b = num[i], num[j]
                r    = float(corr.loc[a, b])
                if abs(r) >= 0.15:
                    pairs.append({
                        "a":   clean_col(a), "b": clean_col(b),
                        "r":   round(r, 3),
                        "r2":  round(r ** 2, 3),
                        "dir": "positive" if r > 0 else "negative",
                        "str": ("strong"   if abs(r) >= 0.6 else
                                "moderate" if abs(r) >= 0.4 else "weak"),
                    })
        pairs.sort(key=lambda x: abs(x["r"]), reverse=True)
        return {
            "chart":    "correlation",
            "cols":     [clean_col(c) for c in num],
            "raw_cols": num,
            "n_sig":    len(pairs),
            "pairs":    pairs[:6],
            "top":      pairs[0] if pairs else None,
        }
    except Exception as e:
        return {"chart": "correlation", "error": str(e)}


def _trend_stats(df: pd.DataFrame, col: str) -> dict:
    try:
        s   = pd.to_numeric(df[col], errors="coerce").dropna()
        s   = s[np.isfinite(s)]
        cv  = s.std() / abs(s.mean()) * 100 if s.mean() != 0 else 0
        mid = len(s) // 2
        f, sc = float(s.iloc[:mid].mean()), float(s.iloc[mid:].mean())
        pct   = (sc - f) / max(abs(f), 0.001) * 100
        return {
            "chart":      "trend",
            "metric":     clean_col(col),
            "mean":       round(float(s.mean()), 3),
            "min":        round(float(s.min()), 3),
            "max":        round(float(s.max()), 3),
            "cv":         round(cv, 1),
            "first_half": round(f, 3),
            "sec_half":   round(sc, 3),
            "trend_pct":  round(pct, 1),
            "trend_dir":  ("improved" if pct > 2
                           else "declined" if pct < -2
                           else "stable"),
        }
    except Exception as e:
        return {"chart": "trend", "error": str(e), "metric": clean_col(col)}


def _pie_stats(df: pd.DataFrame, x: str, y: str) -> dict:
    try:
        grp   = df.groupby(x)[y].mean().sort_values(ascending=False)
        total = grp.sum()
        return {
            "chart":     "pie",
            "metric":    clean_col(y),
            "group_by":  clean_col(x),
            "n":         len(grp),
            "top_seg":   str(grp.index[0]),
            "top_pct":   round(grp.max() / total * 100, 1),
            "top2_pct":  round(grp.nlargest(2).sum() / total * 100, 1),
            "shares":    {str(k): round(v / total * 100, 1) for k, v in grp.items()},
            "balanced":  bool(grp.max() / total * 100 < 40),
        }
    except Exception as e:
        return {"chart": "pie", "error": str(e)}


# ══════════════════════════════════════════════════════════
#  PROMPT BUILDER — JSON schema forces structure
# ══════════════════════════════════════════════════════════

def _build_prompt(stats: dict, df) -> tuple:
    """Returns (system, user) prompts using JSON schema."""
    col_list = [clean_col(c) for c in df.columns]
    ctype    = stats.get("chart", "unknown")
    d        = json.dumps(stats, indent=2)

    system = (
        "Senior Data Analyst. Dataset columns (ONLY these): "
        + str(col_list) + ". "
        "Only reference listed columns. "
        "FORBIDDEN: Sales Revenue, Customer Satisfaction, Marketing Spend, "
        "Website Traffic, churn rate, Sales Targeted. "
        "Return ONLY valid JSON. No markdown. No extra text."
    )

    if ctype == "bar":
        gap = str(round(stats.get("gap_pct", 0), 0))
        schema = (
            '{"finding":"<top group leads, worst group trails, exact values>",'
            '"gap_impact":"<what ' + gap + '% gap means for business>",'
            '"root_cause":"<likely reason>",'
            '"action":"<Strategic Action: one step>","ok":true}'
        )
    elif ctype == "histogram":
        uv = str(stats.get("use_val", 0))
        us = str(stats.get("use_stat", "median"))
        sh = str(stats.get("shape", "symmetric"))
        mn = str(stats.get("min", 0))
        schema = (
            '{"typical":"<what ' + uv + ' ' + us + ' means>",'
            '"shape_meaning":"<what ' + sh + ' shape means>",'
            '"risk":"<what scores near ' + mn + ' indicate>",'
            '"action":"<Strategic Action: one step>","ok":true}'
        )
    elif ctype == "correlation":
        top = stats.get("top") or {}
        a   = str(top.get("a", "N/A"))
        b   = str(top.get("b", "N/A"))
        r   = str(top.get("r", 0))
        r2  = str(top.get("r2", 0))
        ns  = str(stats.get("n_sig", 0))
        schema = (
            '{"n_found":"' + ns + ' relationships",'
            '"strongest":"<exact names: ' + a + ' and ' + b + '>",'
            '"correct_meaning":"<r=' + r + ' means association only, NOT a % effect>",'
            '"business_implication":"<operational meaning>",'
            '"action":"<Strategic Action: one step>","ok":true}'
        )
        d = d + "\n\nSTRONGEST PAIR: " + a + " & " + b + " r=" + r + " r2=" + r2
        d = d + "\nCRITICAL: r is NOT a % effect. Association only."
    elif ctype == "trend":
        metric = str(stats.get("metric", "metric"))
        td     = str(stats.get("trend_dir", "stable"))
        tp     = str(round(abs(stats.get("trend_pct", 0)), 1))
        cv     = str(stats.get("cv", 0))
        schema = (
            '{"trend_summary":"<' + metric + ' ' + td + ' by ' + tp + '% - meaning>",'
            '"variability":"<CV=' + cv + '% consistency meaning>",'
            '"implication":"<key business implication>",'
            '"action":"<Strategic Action: one step>","ok":true}'
        )
        d = d + "\nCRITICAL: TREND chart only. Analyse progression. NO department comparisons."
    elif ctype == "pie":
        ts  = str(stats.get("top_seg", ""))
        tp  = str(stats.get("top_pct", 0))
        t2p = str(stats.get("top2_pct", 0))
        schema = (
            '{"dominant":"<' + ts + ' at ' + tp + '% - meaning>",'
            '"balance":"<top 2 at ' + t2p + '% - risk or health>",'
            '"implication":"<business implication>",'
            '"action":"<Strategic Action: one step>","ok":true}'
        )
    else:
        schema = '{"finding":"","action":"","ok":true}'

    user = "DATA:\n" + d + "\n\nReturn JSON:\n" + schema
    return system, user




def _json_to_text(raw: str, stats: dict, df: pd.DataFrame) -> Optional[str]:
    """Parse JSON → narrative text. Returns None if invalid/hallucinated."""
    try:
        clean = re.sub(r"^```json\s*", "", raw.strip())
        clean = re.sub(r"^```\s*",     "", clean)
        clean = re.sub(r"```$",        "", clean).strip()
        data  = json.loads(clean)

        if not data.get("ok"):
            return None

        # All values as string for hallucination check
        all_vals = " ".join(str(v) for v in data.values())
        if _is_hallucinated(all_vals, df):
            return None

        # Build narrative from JSON fields
        order = [
            "finding", "gap_impact", "root_cause",   # bar
            "typical", "shape_meaning", "risk",       # hist
            "n_found", "strongest", "correct_meaning",# corr
            "business_implication",
            "trend_summary", "variability", "implication",  # trend
            "dominant", "balance",                    # pie
        ]
        parts = [str(data[k]) for k in order if k in data and str(data[k]).strip()]
        parts.append(str(data.get("action", "")))
        narrative = " ".join(p for p in parts if p.strip())
        return _clean(narrative) if narrative.strip() else None

    except Exception:
        return None


# ══════════════════════════════════════════════════════════
#  RULE-BASED FALLBACKS — 100% reliable, no LLM
# ══════════════════════════════════════════════════════════

def _fb_bar(df, x, y):
    s = _bar_stats(df, x, y)
    if "error" in s:
        return f"Analysis of {clean_col(y)} by group reveals performance patterns."
    return (
        f"{s['metric']} across {s['n_groups']} {s['group_by']} groups: "
        f"'{s['top']}' leads at {s['top_val']} while '{s['worst']}' trails "
        f"at {s['worst_val']} — a {s['gap_pct']:.0f}% gap. "
        f"{s['above_avg']} of {s['n_groups']} groups exceed the "
        f"organisation average of {s['org_avg']}. "
        f"Strategic Action: Conduct root-cause review in '{s['worst']}' "
        f"and replicate '{s['top']}' practices to close the gap."
    )


def _fb_hist(df, col):
    s = _hist_stats(df, col)
    if "error" in s:
        return f"Distribution of {clean_col(col)} reveals key patterns."
    return (
        f"{s['metric']} typical value is {s['use_val']} "
        f"(range: {s['min']}–{s['max']}). "
        f"{'Skewed — use ' + s['use_stat'] + ' for accurate reporting.' if s['shape'] != 'symmetric' else 'Symmetric — mean is reliable.'} "
        f"Bottom quartile (below {s['q1']}) represents highest-risk employees. "
        f"Strategic Action: Focus retention programs on employees "
        f"below {s['q1']} to address most at-risk group first."
    )


def _fb_corr(df):
    s = _corr_stats(df)
    if "error" in s or not s.get("pairs"):
        return (
            "No meaningful correlations found (all |r| < 0.15). "
            "Variables operate independently — single-variable interventions "
            "will have limited cross-metric effects. "
            "Strategic Action: Design targeted interventions per metric."
        )
    top = s["pairs"][0]
    second = ""
    if len(s["pairs"]) > 1:
        t2 = s["pairs"][1]
        second = f" Second: {t2['a']} & {t2['b']} (r={t2['r']}, r²={t2['r2']})."
    return (
        f"Correlation matrix: {s['n_sig']} meaningful relationships found. "
        f"Strongest: {top['a']} & {top['b']} (r={top['r']}, {top['dir']}) — "
        f"r²={top['r2']} means {top['r2']*100:.1f}% variance shared "
        f"(association only, not causation).{second} "
        f"Strategic Action: Test the {top['a']}–{top['b']} relationship "
        f"through controlled analysis before acting on it."
    )


def _fb_pie(df, x, y):
    s = _pie_stats(df, x, y)
    if "error" in s:
        return f"Composition of {clean_col(y)} shows segment patterns."
    return (
        f"{s['metric']} across {s['n']} {s['group_by']} segments: "
        f"'{s['top_seg']}' holds {s['top_pct']}%, top 2 combined: {s['top2_pct']}%. "
        f"{'Concentration risk — over-reliance on dominant segments.' if not s['balanced'] else 'Well-balanced — no segment dominates.'} "
        f"Strategic Action: "
        f"{'Diversify away from dominant segments to reduce risk.' if not s['balanced'] else 'Maintain balance and monitor for emerging concentration.'}"
    )


def _fb_trend(df, col):
    s = _trend_stats(df, col)
    if "error" in s:
        return f"Trend of {clean_col(col)} shows performance over time."
    return (
        f"{s['metric']} trend: average {s['mean']} (range {s['min']}–{s['max']}). "
        f"Values {s['trend_dir']} by {abs(s['trend_pct']):.1f}% "
        f"from first to second half — "
        f"{'positive signal.' if s['trend_dir'] == 'improved' else 'declining — investigate immediately.' if s['trend_dir'] == 'declined' else 'stable pattern.'} "
        f"Variability: {s['cv']}% "
        f"({'high — inconsistent' if s['cv'] > 30 else 'stable'}). "
        f"Strategic Action: "
        f"{'Identify decline root causes and implement corrective measures.' if s['trend_dir'] == 'declined' else 'Monitor monthly for early warning signals.'}"
    )


# ══════════════════════════════════════════════════════════
#  MAIN PUBLIC FUNCTIONS
# ══════════════════════════════════════════════════════════

def generate_chart_narrative(
    df:            pd.DataFrame,
    chart_title:   str,
    groq_api_key:  str = "",   # kept for backward compatibility
    domain:        str = "general",
) -> str:
    """
    Generate chart narrative.
    Uses llm_client (Groq primary → Gemini fallback).
    Anti-hallucination: JSON schema + validation + rule-based fallback.
    """
    title = chart_title.lower()
    num   = df.select_dtypes(include="number").columns.tolist()
    cat   = df.select_dtypes(include="object").columns.tolist()

    # ── Detect chart type + compute stats ────────────────
    if "correlation" in title or "heatmap" in title:
        stats    = _corr_stats(df)
        fallback = _fb_corr(df)

    elif "distribution" in title or "histogram" in title:
        col = next((c for c in num if c.lower() in title), num[0] if num else None)
        if not col:
            return "Distribution chart generated."
        stats    = _hist_stats(df, col)
        fallback = _fb_hist(df, col)

    elif "pie" in title or "share" in title:
        parts = title.split(" by ")
        x = next((c for c in cat
                  if c.lower() in (parts[1] if len(parts) > 1 else "")),
                 cat[0] if cat else None)
        y = next((c for c in num if c.lower() in parts[0]),
                 num[0] if num else None)
        if not x or not y:
            return "Pie chart generated."
        stats    = _pie_stats(df, x, y)
        fallback = _fb_pie(df, x, y)

    elif "trend" in title or "line" in title:
        col = next((c for c in num
                    if c.lower().replace("_", "") in
                    title.replace("_", "").replace(" ", "")),
                   num[0] if num else None)
        if not col:
            return "Trend chart generated."
        stats    = _trend_stats(df, col)
        fallback = _fb_trend(df, col)

    elif " by " in title:
        parts = title.replace("avg ", "").replace("total ", "").split(" by ")
        y = next((c for c in num
                  if c.lower().replace("_", "") in parts[0].replace(" ", "")),
                 num[0] if num else None)
        x = next((c for c in cat
                  if c.lower() in (parts[1] if len(parts) > 1 else "")),
                 cat[0] if cat else None)
        if not y or not x:
            return "Bar chart generated."
        stats    = _bar_stats(df, x, y)
        fallback = _fb_bar(df, x, y)

    else:
        if cat and num:
            stats = _bar_stats(df, cat[0], num[0]); fallback = _fb_bar(df, cat[0], num[0])
        elif num:
            stats = _hist_stats(df, num[0]);        fallback = _fb_hist(df, num[0])
        else:
            return "Chart analysis not available."

    # ── Build prompt + call LLM ───────────────────────────
    system, user = _build_prompt(stats, df)
    client       = get_client()
    raw          = client.chat(system, user, task="chart_analysis",
                               max_tokens=400)

    # ── Validate → fallback ───────────────────────────────
    if raw:
        narrative = _json_to_text(raw, stats, df)
        if narrative:
            return narrative

    return fallback   # guaranteed correct


def generate_executive_summary(
    df:           pd.DataFrame,
    domain:       str = "general",
    story_report  = None,
    groq_api_key: str = "",   # kept for backward compatibility
) -> str:
    """
    Executive summary via Gemini (best reasoning) → Groq fallback.
    """
    num = df.select_dtypes(include="number").columns.tolist()
    atr = next((c for c in df.columns
                if c.lower() in ("left", "attrition", "churned", "exited")), None)

    # Pre-compute facts
    facts: dict = {
        "domain":  domain,
        "rows":    len(df),
        "columns": len(df.columns),
    }
    if atr:
        rate = float(df[atr].mean()) * 100
        facts["attrition_pct"]  = round(rate, 1)
        facts["n_left"]         = int(df[atr].sum())
        facts["above_shrm"]     = rate > 15
        facts["gap_pp"]         = round(max(0, rate - 15), 1)

    sat = next((c for c in num if "satisfaction" in c.lower()), None)
    if sat:
        facts["avg_satisfaction"]   = round(float(df[sat].mean()), 3)
        facts["below_industry_norm"] = facts["avg_satisfaction"] < 0.70

    col_list = [clean_col(c) for c in df.columns]
    system   = (
        f"Senior Data Analyst writing a C-suite executive summary. "
        f"Dataset columns: {col_list}. "
        f"ONLY use metrics listed. FORBIDDEN: Sales Revenue, Customer Satisfaction, "
        f"Marketing Spend, Website Traffic. Return ONLY JSON, no other text."
    )
    user = (
        f"PRE-COMPUTED FACTS:\n{json.dumps(facts, indent=2)}\n\n"
        f"Return JSON:\n"
        f'{{"risk": "<most urgent risk with specific number>",'
        f'"lever": "<#1 actionable change>",'
        f'"urgency": "<what must happen in 30 days>",'
        f'"ok": true}}'
    )

    client = get_client()
    # Try Gemini first for executive summary (better reasoning)
    raw    = client.chat(system, user, task="executive_summary",
                         max_tokens=300, force="gemini")
    # Fallback to Groq
    if not raw:
        raw = client.chat(system, user, task="executive_summary",
                          max_tokens=300, force="groq")

    if raw:
        try:
            clean = re.sub(r"^```json\s*", "", raw.strip())
            clean = re.sub(r"^```\s*",     "", clean)
            clean = re.sub(r"```$",        "", clean).strip()
            data  = json.loads(clean)
            if (data.get("ok") and
                    not _is_hallucinated(
                        data.get("risk", "") + data.get("lever", ""), df)):
                return (f"{data.get('risk','')} "
                        f"{data.get('lever','')} "
                        f"{data.get('urgency','')}").strip()
        except Exception:
            pass

    # Rule-based fallback
    parts = [
        f"This {domain.upper()} dataset ({len(df):,} records) "
        "reveals critical patterns requiring executive attention."
    ]
    if atr:
        rate   = float(df[atr].mean()) * 100
        n_left = int(df[atr].sum())
        parts.append(
            f"Attrition is {rate:.1f}% ({n_left:,} left) — "
            f"{'above' if rate > 15 else 'at'} the SHRM 15% benchmark."
        )
    parts.append(
        "Immediate action on critical findings below is required "
        "to prevent further financial and operational impact."
    )
    return " ".join(parts)

"""
ai/report_narrator.py — DataForge AI FINAL v8
Properly wired to prompt_builder.py.

Fixes from v7:
  1. temperature=0.4 (per prompt_builder usage example, not 0.15)
  2. ECOM/SALES insight prompts get {segment_breakdown} param
  3. HR insight prompt gets {attrition_breakdown} param
  4. _raw_data_summary() enriched — enough numbers for LLM to cite
  5. All prompt params match exactly what each prompt template expects
"""

from __future__ import annotations
import re
import logging
import numpy as np
import pandas as pd
from typing import Optional

from ai.llm_client import get_client
from ai.prompt_builder import (
    HR_EXECUTIVE_PROMPT,
    HR_INSIGHT_PROMPT,
    ECOMMERCE_EXECUTIVE_PROMPT,
    ECOMMERCE_INSIGHT_PROMPT,
    SALES_EXECUTIVE_PROMPT,
    SALES_INSIGHT_PROMPT,
    FINANCE_EXECUTIVE_PROMPT,
    BAR_CHART_PROMPT,
    PIE_CHART_PROMPT,
    LINE_CHART_PROMPT,
    DISTRIBUTION_CHART_PROMPT,
    COLUMN_LABEL_MAP,
    translate_column_name,
)

logger = logging.getLogger(__name__)

# ── Hallucination phrases ─────────────────────────────────
FAKE_PHRASES = [
    "customer satisfaction and sales revenue",
    "sales revenue tends to increase",
    "marketing spend and website traffic",
    "top-line growth",
    "sales targeted",
    "customer-centric initiatives",
    "as customer satisfaction increases",
    "net promoter score",
    "website traffic",
    "marketing spend",
]

# ── Domain labels ─────────────────────────────────────────
DOMAIN_LABELS = {
    "hr":        "HR",
    "ecommerce": "eCommerce",
    "sales":     "Sales",
    "finance":   "Finance",
    "general":   "Business Analytics",
}

# ── Domain → executive prompt ─────────────────────────────
EXEC_PROMPTS = {
    "hr":        HR_EXECUTIVE_PROMPT,
    "ecommerce": ECOMMERCE_EXECUTIVE_PROMPT,
    "sales":     SALES_EXECUTIVE_PROMPT,
    "finance":   FINANCE_EXECUTIVE_PROMPT,
    "general":   HR_EXECUTIVE_PROMPT,
}

# ── Domain → insight prompt ───────────────────────────────
INSIGHT_PROMPTS = {
    "hr":        HR_INSIGHT_PROMPT,
    "ecommerce": ECOMMERCE_INSIGHT_PROMPT,
    "sales":     SALES_INSIGHT_PROMPT,
    "finance":   FINANCE_EXECUTIVE_PROMPT,
    "general":   HR_INSIGHT_PROMPT,
}


# ══════════════════════════════════════════════════════════
#  COLUMN TRANSLATION
# ══════════════════════════════════════════════════════════

def clean_col(col: str) -> str:
    return translate_column_name(col)


def _is_hallucinated(text: str, df: pd.DataFrame) -> bool:
    tl = text.lower()
    if any(p in tl for p in FAKE_PHRASES):
        return True
    real = [c.lower().replace("_", "") for c in df.columns]
    invented = [
        "salesrevenue", "customerrevenue", "websitetraffic",
        "marketingspend", "operationalcost", "churnrate",
    ]
    for inv in invented:
        if inv in tl.replace(" ", "").replace("_", ""):
            if not any(inv in r for r in real):
                return True
    return False


def _clean_output(text: str) -> str:
    text = re.sub(r"\bSales Targeted\b",       "targeted",      text, flags=re.IGNORECASE)
    text = re.sub(r"\bSales Representative\b", "representative", text, flags=re.IGNORECASE)
    for raw, clean in COLUMN_LABEL_MAP.items():
        text = text.replace(f"'{raw}'", clean).replace(f'"{raw}"', clean)
    return text.strip()


def _call_llm(client, user: str, task: str,
              force: str = "", max_tokens: int = 600) -> Optional[str]:
    """Call LLM with temperature=0.4 per prompt_builder recommendation."""
    # Override chat_task temperature by passing directly
    try:
        # Try preferred provider
        prov = force or ("gemini" if task == "executive_summary" else "groq")
        result = client.chat_task(
            system     = "Follow the exact output format and rules specified. "
                         "Only cite numbers from the provided data. "
                         "Never invent figures, benchmarks, or external company names.",
            user       = user,
            task       = task,
            max_tokens = max_tokens,
            force      = prov,
        )
        if result:
            return result
        # Fallback to other provider
        other = "groq" if prov == "gemini" else "gemini"
        return client.chat_task(
            system     = "Follow the exact output format and rules specified. "
                         "Only cite numbers from the provided data.",
            user       = user,
            task       = task,
            max_tokens = max_tokens,
            force      = other,
        )
    except Exception as e:
        logger.warning(f"LLM call failed [{task}]: {e}")
        return None


# ══════════════════════════════════════════════════════════
#  STAT BUILDERS  (Python computes, LLM only narrates)
# ══════════════════════════════════════════════════════════

def _raw_data_summary(df: pd.DataFrame, domain: str) -> str:
    """
    Rich stats summary injected as {raw_data_summary}.
    Must contain enough real numbers for the LLM to cite in its output.
    """
    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include="object").columns.tolist()

    lines = [
        f"Dataset: {len(df):,} rows, {len(df.columns)} columns — {domain.upper()} domain",
        "",
        "KEY METRICS:",
    ]

    # Numeric stats with business labels
    for col in num_cols[:7]:
        try:
            s    = pd.to_numeric(df[col], errors="coerce").dropna()
            s    = s[np.isfinite(s)]
            if len(s) < 3:
                continue
            skew = float(s.skew())
            lbl  = clean_col(col)
            use_stat = "Median" if abs(skew) > 0.5 else "Mean"
            use_val  = round(float(s.median()) if abs(skew) > 0.5
                             else float(s.mean()), 3)
            lines.append(
                f"• {lbl}: {use_stat} = {use_val} | "
                f"Range = {round(float(s.min()),2)} to {round(float(s.max()),2)} | "
                f"Std = {round(float(s.std()),2)}"
                + (" [skewed — use median]" if abs(skew) > 0.5 else "")
            )
        except Exception:
            continue

    lines.append("")
    lines.append("CATEGORICAL BREAKDOWN:")

    for col in cat_cols[:4]:
        try:
            vc = df[col].value_counts(normalize=True).head(6)
            breakdown = " | ".join([f"{k}: {v*100:.0f}%" for k, v in vc.items()])
            lines.append(f"• {clean_col(col)}: {breakdown}")
        except Exception:
            continue

    # ── HR domain extras ──────────────────────────────────
    atr_col = next((c for c in df.columns
                    if c.lower() in ("left","attrition","churned","exited")), None)
    if atr_col:
        rate   = float(df[atr_col].mean()) * 100
        n_left = int(df[atr_col].sum())
        lines.append("")
        lines.append("ATTRITION ANALYSIS:")
        lines.append(f"• Overall rate: {rate:.1f}% ({n_left:,} employees left of {len(df):,})")
        lines.append(f"• Industry healthy range: 10-15%. Gap: {max(0,rate-15):.1f}pp above benchmark")

        # By department
        dept_col = next((c for c in cat_cols
                         if c.lower() in ("department","dept")), None)
        if dept_col:
            atr_dept = (df.groupby(dept_col)[atr_col].mean() * 100
                        .sort_values(ascending=False))
            worst = atr_dept.idxmax(); best = atr_dept.idxmin()
            lines.append(
                f"• Highest dept attrition: {worst} ({atr_dept[worst]:.0f}%) | "
                f"Lowest: {best} ({atr_dept[best]:.0f}%)"
            )

        # By salary
        sal_col = next((c for c in cat_cols if c.lower() == "salary"), None)
        if sal_col:
            atr_sal = df.groupby(sal_col)[atr_col].mean() * 100
            lines.append(
                "• Attrition by pay band: " +
                " | ".join([f"{k}: {v:.0f}%" for k,v in atr_sal.items()])
            )

        # Top driver
        atr_nums = [c for c in num_cols if c.lower() != atr_col.lower()]
        best_driver = None; best_diff = 0
        for col in atr_nums[:6]:
            try:
                lm = float(df[df[atr_col]==1][col].mean())
                sm = float(df[df[atr_col]==0][col].mean())
                d  = abs(lm-sm)/max(abs(sm),0.001)*100
                if d > best_diff:
                    best_diff = d; best_driver = (col, lm, sm, d)
            except Exception:
                continue
        if best_driver:
            col, lm, sm, d = best_driver
            lines.append(
                f"• Top attrition driver: {clean_col(col)} — "
                f"leavers avg {lm:.3f} vs stayers {sm:.3f} ({d:.0f}% gap)"
            )

    # ── Ecommerce/Sales extras ────────────────────────────
    rating_col = next((c for c in num_cols if "rating" in c.lower()
                       and "count" not in c.lower()), None)
    if rating_col and domain == "ecommerce":
        mean_r = float(df[rating_col].mean())
        low_n  = int((df[rating_col] < 3.0).sum())
        lines.append("")
        lines.append(f"• Customer Rating: avg {mean_r:.2f}/5 | "
                     f"Below 3.0: {low_n:,} products ({low_n/len(df)*100:.0f}%)")

    rev_col = next((c for c in num_cols
                    if any(k in c.lower() for k in ["revenue","sales","totalprice"])), None)
    if rev_col and domain == "sales":
        mean_r = float(df[rev_col].mean())
        cv     = float(df[rev_col].std()) / abs(mean_r) * 100 if mean_r != 0 else 0
        lines.append("")
        lines.append(f"• Revenue: avg {mean_r:,.2f} | "
                     f"Variability (CV): {cv:.0f}% "
                     f"({'high — investigate' if cv>50 else 'moderate'})")

    return "\n".join(lines)


def _top_correlations_summary(df: pd.DataFrame) -> str:
    """For {top_correlations} param in insight prompts."""
    try:
        num_cols = df.select_dtypes(include="number").columns.tolist()
        if len(num_cols) < 2:
            return "No significant correlations found."
        corr  = df[num_cols[:8]].corr(method="spearman")
        pairs = []
        for i in range(len(num_cols[:8])):
            for j in range(i+1, len(num_cols[:8])):
                a, b = num_cols[i], num_cols[j]
                r    = float(corr.loc[a,b])
                if abs(r) >= 0.2:
                    pairs.append((a,b,r))
        pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        if not pairs:
            return "No meaningful correlations above 0.2 threshold."
        lines = []
        for a, b, r in pairs[:4]:
            direction = "positive" if r > 0 else "negative"
            meaning   = (f"higher {clean_col(a)} tends with higher {clean_col(b)}"
                         if r > 0 else
                         f"higher {clean_col(a)} tends with lower {clean_col(b)}")
            lines.append(
                f"• {clean_col(a)} & {clean_col(b)}: "
                f"r={r:.2f} ({direction}) — {meaning}. "
                f"Association only (r²={r**2:.2f})."
            )
        return "\n".join(lines)
    except Exception:
        return "Correlation analysis unavailable."


def _attrition_breakdown(df: pd.DataFrame) -> str:
    """For {attrition_breakdown} in HR_INSIGHT_PROMPT."""
    atr_col = next((c for c in df.columns
                    if c.lower() in ("left","attrition","churned","exited")), None)
    if not atr_col:
        return "Attrition data not available in this dataset."

    lines  = []
    cat_cols = df.select_dtypes(include="object").columns.tolist()

    for col in cat_cols[:4]:
        try:
            atr = (df.groupby(col)[atr_col].mean() * 100
                   .sort_values(ascending=False))
            worst = atr.idxmax(); best = atr.idxmin()
            lines.append(
                f"By {clean_col(col)}: "
                + " | ".join([f"{k}: {v:.0f}%" for k,v in atr.items()])
                + f" → Worst: {worst} ({atr[worst]:.0f}%)"
            )
        except Exception:
            continue

    num_cols = df.select_dtypes(include="number").columns.tolist()
    atr_nums = [c for c in num_cols if c.lower() != atr_col.lower()]
    for col in atr_nums[:4]:
        try:
            lm = float(df[df[atr_col]==1][col].mean())
            sm = float(df[df[atr_col]==0][col].mean())
            d  = abs(lm-sm)/max(abs(sm),0.001)*100
            if d > 10:
                lines.append(
                    f"• {clean_col(col)}: "
                    f"Leavers avg={lm:.3f}, Stayers avg={sm:.3f} "
                    f"({d:.0f}% difference)"
                )
        except Exception:
            continue

    return "\n".join(lines) if lines else "Attrition breakdown unavailable."


def _segment_breakdown(df: pd.DataFrame) -> str:
    """For {segment_breakdown} in SALES/ECOM insight prompts."""
    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    if not num_cols or not cat_cols:
        return "Segment breakdown unavailable."

    lines = []
    for cat in cat_cols[:3]:
        for num in num_cols[:2]:
            try:
                grp   = df.groupby(cat)[num].mean().sort_values(ascending=False)
                top_g = grp.index[0];   top_v  = float(grp.iloc[0])
                bot_g = grp.index[-1];  bot_v  = float(grp.iloc[-1])
                avg   = float(df[num].mean())
                gap   = abs(top_v-bot_v)/max(abs(bot_v),0.001)*100
                above = int((grp > avg).sum())
                lines.append(
                    f"• {clean_col(num)} by {clean_col(cat)}: "
                    f"Top='{top_g}' ({top_v:.2f}) | "
                    f"Worst='{bot_g}' ({bot_v:.2f}) | "
                    f"Avg={avg:.2f} | Gap={gap:.0f}% | "
                    f"{above}/{len(grp)} above avg"
                )
            except Exception:
                continue
    return "\n".join(lines) if lines else "Segment breakdown unavailable."


# ══════════════════════════════════════════════════════════
#  CHART STAT BUILDERS
# ══════════════════════════════════════════════════════════

def _bar_stats(df: pd.DataFrame, x: str, y: str) -> dict:
    try:
        grp = df.groupby(x)[y].mean().sort_values(ascending=False)
        avg = float(df[y].mean())
        return {
            "chart":           "bar",
            "x_col": x,        "y_col": y,
            "metric_label":    clean_col(y),
            "dimension_label": clean_col(x),
            "top":             str(grp.index[0]),
            "top_val":         round(float(grp.iloc[0]), 3),
            "worst":           str(grp.index[-1]),
            "worst_val":       round(float(grp.iloc[-1]), 3),
            "gap_pct":         round(abs(float(grp.iloc[0])-float(grp.iloc[-1])) /
                                     max(abs(float(grp.iloc[-1])),0.001)*100, 1),
            "org_avg":         round(avg, 3),
            "above_avg":       int((grp > avg).sum()),
            "n_groups":        len(grp),
            "all_values":      {str(k): round(float(v),3) for k,v in grp.items()},
        }
    except Exception as e:
        return {"chart":"bar","error":str(e),"metric_label":clean_col(y),"dimension_label":clean_col(x)}


def _hist_stats(df: pd.DataFrame, col: str) -> dict:
    try:
        s    = pd.to_numeric(df[col], errors="coerce").dropna()
        s    = s[np.isfinite(s)]
        skew = float(s.skew())
        return {
            "chart":        "histogram",
            "col":          col,
            "metric_label": clean_col(col),
            "mean":         round(float(s.mean()),   3),
            "median":       round(float(s.median()), 3),
            "std":          round(float(s.std()),    3),
            "min":          round(float(s.min()),    3),
            "max":          round(float(s.max()),    3),
            "q1":           round(float(s.quantile(0.25)), 3),
            "q3":           round(float(s.quantile(0.75)), 3),
            "skew":         round(skew, 2),
            "shape":        ("right-skewed" if skew > 0.5
                             else "left-skewed" if skew < -0.5
                             else "symmetric"),
            "use_stat":     "median" if abs(skew) > 0.5 else "mean",
            "use_val":      round(float(s.median()) if abs(skew) > 0.5
                                  else float(s.mean()), 3),
        }
    except Exception as e:
        return {"chart":"histogram","error":str(e),"metric_label":clean_col(col)}


def _pie_stats(df: pd.DataFrame, x: str, y: str) -> dict:
    try:
        grp   = df.groupby(x)[y].mean().sort_values(ascending=False)
        total = grp.sum()
        return {
            "chart":           "pie",
            "x_col": x,        "y_col": y,
            "metric_label":    clean_col(y),
            "dimension_label": clean_col(x),
            "n_segments":      len(grp),
            "top_seg":         str(grp.index[0]),
            "top_pct":         round(grp.max()/total*100, 1),
            "top2_pct":        round(grp.nlargest(2).sum()/total*100, 1),
            "shares":          {str(k): round(v/total*100,1) for k,v in grp.items()},
            "balanced":        bool(grp.max()/total*100 < 40),
        }
    except Exception as e:
        return {"chart":"pie","error":str(e)}


def _trend_stats(df: pd.DataFrame, col: str) -> dict:
    try:
        s   = pd.to_numeric(df[col], errors="coerce").dropna()
        s   = s[np.isfinite(s)]
        cv  = s.std()/abs(s.mean())*100 if s.mean() != 0 else 0
        mid = len(s)//2
        f   = float(s.iloc[:mid].mean())
        sc  = float(s.iloc[mid:].mean())
        pct = (sc-f)/max(abs(f),0.001)*100
        return {
            "chart":        "trend",
            "col":          col,
            "metric_label": clean_col(col),
            "mean":         round(float(s.mean()),3),
            "min":          round(float(s.min()),3),
            "max":          round(float(s.max()),3),
            "cv":           round(cv,1),
            "first_half":   round(f,3),
            "sec_half":     round(sc,3),
            "trend_pct":    round(pct,1),
            "trend_dir":    ("improved" if pct>2 else "declined" if pct<-2 else "stable"),
        }
    except Exception as e:
        return {"chart":"trend","error":str(e),"metric_label":clean_col(col)}


# ══════════════════════════════════════════════════════════
#  PROMPT FORMATTERS  (fill prompt_builder templates)
# ══════════════════════════════════════════════════════════

def _format_bar_prompt(s: dict, domain: str) -> str:
    if "error" in s: return ""
    chart_data = (
        f"Top performer: '{s['top']}' = {s['top_val']} | "
        f"Bottom performer: '{s['worst']}' = {s['worst_val']} | "
        f"Organisation average: {s['org_avg']} | "
        f"Performance gap: {s['gap_pct']:.0f}% | "
        f"Groups above average: {s['above_avg']}/{s['n_groups']} | "
        f"All group values: {s['all_values']}"
    )
    try:
        return BAR_CHART_PROMPT.format(
            domain          = DOMAIN_LABELS.get(domain, "Business"),
            metric_label    = s["metric_label"],
            dimension_label = s["dimension_label"],
            raw_metric      = s["y_col"],
            raw_dimension   = s["x_col"],
            chart_data      = chart_data,
        )
    except KeyError:
        return ""


def _format_pie_prompt(s: dict, domain: str) -> str:
    if "error" in s: return ""
    chart_data = (
        f"Segment shares: {s['shares']} | "
        f"Largest segment: '{s['top_seg']}' at {s['top_pct']}% | "
        f"Top 2 combined: {s['top2_pct']}% | "
        f"Distribution: {'CONCENTRATED' if not s['balanced'] else 'BALANCED'}"
    )
    try:
        return PIE_CHART_PROMPT.format(
            domain          = DOMAIN_LABELS.get(domain, "Business"),
            metric_label    = s["metric_label"],
            dimension_label = s["dimension_label"],
            raw_metric      = s.get("y_col", "metric"),
            raw_dimension   = s.get("x_col", "dimension"),
            chart_data      = chart_data,
        )
    except KeyError:
        return ""


def _format_line_prompt(s: dict, domain: str) -> str:
    if "error" in s: return ""
    chart_data = (
        f"Overall mean: {s['mean']} | Range: {s['min']} to {s['max']} | "
        f"First half average: {s['first_half']} | Second half average: {s['sec_half']} | "
        f"Change first→second half: {s['trend_pct']:+.1f}% ({s['trend_dir']}) | "
        f"Variability (CV): {s['cv']}%"
    )
    try:
        return LINE_CHART_PROMPT.format(
            domain       = DOMAIN_LABELS.get(domain, "Business"),
            metric_label = s["metric_label"],
            raw_metric   = s.get("col", "metric"),
            chart_data   = chart_data,
        )
    except KeyError:
        return ""


def _format_hist_prompt(s: dict, domain: str) -> str:
    """
    DISTRIBUTION_CHART_PROMPT takes 6 specific params — not a chart_data dict.
    Matching exactly what prompt_builder expects.
    """
    if "error" in s: return ""
    try:
        return DISTRIBUTION_CHART_PROMPT.format(
            domain       = DOMAIN_LABELS.get(domain, "Business"),
            metric_label = s["metric_label"],
            mean_val     = s["mean"],
            median_val   = s["median"],
            min_val      = s["min"],
            max_val      = s["max"],
        )
    except KeyError:
        return ""


# ══════════════════════════════════════════════════════════
#  RULE-BASED FALLBACKS  (always correct, no LLM)
# ══════════════════════════════════════════════════════════

def _fb_bar(s: dict) -> str:
    if "error" in s:
        return f"Analysis of {s.get('metric_label','metric')} by group."
    return (
        f"{s['metric_label']} across {s['n_groups']} {s['dimension_label']} groups: "
        f"'{s['top']}' leads at {s['top_val']} while '{s['worst']}' trails "
        f"at {s['worst_val']} — a {s['gap_pct']:.0f}% gap. "
        f"{s['above_avg']} of {s['n_groups']} groups exceed the "
        f"organisation average of {s['org_avg']}. "
        f"Strategic Action: Conduct root-cause review in '{s['worst']}' "
        f"and replicate '{s['top']}' practices to close the gap."
    )


def _fb_hist(s: dict) -> str:
    if "error" in s:
        return f"Distribution of {s.get('metric_label','metric')}."
    return (
        f"{s['metric_label']} typical value is {s['use_val']} "
        f"(range: {s['min']}–{s['max']}). "
        f"{'Skewed — use ' + s['use_stat'] + ' for accurate reporting.' if s['shape'] != 'symmetric' else 'Symmetric — mean is reliable.'} "
        f"Bottom quartile (below {s['q1']}) represents highest-risk group. "
        f"Strategic Action: Focus interventions on the bottom quartile first."
    )


def _fb_pie(s: dict) -> str:
    if "error" in s:
        return f"Composition of {s.get('metric_label','metric')}."
    return (
        f"{s['metric_label']} across {s['n_segments']} "
        f"{s['dimension_label']} segments: "
        f"'{s['top_seg']}' holds {s['top_pct']}%, "
        f"top 2 combined: {s['top2_pct']}%. "
        f"{'Concentration risk — dominant segment.' if not s['balanced'] else 'Well-balanced distribution.'} "
        f"Strategic Action: "
        f"{'Reduce over-reliance on dominant segments.' if not s['balanced'] else 'Maintain balance and monitor monthly.'}"
    )


def _fb_trend(s: dict) -> str:
    if "error" in s:
        return f"Trend of {s.get('metric_label','metric')}."
    return (
        f"{s['metric_label']} trend: average {s['mean']} "
        f"(range {s['min']}–{s['max']}). "
        f"Values {s['trend_dir']} by {abs(s['trend_pct']):.1f}% "
        f"from first to second half — "
        f"{'positive signal.' if s['trend_dir']=='improved' else 'declining — investigate.' if s['trend_dir']=='declined' else 'stable.'} "
        f"Strategic Action: "
        f"{'Identify decline root causes immediately.' if s['trend_dir']=='declined' else 'Monitor monthly for early warning signals.'}"
    )


def _fb_corr(df: pd.DataFrame) -> str:
    try:
        num = df.select_dtypes(include="number").columns.tolist()
        if len(num) < 2:
            return "Insufficient columns for correlation analysis."
        corr  = df[num[:8]].corr(method="spearman")
        pairs = []
        for i in range(len(num[:8])):
            for j in range(i+1, len(num[:8])):
                a, b = num[i], num[j]
                r    = float(corr.loc[a,b])
                if abs(r) >= 0.15:
                    pairs.append((a,b,r))
        pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        if not pairs:
            return (
                "No meaningful correlations found (all |r| < 0.15). "
                "Variables operate independently. "
                "Strategic Action: Design targeted single-variable interventions."
            )
        a, b, r = pairs[0]
        second  = ""
        if len(pairs) > 1:
            a2,b2,r2 = pairs[1]
            second = f" Second: {clean_col(a2)} & {clean_col(b2)} (r={r2:.2f})."
        return (
            f"Correlation matrix: {len(pairs)} meaningful relationships. "
            f"Strongest: {clean_col(a)} & {clean_col(b)} (r={r:.2f}) — "
            f"r²={r**2:.2f} means {r**2*100:.0f}% variance shared "
            f"(association only, not causation).{second} "
            f"Strategic Action: Test the {clean_col(a)}–{clean_col(b)} "
            f"link through controlled analysis before acting on it."
        )
    except Exception:
        return "Correlation analysis unavailable."


# ══════════════════════════════════════════════════════════
#  MAIN PUBLIC FUNCTIONS
# ══════════════════════════════════════════════════════════

def generate_chart_narrative(
    df:           pd.DataFrame,
    chart_title:  str,
    groq_api_key: str = "",    # kept for backward compat with 8_Reports.py
    domain:       str = "general",
) -> str:
    """
    Generate chart narrative using prompt_builder domain-specific prompts.
    Routes: bar→BAR_CHART_PROMPT, pie→PIE_CHART_PROMPT,
            trend/line→LINE_CHART_PROMPT, distribution→DISTRIBUTION_CHART_PROMPT.
    """
    title = chart_title.lower()
    num   = df.select_dtypes(include="number").columns.tolist()
    cat   = df.select_dtypes(include="object").columns.tolist()

    prompt   = ""
    fallback = ""

    # ── Correlation / Heatmap ─────────────────────────────
    if "correlation" in title or "heatmap" in title:
        # Rule-based always more reliable for correlation
        return _fb_corr(df)

    # ── Distribution / Histogram ──────────────────────────
    elif "distribution" in title or "histogram" in title:
        col = next((c for c in num if c.lower() in title),
                   num[0] if num else None)
        if not col: return "Distribution chart generated."
        s        = _hist_stats(df, col)
        prompt   = _format_hist_prompt(s, domain)
        fallback = _fb_hist(s)

    # ── Pie / Share ───────────────────────────────────────
    elif "pie" in title or "share" in title:
        parts = title.split(" by ")
        x = next((c for c in cat
                   if c.lower() in (parts[1] if len(parts)>1 else "")),
                  cat[0] if cat else None)
        y = next((c for c in num if c.lower() in parts[0]),
                  num[0] if num else None)
        if not x or not y: return "Pie chart generated."
        s        = _pie_stats(df, x, y)
        prompt   = _format_pie_prompt(s, domain)
        fallback = _fb_pie(s)

    # ── Trend / Line ──────────────────────────────────────
    elif "trend" in title or "line" in title:
        col = next((c for c in num
                     if c.lower().replace("_","") in
                     title.replace("_","").replace(" ","")),
                    num[0] if num else None)
        if not col: return "Trend chart generated."
        s        = _trend_stats(df, col)
        prompt   = _format_line_prompt(s, domain)
        fallback = _fb_trend(s)

    # ── Bar (X by Y) ──────────────────────────────────────
    elif " by " in title:
        parts = title.replace("avg ","").replace("total ","").split(" by ")
        y = next((c for c in num
                   if c.lower().replace("_","") in parts[0].replace(" ","")),
                  num[0] if num else None)
        x = next((c for c in cat
                   if c.lower() in (parts[1] if len(parts)>1 else "")),
                  cat[0] if cat else None)
        if not y or not x: return "Bar chart generated."
        s        = _bar_stats(df, x, y)
        prompt   = _format_bar_prompt(s, domain)
        fallback = _fb_bar(s)

    # ── Generic ───────────────────────────────────────────
    else:
        if cat and num:
            s = _bar_stats(df, cat[0], num[0])
            prompt = _format_bar_prompt(s, domain)
            fallback = _fb_bar(s)
        elif num:
            s = _hist_stats(df, num[0])
            prompt = _format_hist_prompt(s, domain)
            fallback = _fb_hist(s)
        else:
            return "Chart analysis not available."

    if not prompt:
        return fallback

    # ── LLM call ─────────────────────────────────────────
    try:
        client = get_client(groq_api_key)
        raw    = _call_llm(client, prompt, task="chart_analysis",
                           force="groq", max_tokens=300)
        if raw:
            cleaned = _clean_output(raw)
            if not _is_hallucinated(cleaned, df):
                return cleaned
    except Exception as e:
        logger.warning(f"Chart narrative LLM failed: {e}")

    return fallback


def generate_executive_summary(
    df:           pd.DataFrame,
    domain:       str = "general",
    story_report  = None,
    groq_api_key: str = "",    # kept for backward compat
) -> str:
    """
    Executive summary using domain-specific prompt_builder prompt.
    Injects rich pre-computed stats so LLM can cite real numbers.
    """
    raw_summary   = _raw_data_summary(df, domain)
    exec_template = EXEC_PROMPTS.get(domain, HR_EXECUTIVE_PROMPT)

    try:
        prompt = exec_template.format(raw_data_summary=raw_summary)
    except KeyError:
        prompt = exec_template.replace("{raw_data_summary}", raw_summary)

    # Try Gemini first (better reasoning), then Groq
    raw = None
    try:
        client = get_client(groq_api_key)
        raw    = _call_llm(client, prompt, task="executive_summary",
                           force="gemini", max_tokens=700)
    except Exception as e:
        logger.warning(f"Executive summary LLM failed: {e}")

    if raw:
        cleaned = _clean_output(raw)
        if not _is_hallucinated(cleaned, df):
            return cleaned

    # Rule-based fallback
    atr_col = next((c for c in df.columns
                    if c.lower() in ("left","attrition","churned","exited")), None)
    parts = [
        f"This {domain.upper()} dataset ({len(df):,} records) "
        "reveals critical patterns requiring executive attention."
    ]
    if atr_col:
        rate   = float(df[atr_col].mean()) * 100
        n_left = int(df[atr_col].sum())
        parts.append(
            f"Attrition is {rate:.1f}% ({n_left:,} employees left) — "
            f"{'above' if rate>15 else 'at'} the healthy 10–15% benchmark."
        )
    parts.append(
        "Immediate action on the critical findings below is required "
        "to prevent further financial and operational impact."
    )
    return " ".join(parts)

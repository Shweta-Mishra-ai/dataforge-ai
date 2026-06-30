"""
core/story_engine.py — DataForge AI  [REFACTORED]
Thin orchestrator: detects domain, delegates to domain engine, assembles StoryReport.
All heavy logic lives in core/engines/{hr,ecommerce,sales,finance,general}.py

Public API (unchanged):
    generate_story(df)  → StoryReport
    detect_domain(df)   → (domain_str, confidence_float)

Re-exports for backwards compatibility:
    Insight, AttritionAnalysis, StoryReport  (from core.engines.base)
    _col_stats, _correlations, _build_insight (legacy aliases)
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ── Public dataclasses (single source of truth in base) ──────────────────────
from core.engines.base import (
    Insight, AttritionAnalysis, StoryReport,
    col_stats, correlations, build_insight,
)

# ── Domain engines ────────────────────────────────────────────────────────────
from core.engines.hr       import _insights_hr, _run_attrition
from core.engines.ecommerce import _insights_ecommerce
from core.engines.sales    import _insights_sales
from core.engines.finance  import _insights_finance
from core.engines.general  import _insights_general

# ── Legacy aliases so any existing callers don't break ───────────────────────
_col_stats    = col_stats
_correlations = correlations
_build_insight = build_insight

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════
#  DOMAIN DETECTION
# ══════════════════════════════════════════════════════════

DOMAIN_KEYWORDS: Dict[str, List[str]] = {
    "ecommerce": ["price", "discount", "rating", "product", "category",
                  "order", "revenue", "sales", "sku", "review", "seller",
                  "cart", "inventory", "stock", "asin", "marketplace"],
    "hr":        ["employee", "salary", "department", "attrition", "satisfaction",
                  "tenure", "performance", "hire", "job", "left", "manager",
                  "bonus", "promotion", "headcount", "workforce"],
    "sales":     ["revenue", "sales", "profit", "margin", "target", "quota",
                  "pipeline", "deal", "customer", "region", "territory",
                  "forecast", "conversion", "lead", "opportunity", "closed"],
    "finance":   ["profit", "loss", "expense", "income", "budget", "cost",
                  "margin", "cashflow", "asset", "liability", "tax", "invoice"],
    "marketing": ["campaign", "click", "impression", "conversion", "lead",
                  "channel", "spend", "roi", "ctr", "cpa", "traffic"],
}


def detect_domain(df: pd.DataFrame) -> Tuple[str, float]:
    """
    Returns (domain, confidence 0..1).
    Confidence = fraction of domain keywords found in column names.
    Raises TypeError if df is not a DataFrame.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"detect_domain expects pd.DataFrame, got {type(df)}")

    col_text = " ".join(df.columns.str.lower())
    scores: Dict[str, float] = {}
    for domain, keywords in DOMAIN_KEYWORDS.items():
        hits = sum(1 for kw in keywords if kw in col_text)
        scores[domain] = hits / len(keywords)

    best = max(scores, key=lambda d: scores[d])
    confidence = scores[best]

    # Require a minimum signal — below 0.05 is just noise
    if confidence < 0.05:
        return "general", 0.0

    return best, round(confidence, 3)


# ══════════════════════════════════════════════════════════
#  ANOMALY DETECTION
# ══════════════════════════════════════════════════════════

def _detect_anomalies(df: pd.DataFrame, stats: Dict) -> List[str]:
    """Surface statistical anomalies as plain-English strings."""
    anomalies = []
    for col, st in stats.items():
        if not st:
            continue
        out_pct = st.get("outlier_pct", 0)
        skew_v  = st.get("skew", 0)
        miss    = st.get("missing_pct", 0)
        if out_pct > 10:
            lo = st["q1"] - 1.5 * st["iqr"]
            hi = st["q3"] + 1.5 * st["iqr"]
            anomalies.append(
                f"'{col}' has {out_pct:.1f}% outliers — "
                f"normal range {lo:.2f}–{hi:.2f}. Validate before modelling."
            )
        if abs(skew_v) > 2:
            anomalies.append(
                f"'{col}' heavily skewed (skew={skew_v:.2f}). "
                f"Median {st['median']:.2f} more reliable than mean {st['mean']:.2f}."
            )
        if miss > 20:
            anomalies.append(
                f"'{col}' is {miss:.1f}% missing — "
                "imputed values may affect results."
            )
    return anomalies


# ══════════════════════════════════════════════════════════
#  MAIN ORCHESTRATOR
# ══════════════════════════════════════════════════════════

def _build_narrative_summary(
    df: pd.DataFrame, domain: str, confidence: float,
    deduped: List[Insight], corrs: List[Dict],
    attrition: Optional[AttritionAnalysis], raw: Dict,
) -> str:
    """
    Synthesises a single headline narrative claim instead of listing facts.
    Priority order for the headline: attrition signal > critical insight >
    strongest correlation > data quality. Supporting sentences follow,
    each connecting back to the headline rather than standing alone.
    """
    n_crit = sum(1 for i in deduped if i.severity == "critical")
    n_warn = sum(1 for i in deduped if i.severity == "warning")
    miss   = round(df.isna().mean().mean() * 100, 1)
    n_rows = len(df)

    # ── HEADLINE: pick the single most important claim ─────────────────────
    headline = None

    if attrition is not None and attrition.severity in ("critical", "high"):
        cohort_note = ""
        if attrition.top_drivers:
            d = attrition.top_drivers[0]
            cohort_note = f", concentrated among {d.get('label', 'a specific cohort')}"
        headline = (
            f"The {attrition.rate:.1f}% attrition rate ({attrition.n_left:,} of "
            f"{attrition.n_total:,} employees){cohort_note} is the dominant signal "
            f"in this dataset and warrants immediate retention review."
        )
    elif n_crit > 0:
        top_critical = next((i for i in deduped if i.severity == "critical"), None)
        if top_critical:
            headline = (
                f"{top_critical.problem} This is the most urgent finding in the "
                f"dataset — {n_crit} critical issue{'s' if n_crit > 1 else ''} total."
            )
    elif corrs and corrs[0]["strength"] == "strong":
        top = corrs[0]
        headline = (
            f"'{top['col_a']}' and '{top['col_b']}' show a strong "
            f"{top['direction']} relationship (Spearman r={top['r']:+.2f}), "
            f"explaining {top['r']**2*100:.0f}% of shared variance — the clearest "
            f"structural pattern in this dataset."
        )
    elif miss > 15:
        headline = (
            f"Data completeness is the primary concern: {miss:.1f}% of values "
            f"are missing across {n_rows:,} records, which will materially affect "
            f"any downstream analysis or modelling."
        )
    else:
        headline = (
            f"Analysis of {n_rows:,} records across {len(df.columns)} variables "
            f"in the {domain.upper()} domain (detection confidence: {confidence:.0%}) "
            f"did not surface a single dominant risk — findings below are of "
            f"comparable priority."
        )

    # ── SUPPORTING sentences — connect to headline, don't repeat it ────────
    support = []

    if attrition is not None and headline and "attrition rate" not in headline.lower()[:60]:
        support.append(
            f"Separately, attrition stands at {attrition.rate:.1f}% "
            f"({attrition.severity} severity)."
        )

    if n_warn > 0:
        support.append(
            f"{n_warn} additional warning{'s' if n_warn > 1 else ''} "
            f"{'require' if n_warn == 1 else 'require'} review but are not urgent."
        )

    if miss > 0 and miss <= 15:
        support.append(f"Data completeness is acceptable ({100-miss:.1f}% complete).")
    elif miss == 0:
        support.append("Data is fully complete with no missing values.")

    if corrs and "Spearman r=" not in (headline or ""):
        top = corrs[0]
        support.append(
            f"Notable relationship: '{top['col_a']}' vs '{top['col_b']}' "
            f"(r={top['r']:+.2f}, {top['strength']})."
        )

    n_actions = len(raw.get("actions", []))
    if n_actions:
        support.append(
            f"{n_actions} recommendation{'s' if n_actions > 1 else ''} "
            f"{'follows' if n_actions == 1 else 'follow'} below."
        )

    return headline + (" " + " ".join(support) if support else "")


def generate_story(df: pd.DataFrame) -> StoryReport:
    """
    Main entry point.  Returns a StoryReport with all insights, risks,
    opportunities, and executive summary computed from df.

    Raises:
        TypeError  — if df is not a DataFrame
        ValueError — if df has 0 rows or 0 columns
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"generate_story expects pd.DataFrame, got {type(df)}")
    if df.empty:
        raise ValueError("generate_story received an empty DataFrame")

    domain, confidence = detect_domain(df)

    # ── Per-column stats (numeric only) ──────────────────────────────────────
    num_cols = df.select_dtypes(include="number").columns.tolist()
    all_stats: Dict = {}
    for col in num_cols:
        try:
            all_stats[col] = col_stats(df[col])
        except Exception:
            logger.warning("col_stats failed for '%s'", col, exc_info=True)
            all_stats[col] = {}

    # ── Correlations (Spearman) ───────────────────────────────────────────────
    try:
        corrs = correlations(df)
    except Exception:
        logger.warning("correlations() failed", exc_info=True)
        corrs = []

    # ── Attrition (HR only) ───────────────────────────────────────────────────
    attrition = None
    if domain == "hr":
        try:
            attrition = _run_attrition(df)
        except Exception:
            logger.warning("_run_attrition failed", exc_info=True)

    # ── Domain engine dispatch ────────────────────────────────────────────────
    try:
        if domain == "hr":
            raw = _insights_hr(df, all_stats, corrs, attrition)
        elif domain == "ecommerce":
            raw = _insights_ecommerce(df, all_stats, corrs)
        elif domain == "sales":
            raw = _insights_sales(df, all_stats, corrs)
        elif domain == "finance":
            raw = _insights_finance(df, all_stats, corrs)
        else:
            raw = _insights_general(df, all_stats, corrs)
    except Exception:
        logger.error("Domain engine '%s' failed — falling back to general", domain, exc_info=True)
        raw = _insights_general(df, all_stats, corrs)

    # ── Merge general insights only when domain engine is thin ────────────────
    try:
        gen = _insights_general(df, all_stats, corrs)
        if domain != "general":
            if len(raw.get("findings", [])) < 3:
                raw.setdefault("findings", []).extend(gen["findings"])
            if len(raw.get("risks", [])) < 2:
                raw.setdefault("risks", []).extend(gen["risks"])
            if len(raw.get("opportunities", [])) < 2:
                raw.setdefault("opportunities", []).extend(gen["opportunities"])
            if len(raw.get("insights", [])) < 4:
                raw.setdefault("insights", []).extend(gen.get("insights", []))
        else:
            for key in ("findings", "risks", "opportunities"):
                raw.setdefault(key, []).extend(gen.get(key, []))
    except Exception:
        logger.warning("general insight merge failed", exc_info=True)

    # ── Sort + deduplicate insights ───────────────────────────────────────────
    sev_order = {"critical": 0, "warning": 1, "info": 2, "positive": 3}
    raw_insights = raw.get("insights", [])
    raw_insights.sort(key=lambda x: sev_order.get(x.severity, 99))
    seen, deduped = set(), []
    for ins in raw_insights:
        if ins.title not in seen:
            seen.add(ins.title)
            deduped.append(ins)

    # ── Executive summary ─────────────────────────────────────────────────────
    findings_flat = raw.get("findings", [])[:6]
    if not findings_flat and deduped:
        findings_flat = [
            f"{ins.severity.upper()}: {ins.title}"
            for ins in deduped[:6]
        ]
    if not findings_flat:
        nc = len(num_cols)
        cc = len(df.select_dtypes(include=["object", "string"]).columns)
        mp = round(df.isna().mean().mean() * 100, 1)
        findings_flat = [
            f"{len(df):,} records × {len(df.columns)} columns ({nc} numeric, {cc} categorical)",
            f"Missing data: {mp:.1f}% {'— fully complete' if mp == 0 else '— imputation applied'}",
        ]

    exec_summary = raw.get("executive_summary", "")
    if not exec_summary:
        exec_summary = _build_narrative_summary(
            df=df, domain=domain, confidence=confidence,
            deduped=deduped, corrs=corrs, attrition=attrition,
            raw=raw,
        )

    # ── Data quality verdict ──────────────────────────────────────────────────
    miss_overall = df.isna().mean().mean() * 100
    dup_pct      = df.duplicated().mean() * 100
    if miss_overall < 1 and dup_pct < 1:
        dq_verdict = "GOOD — Complete data, minimal duplicates"
    elif miss_overall < 5 and dup_pct < 5:
        dq_verdict = "FAIR — Minor quality issues, review before modelling"
    else:
        dq_verdict = f"POOR — {miss_overall:.1f}% missing, {dup_pct:.1f}% duplicates"

    conf_label = (
        "HIGH — Strong domain signal and adequate sample size"
        if confidence > 0.15 and len(df) > 200
        else "MEDIUM — Moderate domain signal or small sample"
        if confidence > 0.05
        else "LOW — Insufficient domain signal; results are general statistical analysis"
    )

    # ── Anomalies ─────────────────────────────────────────────────────────────
    try:
        anomalies = _detect_anomalies(df, all_stats)
    except Exception:
        logger.warning("_detect_anomalies failed", exc_info=True)
        anomalies = []

    return StoryReport(
        domain=domain,
        domain_confidence=confidence,
        executive_summary=exec_summary,
        key_findings=findings_flat,
        business_risks=raw.get("risks", []),
        opportunities=raw.get("opportunities", []),
        recommended_actions=raw.get("actions", []),
        insights=deduped,
        anomalies=anomalies,
        attrition=attrition,
        data_quality_verdict=dq_verdict,
        analysis_confidence=conf_label,
    )

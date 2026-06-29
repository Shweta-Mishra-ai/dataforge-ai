"""
core/engines/base.py
Shared dataclasses, helpers, and stat primitives used by all domain engines.
Single source of truth — import from here, never from story_engine directly.
"""
from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════
#  DATACLASSES
# ══════════════════════════════════════════════════════════

@dataclass
class Insight:
    title:    str
    problem:  str
    cause:    str
    evidence: str
    action:   str
    impact:   str
    severity: str        # critical | warning | positive | info
    category: str = "general"


@dataclass
class AttritionAnalysis:
    rate:             float
    n_left:           int
    n_total:          int
    severity:         str
    top_drivers:      List[Dict]
    dept_attrition:   Dict
    salary_attrition: Dict
    n_flight_risk:    int
    flight_risk_pct:  float
    cost_estimate:    str
    interpretation:   str


@dataclass
class StoryReport:
    domain:              str = "general"
    domain_confidence:   float = 0.0
    executive_summary:   str = ""
    key_findings:        List[str] = field(default_factory=list)
    business_risks:      List[str] = field(default_factory=list)
    opportunities:       List[str] = field(default_factory=list)
    recommended_actions: List[str] = field(default_factory=list)
    insights:            List[Insight] = field(default_factory=list)
    anomalies:           List[str] = field(default_factory=list)
    attrition:           Optional[AttritionAnalysis] = None
    data_quality_verdict: str = ""
    analysis_confidence:  str = ""


# ══════════════════════════════════════════════════════════
#  STAT HELPERS  (Spearman throughout — non-parametric)
# ══════════════════════════════════════════════════════════

def col_stats(s: pd.Series) -> Dict:
    """
    Per-column statistics. Uses robust (non-parametric) measures throughout.
    Raises on empty series — callers must guard.
    """
    s = s.dropna()
    if len(s) < 3:
        return {}
    n = len(s)
    q1, med, q3 = float(np.percentile(s, 25)), float(np.percentile(s, 50)), float(np.percentile(s, 75))
    iqr = q3 - q1
    mean_v = float(s.mean())
    std_v  = float(s.std())
    try:
        skew_v = float(scipy_stats.skew(s))
    except Exception:
        skew_v = 0.0
        logger.debug("skew computation failed for column — data nearly constant")
    try:
        kurt_v = float(scipy_stats.kurtosis(s))
    except Exception:
        kurt_v = 0.0
        logger.debug("kurtosis computation failed for column — data nearly constant")
    out_mask = (s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)
    out_pct  = float(out_mask.mean() * 100)
    cv       = std_v / abs(mean_v) if mean_v != 0 else 0.0
    return {
        "n": n, "mean": mean_v, "median": med,
        "std": std_v, "cv": cv,
        "q1": q1, "q3": q3, "iqr": iqr,
        "min": float(s.min()), "max": float(s.max()),
        "skew": skew_v, "kurtosis": kurt_v,
        "outlier_pct": out_pct,
        "outliers": int(out_mask.sum()),
        "p10": float(np.percentile(s, 10)),
        "p90": float(np.percentile(s, 90)),
    }


def correlations(df: pd.DataFrame, min_r: float = 0.25) -> List[Dict]:
    """
    Spearman correlations — robust to non-normal distributions.
    Returns pairs sorted by |r| descending, filtered to |r| >= min_r.
    Raises ValueError if df has fewer than 2 numeric columns.
    """
    num_cols = df.select_dtypes(include="number").columns.tolist()
    if len(num_cols) < 2:
        return []
    results = []
    for i in range(len(num_cols)):
        for j in range(i + 1, len(num_cols)):
            a, b = num_cols[i], num_cols[j]
            common = df[[a, b]].dropna()
            if len(common) < 10:
                continue
            try:
                r, p = scipy_stats.spearmanr(common[a], common[b])
                r = float(r)
                if abs(r) < min_r:
                    continue
                strength = (
                    "strong"   if abs(r) >= 0.7 else
                    "moderate" if abs(r) >= 0.4 else
                    "weak"
                )
                results.append({
                    "col_a": a, "col_b": b,
                    "r": round(r, 4), "p": round(float(p), 6),
                    "r2": round(r ** 2, 4),
                    "strength": strength,
                    "direction": "positive" if r > 0 else "negative",
                    "significant": p < 0.05,
                })
            except Exception:
                logger.warning("Spearman failed for %s/%s", a, b, exc_info=True)
    results.sort(key=lambda x: abs(x["r"]), reverse=True)
    return results


def build_insight(
    title: str, problem: str, cause: str,
    evidence: str, action: str, impact: str,
    severity: str = "info", category: str = "general",
) -> Insight:
    return Insight(
        title=title, problem=problem, cause=cause,
        evidence=evidence, action=action, impact=impact,
        severity=severity, category=category,
    )

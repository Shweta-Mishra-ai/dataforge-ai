"""
core/engines/general.py — General / unknown domain engine.
Handles any dataset without a detectable domain.
Provides outlier flagging, correlation surfacing, and generic opportunity detection.
"""
from __future__ import annotations
import logging
from typing import Dict, List

import numpy as np
import pandas as pd

from core.engines.base import Insight, build_insight, col_stats, correlations

logger = logging.getLogger(__name__)


def _insights_general(df: pd.DataFrame, stats: Dict, corrs: List) -> Dict:
    findings, risks, opps, actions = [], [], [], []
    insights = []

    num_cols = df.select_dtypes(include="number").columns.tolist()

    for col in list(stats.keys())[:8]:
        st = stats.get(col, {})
        if not st:
            continue
        skew    = st.get("skew", 0)
        out_pct = st.get("outlier_pct", 0)
        mean    = st.get("mean", 0)
        median  = st.get("median", 0)

        if out_pct > 10:
            insights.append(build_insight(
                title="'{}': {:.0f}% Outliers — Data Quality Issue".format(col, out_pct),
                problem="{:.0f}% of '{}' values are statistical outliers".format(out_pct, col),
                cause="Data entry errors, measurement anomalies, or genuine extreme values",
                evidence="IQR method: {:.0f}% outliers. Range: {:.2f} to {:.2f}".format(
                    out_pct, st.get("min", 0), st.get("max", 0)),
                action="1. Inspect outlier records  2. Determine error or genuine  "
                       "3. Cap or remove confirmed errors  4. Document decisions",
                impact="Outliers distort all statistical analyses and reduce ML accuracy",
                severity="warning", category="data_quality"
            ))
        if abs(skew) > 1.5:
            findings.append(
                "'{}' is {}-skewed (mean {:.2f} vs median {:.2f}). "
                "Report median for this column.".format(
                    col, "right" if skew > 0 else "left", mean, median))

    for corr in corrs[:3]:
        if corr.get("strength") in ("strong", "moderate"):
            findings.append(
                "{} {} relationship: '{}' and '{}' (r={:.2f}) — "
                "statistically significant".format(
                    corr["strength"].title(), corr["direction"],
                    corr["col_a"], corr["col_b"], corr["r"]))

    # ── Generic opportunity detection — works on ANY column names ────────────
    # Looks for P90/median uplift potential, high-variability improvement room
    for col in num_cols[:8]:
        try:
            s = df[col].dropna()
            if len(s) < 20:
                continue
            p10  = float(s.quantile(0.10))
            p50  = float(s.quantile(0.50))
            p90  = float(s.quantile(0.90))
            mean_v = float(s.mean())
            cv   = s.std() / abs(mean_v) * 100 if mean_v != 0 else 0

            # Uplift opportunity: large spread between P10 and P90
            if p50 > 0 and p90 / p50 >= 2.0 and cv > 40:
                uplift_pct = (p90 - p50) / p50 * 100
                opps.append(
                    f"'{col}': Top decile ({p90:.2g}) is {p90/p50:.1f}× the median "
                    f"({p50:.2g}). Bringing the bottom quartile (currently {p10:.2g}) "
                    f"to median would represent a {uplift_pct:.0f}% improvement. "
                    f"Identify what high performers have in common."
                )
            # High concentration risk: >50% in one value for numeric col
            top_val_pct = float(s.value_counts(normalize=True).iloc[0]) * 100
            if top_val_pct > 60 and s.nunique() > 3:
                risks.append(
                    f"'{col}': {top_val_pct:.0f}% of records share the same value "
                    f"({s.value_counts().index[0]:.3g}). "
                    "Potential data collection bias or limited diversity."
                )
        except Exception:
            logger.debug("Generic opportunity check failed for %s", col, exc_info=True)

    actions.extend([
        "Validate all outliers before analysis or modeling",
        "Use median for skewed distributions in executive reports",
        "Segment analysis — subgroups may tell different stories",
    ])
    return {"findings": findings, "risks": risks, "opportunities": opps,
            "actions": actions, "insights": insights}


# ══════════════════════════════════════════════════════════
#  ANOMALY DETECTION
# ══════════════════════════════════════════════════════════


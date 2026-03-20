"""
pages/2_Data_Quality.py
Data Quality + Statistical Analysis page.
Uses session_manager — no direct session_state access.
"""
import streamlit as st
import pandas as pd
import numpy as np
from core.session_manager import require_data, get_df, get_filename, update_active_df, cache_stats, get_cached_stats
from core.data_cleaner import auto_clean, get_cleaning_summary
from core.stats_engine import analyze

st.set_page_config(page_title="Data Quality", layout="wide")

# ── Guard — stops page if no data loaded ──────────────────
require_data()
df_raw   = get_df()
filename = get_filename()

# ── Cached cleaning ───────────────────────────────────────
@st.cache_data(show_spinner=False)
def run_cleaning(df_json: str) -> tuple:
    df = pd.read_json(df_json)
    cleaned_df, report = auto_clean(df)
    summary = get_cleaning_summary(report)
    return cleaned_df, summary

@st.cache_data(show_spinner=False)
def run_stats(df_json: str):
    df = pd.read_json(df_json)
    return analyze(df)

st.markdown("## Data Quality Report")
st.caption("{} — {:,} rows, {} columns".format(
    filename, len(df_raw), len(df_raw.columns)))
st.divider()

with st.spinner("Running cleaning pipeline..."):
    df_clean, summary = run_cleaning(df_raw.to_json(date_format="iso"))
    update_active_df(df_clean)   # update via session_manager

with st.spinner("Running statistical analysis..."):
    stats = run_stats(df_clean.to_json(date_format="iso"))
    cache_stats(stats)           # cache for other pages to reuse

# ══════════════════════════════════════════════════════════
#  SECTION 1 — BEFORE vs AFTER
# ══════════════════════════════════════════════════════════
st.markdown("### Dataset Snapshot — Before vs After Cleaning")

c1, c2, c3, c4, c5 = st.columns(5)
delta_rows = summary["cleaned_rows"] - summary["original_rows"]
delta_cols = summary["cleaned_cols"] - summary["original_cols"]

c1.metric("Rows (Before)", "{:,}".format(summary["original_rows"]))
c2.metric("Rows (After)",  "{:,}".format(summary["cleaned_rows"]),
          delta=delta_rows if delta_rows != 0 else None)
c3.metric("Cols (Before)", str(summary["original_cols"]))
c4.metric("Cols (After)",  str(summary["cleaned_cols"]),
          delta=delta_cols if delta_cols != 0 else None)
c5.metric("Total Actions", str(summary["total_actions"]))
st.divider()

# ══════════════════════════════════════════════════════════
#  SECTION 2 — CLEANING LOG
# ══════════════════════════════════════════════════════════
st.markdown("### Cleaning Actions Taken")
st.caption("Every change made to your data is documented below.")

groups = summary["groups"]

# Duplicates
if summary["duplicates_removed"] > 0:
    with st.expander("DUPLICATES REMOVED — {:,} rows".format(
            summary["duplicates_removed"]), expanded=True):
        st.error("**{:,} duplicate rows** removed.\n\n"
                 "Original: {:,}  →  Cleaned: {:,}".format(
                     summary["duplicates_removed"],
                     summary["original_rows"],
                     summary["cleaned_rows"]))
else:
    st.success("**No duplicate rows** found.")

# Missing values
missing_actions = groups["missing"]
if missing_actions:
    with st.expander("MISSING VALUES HANDLED — {} column(s)".format(
            len(missing_actions)), expanded=True):
        rows = [{"Column": a.column, "Issue": a.issue,
                 "Action Taken": a.action, "Rows Affected": a.rows_affected}
                for a in missing_actions]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
else:
    st.success("**No missing values** found.")

# Dropped columns
dropped = groups["dropped_col"]
if dropped:
    with st.expander("COLUMNS DROPPED — {}".format(len(dropped)), expanded=False):
        for a in dropped:
            st.warning("**{}** — {} → {}".format(a.column, a.issue, a.action))
else:
    st.success("**No columns dropped.**")

# Outliers flagged
flagged = groups["flagged"]
if flagged:
    with st.expander("OUTLIERS FLAGGED — {} column(s)".format(len(flagged)), expanded=False):
        st.info("Extreme outliers (3x IQR) flagged but not removed — review before modeling.")
        rows = [{"Column": a.column, "Issue": a.issue, "Action": a.action}
                for a in flagged]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# Other changes
all_minor = groups["type_fix"] + groups["whitespace"] + groups["other"]
if all_minor:
    with st.expander("OTHER CHANGES — {}".format(len(all_minor)), expanded=False):
        rows = [{"Column": a.column, "Issue": a.issue, "Action": a.action}
                for a in all_minor]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

st.divider()

# ══════════════════════════════════════════════════════════
#  SECTION 3 — STATISTICAL ANALYSIS
# ══════════════════════════════════════════════════════════
st.markdown("### Statistical Analysis")

num_cols = stats.numeric_cols
cat_cols = stats.categorical_cols

tab_num, tab_cat, tab_corr, tab_insights = st.tabs([
    "Numeric Columns", "Categorical Columns",
    "Correlations", "Insights & Recommendations"
])

with tab_num:
    if not num_cols:
        st.info("No numeric columns found after cleaning.")
    else:
        rows = []
        for col in num_cols:
            cs = stats.column_stats.get(col)
            if not cs or cs.mean is None:
                continue
            rows.append({
                "Column": col, "Mean": cs.mean, "Median": cs.median,
                "Std Dev": cs.std, "Min": cs.min_val, "Max": cs.max_val,
                "Skewness": cs.skewness, "Shape": cs.skew_label,
                "Normal?": "Yes" if cs.is_normal else "No",
                "p-value": cs.normality_pvalue,
                "Outliers (IQR)": cs.outlier_count_iqr,
                "Best Method": cs.outlier_method_recommended,
            })
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            c1, c2, c3 = st.columns(3)
            c1.info("**Skewness > 1** — Right-skewed. Use median, not mean.")
            c2.info("**|Skewness| < 0.5** — Symmetric. Mean is reliable.")
            c3.info("**p > 0.05** — Normal distribution. Parametric tests valid.")

with tab_cat:
    if not cat_cols:
        st.info("No categorical columns found.")
    else:
        rows = []
        for col in cat_cols:
            cs = stats.column_stats.get(col)
            if not cs:
                continue
            rows.append({
                "Column": col,
                "Unique Values": cs.unique_count,
                "Cardinality": cs.cardinality_label or "-",
                "Top Value": cs.top_value or "-",
                "Top Value %": "{}%".format(cs.top_value_pct) if cs.top_value_pct else "-",
                "Missing": "{}%".format(cs.missing_pct),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        c1, c2 = st.columns(2)
        c1.info("**Low cardinality** (10 or fewer unique) — Good for grouping.")
        c2.warning("**High cardinality** (80%+ unique) — Likely ID field, skip in analysis.")

with tab_corr:
    if not stats.correlations:
        st.info("Need at least 2 numeric columns for correlation analysis.")
    else:
        sig_corr = [c for c in stats.correlations if c.is_significant]
        if not sig_corr:
            st.info("No statistically significant correlations found.")
        else:
            st.markdown("**Significant correlations only (p < 0.05)**")
            rows = []
            for c in sig_corr[:20]:
                rows.append({
                    "Column A": c.col_a, "Column B": c.col_b,
                    "Pearson r": c.pearson_r, "Spearman r": c.spearman_r,
                    "p-value": c.p_value, "Strength": c.strength.title(),
                    "Direction": c.direction.title(),
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            c1, c2, c3 = st.columns(3)
            c1.error("|r| >= 0.7 — Strong relationship")
            c2.warning("|r| 0.4-0.7 — Moderate, worth investigating")
            c3.info("|r| < 0.4 — Weak, likely not meaningful")

with tab_insights:
    if stats.dataset_insights:
        st.markdown("#### Automated Statistical Insights")
        for insight in stats.dataset_insights:
            st.markdown("- " + insight)
    else:
        st.info("No notable statistical patterns detected.")

    if stats.recommended_analysis:
        st.divider()
        st.markdown("#### Recommended Analysis Methods")
        for rec in stats.recommended_analysis:
            st.success("→ " + rec)

    if stats.top_correlations:
        st.divider()
        st.markdown("#### Top Significant Correlations")
        for c in stats.top_correlations:
            tag = "[STRONG]" if c.strength == "strong" else "[MODERATE]"
            st.markdown("{} {}".format(tag, c.label))

st.divider()

# ══════════════════════════════════════════════════════════
#  SECTION 4 — COLUMN DETAIL
# ══════════════════════════════════════════════════════════
st.markdown("### Column-by-Column Detail")
st.caption("Full breakdown for every column.")

for col in df_clean.columns[:25]:
    cs = stats.column_stats.get(col)
    if not cs:
        continue

    score = 100.0
    if cs.missing_pct > 0:
        score -= min(cs.missing_pct * 0.5, 40)
    if hasattr(cs, "outlier_pct") and cs.outlier_pct:
        score -= min(cs.outlier_pct * 0.3, 20)
    score = max(0, round(score, 0))
    tag = "[GOOD]" if score >= 90 else "[FAIR]" if score >= 70 else "[POOR]"

    with st.expander("{} {}  |  {}  |  Score: {}/100".format(
            tag, col, cs.dtype, int(score))):
        if cs.mean is not None:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Mean",    str(cs.mean))
            c2.metric("Median",  str(cs.median))
            c3.metric("Std Dev", str(cs.std))
            c4.metric("Range",   str(cs.range_val))

            c5, c6, c7, c8 = st.columns(4)
            c5.metric("Q1",       str(cs.q1))
            c6.metric("Q3",       str(cs.q3))
            c7.metric("IQR",      str(cs.iqr))
            c8.metric("Skewness", str(cs.skewness))

            c9, c10, c11 = st.columns(3)
            c9.metric("Shape",    cs.skew_label or "-")
            c10.metric("Normality", cs.normality_label or "-",
                       help="{}: p={:.4f}".format(
                           cs.normality_test or "",
                           cs.normality_pvalue or 0))
            c11.metric("Outliers (IQR)", str(cs.outlier_count_iqr))

            if cs.outlier_count_iqr > 0:
                st.caption("Recommended method: {}".format(
                    cs.outlier_method_recommended))
        else:
            c1, c2, c3 = st.columns(3)
            c1.metric("Unique Values", cs.unique_count)
            c2.metric("Top Value",     cs.top_value or "-")
            c3.metric("Top Value %",
                      "{}%".format(cs.top_value_pct) if cs.top_value_pct else "-")
            st.caption("Cardinality: {}".format(cs.cardinality_label or "-"))

            if cs.unique_count <= 15 and col in df_clean.columns:
                vc = df_clean[col].value_counts().reset_index()
                vc.columns = ["Value", "Count"]
                vc["Percentage"] = (
                    vc["Count"] / len(df_clean) * 100
                ).round(1).astype(str) + "%"
                st.dataframe(vc, use_container_width=True, hide_index=True)

        if cs.missing_count > 0:
            st.warning("Missing: {} values ({:.1f}%) — handled in cleaning.".format(
                cs.missing_count, cs.missing_pct))

import streamlit as st
import pandas as pd
import numpy as np

# â”€â”€ Guard â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
if "df_active" not in st.session_state:
    st.warning("Please upload a dataset first.")
    st.page_link("pages/1_ðŸ“¥_Data_Upload.py", label="Go to Upload", icon="ðŸ“¥")
    st.stop()

from core.data_cleaner import auto_clean, get_cleaning_summary
from core.stats_engine import analyze

st.set_page_config(page_title="Data Quality", layout="wide")

# â”€â”€ Run cleaning (cached per df hash) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
@st.cache_data(show_spinner=False)
def run_cleaning(df_json: str) -> tuple:
    df_raw = pd.read_json(df_json)
    cleaned_df, report = auto_clean(df_raw)
    summary = get_cleaning_summary(report)
    return cleaned_df, summary

@st.cache_data(show_spinner=False)
def run_stats(df_json: str):
    df = pd.read_json(df_json)
    return analyze(df)

# â”€â”€ Page header â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
st.markdown("## ðŸ§¹ Data Quality Report")
st.caption("Auto-cleaning pipeline â€” every action logged and explained.")
st.divider()

df_raw = st.session_state["df_active"]
df_json = df_raw.to_json()

with st.spinner("Running cleaning pipeline..."):
    df_clean, summary = run_cleaning(df_json)
    st.session_state["df_active"] = df_clean   # update active df

with st.spinner("Running statistical analysis..."):
    stats = run_stats(df_clean.to_json())

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#  SECTION 1 â€” BEFORE vs AFTER SNAPSHOT
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
st.markdown("### ðŸ“Š Dataset Snapshot")

col1, col2, col3, col4, col5 = st.columns(5)

delta_rows = summary["cleaned_rows"] - summary["original_rows"]
delta_cols = summary["cleaned_cols"] - summary["original_cols"]

col1.metric("Rows (Before)",  "{:,}".format(summary["original_rows"]))
col2.metric("Rows (After)",   "{:,}".format(summary["cleaned_rows"]),
            delta=delta_rows if delta_rows != 0 else None)
col3.metric("Cols (Before)",  str(summary["original_cols"]))
col4.metric("Cols (After)",   str(summary["cleaned_cols"]),
            delta=delta_cols if delta_cols != 0 else None)
col5.metric("Total Actions",  str(summary["total_actions"]),
            help="Cleaning steps applied automatically")

st.divider()

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#  SECTION 2 â€” CLEANING ACTION LOG
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
st.markdown("### ðŸ”§ Cleaning Actions Taken")
st.caption("Every change made to your data is documented below.")

groups = summary["groups"]

# â”€â”€ Duplicates â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
if summary["duplicates_removed"] > 0:
    with st.expander(
        "ðŸ” Duplicates Removed â€” {:,} rows".format(summary["duplicates_removed"]),
        expanded=True
    ):
        st.error(
            "**{:,} duplicate rows** were found and removed.  \n"
            "Original: {:,} rows â†’ Cleaned: {:,} rows".format(
                summary["duplicates_removed"],
                summary["original_rows"],
                summary["cleaned_rows"],
            )
        )
else:
    st.success("âœ… **No duplicate rows** found.")

# â”€â”€ Missing values â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
missing_actions = groups["missing"]
if missing_actions:
    with st.expander(
        "âš ï¸ Missing Values Handled â€” {} column(s)".format(len(missing_actions)),
        expanded=True
    ):
        rows = []
        for a in missing_actions:
            rows.append({
                "Column":       a.column,
                "Issue":        a.issue,
                "Action Taken": a.action,
                "Rows Affected": a.rows_affected,
            })
        st.dataframe(
            pd.DataFrame(rows),
            use_container_width=True,
            hide_index=True,
        )
else:
    st.success("âœ… **No missing values** found in any column.")

# â”€â”€ Dropped columns â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
dropped = groups["dropped_col"]
if dropped:
    with st.expander(
        "ðŸ—‘ï¸ Columns Dropped â€” {}".format(len(dropped)),
        expanded=False
    ):
        for a in dropped:
            st.warning("**{}** â€” {} â†’ {}".format(a.column, a.issue, a.action))
else:
    st.success("âœ… **No columns dropped.**")

# â”€â”€ Outliers flagged â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
flagged = groups["flagged"]
if flagged:
    with st.expander(
        "ðŸ“Š Outliers Flagged â€” {} column(s)".format(len(flagged)),
        expanded=False
    ):
        st.info(
            "Extreme outliers (3x IQR) are **flagged but not removed** â€” "
            "review before modeling."
        )
        rows = []
        for a in flagged:
            rows.append({
                "Column":  a.column,
                "Issue":   a.issue,
                "Action":  a.action,
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# â”€â”€ Type conversions â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
type_fixes = groups["type_fix"]
whitespace  = groups["whitespace"]
all_minor   = type_fixes + whitespace + groups["other"]
if all_minor:
    with st.expander(
        "ðŸ”„ Other Changes â€” {}".format(len(all_minor)),
        expanded=False
    ):
        rows = []
        for a in all_minor:
            rows.append({
                "Column": a.column,
                "Issue":  a.issue,
                "Action": a.action,
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

st.divider()

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#  SECTION 3 â€” REAL STATISTICAL ANALYSIS
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
st.markdown("### ðŸ“ˆ Statistical Analysis")

num_cols = stats.numeric_cols
cat_cols = stats.categorical_cols

tab_num, tab_cat, tab_corr, tab_insights = st.tabs([
    "Numeric Columns", "Categorical Columns",
    "Correlations", "Insights & Recommendations"
])

# â”€â”€ Tab 1 â€” Numeric â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
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
                "Column":        col,
                "Mean":          cs.mean,
                "Median":        cs.median,
                "Std Dev":       cs.std,
                "Min":           cs.min_val,
                "Max":           cs.max_val,
                "Skewness":      cs.skewness,
                "Shape":         cs.skew_label,
                "Kurtosis":      cs.kurtosis,
                "Normal?":       "Yes" if cs.is_normal else "No",
                "Normality Test":cs.normality_test or "-",
                "p-value":       cs.normality_pvalue,
                "Outliers (IQR)":cs.outlier_count_iqr,
                "Outliers (Z)":  cs.outlier_count_zscore,
                "Best Method":   cs.outlier_method_recommended,
            })

        if rows:
            df_display = pd.DataFrame(rows)
            st.dataframe(df_display, use_container_width=True, hide_index=True)

            st.markdown("#### Distribution Shape Guide")
            c1, c2, c3 = st.columns(3)
            c1.info("**Skewness > 1** â†’ Right-skewed  \nUse median, not mean")
            c2.info("**|Skewness| < 0.5** â†’ Symmetric  \nMean is reliable")
            c3.info("**p > 0.05** â†’ Normal  \nParametric tests valid")

# â”€â”€ Tab 2 â€” Categorical â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
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
                "Column":       col,
                "Unique Values":cs.unique_count,
                "Cardinality":  cs.cardinality_label or "-",
                "Top Value":    cs.top_value or "-",
                "Top Value %":  "{}%".format(cs.top_value_pct) if cs.top_value_pct else "-",
                "Missing":      "{}%".format(cs.missing_pct),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        st.markdown("#### Cardinality Guide")
        c1, c2 = st.columns(2)
        c1.info("**Low cardinality** (â‰¤10 unique) â†’ Good for grouping & charts")
        c2.warning("**High cardinality** (>80% unique) â†’ Likely ID field, skip in analysis")

# â”€â”€ Tab 3 â€” Correlations â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
with tab_corr:
    if not stats.correlations:
        st.info("Need at least 2 numeric columns for correlation analysis.")
    else:
        st.markdown("**Significant correlations only (p < 0.05)**")

        sig_corr = [c for c in stats.correlations if c.is_significant]

        if not sig_corr:
            st.info("No statistically significant correlations found.")
        else:
            rows = []
            for c in sig_corr[:20]:
                rows.append({
                    "Column A":     c.col_a,
                    "Column B":     c.col_b,
                    "Pearson r":    c.pearson_r,
                    "Spearman r":   c.spearman_r,
                    "p-value":      c.p_value,
                    "Strength":     c.strength.title(),
                    "Direction":    c.direction.title(),
                    "Significant?": "Yes",
                })
            df_corr = pd.DataFrame(rows)

            def _color_strength(val):
                if val == "Strong":
                    return "background-color: #1a4a8a; color: white"
                elif val == "Moderate":
                    return "background-color: #42A5F5; color: white"
                return ""

            st.dataframe(
                df_corr.style.applymap(_color_strength, subset=["Strength"]),
                use_container_width=True, hide_index=True
            )

            st.markdown("#### Correlation Strength Guide")
            c1, c2, c3 = st.columns(3)
            c1.error("**|r| â‰¥ 0.7** â†’ Strong â€” significant relationship")
            c2.warning("**|r| 0.4â€“0.7** â†’ Moderate â€” worth investigating")
            c3.info("**|r| < 0.4** â†’ Weak â€” likely not meaningful")

# â”€â”€ Tab 4 â€” Insights â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
with tab_insights:
    if stats.dataset_insights:
        st.markdown("#### ðŸ” Automated Statistical Insights")
        for insight in stats.dataset_insights:
            st.markdown("- " + insight)
    else:
        st.info("No notable statistical patterns detected.")

    if stats.recommended_analysis:
        st.divider()
        st.markdown("#### âš¡ Recommended Analysis Methods")
        for rec in stats.recommended_analysis:
            st.success("â†’ " + rec)

    if stats.top_correlations:
        st.divider()
        st.markdown("#### ðŸ”— Top Significant Correlations")
        for c in stats.top_correlations:
            icon = "ðŸ”´" if c.strength == "strong" else "ðŸŸ¡"
            st.markdown("{} {}".format(icon, c.label))

st.divider()

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#  SECTION 4 â€” PER-COLUMN DETAIL (expandable)
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
st.markdown("### ðŸ”Ž Column-by-Column Detail")
st.caption("Full breakdown for every column â€” missing, outliers, distribution.")

for col in df_clean.columns[:25]:  # max 25 cols
    cs = stats.column_stats.get(col)
    if not cs:
        continue

    # Health score
    score = 100.0
    if cs.missing_pct > 0:
        score -= min(cs.missing_pct * 0.5, 40)
    if hasattr(cs, "outlier_pct") and cs.outlier_pct:
        score -= min(cs.outlier_pct * 0.3, 20)
    score = max(0, round(score, 0))

    score_icon = "ðŸŸ¢" if score >= 90 else ("ðŸŸ¡" if score >= 70 else "ðŸ”´")

    with st.expander(
        "{} **{}**  â€” {}  |  Score: {}/100".format(
            score_icon, col, cs.dtype, int(score))
    ):
        if cs.mean is not None:
            # Numeric column detail
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Mean",   str(cs.mean))
            c2.metric("Median", str(cs.median))
            c3.metric("Std",    str(cs.std))
            c4.metric("Range",  str(cs.range_val))

            c5, c6, c7, c8 = st.columns(4)
            c5.metric("Q1",  str(cs.q1))
            c6.metric("Q3",  str(cs.q3))
            c7.metric("IQR", str(cs.iqr))
            c8.metric("Skewness", str(cs.skewness))

            c9, c10, c11 = st.columns(3)
            c9.metric("Distribution Shape", cs.skew_label or "-")
            c10.metric("Normality",
                       cs.normality_label or "-",
                       help="{}: p={:.4f}".format(
                           cs.normality_test or "",
                           cs.normality_pvalue or 0))
            c11.metric("Outliers (IQR)", str(cs.outlier_count_iqr))

            if cs.outlier_count_iqr > 0:
                st.caption("Recommended outlier method: {}".format(
                    cs.outlier_method_recommended))

        else:
            # Categorical column detail
            c1, c2, c3 = st.columns(3)
            c1.metric("Unique Values", cs.unique_count)
            c2.metric("Top Value",     cs.top_value or "-")
            c3.metric("Top Value %",   "{}%".format(cs.top_value_pct) if cs.top_value_pct else "-")
            st.caption("Cardinality: {}".format(cs.cardinality_label or "-"))

            # Show value counts for low-cardinality columns
            if cs.unique_count <= 15 and col in df_clean.columns:
                vc = df_clean[col].value_counts().reset_index()
                vc.columns = ["Value", "Count"]
                vc["Percentage"] = (vc["Count"] / len(df_clean) * 100).round(1).astype(str) + "%"
                st.dataframe(vc, use_container_width=True, hide_index=True)

        if cs.missing_count > 0:
            st.warning("Missing: {} values ({:.1f}%) â€” handled in cleaning step.".format(
                cs.missing_count, cs.missing_pct))

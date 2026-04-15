"""
pages/6_Deep_EDA.py — Senior analyst level EDA.
"""
import io
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from core.session_manager import require_data, get_df, get_filename

st.set_page_config(page_title="Deep EDA", layout="wide")
require_data()
df    = get_df()
fname = get_filename()

from core.eda_engine import run_eda

COLORS = ["#1a4a8a", "#2196F3", "#22d3a5", "#f7934f", "#a78bfa", "#f77070"]

st.markdown("## Deep EDA — Statistical Analysis")
st.caption("{} — {:,} rows, {} columns".format(fname, len(df), len(df.columns)))
st.divider()

@st.cache_data(show_spinner=False)
def get_eda(df_json: str):
    # FIX: pandas 3.0 treats long strings as file paths
    return run_eda(pd.read_json(io.StringIO(df_json)))

with st.spinner("Running full statistical analysis..."):
    report = get_eda(df.to_json())

# Key findings banner
if report.key_findings:
    with st.expander("Key Findings ({})".format(len(report.key_findings)),
                     expanded=True):
        for f in report.key_findings:
            st.info(f)

st.divider()

# ══════════════════════════════════════════════════════════
#  TABS
# ══════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Univariate Analysis",
    "Correlations & Tests",
    "Group Comparisons",
    "Multicollinearity",
    "Time Series",
])

# ── Tab 1: Univariate ─────────────────────────────────────
with tab1:
    st.markdown("### Univariate Analysis")
    st.caption("Full distribution analysis for every column — normality, outliers, shape.")

    # Numeric summary table
    num_rows = []
    for col, r in report.univariate.items():
        if r.mean is None:
            continue
        num_rows.append({
            "Column":       col,
            "Mean":         r.mean,
            "Median":       r.median,
            "Std":          r.std,
            "CV":           r.cv,
            "Skewness":     r.skewness,
            "Shape":        r.skew_label,
            "Kurtosis":     r.kurtosis,
            "Normality":    r.normality_verdict,
            "Outliers IQR": r.outliers_iqr,
            "Outliers Z":   r.outliers_zscore,
            "Outliers ModZ":r.outliers_modz,
            "Best Method":  r.recommended_method,
            "Missing %":    r.missing_pct,
        })

    if num_rows:
        st.markdown("#### Numeric Columns Summary")
        st.dataframe(pd.DataFrame(num_rows),
                     use_container_width=True, hide_index=True)

    # Categorical summary
    cat_rows = []
    for col, r in report.univariate.items():
        if r.mean is not None:
            continue
        cat_rows.append({
            "Column":      col,
            "Unique":      r.unique_count,
            "Top Value":   r.top_value or "-",
            "Top %":       "{}%".format(r.top_pct) if r.top_pct else "-",
            "Entropy":     r.entropy,
            "Missing %":   r.missing_pct,
            "Notes":       r.interpretation[:60] if r.interpretation else "-",
        })

    if cat_rows:
        st.markdown("#### Categorical Columns Summary")
        st.dataframe(pd.DataFrame(cat_rows),
                     use_container_width=True, hide_index=True)
        st.caption("Entropy: higher value = more diverse categories (max = log2(unique))")

    # Per-column deep dive
    st.markdown("#### Column Deep Dive")
    num_cols_list = [col for col, r in report.univariate.items()
                     if r.mean is not None]
    if num_cols_list:
        selected_col = st.selectbox("Select column to analyze",
                                    num_cols_list, key="eda_col")
        r = report.univariate[selected_col]

        # Interpretation box
        st.markdown(
            "<div style='background:#f0f4ff;border-left:4px solid #1a4a8a;"
            "padding:14px 18px;border-radius:4px;margin-bottom:16px'>"
            "<b>Interpretation:</b> {}</div>".format(r.interpretation),
            unsafe_allow_html=True
        )

        # Metrics grid
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Mean",     str(r.mean))
        c2.metric("Median",   str(r.median))
        c3.metric("Std Dev",  str(r.std))
        c4.metric("Skewness", str(r.skewness))
        c5.metric("CV",       str(r.cv), help="Coefficient of Variation = Std/Mean")

        c6, c7, c8, c9, c10 = st.columns(5)
        c6.metric("Q1",  str(r.q1))
        c7.metric("Q3",  str(r.q3))
        c8.metric("IQR", str(r.iqr))
        c9.metric("P5",  str(r.p5), help="5th percentile")
        c10.metric("P95", str(r.p95), help="95th percentile")

        # Normality tests side by side
        st.markdown("**Normality Tests**")
        c1, c2, c3 = st.columns(3)
        if r.shapiro_p is not None:
            c1.metric("Shapiro-Wilk p",
                      "{:.4f}".format(r.shapiro_p),
                      delta="NORMAL" if r.shapiro_p > 0.05 else "NON-NORMAL",
                      delta_color="normal" if r.shapiro_p > 0.05 else "inverse")
        if r.dagostino_p is not None:
            c2.metric("D'Agostino p",
                      "{:.4f}".format(r.dagostino_p),
                      delta="NORMAL" if r.dagostino_p > 0.05 else "NON-NORMAL",
                      delta_color="normal" if r.dagostino_p > 0.05 else "inverse")
        if r.anderson_stat is not None:
            passed = r.anderson_stat < (r.anderson_critical or 0)
            c3.metric("Anderson-Darling",
                      "{:.4f}".format(r.anderson_stat),
                      delta="NORMAL" if passed else "NON-NORMAL",
                      delta_color="normal" if passed else "inverse")

        # Outlier comparison
        st.markdown("**Outlier Detection — 3 Methods Compared**")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("IQR Method",       str(r.outliers_iqr))
        c2.metric("Z-Score (|z|>3)",  str(r.outliers_zscore))
        c3.metric("Modified Z-Score", str(r.outliers_modz))
        c4.metric("Recommended",      r.recommended_method.split(" ")[0])

        if r.iqr_lower is not None:
            st.caption("IQR bounds: {:.4f} to {:.4f}".format(
                r.iqr_lower, r.iqr_upper or 0))

        # Best fit distribution
        if r.best_fit_dist:
            st.info("Best-fit distribution: **{}** (KS test p={:.4f})".format(
                r.best_fit_dist,
                r.best_fit_params.get("ks_p", 0) if r.best_fit_params else 0))

        # Distribution chart
        col_data = df[selected_col].dropna()
        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=["Distribution", "Box Plot"])

        fig.add_trace(
            go.Histogram(x=col_data, name="Frequency",
                         marker_color=COLORS[0], opacity=0.8,
                         nbinsx=30),
            row=1, col=1
        )
        fig.add_vline(x=r.mean, line_dash="dash", line_color="#f7934f",
                      annotation_text="Mean={:.2f}".format(r.mean),
                      row=1, col=1)
        fig.add_vline(x=r.median, line_dash="dot", line_color="#22d3a5",
                      annotation_text="Median={:.2f}".format(r.median),
                      row=1, col=1)
        fig.add_trace(
            go.Box(y=col_data, name=selected_col,
                   marker_color=COLORS[0], boxmean=True),
            row=1, col=2
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#f8faff",
            font=dict(family="Helvetica", size=11),
            margin=dict(l=10, r=10, t=40, b=10),
            showlegend=False, height=380,
        )
        st.plotly_chart(fig, use_container_width=True)

# ── Tab 2: Correlations ───────────────────────────────────
with tab2:
    st.markdown("### Correlations & Statistical Tests")
    st.caption("Pearson for normal data, Spearman for non-normal. All with effect sizes.")

    if not report.correlations:
        st.info("Need at least 2 numeric columns.")
    else:
        # Significant only toggle
        show_all = st.toggle("Show all correlations (including non-significant)", value=False)
        corrs    = report.correlations if show_all else [
            c for c in report.correlations if c.is_significant
        ]

        if not corrs:
            st.info("No statistically significant correlations found.")
        else:
            rows = []
            for c in corrs[:25]:
                rows.append({
                    "Col A":        c.col_a,
                    "Col B":        c.col_b,
                    "Test":         c.test_name,
                    "Statistic":    c.statistic,
                    "p-value":      c.p_value,
                    "Significant?": "YES" if c.is_significant else "no",
                    "Effect Size":  c.effect_size,
                    "Effect":       c.effect_label,
                })
            st.dataframe(pd.DataFrame(rows),
                         use_container_width=True, hide_index=True)

        # Heatmap
        num_cols = report.numeric_cols[:10]
        if len(num_cols) >= 2:
            corr_matrix = df[num_cols].corr().round(3)
            fig = px.imshow(
                corr_matrix,
                color_continuous_scale="RdBu_r",
                zmin=-1, zmax=1,
                text_auto=True, aspect="auto",
                title="Correlation Heatmap",
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=10, r=10, t=40, b=10),
            )
            st.plotly_chart(fig, use_container_width=True)

        # Scatter for top correlation
        if corrs:
            top = corrs[0]
            st.markdown("#### Scatter — {} vs {}".format(top.col_a, top.col_b))
            st.caption(top.interpretation)
            st.caption("Recommendation: " + top.recommendation)

            sample = df[[top.col_a, top.col_b]].dropna().sample(
                min(2000, len(df)), random_state=42
            )
            fig_sc = px.scatter(
                sample, x=top.col_a, y=top.col_b,
                trendline=None,
                opacity=0.5, color_discrete_sequence=[COLORS[0]],
                title="{} vs {} (r={:.3f}, p={:.4f})".format(
                    top.col_a, top.col_b, top.statistic, top.p_value),
            )
            # Manual trendline
            try:
                x_v = sample[top.col_a].values
                y_v = sample[top.col_b].values
                m, b = np.polyfit(x_v, y_v, 1)
                x_l = np.linspace(x_v.min(), x_v.max(), 100)
                fig_sc.add_trace(go.Scatter(
                    x=x_l, y=m*x_l+b, mode="lines", name="Trend",
                    line=dict(color="#f7934f", width=2, dash="dash")
                ))
            except Exception:
                pass
            fig_sc.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#f8faff",
                margin=dict(l=10, r=10, t=40, b=10),
            )
            st.plotly_chart(fig_sc, use_container_width=True)

        # Effect size guide
        st.markdown("#### Effect Size Guide")
        c1, c2, c3, c4 = st.columns(4)
        c1.info("|r| < 0.1 — Negligible")
        c2.info("|r| 0.1-0.3 — Small")
        c3.warning("|r| 0.3-0.5 — Medium")
        c4.error("|r| > 0.5 — Large")

# ── Tab 3: Group Comparisons ──────────────────────────────
with tab3:
    st.markdown("### Group Comparisons")
    st.caption("ANOVA (normal data) or Kruskal-Wallis (non-normal). With eta-squared effect size.")

    if not report.group_comparisons:
        st.info("Need categorical columns with 2-15 groups and numeric columns.")
    else:
        for gc in report.group_comparisons:
            sig_color = "error" if gc.is_significant else "info"
            with st.expander(
                "{} — '{}' by '{}' | {} | p={:.4f} | Effect={}".format(
                    "SIGNIFICANT" if gc.is_significant else "not significant",
                    gc.numeric_col, gc.group_col,
                    gc.test_used, gc.p_value, gc.effect_label),
                expanded=bool(gc.is_significant)   # FIX: numpy.bool_ → Python bool
            ):
                st.markdown(gc.interpretation)
                if gc.post_hoc:
                    for ph in gc.post_hoc:
                        st.info(ph)

                # Group stats table
                if gc.group_stats:
                    rows = []
                    for grp, stats_d in gc.group_stats.items():
                        rows.append({
                            "Group":  grp,
                            "N":      stats_d["n"],
                            "Mean":   stats_d["mean"],
                            "Median": stats_d["median"],
                            "Std":    stats_d["std"],
                        })
                    st.dataframe(pd.DataFrame(rows),
                                 use_container_width=True, hide_index=True)

                # Box plot by group
                try:
                    plot_df = df[[gc.group_col, gc.numeric_col]].dropna()
                    top_grps = plot_df[gc.group_col].value_counts().head(10).index
                    plot_df = plot_df[plot_df[gc.group_col].isin(top_grps)]
                    fig = px.box(
                        plot_df, x=gc.group_col, y=gc.numeric_col,
                        color=gc.group_col,
                        color_discrete_sequence=COLORS,
                        title="{} by {}".format(gc.numeric_col, gc.group_col),
                        points="outliers",
                    )
                    fig.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#f8faff",
                        margin=dict(l=10, r=10, t=40, b=10),
                        showlegend=False,
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except Exception:
                    pass

# ── Tab 4: Multicollinearity ──────────────────────────────
with tab4:
    st.markdown("### Multicollinearity — VIF Analysis")
    st.caption(
        "Variance Inflation Factor (VIF) measures how much each feature "
        "is explained by other features. VIF > 10 = problem."
    )

    if not report.multicollinearity:
        st.info("Need at least 2 numeric columns for VIF analysis.")
    else:
        rows = []
        for r in report.multicollinearity:
            rows.append({
                "Feature":        r.feature,
                "VIF":            r.vif,
                "Verdict":        r.verdict,
                "Interpretation": r.interpretation,
            })
        df_vif = pd.DataFrame(rows)
        st.dataframe(df_vif, use_container_width=True, hide_index=True)

        # VIF bar chart
        fig = go.Figure(go.Bar(
            x=[r.feature for r in report.multicollinearity],
            y=[r.vif for r in report.multicollinearity],
            marker_color=[
                "#f77070" if r.vif >= 20
                else "#f7934f" if r.vif >= 10
                else "#ffd43b" if r.vif >= 5
                else "#22d3a5"
                for r in report.multicollinearity
            ],
            text=["{:.1f}".format(r.vif) for r in report.multicollinearity],
            textposition="outside",
        ))
        fig.add_hline(y=10, line_dash="dash", line_color="#f77070",
                      annotation_text="VIF=10 (critical threshold)")
        fig.add_hline(y=5, line_dash="dot", line_color="#f7934f",
                      annotation_text="VIF=5 (warning)")
        fig.update_layout(
            title="VIF by Feature",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#f8faff",
            font=dict(family="Helvetica", size=11),
            margin=dict(l=10, r=10, t=40, b=10),
            xaxis_tickangle=-30,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Guide
        c1, c2, c3, c4 = st.columns(4)
        c1.success("VIF < 5 — OK")
        c2.warning("VIF 5-10 — Moderate")
        c3.error("VIF 10-20 — High")
        c4.error("VIF > 20 — Severe")

# ── Tab 5: Time Series ────────────────────────────────────
with tab5:
    st.markdown("### Time Series Analysis")

    if not report.datetime_cols:
        st.info("No datetime columns detected. Upload data with date/time column for time series analysis.")
    elif not report.time_series:
        st.info("Could not perform time series analysis — need at least 10 time points.")
    else:
        for ts in report.time_series:
            with st.expander(
                "{} over time | {} | Stationary: {}".format(
                    ts.column, ts.trend or "N/A",
                    "YES" if ts.is_stationary else "NO"),
                expanded=True
            ):
                st.markdown(ts.interpretation)

                c1, c2, c3 = st.columns(3)
                if ts.adf_stat is not None:
                    c1.metric("ADF Statistic", "{:.4f}".format(ts.adf_stat))
                if ts.adf_p is not None:
                    c2.metric("ADF p-value", "{:.4f}".format(ts.adf_p),
                              delta="Stationary" if ts.is_stationary else "Non-Stationary",
                              delta_color="normal" if ts.is_stationary else "inverse")
                if ts.trend_slope is not None:
                    c3.metric("Trend Slope", "{:.6f}".format(ts.trend_slope))

                # Time series chart
                try:
                    dt_col  = ts.date_col
                    val_col = ts.column
                    ts_data = (df.set_index(dt_col)[val_col]
                                 .resample("M").mean()
                                 .reset_index()
                                 .dropna())
                    fig = px.line(
                        ts_data, x=dt_col, y=val_col,
                        title="{} over time (monthly avg)".format(val_col),
                        line_shape="spline",
                    )
                    fig.update_traces(line_color=COLORS[0], line_width=2)
                    fig.add_hline(
                        y=float(ts_data[val_col].mean()),
                        line_dash="dash", line_color="#f7934f",
                        annotation_text="Mean={:.2f}".format(
                            ts_data[val_col].mean())
                    )
                    fig.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#f8faff",
                        margin=dict(l=10, r=10, t=40, b=10),
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except Exception:
                    pass

        st.markdown("#### ADF Test Guide")
        c1, c2 = st.columns(2)
        c1.success("p < 0.05 — Stationary: mean/variance stable. Ready for modeling.")
        c2.error("p > 0.05 — Non-stationary: apply differencing or log-transform before ARIMA.")


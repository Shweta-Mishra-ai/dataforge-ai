import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Guard ─────────────────────────────────────────────────
from core.session_manager import require_data, get_df
require_data()
df_master = get_df()  
from core.stats_engine import analyze

st.set_page_config(page_title="Dashboard", layout="wide")

# ── Theme ─────────────────────────────────────────────────
COLORS = ["#1a4a8a", "#2196F3", "#42A5F5", "#22d3a5",
          "#f7934f", "#a78bfa", "#f77070", "#ffd43b"]

PLOTLY_THEME = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#f8faff",
    font=dict(family="Helvetica, Arial", size=11, color="#1e1e28"),
    margin=dict(l=10, r=10, t=40, b=10),
    legend=dict(orientation="h", yanchor="bottom", y=1.02,
                xanchor="right", x=1),
    colorway=COLORS,
)

# ── Cache ──────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def get_stats(df_json: str):
    df = pd.read_json(df_json)
    return analyze(df), df

# ── Load data ─────────────────────────────────────────────
df_master = st.session_state["df_active"]
stats, _   = get_stats(df_master.to_json())

num_cols  = stats.numeric_cols
cat_cols  = [c for c in stats.categorical_cols
             if df_master[c].nunique() <= 50]   # exclude ID-like
id_cols   = [c for c in stats.categorical_cols
             if df_master[c].nunique() > 50]
dt_cols   = stats.datetime_cols

# ══════════════════════════════════════════════════════════
#  SIDEBAR FILTERS — Power BI style
# ══════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 🎛️ Filters")
    st.caption("Filter once → all charts update")

    df_filtered = df_master.copy()

    # Categorical filters
    active_cat_filters = {}
    for col in cat_cols[:4]:
        unique_vals = sorted(df_master[col].dropna().unique().tolist())
        if len(unique_vals) <= 20:
            selected = st.multiselect(
                col, unique_vals,
                default=unique_vals,
                key="filter_{}".format(col)
            )
            active_cat_filters[col] = selected
            if selected:
                df_filtered = df_filtered[df_filtered[col].isin(selected)]

    # Numeric range filters
    for col in num_cols[:2]:
        s = df_master[col].dropna()
        lo, hi = float(s.min()), float(s.max())
        if lo < hi:
            rng = st.slider(
                col, lo, hi, (lo, hi),
                key="range_{}".format(col)
            )
            df_filtered = df_filtered[
                (df_filtered[col] >= rng[0]) &
                (df_filtered[col] <= rng[1])
            ]

    st.divider()
    st.caption("Showing {:,} of {:,} rows".format(
        len(df_filtered), len(df_master)))
    if st.button("Reset Filters", use_container_width=True):
        st.rerun()

# ══════════════════════════════════════════════════════════
#  HEADER
# ══════════════════════════════════════════════════════════
fname = st.session_state.get("filename", "Dataset")
st.markdown("## 📊 Dashboard — {}".format(fname))
st.caption("Real-time analysis · {:,} rows · {} columns · {} filtered".format(
    len(df_master), len(df_master.columns), len(df_filtered)))
st.divider()

# ══════════════════════════════════════════════════════════
#  SECTION 1 — KPI CARDS with context
# ══════════════════════════════════════════════════════════
if num_cols:
    st.markdown("### 📌 Key Performance Indicators")

    kpi_cols = num_cols[:4]
    cols_ui  = st.columns(len(kpi_cols))

    for i, col in enumerate(kpi_cols):
        s_full   = df_master[col].dropna()
        s_filter = df_filtered[col].dropna() if col in df_filtered else s_full

        mean_full   = s_full.mean()
        mean_filter = s_filter.mean() if len(s_filter) > 0 else mean_full
        delta       = mean_filter - mean_full

        cs = stats.column_stats.get(col)
        skew_note = ""
        if cs and cs.skewness and abs(cs.skewness) > 1:
            skew_note = " (median: {:.2f})".format(cs.median)

        with cols_ui[i]:
            st.metric(
                label=col[:20],
                value="{:,.2f}{}".format(mean_filter, skew_note),
                delta="{:+.2f} vs full dataset".format(delta)
                      if abs(delta) > 0.001 else None,
                help="Sum: {:,.0f} | Min: {:,.2f} | Max: {:,.2f}".format(
                    s_filter.sum(), s_filter.min(), s_filter.max())
            )

    # Skewed warning
    skewed_cols = [c for c in kpi_cols
                   if stats.column_stats.get(c) and
                   stats.column_stats[c].skewness and
                   abs(stats.column_stats[c].skewness) > 1]
    if skewed_cols:
        st.warning(
            "⚠️ **Statistical note:** {} — distribution is skewed. "
            "Median shown in brackets is more reliable than mean.".format(
                ", ".join(skewed_cols))
        )

    st.divider()

# ══════════════════════════════════════════════════════════
#  SECTION 2 — SMART CHART GRID
# ══════════════════════════════════════════════════════════
st.markdown("### 📈 Visual Analysis")

chart_tab_labels = ["Overview", "Distribution", "Relationships", "Deep Dive"]
if dt_cols:
    chart_tab_labels.insert(1, "Trends")

tabs = st.tabs(chart_tab_labels)
tab_idx = 0

# ── Tab: Overview ─────────────────────────────────────────
with tabs[tab_idx]:
    tab_idx += 1
    if cat_cols and num_cols:
        c1, c2 = st.columns(2)

        with c1:
            best_cat = next(
                (c for c in cat_cols if 2 <= df_filtered[c].nunique() <= 15),
                cat_cols[0]
            )
            best_num = num_cols[0]
            agg = (df_filtered.groupby(best_cat)[best_num]
                   .mean().reset_index()
                   .sort_values(best_num, ascending=True)
                   .tail(15))
            fig = px.bar(
                agg, x=best_num, y=best_cat,
                orientation="h",
                title="Avg {} by {}".format(best_num, best_cat),
                color=best_num,
                color_continuous_scale=["#90CAF9", "#1a4a8a"],
                text=agg[best_num].round(2),
            )
            fig.update_traces(textposition="outside")
            fig.update_layout(**PLOTLY_THEME)
            fig.update_coloraxes(showscale=False)
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            if len(cat_cols) >= 1 and len(num_cols) >= 1:
                best_cat2 = next(
                    (c for c in cat_cols if 2 <= df_filtered[c].nunique() <= 10),
                    cat_cols[0]
                )
                pie_agg = (df_filtered.groupby(best_cat2)[num_cols[0]]
                           .sum().reset_index()
                           .sort_values(num_cols[0], ascending=False)
                           .head(8))
                fig2 = px.pie(
                    pie_agg, values=num_cols[0], names=best_cat2,
                    title="{} Share by {}".format(num_cols[0], best_cat2),
                    color_discrete_sequence=COLORS,
                    hole=0.4,
                )
                fig2.update_layout(**PLOTLY_THEME)
                st.plotly_chart(fig2, use_container_width=True)
    elif num_cols:
        # No categorical — show top numeric overview
        fig = px.bar(
            x=num_cols[:8],
            y=[df_filtered[c].mean() for c in num_cols[:8]],
            title="Column Averages",
            labels={"x": "Column", "y": "Average"},
            color=num_cols[:8],
            color_discrete_sequence=COLORS,
        )
        fig.update_layout(**PLOTLY_THEME)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No numeric + categorical columns available for overview charts.")

# ── Tab: Trends (only if datetime) ────────────────────────
if dt_cols and "Trends" in chart_tab_labels:
    with tabs[tab_idx]:
        tab_idx += 1
        dt_col  = dt_cols[0]
        val_col = st.selectbox("Select metric", num_cols, key="trend_metric")
        agg_by  = st.radio("Aggregate by",
                           ["Day", "Week", "Month", "Quarter"],
                           horizontal=True, key="trend_agg")

        freq_map = {"Day": "D", "Week": "W", "Month": "M", "Quarter": "Q"}
        freq = freq_map[agg_by]

        trend = (df_filtered.set_index(dt_col)[val_col]
                 .resample(freq).mean().reset_index())

        fig = px.line(
            trend, x=dt_col, y=val_col,
            title="{} — {} Trend".format(val_col, agg_by),
            line_shape="spline",
            markers=True,
        )
        fig.update_traces(line_color=COLORS[0], line_width=2.5,
                          marker_color=COLORS[1])
        fig.add_hline(
            y=trend[val_col].mean(),
            line_dash="dash", line_color="#f7934f",
            annotation_text="Avg: {:.2f}".format(trend[val_col].mean()),
            annotation_position="right",
        )
        fig.update_layout(**PLOTLY_THEME)
        st.plotly_chart(fig, use_container_width=True)

# ── Tab: Distribution ─────────────────────────────────────
with tabs[tab_idx]:
    tab_idx += 1
    if not num_cols:
        st.info("No numeric columns for distribution analysis.")
    else:
        col_sel = st.selectbox("Select column", num_cols, key="dist_col")
        c1, c2 = st.columns(2)

        with c1:
            cs = stats.column_stats.get(col_sel)
            fig = px.histogram(
                df_filtered, x=col_sel,
                nbins=30,
                title="Distribution: {}".format(col_sel),
                color_discrete_sequence=[COLORS[0]],
                marginal="box",
            )
            if cs and cs.mean:
                fig.add_vline(x=cs.mean, line_dash="dash",
                              line_color="#f7934f",
                              annotation_text="Mean: {:.2f}".format(cs.mean))
                fig.add_vline(x=cs.median, line_dash="dot",
                              line_color="#22d3a5",
                              annotation_text="Median: {:.2f}".format(cs.median))
            fig.update_layout(**PLOTLY_THEME)
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            if cs:
                st.markdown("**Statistical Summary**")
                stat_data = {
                    "Metric": ["Mean", "Median", "Std Dev", "Variance",
                               "Min", "Max", "IQR", "Skewness",
                               "Kurtosis", "Normality", "Outliers (IQR)"],
                    "Value": [
                        cs.mean, cs.median, cs.std, cs.variance,
                        cs.min_val, cs.max_val, cs.iqr, cs.skewness,
                        cs.kurtosis, cs.normality_label,
                        cs.outlier_count_iqr,
                    ],
                    "Interpretation": [
                        "Average value",
                        "Middle value — robust to outliers",
                        "Spread of data",
                        "Squared spread",
                        "Smallest value",
                        "Largest value",
                        "Middle 50% range",
                        cs.skew_label or "-",
                        cs.kurtosis_label or "-",
                        "p={:.4f} ({})".format(
                            cs.normality_pvalue or 0,
                            cs.normality_test or "") if cs.normality_pvalue else "-",
                        cs.outlier_method_recommended,
                    ]
                }
                st.dataframe(
                    pd.DataFrame(stat_data),
                    use_container_width=True, hide_index=True
                )

# ── Tab: Relationships ────────────────────────────────────
with tabs[tab_idx]:
    tab_idx += 1
    if len(num_cols) < 2:
        st.info("Need at least 2 numeric columns for relationship analysis.")
    else:
        c1, c2 = st.columns(2)

        with c1:
            # Correlation heatmap
            corr = df_filtered[num_cols[:10]].corr().round(2)
            fig = px.imshow(
                corr,
                color_continuous_scale="RdBu_r",
                zmin=-1, zmax=1,
                title="Correlation Matrix",
                text_auto=True,
                aspect="auto",
            )
            fig.update_layout(**PLOTLY_THEME)
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            # Scatter with trend line
            x_col = st.selectbox("X axis", num_cols, key="scatter_x")
            y_col = st.selectbox("Y axis", num_cols,
                                 index=min(1, len(num_cols)-1), key="scatter_y")
            color_by = st.selectbox("Color by (optional)",
                                    ["None"] + cat_cols[:5], key="scatter_c")

            sample_df = df_filtered.sample(
                min(2000, len(df_filtered)), random_state=42
            ).dropna(subset=[x_col, y_col])

            fig = px.scatter(
                sample_df,
                x=x_col, y=y_col,
                color=color_by if color_by != "None" else None,
                title="{} vs {}".format(x_col, y_col),
                color_discrete_sequence=COLORS,
                opacity=0.6,
            )
            # Manual trendline using numpy — no statsmodels needed
            try:
                x_vals = sample_df[x_col].values
                y_vals = sample_df[y_col].values
                m, b   = np.polyfit(x_vals, y_vals, 1)
                x_line = np.linspace(x_vals.min(), x_vals.max(), 100)
                y_line = m * x_line + b
                fig.add_trace(go.Scatter(
                    x=x_line, y=y_line,
                    mode="lines",
                    name="Trend",
                    line=dict(color="#f7934f", width=2, dash="dash"),
                ))
            except Exception:
                pass
            fig.update_layout(**PLOTLY_THEME)
            st.plotly_chart(fig, use_container_width=True)

        # Significant correlations callout
        if stats.top_correlations:
            st.markdown("**Statistically Significant Relationships (p < 0.05)**")
            for c in stats.top_correlations[:5]:
                icon = "🔴" if c.strength == "strong" else "🟡"
                st.markdown("{} {}".format(icon, c.label))

# ── Tab: Deep Dive ────────────────────────────────────────
with tabs[tab_idx]:
    tab_idx += 1
    st.markdown("#### Custom Analysis")

    c1, c2, c3 = st.columns(3)
    chart_type = c1.selectbox("Chart type",
                              ["Bar", "Line", "Box Plot",
                               "Violin", "Scatter", "Histogram"],
                              key="dd_type")
    x_ax = c2.selectbox("X axis",
                        (cat_cols + num_cols + dt_cols)[:20],
                        key="dd_x")
    y_ax = c3.selectbox("Y axis", num_cols, key="dd_y") if num_cols else None

    if y_ax:
        try:
            if chart_type == "Bar":
                agg = df_filtered.groupby(x_ax)[y_ax].mean().reset_index()
                fig = px.bar(agg, x=x_ax, y=y_ax,
                             title="{} by {}".format(y_ax, x_ax),
                             color=y_ax,
                             color_continuous_scale=["#90CAF9", "#1a4a8a"])
            elif chart_type == "Line":
                fig = px.line(df_filtered.sort_values(x_ax),
                              x=x_ax, y=y_ax,
                              title="{} over {}".format(y_ax, x_ax))
            elif chart_type == "Box Plot":
                fig = px.box(df_filtered, x=x_ax, y=y_ax,
                             title="{} distribution by {}".format(y_ax, x_ax),
                             color=x_ax,
                             color_discrete_sequence=COLORS)
            elif chart_type == "Violin":
                fig = px.violin(df_filtered, x=x_ax, y=y_ax,
                                title="{} violin by {}".format(y_ax, x_ax),
                                color=x_ax, box=True,
                                color_discrete_sequence=COLORS)
            elif chart_type == "Scatter":
                fig = px.scatter(df_filtered.sample(min(2000, len(df_filtered))),
                                 x=x_ax, y=y_ax, 
                                 title="{} vs {}".format(x_ax, y_ax),
                                 opacity=0.6,
                                 color_discrete_sequence=[COLORS[0]])
            else:  # Histogram
                fig = px.histogram(df_filtered, x=y_ax, nbins=30,
                                   title="Distribution: {}".format(y_ax),
                                   color_discrete_sequence=[COLORS[0]])

            fig.update_layout(**PLOTLY_THEME)
            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error("Chart error: {}".format(str(e)))

st.divider()

# ══════════════════════════════════════════════════════════
#  SECTION 3 — STATISTICAL INSIGHTS PANEL
# ══════════════════════════════════════════════════════════
st.markdown("### 💡 Statistical Insights")

if stats.dataset_insights:
    c1, c2 = st.columns(2)
    mid = len(stats.dataset_insights) // 2
    with c1:
        for insight in stats.dataset_insights[:mid+1]:
            st.info("📌 " + insight)
    with c2:
        for insight in stats.dataset_insights[mid+1:]:
            st.info("📌 " + insight)
else:
    st.success("✅ No major statistical concerns detected in this dataset.")

if stats.recommended_analysis:
    st.markdown("**Recommended next steps:**")
    for rec in stats.recommended_analysis:
        st.markdown("→ " + rec)

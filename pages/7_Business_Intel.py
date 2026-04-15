"""
pages/7_Business_Intel.py — Business Intelligence Dashboard.
Benchmarking, Root Cause, Cohort Analysis, Pareto, Segment Health.
"""
import io
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from core.session_manager import require_data, get_df, get_filename

st.set_page_config(page_title="Business Intelligence", layout="wide")
require_data()
df    = get_df()
fname = get_filename()

from core.bi_engine import (
    run_bi, analyze_cohort, analyze_root_cause,
    analyze_pareto, analyze_segment_health
)

COLORS = ["#1a4a8a", "#2196F3", "#22d3a5", "#f7934f", "#a78bfa", "#f77070"]
PLOTLY_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#f8faff",
    font=dict(family="Helvetica", size=11),
    margin=dict(l=10, r=10, t=40, b=10),
)

st.markdown("## Business Intelligence")
st.caption("{} — {:,} rows, {} columns".format(fname, len(df), len(df.columns)))
st.divider()

@st.cache_data(show_spinner=False)
def get_bi(df_json: str):
    return run_bi(pd.read_json(io.StringIO(df_json)))

with st.spinner("Running BI analysis..."):
    report = get_bi(df.to_json())

# Executive brief
if report.executive_brief:
    st.info("**Executive Brief:** " + report.executive_brief)

# Key insights
if report.key_insights:
    with st.expander("Key Insights ({})".format(len(report.key_insights)),
                     expanded=True):
        for ins in report.key_insights:
            st.markdown("- " + ins)

st.divider()

num_cols = df.select_dtypes(include="number").columns.tolist()
cat_cols = [c for c in df.select_dtypes(include="object").columns
            if 2 <= df[c].nunique() <= 25]

# ══════════════════════════════════════════════════════════
#  TABS
# ══════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Benchmarking",
    "Root Cause Analysis",
    "Cohort Analysis",
    "Pareto Analysis",
    "Segment Health",
])

# ── Tab 1: Benchmarking ───────────────────────────────────
with tab1:
    st.markdown("### Benchmarking")
    st.caption("How each metric distributes across the dataset — percentiles, thresholds, variation.")

    if not report.benchmarks:
        st.info("No numeric columns found for benchmarking.")
    else:
        # Summary cards
        cols_ui = st.columns(min(4, len(report.benchmarks)))
        for i, bm in enumerate(report.benchmarks[:4]):
            with cols_ui[i]:
                st.metric(bm.column, "{:.2f}".format(bm.mean),
                          delta="median {:.2f}".format(bm.median))
                st.caption("CV={:.2f} | {:.0f}% above avg".format(
                    bm.cv, bm.above_avg_pct))

        # Detailed table
        rows = []
        for bm in report.benchmarks:
            rows.append({
                "Column":        bm.column,
                "Mean":          bm.mean,
                "Median":        bm.median,
                "P25":           bm.p25,
                "P75":           bm.p75,
                "Top 10%":       bm.top_10_pct,
                "Bottom 10%":    bm.bottom_10_pct,
                "Above Avg %":   "{}%".format(bm.above_avg_pct),
                "CV":            bm.cv,
                "Variation":     bm.benchmark_label.split(" — ")[0],
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True,
                     hide_index=True)

        # Select one for deep dive
        sel = st.selectbox("Select column for benchmark chart",
                           [bm.column for bm in report.benchmarks],
                           key="bm_col")
        bm  = next(b for b in report.benchmarks if b.column == sel)

        st.markdown(
            "<div style='background:#f0f4ff;border-left:4px solid #1a4a8a;"
            "padding:12px 16px;border-radius:4px'>{}</div>".format(
                bm.interpretation),
            unsafe_allow_html=True
        )

        # Distribution with percentile lines
        col_data = df[sel].dropna()
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=col_data, name="Frequency",
            marker_color=COLORS[0], opacity=0.75, nbinsx=30,
        ))
        for val, label, color in [
            (bm.bottom_10_pct, "P10", "#f77070"),
            (bm.p25,           "P25", "#f7934f"),
            (bm.median,        "Median", "#22d3a5"),
            (bm.mean,          "Mean", "#1a4a8a"),
            (bm.p75,           "P75", "#f7934f"),
            (bm.top_10_pct,    "P90", "#f77070"),
        ]:
            fig.add_vline(x=val, line_dash="dash", line_color=color,
                          annotation_text="{} {:.2f}".format(label, val),
                          annotation_position="top")
        fig.update_layout(
            title="Distribution with Percentile Benchmarks: {}".format(sel),
            **PLOTLY_BASE
        )
        st.plotly_chart(fig, use_container_width=True)

# ── Tab 2: Root Cause Analysis ────────────────────────────
with tab2:
    st.markdown("### Root Cause Analysis")
    st.caption("Why are some records underperforming? Compares bottom 25% vs top 75%.")

    c1, c2 = st.columns(2)
    with c1:
        target_col = st.selectbox("Target metric (what to diagnose)",
                                  num_cols, key="rc_target")
    with c2:
        threshold = st.slider("Low performer threshold (%)",
                              10, 40, 25, key="rc_thresh",
                              help="Bottom X% = low performers")

    if st.button("Run Root Cause Analysis", type="primary",
                 use_container_width=False):
        with st.spinner("Analyzing root causes..."):
            rc = analyze_root_cause(df, target_col, threshold)

        # Summary
        c1, c2, c3 = st.columns(3)
        c1.metric("Low Performers",
                  "{:,}".format(rc.n_low_performers),
                  delta="{:.1f}% of data".format(rc.low_pct),
                  delta_color="inverse")
        c2.metric("Threshold", "{:.2f}".format(rc.low_performer_threshold))
        c3.metric("Drivers Found", str(len(rc.drivers)))

        st.markdown(
            "<div style='background:#fff0f0;border-left:4px solid #f77070;"
            "padding:12px 16px;border-radius:4px;margin:12px 0'>"
            "<b>Root Cause:</b> {}</div>".format(rc.interpretation),
            unsafe_allow_html=True
        )

        if rc.drivers:
            st.markdown("#### Top Drivers")
            rows = []
            for d in rc.drivers[:8]:
                rows.append({
                    "Factor":    d["factor"],
                    "Impact":    "{:.1f}%".format(d["impact"] * 100),
                    "Type":      d["dtype"],
                    "p-value":   d["p_value"],
                    "Detail":    d["detail"][:80],
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True,
                         hide_index=True)

            # Impact bar chart
            numeric_drivers = [d for d in rc.drivers if d["dtype"] == "numeric"]
            if numeric_drivers:
                fig = go.Figure(go.Bar(
                    x=[d["factor"] for d in numeric_drivers[:8]],
                    y=[d["impact"] * 100 for d in numeric_drivers[:8]],
                    marker_color=[
                        "#f77070" if d["direction"] == "negative"
                        else "#22d3a5"
                        for d in numeric_drivers[:8]
                    ],
                    text=["{:.0f}%".format(d["impact"]*100)
                          for d in numeric_drivers[:8]],
                    textposition="outside",
                ))
                fig.update_layout(
                    title="Driver Impact on Low '{}' Performance".format(target_col),
                    xaxis_title="Factor", yaxis_title="Impact (%)",
                    **PLOTLY_BASE
                )
                st.plotly_chart(fig, use_container_width=True)

        if rc.recommendations:
            st.markdown("#### Recommendations")
            for i, rec in enumerate(rc.recommendations, 1):
                st.success("{}. {}".format(i, rec))
    else:
        # Show cached results
        if report.root_causes:
            rc = next((r for r in report.root_causes
                       if r.target_col == num_cols[0]), report.root_causes[0])
            st.info("Auto-analysis result for '{}':".format(rc.target_col))
            st.markdown(rc.interpretation)
            if rc.recommendations:
                for rec in rc.recommendations:
                    st.success(rec)

# ── Tab 3: Cohort Analysis ────────────────────────────────
with tab3:
    st.markdown("### Cohort Analysis")
    st.caption("Compare metric performance across segments with statistical significance testing.")

    if not cat_cols:
        st.info("Need categorical columns with 2-25 unique values.")
    elif not num_cols:
        st.info("Need numeric columns for cohort metric.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            cohort_col = st.selectbox("Segment by (cohort column)",
                                      cat_cols, key="coh_col")
        with c2:
            metric_col = st.selectbox("Metric to compare",
                                      num_cols, key="coh_metric")

        @st.cache_data(show_spinner=False)
        def get_cohort(df_json, cc, mc):
            return analyze_cohort(pd.read_json(io.StringIO(df_json)), cc, mc)

        coh = get_cohort(df.to_json(), cohort_col, metric_col)

        # Significance badge
        if coh.is_significant:
            st.error("SIGNIFICANT DIFFERENCE DETECTED — "
                     "{} | p={:.4f}".format(coh.test_used, coh.p_value))
        else:
            st.success("No significant difference — "
                       "{} | p={:.4f}".format(coh.test_used, coh.p_value))

        st.markdown(coh.interpretation)

        # Cohort table
        if coh.cohorts:
            rows = []
            for c in coh.cohorts:
                rows.append({
                    "Rank":        c["rank"],
                    "Cohort":      c["name"],
                    "N":           c["n"],
                    "Mean":        c["mean"],
                    "Median":      c["median"],
                    "Std":         c["std"],
                    "vs Avg":      "{}%".format(c["vs_avg_pct"]),
                    "Status":      c["status"].upper(),
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True,
                         hide_index=True)

            # Bar chart with avg line
            fig = go.Figure()
            bar_colors = [
                COLORS[0] if c["status"] == "top"
                else "#f77070" if c["status"] == "bottom"
                else COLORS[2]
                for c in coh.cohorts
            ]
            fig.add_trace(go.Bar(
                x=[c["name"] for c in coh.cohorts],
                y=[c["mean"] for c in coh.cohorts],
                marker_color=bar_colors,
                text=["{:.2f}".format(c["mean"]) for c in coh.cohorts],
                textposition="outside",
                name="Mean {}".format(metric_col),
            ))
            # Dataset average line
            overall_mean = float(df[metric_col].mean())
            fig.add_hline(
                y=overall_mean, line_dash="dash",
                line_color="#f7934f",
                annotation_text="Dataset Avg={:.2f}".format(overall_mean),
            )
            fig.update_layout(
                title="Avg '{}' by '{}' Cohort".format(metric_col, cohort_col),
                **PLOTLY_BASE
            )
            st.plotly_chart(fig, use_container_width=True)

            # Color guide
            c1, c2, c3 = st.columns(3)
            c1.info("Blue — Above average (top tier)")
            c2.success("Green — Near average")
            c3.error("Red — Below average (needs attention)")

        if coh.recommendations:
            st.markdown("#### Recommendations")
            for rec in coh.recommendations:
                st.info(rec)

# ── Tab 4: Pareto Analysis ────────────────────────────────
with tab4:
    st.markdown("### Pareto Analysis — 80/20 Rule")
    st.caption("Which 20% of groups drive 80% of value?")

    if not cat_cols or not num_cols:
        st.info("Need both categorical and numeric columns.")
    else:
        c1, c2, c3 = st.columns(3)
        with c1:
            par_group = st.selectbox("Group by", cat_cols, key="par_grp")
        with c2:
            par_value = st.selectbox("Value metric", num_cols, key="par_val")
        with c3:
            par_agg   = st.radio("Aggregation", ["sum", "mean"],
                                  horizontal=True, key="par_agg")

        @st.cache_data(show_spinner=False)
        def get_pareto(df_json, gc, vc, fn):
            return analyze_pareto(pd.read_json(io.StringIO(df_json)), gc, vc, fn)

        par = get_pareto(df.to_json(), par_group, par_value, par_agg)

        # Verdict
        if par.pareto_holds:
            st.success(
                "Pareto HOLDS — top {:.0f}% of '{}' groups ({} groups) "
                "account for {:.0f}% of total '{}'.".format(
                    20, par_group, par.top_20_pct_groups,
                    par.top_groups_share, par_value)
            )
        else:
            st.info(
                "Pareto does NOT hold — value is spread across groups. "
                "Top {:.0f}% drives only {:.0f}% of '{}'.".format(
                    20, par.top_groups_share, par_value)
            )

        st.markdown(par.interpretation)

        if par.groups:
            df_par = pd.DataFrame(par.groups)

            # Pareto chart — bar + cumulative line
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=df_par["name"], y=df_par["pct_of_total"],
                name="% of Total",
                marker_color=[
                    COLORS[0] if g["in_top_20"] else COLORS[2]
                    for g in par.groups
                ],
            ))
            fig.add_trace(go.Scatter(
                x=df_par["name"], y=df_par["cumulative_pct"],
                name="Cumulative %",
                yaxis="y2", mode="lines+markers",
                line=dict(color="#f7934f", width=2),
                marker=dict(size=5),
            ))
            fig.add_hline(y=80, line_dash="dash",
                          line_color="#f77070",
                          annotation_text="80% threshold",
                          yref="y2")
            fig.update_layout(
                title="Pareto Chart — {} by {}".format(par_value, par_group),
                yaxis=dict(title="% of Total"),
                yaxis2=dict(title="Cumulative %", overlaying="y",
                            side="right", range=[0, 110]),
                **PLOTLY_BASE,
                legend=dict(orientation="h", y=1.1),
            )
            st.plotly_chart(fig, use_container_width=True)

            # Table
            st.dataframe(
                df_par[["rank", "name", "value", "pct_of_total",
                         "cumulative_pct", "in_top_20"]].rename(columns={
                    "rank": "Rank", "name": "Group",
                    "value": par_agg.title() + " " + par_value,
                    "pct_of_total": "% of Total",
                    "cumulative_pct": "Cumulative %",
                    "in_top_20": "Top 20%?",
                }),
                use_container_width=True, hide_index=True
            )

# ── Tab 5: Segment Health ─────────────────────────────────
with tab5:
    st.markdown("### Segment Health Scoring")
    st.caption("Overall health score for each segment across multiple metrics. 0=poor, 100=excellent.")

    if not cat_cols or not num_cols:
        st.info("Need categorical and numeric columns.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            seg_col = st.selectbox("Segment column", cat_cols, key="seg_col")
        with c2:
            seg_metrics = st.multiselect(
                "Metrics to score",
                num_cols,
                default=num_cols[:4],
                key="seg_metrics"
            )

        if not seg_metrics:
            st.warning("Select at least one metric.")
        else:
            @st.cache_data(show_spinner=False)
            def get_seg_health(df_json, sc, sm):
                return analyze_segment_health(
                    pd.read_json(io.StringIO(df_json)), sc, sm)

            segs = get_seg_health(df.to_json(), seg_col, seg_metrics)

            if not segs:
                st.info("Could not compute segment health — "
                        "need segments with 5+ records each.")
            else:
                # Health score cards
                n_segs = min(len(segs), 6)
                cols_ui = st.columns(n_segs)
                for i, seg in enumerate(segs[:n_segs]):
                    with cols_ui[i]:
                        color = ("#22d3a5" if seg.health_score >= 70
                                 else "#f7934f" if seg.health_score >= 45
                                 else "#f77070")
                        st.markdown(
                            "<div style='background:{};border-radius:8px;"
                            "padding:12px;text-align:center;color:white'>"
                            "<div style='font-size:13px;font-weight:700'>{}</div>"
                            "<div style='font-size:26px;font-weight:900'>{:.0f}</div>"
                            "<div style='font-size:11px'>/ 100</div>"
                            "<div style='font-size:11px'>n={:,}</div>"
                            "</div>".format(
                                color, seg.segment_name[:12],
                                seg.health_score, seg.n),
                            unsafe_allow_html=True
                        )

                st.markdown("<br>", unsafe_allow_html=True)

                # Detailed segment view
                sel_seg = st.selectbox(
                    "Drill into segment",
                    [s.segment_name for s in segs],
                    key="seg_drill"
                )
                seg_data = next(s for s in segs if s.segment_name == sel_seg)

                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**Strengths**")
                    if seg_data.strengths:
                        for s in seg_data.strengths:
                            pct = seg_data.metrics[s]["vs_avg"]
                            st.success("{} (+{:.1f}% vs avg)".format(s, pct))
                    else:
                        st.info("No significant strengths detected.")
                with c2:
                    st.markdown("**Weaknesses**")
                    if seg_data.weaknesses:
                        for w in seg_data.weaknesses:
                            pct = seg_data.metrics[w]["vs_avg"]
                            st.error("{} ({:.1f}% vs avg)".format(w, pct))
                    else:
                        st.success("No significant weaknesses.")

                st.info("**Opportunity:** " + seg_data.opportunity)

                # Radar-style metric comparison
                if len(seg_metrics) >= 3:
                    # Normalize all segments for radar
                    all_means = {
                        col: float(df[col].mean())
                        for col in seg_metrics if col in df.columns
                    }
                    fig = go.Figure()
                    for seg in segs[:5]:
                        vals = [
                            min(200, max(0,
                                50 + seg.metrics.get(col, {}).get("vs_avg", 0)))
                            for col in seg_metrics
                            if col in seg.metrics
                        ]
                        cols_used = [col for col in seg_metrics
                                     if col in seg.metrics]
                        if vals:
                            fig.add_trace(go.Scatterpolar(
                                r=vals + [vals[0]],
                                theta=cols_used + [cols_used[0]],
                                name=seg.segment_name,
                                fill="toself", opacity=0.3,
                            ))
                    fig.update_layout(
                        polar=dict(
                            radialaxis=dict(visible=True, range=[0, 150])
                        ),
                        title="Segment Comparison (100 = dataset average)",
                        **PLOTLY_BASE,
                    )
                    st.plotly_chart(fig, use_container_width=True)

                # Full metrics table
                st.markdown("#### Metric Detail — {}".format(sel_seg))
                metric_rows = []
                for col, m in seg_data.metrics.items():
                    metric_rows.append({
                        "Metric":      col,
                        "Segment Mean":m["mean"],
                        "vs Average":  "{}%".format(m["vs_avg"]),
                        "Rank":        "{} / {}".format(m["rank"], m["n_total"]),
                        "Status":      m["status"].upper(),
                    })
                st.dataframe(pd.DataFrame(metric_rows),
                             use_container_width=True, hide_index=True)


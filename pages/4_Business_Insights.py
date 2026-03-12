import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

if "df_active" not in st.session_state:
    st.warning("Please upload a dataset first.")
    st.stop()

from core.story_engine import generate_story

st.set_page_config(page_title="Business Insights", layout="wide")

COLORS = ["#1a4a8a", "#2196F3", "#22d3a5", "#f7934f", "#a78bfa", "#f77070"]

@st.cache_data(show_spinner=False)
def get_story(df_json: str):
    df = pd.read_json(df_json)
    return generate_story(df)

df = st.session_state["df_active"]
fname = st.session_state.get("filename", "Dataset")

with st.spinner("Generating business insights..."):
    report = get_story(df.to_json())

# ══════════════════════════════════════════════════════════
#  HEADER — Domain badge + headline
# ══════════════════════════════════════════════════════════
domain_colors = {
    "ecommerce":  "#2196F3",
    "hr":         "#22d3a5",
    "finance":    "#1a4a8a",
    "marketing":  "#a78bfa",
    "healthcare": "#f77070",
    "logistics":  "#f7934f",
    "general":    "#646882",
}
domain_color = domain_colors.get(report.domain, "#646882")

col_title, col_badge = st.columns([4, 1])
with col_title:
    st.markdown("## Business Insights Report")
    st.caption("{} — {:,} rows, {} columns".format(
        fname, len(df), len(df.columns)))
with col_badge:
    st.markdown(
        "<div style='background:{};color:white;padding:8px 14px;"
        "border-radius:20px;text-align:center;font-weight:700;"
        "font-size:13px;margin-top:10px'>{} Dataset</div>".format(
            domain_color, report.domain.upper()
        ),
        unsafe_allow_html=True
    )

st.divider()

# ══════════════════════════════════════════════════════════
#  HEADLINE FINDING
# ══════════════════════════════════════════════════════════
is_risk = report.headline.startswith("RISK:")
if is_risk:
    st.error("**Top Finding:** " + report.headline)
else:
    st.info("**Top Finding:** " + report.headline)

# ══════════════════════════════════════════════════════════
#  QUICK STATS ROW
# ══════════════════════════════════════════════════════════
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Domain",        report.domain.title())
c2.metric("Findings",      str(len(report.key_findings)))
c3.metric("Risks",         str(len(report.business_risks)),
          delta="-{}".format(len(report.business_risks))
          if report.business_risks else None,
          delta_color="inverse")
c4.metric("Opportunities", str(len(report.opportunities)))
c5.metric("Actions",       str(len(report.recommended_actions)))

st.divider()

# ══════════════════════════════════════════════════════════
#  TABS
# ══════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Executive Summary",
    "Key Findings",
    "Risks & Opportunities",
    "Column Intelligence",
    "Action Plan",
])

# ── Tab 1: Executive Summary ──────────────────────────────
with tab1:
    st.markdown("### Executive Summary")
    st.markdown(
        "<div style='background:#f0f4ff;border-left:4px solid {};"
        "padding:18px 20px;border-radius:4px;font-size:15px;"
        "line-height:1.8;color:#1e1e28'>{}</div>".format(
            domain_color, report.executive_summary
        ),
        unsafe_allow_html=True
    )

    st.markdown("<br>", unsafe_allow_html=True)

    col_q, col_c = st.columns(2)
    with col_q:
        st.markdown("**Data Quality Verdict**")
        if "GOOD" in report.data_quality_verdict:
            st.success(report.data_quality_verdict)
        elif "FAIR" in report.data_quality_verdict:
            st.warning(report.data_quality_verdict)
        else:
            st.error(report.data_quality_verdict)

    with col_c:
        st.markdown("**Analysis Confidence**")
        if "HIGH" in report.analysis_confidence:
            st.success(report.analysis_confidence)
        elif "MEDIUM" in report.analysis_confidence:
            st.warning(report.analysis_confidence)
        else:
            st.error(report.analysis_confidence)

    # Domain confidence bar
    if report.domain != "general":
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**Domain Detection Confidence**")
        conf_pct = int(report.domain_confidence * 100)
        st.progress(min(conf_pct / 100, 1.0))
        st.caption(
            "{}% confident this is a {} dataset based on column names.".format(
                conf_pct * 5, report.domain
            )
        )

# ── Tab 2: Key Findings ───────────────────────────────────
with tab2:
    st.markdown("### Key Findings")

    if not report.key_findings:
        st.info("No domain-specific findings. Try uploading a dataset with "
                "more business context.")
    else:
        for i, finding in enumerate(report.key_findings, 1):
            st.markdown(
                "<div style='background:#f8faff;border:1px solid #d0d8f0;"
                "border-radius:8px;padding:14px 18px;margin-bottom:10px'>"
                "<span style='color:{};font-weight:700;font-size:13px'>"
                "FINDING {}</span><br>"
                "<span style='font-size:14px;color:#1e1e28;line-height:1.7'>"
                "{}</span></div>".format(domain_color, i, finding),
                unsafe_allow_html=True
            )

    # Statistical anomalies
    if report.anomalies:
        st.markdown("### Statistical Anomalies")
        st.caption("Unusual patterns detected that may affect analysis accuracy.")
        for anomaly in report.anomalies:
            st.warning(anomaly)

# ── Tab 3: Risks & Opportunities ─────────────────────────
with tab3:
    col_r, col_o = st.columns(2)

    with col_r:
        st.markdown("### Business Risks")
        if not report.business_risks:
            st.success("No major business risks detected.")
        else:
            for i, risk in enumerate(report.business_risks, 1):
                severity = "CRITICAL" if i == 1 else "HIGH" if i == 2 else "MEDIUM"
                color    = "#f77070" if i == 1 else "#f7934f" if i == 2 else "#ffd43b"
                st.markdown(
                    "<div style='border-left:4px solid {};background:#fff8f8;"
                    "padding:12px 16px;border-radius:4px;margin-bottom:10px'>"
                    "<span style='color:{};font-size:11px;font-weight:700'>"
                    "{}</span><br>"
                    "<span style='font-size:13px;color:#1e1e28'>{}</span>"
                    "</div>".format(color, color, severity, risk),
                    unsafe_allow_html=True
                )

    with col_o:
        st.markdown("### Growth Opportunities")
        if not report.opportunities:
            st.info("Generate more insights by uploading data with "
                    "revenue, satisfaction, or performance metrics.")
        else:
            for i, opp in enumerate(report.opportunities, 1):
                st.markdown(
                    "<div style='border-left:4px solid #22d3a5;"
                    "background:#f0fff8;padding:12px 16px;"
                    "border-radius:4px;margin-bottom:10px'>"
                    "<span style='color:#22d3a5;font-size:11px;font-weight:700'>"
                    "OPPORTUNITY {}</span><br>"
                    "<span style='font-size:13px;color:#1e1e28'>{}</span>"
                    "</div>".format(i, opp),
                    unsafe_allow_html=True
                )

# ── Tab 4: Column Intelligence ────────────────────────────
with tab4:
    st.markdown("### Column Intelligence")
    st.caption("Every column analyzed — what it means for your business.")

    severity_order = {"critical": 0, "warning": 1, "info": 2, "positive": 3}
    sorted_insights = sorted(
        report.column_insights,
        key=lambda x: severity_order.get(x.severity, 99)
    )

    for ci in sorted_insights:
        sev_config = {
            "critical": ("[CRITICAL]", "#f77070", "#fff0f0"),
            "warning":  ("[WARNING]",  "#f7934f", "#fff8f0"),
            "positive": ("[GOOD]",     "#22d3a5", "#f0fff8"),
            "info":     ("[INFO]",     "#2196F3", "#f0f8ff"),
        }
        tag, color, bg = sev_config.get(ci.severity, ("[INFO]", "#2196F3", "#f0f8ff"))

        with st.expander(
            "{} {} — {}".format(tag, ci.column, ci.finding)
        ):
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("**Type:** {}".format(ci.dtype))
                st.markdown("**Metric:** {}".format(ci.metric))
            with col_b:
                st.markdown("**Finding:** {}".format(ci.finding))
                st.markdown("**Recommendation:** {}".format(ci.recommendation))

    # Visual severity summary
    st.markdown("### Severity Summary")
    sev_counts = {}
    for ci in report.column_insights:
        sev_counts[ci.severity] = sev_counts.get(ci.severity, 0) + 1

    if sev_counts:
        fig = go.Figure(go.Bar(
            x=list(sev_counts.keys()),
            y=list(sev_counts.values()),
            marker_color=["#f77070" if s == "critical"
                          else "#f7934f" if s == "warning"
                          else "#22d3a5" if s == "positive"
                          else "#2196F3"
                          for s in sev_counts.keys()],
            text=list(sev_counts.values()),
            textposition="outside",
        ))
        fig.update_layout(
            title="Column Health Distribution",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#f8faff",
            font=dict(family="Helvetica", size=11),
            margin=dict(l=10, r=10, t=40, b=10),
            showlegend=False,
            height=300,
        )
        st.plotly_chart(fig, use_container_width=True)

# ── Tab 5: Action Plan ────────────────────────────────────
with tab5:
    st.markdown("### Recommended Action Plan")
    st.caption("Prioritized steps based on findings — do these first.")

    priority_labels = ["IMMEDIATE", "SHORT TERM", "SHORT TERM",
                       "MEDIUM TERM", "MEDIUM TERM", "LONG TERM"]
    priority_colors = ["#f77070", "#f7934f", "#f7934f",
                       "#ffd43b", "#ffd43b", "#22d3a5"]

    for i, action in enumerate(report.recommended_actions):
        label = priority_labels[min(i, len(priority_labels)-1)]
        color = priority_colors[min(i, len(priority_colors)-1)]
        st.markdown(
            "<div style='display:flex;align-items:flex-start;"
            "gap:12px;margin-bottom:12px'>"
            "<div style='background:{};color:white;padding:4px 10px;"
            "border-radius:12px;font-size:11px;font-weight:700;"
            "white-space:nowrap;margin-top:2px'>{}</div>"
            "<div style='font-size:14px;color:#1e1e28;line-height:1.6'>{}</div>"
            "</div>".format(color, label, action),
            unsafe_allow_html=True
        )

    st.divider()
    st.markdown("### Next Steps in DataForge AI")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.info("**ML Predictions**\n\nBuild predictive models on this dataset automatically.")
    with c2:
        st.info("**Deep EDA**\n\nSenior analyst level statistical deep dive.")
    with c3:
        st.info("**Generate Report**\n\nExport this analysis as a branded PDF report.")

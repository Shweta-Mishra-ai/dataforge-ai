import streamlit as st
import pandas as pd
from core.data_profiler import profile_dataset
from core.report_engine import ReportConfig
from core.chart_exporter import fig_to_bytes
from core.chart_engine import recommend_charts
from core.pdf_builder import build_pdf
from ai.llm_client import LLMClient
from ai.report_narrator import (
    generate_executive_summary,
    generate_chart_narrative,
    generate_recommendations,
)
from components.kpi_cards import inject_global_css, quality_score_banner

st.set_page_config(
    page_title="Reports — DataForge AI",
    page_icon="📄",
    layout="wide"
)
inject_global_css()

# ── State guard ────────────────────────────────────────────
if "df_active" not in st.session_state:
    st.warning("⚠️ No data loaded.")
    st.page_link("pages/1_📥_Data_Upload.py", label="← Go to Upload", icon="📥")
    st.stop()

df      = st.session_state["df_active"]
profile = st.session_state.get("profile") or profile_dataset(df)
fname   = st.session_state.get("filename", "data")

# ── API Key ────────────────────────────────────────────────
groq_key = st.secrets.get("GROQ_API_KEY", "")
if not groq_key:
    st.error("⚠️ GROQ_API_KEY not found in .streamlit/secrets.toml")
    st.stop()

client = LLMClient(api_key=groq_key)

# ── Header ─────────────────────────────────────────────────
st.title("📄 Professional Report Generator")
st.markdown("Generate a complete **PDF analysis report** with charts, AI insights, and recommendations.")
st.divider()

# ── Quality score ──────────────────────────────────────────
col_a, col_b = st.columns([1, 3])
with col_a:
    quality_score_banner(profile.overall_quality_score)
with col_b:
    st.markdown(f"""
    **Dataset:** `{fname}`
    **Rows:** {profile.rows:,} &nbsp;|&nbsp;
    **Columns:** {profile.cols} &nbsp;|&nbsp;
    **Quality:** {profile.overall_quality_score}/100
    """)

st.divider()

# ── Report config ──────────────────────────────────────────
st.subheader("⚙️ Report Settings")

c1, c2 = st.columns(2)
with c1:
    report_title   = st.text_input("Report Title", value="Data Analysis Report")
    client_name    = st.text_input("Client Name",  value="Client")
with c2:
    prepared_by    = st.text_input("Prepared By",  value="DataForge AI")
    confidential   = st.checkbox("Mark as Confidential", value=True)

st.divider()

# ── Chart selection ────────────────────────────────────────
st.subheader("📊 Select Charts to Include")

with st.spinner("Generating chart previews..."):
    all_charts = recommend_charts(df)

if not all_charts:
    st.warning("Not enough data to generate charts.")
    st.stop()

selected_charts = []
cols = st.columns(min(len(all_charts), 3))
for i, (title, fig) in enumerate(all_charts):
    with cols[i % 3]:
        include = st.checkbox(f"Include: {title}", value=True, key=f"ch_{i}")
        st.plotly_chart(fig, use_container_width=True)
        if include:
            selected_charts.append((title, fig))

st.divider()

# ── Generate button ────────────────────────────────────────
st.subheader("🚀 Generate Report")

if st.button("📄 Generate PDF Report", type="primary", use_container_width=True):

    if not selected_charts:
        st.error("Please select at least one chart.")
        st.stop()

    config = ReportConfig(
        title=report_title,
        client_name=client_name,
        prepared_by=prepared_by,
        confidential=confidential,
    )

    # ── Step 1 — Executive Summary ─────────────────────────
    with st.status("🤖 Generating AI executive summary...", expanded=True) as status:
        executive_summary = generate_executive_summary(df, profile, client)
        status.update(label="✅ Executive summary done", state="complete")

    # ── Step 2 — Export charts ─────────────────────────────
    with st.status("📊 Exporting charts...", expanded=True) as status:
        chart_images    = []
        chart_titles    = []
        chart_narratives = []

        for title, fig in selected_charts:
            try:
                img_bytes = fig_to_bytes(fig, width=900, height=450)
                chart_images.append(img_bytes)
                chart_titles.append(title)
            except Exception as e:
                st.warning(f"Chart '{title}' export failed: {e}")

        status.update(label=f"✅ {len(chart_images)} charts exported", state="complete")

    # ── Step 3 — Chart narratives ──────────────────────────
    with st.status("✍️ Writing chart analysis...", expanded=True) as status:
        num_cols = profile.numeric_cols
        cat_cols = profile.categorical_cols

        for i, (title, fig) in enumerate(selected_charts[:len(chart_images)]):
            narrative = generate_chart_narrative(
                chart_title=title,
                chart_type=title.split(":")[0] if ":" in title else "Chart",
                df=df,
                x_col=cat_cols[0] if cat_cols else (num_cols[0] if num_cols else ""),
                y_col=num_cols[0] if num_cols else "",
                client=client,
            )
            chart_narratives.append(narrative)

        status.update(label="✅ Chart analysis written", state="complete")

    # ── Step 4 — Recommendations ───────────────────────────
    with st.status("🎯 Generating recommendations...", expanded=True) as status:
        recommendations = generate_recommendations(profile, df, client)
        status.update(label="✅ Recommendations ready", state="complete")

    # ── Step 5 — Build PDF ─────────────────────────────────
    with st.status("📄 Building PDF...", expanded=True) as status:
        try:
            pdf_bytes = build_pdf(
                df=df,
                profile=profile,
                config=config,
                chart_images=chart_images,
                chart_titles=chart_titles,
                chart_narratives=chart_narratives,
                executive_summary=executive_summary,
                recommendations=recommendations,
            )
            status.update(label="✅ PDF ready!", state="complete")
        except Exception as e:
            st.error(f"PDF generation failed: {e}")
            st.stop()

    # ── Download button ────────────────────────────────────
    st.success("🎉 Report generated successfully!")
    st.download_button(
        label="⬇️ Download PDF Report",
        data=pdf_bytes,
        file_name=f"dataforge_report_{fname}.pdf",
        mime="application/pdf",
        type="primary",
        use_container_width=True,
    )

    # ── Preview findings ───────────────────────────────────
    st.divider()
    st.subheader("📋 Report Preview")

    with st.expander("Executive Summary", expanded=True):
        for item in executive_summary:
            finding = item.get("finding", str(item))
            ftype   = item.get("type", "neutral")
            icon    = "✅" if ftype == "positive" else "⚠️" if ftype == "negative" else "📌"
            st.markdown(f"{icon} {finding}")

    with st.expander("Recommendations"):
        if recommendations.get("immediate"):
            st.markdown("**🔴 Immediate:**")
            for r in recommendations["immediate"]:
                st.markdown(f"- {r}")
        if recommendations.get("short_term"):
            st.markdown("**🟠 Short Term:**")
            for r in recommendations["short_term"]:
                st.markdown(f"- {r}")
        if recommendations.get("long_term"):
            st.markdown("**🟢 Long Term:**")
            for r in recommendations["long_term"]:
                st.markdown(f"- {r}")

# ── CSV Export ─────────────────────────────────────────────
st.divider()
st.subheader("📥 Quick Export")
c1, c2 = st.columns(2)

with c1:
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download CSV",
        data=csv,
        file_name=f"cleaned_{fname}.csv",
        mime="text/csv",
        use_container_width=True,
    )

with c2:
    try:
        import io
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
            df.to_excel(writer, sheet_name="Cleaned Data", index=False)
            pd.DataFrame([{
                "Column":  p.name,
                "Missing %": p.missing_pct,
                "Score":   p.quality_score,
            } for p in profile.column_profiles]).to_excel(
                writer, sheet_name="Quality Report", index=False
            )
        buf.seek(0)
        st.download_button(
            "Download Excel",
            data=buf,
            file_name=f"cleaned_{fname}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
    except ImportError:
        st.error("Run: pip install xlsxwriter")

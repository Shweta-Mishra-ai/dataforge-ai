import streamlit as st
import pandas as pd
import io
from core.data_profiler import profile_dataset
from core.report_engine import ReportConfig, THEMES
from core.chart_exporter import generate_all_charts
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
    st.warning("No data loaded.")
    st.page_link("pages/1_📥_Data_Upload.py",
                 label="Go to Upload", icon="📥")
    st.stop()

df      = st.session_state["df_active"]
profile = st.session_state.get("profile") or profile_dataset(df)
fname   = st.session_state.get("filename", "data")

groq_key = st.secrets.get("GROQ_API_KEY", "")
if not groq_key:
    st.error("GROQ_API_KEY not found in .streamlit/secrets.toml")
    st.stop()

client = LLMClient(api_key=groq_key)

# ── Header ─────────────────────────────────────────────────
st.title("📄 Professional Report Generator")
st.markdown(
    "Generate a complete **PDF analysis report** — "
    "charts, AI insights, and recommendations."
)
st.divider()

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

# ── Report settings ────────────────────────────────────────
st.subheader("Report Settings")

c1, c2 = st.columns(2)
with c1:
    report_title = st.text_input(
        "Report Title",
        value="Data Analysis Report"
    )
    client_name  = st.text_input(
        "Client Name",
        value="Client"
    )
with c2:
    prepared_by  = st.text_input(
        "Prepared By",
        value="DataForge AI"
    )
    confidential = st.checkbox(
        "Mark as Confidential",
        value=True
    )

# ── Theme selector ─────────────────────────────────────────
st.subheader("Report Style")

theme_col1, theme_col2, theme_col3 = st.columns(3)

with theme_col1:
    st.markdown("""
    <div style="background:#f0f4ff;border-top:4px solid #1a4a8a;
    border-radius:8px;padding:14px;text-align:center">
    <b style="color:#1a4a8a">Corporate Light</b><br>
    <span style="font-size:11px;color:#666">
    White background<br>Navy blue accents<br>
    McKinsey / Deloitte style
    </span>
    </div>
    """, unsafe_allow_html=True)

with theme_col2:
    st.markdown("""
    <div style="background:#0e0f1a;border-top:4px solid #4f8ef7;
    border-radius:8px;padding:14px;text-align:center">
    <b style="color:#4f8ef7">Dark Tech</b><br>
    <span style="font-size:11px;color:#636a8a">
    Dark background<br>Blue accents<br>
    Modern SaaS style
    </span>
    </div>
    """, unsafe_allow_html=True)

with theme_col3:
    st.markdown("""
    <div style="background:#f0faf5;border-top:4px solid #1a6b4a;
    border-radius:8px;padding:14px;text-align:center">
    <b style="color:#1a6b4a">Executive Green</b><br>
    <span style="font-size:11px;color:#666">
    White background<br>Green accents<br>
    Finance / Banking style
    </span>
    </div>
    """, unsafe_allow_html=True)

theme_name = st.selectbox(
    "Select Theme:",
    list(THEMES.keys()),
    index=0
)

st.divider()

# ── Chart preview + selection ──────────────────────────────
st.subheader("Charts to Include")
st.caption("Charts are generated using the selected theme colors.")

with st.spinner("Generating chart previews..."):
    try:
        chart_data = generate_all_charts(
            df,
            theme_name=theme_name,
            max_charts=5
        )
    except Exception as e:
        st.error(f"Chart generation failed: {e}")
        st.stop()

if not chart_data:
    st.warning("Not enough data to generate charts.")
    st.stop()

selected_charts = []
cols = st.columns(min(len(chart_data), 3))

for i, (title, img_bytes) in enumerate(chart_data):
    with cols[i % 3]:
        include = st.checkbox(
            f"{title}",
            value=True,
            key=f"ch_{i}"
        )
        # Show preview using st.image
        st.image(img_bytes, use_column_width=True)
        if include:
            selected_charts.append((title, img_bytes))

st.divider()

# ── Generate button ────────────────────────────────────────
st.subheader("Generate Report")

if st.button(
    "Generate PDF Report",
    type="primary",
    use_container_width=True
):
    if not selected_charts:
        st.error("Please select at least one chart.")
        st.stop()

    config = ReportConfig(
        title=report_title,
        client_name=client_name,
        prepared_by=prepared_by,
        confidential=confidential,
        theme_name=theme_name,
    )

    # Step 1 — Executive summary
    with st.status("Generating AI executive summary...") as status:
        try:
            executive_summary = generate_executive_summary(
                df, profile, client
            )
            status.update(
                label="Executive summary done",
                state="complete"
            )
        except Exception as e:
            executive_summary = [
                {"finding": "Dataset analysis complete.", "type": "neutral"}
            ]
            status.update(label=f"Summary fallback used", state="complete")

    # Step 2 — Chart narratives
    with st.status("Writing chart analysis...") as status:
        chart_narratives = []
        num_cols = profile.numeric_cols
        cat_cols = profile.categorical_cols

        for title, _ in selected_charts:
            try:
                narrative = generate_chart_narrative(
                    chart_title=title,
                    chart_type=title.split(":")[0],
                    df=df,
                    x_col=cat_cols[0] if cat_cols else (
                        num_cols[0] if num_cols else ""),
                    y_col=num_cols[0] if num_cols else "",
                    client=client,
                )
                chart_narratives.append(narrative)
            except Exception:
                chart_narratives.append(
                    f"Analysis of {title}."
                )

        status.update(
            label=f"Analysis written for {len(chart_narratives)} charts",
            state="complete"
        )

    # Step 3 — Recommendations
    with st.status("Generating recommendations...") as status:
        try:
            recommendations = generate_recommendations(
                profile, df, client
            )
            status.update(
                label="Recommendations ready",
                state="complete"
            )
        except Exception:
            recommendations = {
                "immediate": ["Review data quality issues"],
                "short_term": ["Address missing values"],
                "long_term": ["Implement data validation"],
            }
            status.update(
                label="Recommendations fallback used",
                state="complete"
            )

    # Step 4 — Build PDF
    with st.status("Building PDF...") as status:
        try:
            pdf_bytes = build_pdf(
                df=df,
                profile=profile,
                config=config,
                chart_data=selected_charts,
                chart_narratives=chart_narratives,
                executive_summary=executive_summary,
                recommendations=recommendations,
            )
            status.update(
                label="PDF ready!",
                state="complete"
            )
        except Exception as e:
            st.error(f"PDF generation failed: {e}")
            st.stop()

    # ── Download ───────────────────────────────────────────
    st.success("Report generated successfully!")
    st.download_button(
        label="Download PDF Report",
        data=pdf_bytes,
        file_name=f"dataforge_report_{fname}.pdf",
        mime="application/pdf",
        type="primary",
        use_container_width=True,
    )

    # ── Preview findings ───────────────────────────────────
    st.divider()
    st.subheader("Report Preview")

    with st.expander("Executive Summary", expanded=True):
        for item in executive_summary:
            finding = item.get("finding", str(item)) \
                if isinstance(item, dict) else str(item)
            ftype   = item.get("type", "neutral") \
                if isinstance(item, dict) else "neutral"
            icon    = "+" if ftype == "positive" \
                else "!" if ftype == "negative" else ">"
            st.markdown(f"**{icon}** {finding}")

    with st.expander("Recommendations"):
        if recommendations.get("immediate"):
            st.markdown("**Immediate:**")
            for r in recommendations["immediate"]:
                st.markdown(f"- {r}")
        if recommendations.get("short_term"):
            st.markdown("**Short Term:**")
            for r in recommendations["short_term"]:
                st.markdown(f"- {r}")
        if recommendations.get("long_term"):
            st.markdown("**Long Term:**")
            for r in recommendations["long_term"]:
                st.markdown(f"- {r}")

st.divider()

# ── Quick export ───────────────────────────────────────────
st.subheader("Quick Export")
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
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
            df.to_excel(
                writer,
                sheet_name="Cleaned Data",
                index=False
            )
            pd.DataFrame([{
                "Column":    p.name,
                "Missing %": p.missing_pct,
                "Score":     p.quality_score,
            } for p in profile.column_profiles]).to_excel(
                writer,
                sheet_name="Quality Report",
                index=False
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

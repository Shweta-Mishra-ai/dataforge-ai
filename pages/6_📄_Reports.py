import streamlit as st
import pandas as pd
import io
from core.data_profiler import profile_dataset
from components.kpi_cards import inject_global_css

st.set_page_config(
    page_title="Reports — DataForge AI",
    page_icon="📄",
    layout="wide"
)
inject_global_css()

if "df_active" not in st.session_state:
    st.warning("⚠️ No data loaded.")
    st.page_link("pages/1_📥_Data_Upload.py", label="← Go to Upload", icon="📥")
    st.stop()

df      = st.session_state["df_active"]
profile = st.session_state.get("profile") or profile_dataset(df)
fname   = st.session_state.get("filename", "data")

st.title("📄 Export Reports")
st.divider()

c1, c2 = st.columns(2)

# ── CSV Export ─────────────────────────────────────────────
with c1:
    st.subheader("📥 CSV Export")
    st.markdown(f"**{len(df):,} rows × {len(df.columns)} columns**")
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download CSV",
        data=csv,
        file_name=f"cleaned_{fname}.csv",
        mime="text/csv",
        type="primary",
        use_container_width=True
    )

# ── Excel Export ───────────────────────────────────────────
with c2:
    st.subheader("📊 Excel Export")
    st.markdown("Data sheet + Quality Report sheet")
    try:
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
            df.to_excel(writer, sheet_name="Cleaned Data", index=False)
            pd.DataFrame([{
                "Column":        p.name,
                "Type":          p.dtype,
                "Missing %":     p.missing_pct,
                "Unique Count":  p.unique_count,
                "Outliers":      p.outlier_count,
                "Quality Score": p.quality_score,
            } for p in profile.column_profiles]).to_excel(
                writer, sheet_name="Quality Report", index=False
            )
        buf.seek(0)
        st.download_button(
            "Download Excel",
            data=buf,
            file_name=f"cleaned_{fname}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            type="primary",
            use_container_width=True
        )
    except ImportError:
        st.error("Run: pip install xlsxwriter")

st.divider()

# ── Quality summary ────────────────────────────────────────
st.subheader("📋 Quality Report Summary")
st.markdown(f"""
| Metric | Value |
|---|---|
| Overall Quality Score | **{profile.overall_quality_score} / 100** |
| Total Rows | {profile.rows:,} |
| Total Columns | {profile.cols} |
| Missing Cells | {profile.missing_cells:,} ({profile.missing_pct}%) |
| Duplicate Rows | {profile.duplicate_rows:,} ({profile.duplicate_pct}%) |
| Numeric Columns | {len(profile.numeric_cols)} |
| Categorical Columns | {len(profile.categorical_cols)} |
| DateTime Columns | {len(profile.datetime_cols)} |
""")

if profile.recommendations:
    st.subheader("🎯 Recommendations")
    for r in profile.recommendations:
        st.markdown(f"- {r}")

st.info("📄 PDF report export — coming in next version.")

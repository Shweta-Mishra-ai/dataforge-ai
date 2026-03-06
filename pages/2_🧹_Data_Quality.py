import streamlit as st
import pandas as pd
import plotly.express as px
from core.data_profiler import profile_dataset
from core.data_cleaner import clean_with_strategy, auto_clean
from components.kpi_cards import inject_global_css, kpi_grid, quality_score_banner

st.set_page_config(
    page_title="Data Quality — DataForge AI",
    page_icon="🧹",
    layout="wide"
)
inject_global_css()

if "df_active" not in st.session_state:
    st.warning("⚠️ No data loaded.")
    st.page_link("pages/1_📥_Data_Upload.py", label="← Go to Upload", icon="📥")
    st.stop()

df      = st.session_state["df_active"]
profile = st.session_state.get("profile") or profile_dataset(df)

st.title("🧹 Data Quality Report")
st.markdown(f"File: **{st.session_state.get('filename', 'dataset')}**")
st.divider()

# ── Score + KPIs ───────────────────────────────────────────
col_score, col_kpis = st.columns([1, 3])

with col_score:
    quality_score_banner(profile.overall_quality_score)

with col_kpis:
    kpi_grid([
        {"label": "Total Cells",  "value": f"{profile.total_cells:,}",      "accent": "#4f8ef7"},
        {"label": "Missing",      "value": f"{profile.missing_cells:,}",    "sub": f"{profile.missing_pct}%",  "accent": "#f7934f"},
        {"label": "Duplicates",   "value": str(profile.duplicate_rows),     "sub": f"{profile.duplicate_pct}%","accent": "#f77070"},
        {"label": "Numeric Cols", "value": str(len(profile.numeric_cols)),   "accent": "#22d3a5"},
        {"label": "Cat. Cols",    "value": str(len(profile.categorical_cols)),"accent": "#a78bfa"},
    ])

tab1, tab2, tab3 = st.tabs(["📊 Column Report", "🔧 Clean Data", "📥 Export"])

# ── TAB 1 — Column report ──────────────────────────────────
with tab1:
    st.subheader("Per-Column Quality Breakdown")

    rows = []
    for p in profile.column_profiles:
        if p.missing_pct > 50:
            flag = "🚨 High Missing"
        elif p.missing_pct > 20:
            flag = "⚠️ Some Missing"
        elif p.is_constant:
            flag = "🗑️ Constant"
        elif p.has_outliers:
            flag = "📊 Has Outliers"
        else:
            flag = "✅ Good"

        rows.append({
            "Column":  p.name,
            "Type":    p.dtype,
            "Missing %": f"{p.missing_pct}%",
            "Unique":  p.unique_count,
            "Outliers": p.outlier_count if p.has_outliers else "—",
            "Score":   f"{p.quality_score}/100",
            "Status":  flag,
        })

    st.dataframe(pd.DataFrame(rows), use_container_width=True, height=400)

    # Missing values chart
    miss = [(p.name, p.missing_pct) for p in profile.column_profiles if p.missing_pct > 0]
    if miss:
        st.subheader("Missing Values by Column")
        mdf = pd.DataFrame(miss, columns=["Column", "Missing %"])
        mdf = mdf.sort_values("Missing %")
        fig = px.bar(
            mdf, x="Missing %", y="Column", orientation="h",
            title="Missing Value % per Column",
            template="plotly_dark", color="Missing %",
            color_continuous_scale=["#22d3a5", "#f7934f", "#f77070"]
        )
        fig.update_layout(paper_bgcolor="#07080f", plot_bgcolor="#0e0f1a")
        st.plotly_chart(fig, use_container_width=True)

    if profile.recommendations:
        st.subheader("🎯 Recommendations")
        for r in profile.recommendations:
            st.markdown(f"- {r}")

# ── TAB 2 — Clean data ─────────────────────────────────────
with tab2:
    st.subheader("Cleaning Controls")

    if st.button("⚡ Auto Clean", type="primary"):
        with st.spinner("Cleaning..."):
            df_c = auto_clean(df)
            st.session_state["df_active"] = df_c
            st.session_state["profile"]   = profile_dataset(df_c)
        st.success(f"✅ Done! {len(df) - len(df_c):,} rows removed.")
        st.rerun()

    st.divider()
    st.markdown("**Or choose strategy per column:**")

    OPTS = [
        "keep", "fill_mean", "fill_median", "fill_mode",
        "fill_zero", "fill_unknown", "ffill", "drop_rows", "drop_col"
    ]

    strategies = {}
    for p in profile.column_profiles:
        if p.is_constant or p.missing_pct > 60:
            default = "drop_col"
        elif p.missing_pct > 0 and "float" in p.dtype:
            default = "fill_median"
        elif p.missing_pct > 0:
            default = "fill_mode"
        else:
            default = "keep"

        strategies[p.name] = st.selectbox(
            f"**{p.name}** — {p.dtype} | {p.missing_pct}% missing",
            OPTS,
            index=OPTS.index(default),
            key=f"s_{p.name}"
        )

    if st.button("✅ Apply Strategies", type="secondary"):
        with st.spinner("Cleaning..."):
            df_c = clean_with_strategy(df, strategies)
            st.session_state["df_active"] = df_c
            st.session_state["profile"]   = profile_dataset(df_c)
        st.success(f"✅ Done! Dataset now has {len(df_c):,} rows.")
        st.rerun()

# ── TAB 3 — Export ─────────────────────────────────────────
with tab3:
    st.subheader("Export Cleaned Data")
    df_export = st.session_state["df_active"]
    csv = df_export.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Download CSV",
        data=csv,
        file_name=f"cleaned_{st.session_state.get('filename', 'data')}.csv",
        mime="text/csv",
        type="primary"
    )

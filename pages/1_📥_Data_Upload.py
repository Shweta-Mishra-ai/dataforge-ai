import streamlit as st
from core.data_loader import load_file
from core.data_profiler import profile_dataset
from components.kpi_cards import inject_global_css, kpi_grid, quality_score_banner

st.set_page_config(
    page_title="Upload — DataForge AI",
    page_icon="📥",
    layout="wide"
)
inject_global_css()

st.title("📥 Upload Your Data")
st.markdown("Supports **CSV**, **Excel** (.xlsx/.xls), **JSON** — up to **200 MB**")
st.divider()

uploaded = st.file_uploader(
    "Drop your file here",
    type=["csv", "xlsx", "xls", "json"],
    help="Your data stays in your session — never stored on any server."
)

if not uploaded:
    st.info("👆 Upload a file to get started.")
    st.stop()

with st.spinner("Loading file..."):
    result = load_file(uploaded)

if not result.success:
    st.error(f"❌ {result.error}")
    st.stop()

# ── Sheet selector (Excel only) ────────────────────────────
if result.sheet_names and len(result.sheet_names) > 1:
    st.success(f"Excel file has **{len(result.sheet_names)} sheets**")
    selected_sheet = st.selectbox("Select sheet:", result.sheet_names)
    with st.spinner("Loading sheet..."):
        result = load_file(uploaded, sheet_name=selected_sheet)
    if not result.success:
        st.error(f"❌ {result.error}")
        st.stop()

df = result.df

# ── Save to session state ──────────────────────────────────
st.session_state["df_raw"]    = df
st.session_state["df_active"] = df
st.session_state["filename"]  = result.filename
st.session_state["file_size"] = result.file_size_mb

# ── Profile ────────────────────────────────────────────────
with st.spinner("Profiling data quality..."):
    profile = profile_dataset(df)
    st.session_state["profile"] = profile

# ── KPI cards ──────────────────────────────────────────────
st.subheader("Dataset Overview")
kpi_grid([
    {"label": "Rows",       "value": f"{len(df):,}",               "sub": "records",          "icon": "📋", "accent": "#4f8ef7"},
    {"label": "Columns",    "value": str(len(df.columns)),          "sub": "features",         "icon": "🏛️",  "accent": "#a78bfa"},
    {"label": "File Size",  "value": f"{result.file_size_mb:.1f}MB","sub": "uploaded",         "icon": "💾", "accent": "#22d3a5"},
    {"label": "Missing",    "value": f"{profile.missing_pct}%",    "sub": f"{profile.missing_cells:,} cells", "icon": "⚠️", "accent": "#f7934f"},
    {"label": "Duplicates", "value": str(profile.duplicate_rows),   "sub": "rows",             "icon": "🔁", "accent": "#f77070"},
])

quality_score_banner(profile.overall_quality_score)

# ── Preview ────────────────────────────────────────────────
st.subheader("Data Preview")
c1, c2 = st.columns([3, 1])

with c1:
    st.dataframe(df.head(50), use_container_width=True, height=400)

with c2:
    st.markdown("**Column Types**")
    import pandas as pd
    type_df          = df.dtypes.reset_index()
    type_df.columns  = ["Column", "Type"]
    st.dataframe(type_df, use_container_width=True, height=400)

st.divider()
st.success("✅ File loaded! Use the sidebar to navigate to other pages.")

"""
pages/1_Data_Upload.py
Upload page — now uses session_manager for proper state handling.
"""
import streamlit as st
from core.data_loader import load_file
from core.data_profiler import profile_dataset
from core.session_manager import init_session, set_dataframe, get_filename
from components.kpi_cards import inject_global_css, kpi_grid, quality_score_banner

st.set_page_config(page_title="Upload — DataForge AI", page_icon="📥", layout="wide")
inject_global_css()
init_session()   # ensure all keys exist

st.title("Upload Your Data")
st.markdown("Supports **CSV**, **Excel** (.xlsx/.xls, multi-sheet), **JSON** — up to **200 MB**")
st.divider()

uploaded = st.file_uploader(
    label="Drop your file here",
    type=["csv", "xlsx", "xls", "json"],
    help="Max 200MB. Your data stays in your browser session — never stored on any server."
)

if not uploaded:
    st.info("Upload a file to get started. All pages will unlock once a file is loaded.")
    st.stop()

# ── Load file ──────────────────────────────────────────────
with st.spinner("Loading file..."):
    result = load_file(uploaded)

if not result.success:
    st.error("Error: {}".format(result.error))
    if result.validation and result.validation.warnings:
        for w in result.validation.warnings:
            st.warning(w)
    st.stop()

# ── Show validation warnings if any ───────────────────────
if result.validation and result.validation.warnings:
    with st.expander("Data Warnings ({})".format(len(result.validation.warnings))):
        for w in result.validation.warnings:
            st.warning(w)

# ── Sheet selector (Excel only) ────────────────────────────
if result.sheet_names and len(result.sheet_names) > 1:
    st.success("Excel file has **{}** sheets".format(len(result.sheet_names)))
    selected_sheet = st.selectbox("Select sheet to analyse:", result.sheet_names)
    with st.spinner("Loading selected sheet..."):
        result = load_file(uploaded, sheet_name=selected_sheet)
    if not result.success:
        st.error("Error: {}".format(result.error))
        st.stop()

df = result.df

# ── Save to session via session_manager ───────────────────
set_dataframe(df, result.filename, result.file_size_mb)

# ── Show type conversions ──────────────────────────────────
if result.type_conversions:
    with st.expander("Smart Type Detection — {} column(s) auto-converted".format(
        len(result.type_conversions)
    )):
        for c in result.type_conversions:
            st.success("'{}' — {} → {} ({})".format(
                c["column"], c["from"], c["to"], c["method"]
            ))

# ── Profile dataset ────────────────────────────────────────
with st.spinner("Profiling data quality..."):
    profile = profile_dataset(df)
    st.session_state["profile"] = profile

# ── KPI cards ──────────────────────────────────────────────
st.subheader("Dataset Overview")
kpi_grid([
    {"label": "Rows",       "value": "{:,}".format(len(df)),
     "sub": "total records",  "icon": "📋", "accent": "#4f8ef7"},
    {"label": "Columns",    "value": str(len(df.columns)),
     "sub": "features",       "icon": "🏛️",  "accent": "#a78bfa"},
    {"label": "File Size",  "value": "{:.1f} MB".format(result.file_size_mb),
     "sub": "uploaded",       "icon": "💾", "accent": "#22d3a5"},
    {"label": "Missing",    "value": "{}%".format(profile.missing_pct),
     "sub": "{:,} cells".format(profile.missing_cells),
     "icon": "⚠️", "accent": "#f7934f"},
    {"label": "Duplicates", "value": str(profile.duplicate_rows),
     "sub": "duplicate rows", "icon": "🔁", "accent": "#f77070"},
])

quality_score_banner(profile.overall_quality_score)

# ── Data preview ───────────────────────────────────────────
st.subheader("Data Preview")
col1, col2 = st.columns([3, 1])
with col1:
    st.dataframe(df.head(50), use_container_width=True, height=400)
with col2:
    st.markdown("**Column Types**")
    type_df = df.dtypes.reset_index()
    type_df.columns = ["Column", "Type"]
    st.dataframe(type_df, use_container_width=True, height=400)

st.divider()
st.success("File loaded! Use the sidebar to navigate to Data Quality, Dashboard, or AI Chat.")

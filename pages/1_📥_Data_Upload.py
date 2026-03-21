"""
pages/1_Data_Upload.py
Upload page — handles original dirty data.
Shows warnings, full dataset, quality profile.
"""
import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="Upload — DataForge AI",
    page_icon="📥",
    layout="wide"
)

from core.data_loader import load_file
SAMPLE_THRESHOLD = 100_000
from core.data_profiler import profile_dataset
from core.session_manager import init_session, set_dataframe

init_session()

# ── Header ─────────────────────────────────────────────────
st.markdown("## Upload Your Data")
st.caption("Supports CSV, Excel (.xlsx/.xls, multi-sheet), JSON — up to 200 MB")
st.divider()

# ── Upload widget ──────────────────────────────────────────
uploaded = st.file_uploader(
    label="Drop your file here",
    type=["csv", "xlsx", "xls", "json"],
    help="Max 200MB. Data stays in your browser session — never stored on any server.",
    label_visibility="collapsed",
)

if not uploaded:
    st.info("Upload a file to get started. All pages unlock once a file is loaded.")
    st.stop()

# ── Load file ──────────────────────────────────────────────
with st.spinner("Loading {}...".format(uploaded.name)):
    result = load_file(uploaded)

if not result.success:
    st.error(result.error)
    st.stop()

# ── Sheet selector (Excel multi-sheet) ────────────────────
if result.sheet_names and len(result.sheet_names) > 1:
    st.success("Excel file — {} sheets found.".format(len(result.sheet_names)))
    selected_sheet = st.selectbox("Select sheet to analyse:", result.sheet_names)
    with st.spinner("Loading sheet '{}'...".format(selected_sheet)):
        result = load_file(uploaded, sheet_name=selected_sheet)
    if not result.success:
        st.error(result.error)
        st.stop()

df = result.df

# ── Warnings ───────────────────────────────────────────────
if result.warnings:
    with st.expander("Data Warnings ({})".format(len(result.warnings))):
        for w in result.warnings:
            st.warning(w)

# ── Save to session ────────────────────────────────────────
set_dataframe(df, result.filename, result.file_size_mb)
st.session_state["profile"]  = None  # will compute below

# ── Profile (sampled for speed on large files) ─────────────
with st.spinner("Analysing data quality..."):
    try:
        df_profile = (df.sample(n=min(10000, len(df)), random_state=42)
                      if len(df) > 10000 else df)
        profile = profile_dataset(df_profile)
        st.session_state["profile"] = profile
    except Exception as e:
        profile = None
        st.warning("Quality profile unavailable: {}".format(str(e)))

# ── KPI Cards ──────────────────────────────────────────────
st.divider()
st.markdown("### Dataset Overview")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Rows",       "{:,}".format(len(df)),      "records")
c2.metric("Columns",    str(len(df.columns)),          "features")
c3.metric("File Size",  "{:.1f} MB".format(result.file_size_mb))
if profile:
    c4.metric("Missing",    "{:.1f}%".format(profile.missing_pct),
              "{:,} cells".format(profile.missing_cells))
    c5.metric("Quality",    "{}/100".format(profile.overall_quality_score),
              "Grade {}".format(getattr(profile,"data_quality_grade","?")))
else:
    c4.metric("Missing",
              "{:.1f}%".format(df.isna().sum().sum()/max(df.size,1)*100))
    c5.metric("Duplicates", "{:,}".format(int(df.duplicated().sum())))

# ── Quality Recommendations ────────────────────────────────
if profile and profile.recommendations:
    st.divider()
    st.markdown("### Data Quality Findings")
    for rec in profile.recommendations[:6]:
        if rec.startswith("CRITICAL"):
            st.error(rec)
        elif rec.startswith("GOOD"):
            st.success(rec)
        else:
            st.warning(rec)

# ── Data Preview ───────────────────────────────────────────
st.divider()
st.markdown("### Data Preview")

tab1, tab2, tab3 = st.tabs(["Raw Data", "Column Info", "Missing Values"])

with tab1:
    st.caption("Showing first 100 rows of {:,} total rows".format(len(df)))
    st.dataframe(df.head(100), use_container_width=True, height=400)

with tab2:
    info_rows = []
    for col in df.columns:
        s = df[col]
        info_rows.append({
            "Column":    col,
            "Type":      str(s.dtype),
            "Missing":   "{:.1f}%".format(s.isna().mean()*100),
            "Unique":    "{:,}".format(int(s.nunique())),
            "Sample":    str(s.dropna().iloc[0])[:40] if len(s.dropna())>0 else "—",
        })
    info_df = pd.DataFrame(info_rows)
    st.dataframe(info_df, use_container_width=True, height=400)

with tab3:
    miss = df.isna().sum()
    miss = miss[miss > 0].sort_values(ascending=False)
    if len(miss) == 0:
        st.success("No missing values found.")
    else:
        miss_df = pd.DataFrame({
            "Column":    miss.index,
            "Missing":   miss.values,
            "Missing %": (miss.values / len(df) * 100).round(1),
        })
        st.dataframe(miss_df, use_container_width=True, height=400)

# ── Numeric Stats ──────────────────────────────────────────
num_cols = df.select_dtypes(include="number").columns.tolist()
if num_cols:
    st.divider()
    st.markdown("### Descriptive Statistics")
    st.dataframe(
        df[num_cols].describe().round(3),
        use_container_width=True)

# ── Navigation ─────────────────────────────────────────────
st.divider()
st.success(
    "File loaded — {:,} rows, {} columns. "
    "Use the sidebar to navigate.".format(len(df), len(df.columns)))

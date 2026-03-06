import streamlit as st
from core.chart_engine import recommend_charts
from core.insight_engine import generate_insights
from core.data_profiler import profile_dataset
from components.kpi_cards import inject_global_css, kpi_grid, insight_card

st.set_page_config(
    page_title="Dashboard — DataForge AI",
    page_icon="📊",
    layout="wide"
)
inject_global_css()

if "df_active" not in st.session_state:
    st.warning("⚠️ No data loaded.")
    st.page_link("pages/1_📥_Data_Upload.py", label="← Go to Upload", icon="📥")
    st.stop()

df       = st.session_state["df_active"]
profile  = st.session_state.get("profile") or profile_dataset(df)
num_cols = profile.numeric_cols

st.title("📊 Auto Dashboard")
st.markdown(f"`{st.session_state.get('filename', 'dataset')}` — {len(df):,} rows × {len(df.columns)} cols")
st.divider()

if not num_cols:
    st.warning("No numeric columns found.")
    st.stop()

# ── Primary metric selector ────────────────────────────────
metric_col = st.selectbox("Primary Metric:", num_cols)
s = df[metric_col].dropna()

kpi_grid([
    {"label": "Total",   "value": f"{s.sum():,.0f}",  "sub": metric_col, "icon": "∑", "accent": "#4f8ef7"},
    {"label": "Average", "value": f"{s.mean():,.2f}", "sub": "mean",     "icon": "≈", "accent": "#22d3a5"},
    {"label": "Max",     "value": f"{s.max():,.2f}",  "sub": "highest",  "icon": "↑", "accent": "#a78bfa"},
    {"label": "Min",     "value": f"{s.min():,.2f}",  "sub": "lowest",   "icon": "↓", "accent": "#f7934f"},
    {"label": "Std Dev", "value": f"{s.std():,.2f}",  "sub": "spread",   "icon": "σ", "accent": "#ffd43b"},
])

st.divider()

# ── Auto charts ────────────────────────────────────────────
st.subheader("Auto-Generated Charts")

with st.spinner("Building charts..."):
    charts = recommend_charts(df)

if not charts:
    st.info("Not enough data variety to auto-generate charts.")
else:
    for i in range(0, len(charts), 2):
        cols = st.columns(2)
        for j, col in enumerate(cols):
            if i + j < len(charts):
                with col:
                    st.plotly_chart(
                        charts[i + j][1],
                        use_container_width=True
                    )

st.divider()

# ── Insights ───────────────────────────────────────────────
st.subheader("🤖 Auto Insights")

with st.spinner("Generating insights..."):
    insights = generate_insights(df)

for ins in insights:
    insight_card(
        title=f"{ins['icon']}  {ins['title']}",
        body=ins["body"],
        type_=ins["type"]
    )

import streamlit as st
import pandas as pd
import plotly.express as px
from core.chart_engine import make_bar, make_line, make_scatter, make_histogram, make_pie, make_heatmap
from core.data_profiler import profile_dataset
from components.kpi_cards import inject_global_css


# ── Global adaptive CSS (dark + light theme safe) ─────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif!important}
.block-container{padding-top:1.2rem!important}
section[data-testid="stSidebar"]{background:linear-gradient(180deg,#0D1B2E,#0F2240)!important}
section[data-testid="stSidebar"] *{color:rgba(255,255,255,.85)!important}
section[data-testid="stSidebar"] hr{border-color:rgba(255,255,255,.12)!important}
/* adaptive card base */
.df-card{background:rgba(128,128,128,.06);border:1px solid rgba(128,128,128,.18);border-radius:12px;padding:16px 20px;margin-bottom:12px}
/* finding/risk/opp rows */
.risk-row{border-left:4px solid #ef4444;background:rgba(239,68,68,.07);padding:12px 16px;border-radius:0 8px 8px 0;margin-bottom:8px}
.opp-row{border-left:4px solid #10b981;background:rgba(16,185,129,.07);padding:12px 16px;border-radius:0 8px 8px 0;margin-bottom:8px}
.info-row{border-left:4px solid #3b82f6;background:rgba(59,130,246,.07);padding:12px 16px;border-radius:0 8px 8px 0;margin-bottom:8px}
</style>
""", unsafe_allow_html=True)

st.set_page_config(
    page_title="Deep Analysis — DataForge AI",
    page_icon="🔬",
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
cat_cols = profile.categorical_cols
all_cols = list(df.columns)

st.title("🔬 Deep Analysis")
st.divider()

tab1, tab2, tab3 = st.tabs(["🎨 Custom Chart", "🔗 Correlations", "⚠️ Anomaly Detection"])

# ── TAB 1 — Custom chart ───────────────────────────────────
with tab1:
    st.subheader("Build Your Own Chart")

    c1, c2, c3 = st.columns(3)
    chart_type = c1.selectbox("Chart Type", ["Bar", "Line", "Scatter", "Histogram", "Pie"])
    x_col      = c2.selectbox("X Axis", all_cols)
    y_col      = c3.selectbox("Y Axis", num_cols or all_cols) if chart_type != "Histogram" else x_col

    color_col = None
    if chart_type == "Scatter" and cat_cols:
        sel       = st.selectbox("Color by (optional):", ["None"] + cat_cols)
        color_col = None if sel == "None" else sel

    if st.button("📊 Generate Chart", type="primary"):
        try:
            if chart_type == "Bar":
                fig = make_bar(df, x_col, y_col)
            elif chart_type == "Line":
                fig = make_line(df, x_col, y_col)
            elif chart_type == "Scatter":
                fig = make_scatter(df, x_col, y_col, color_col)
            elif chart_type == "Histogram":
                fig = make_histogram(df, x_col)
            elif chart_type == "Pie":
                # FIX: user explicitly chose Pie — build it directly without
                # the auto-mode guard that redirects score-like columns to bar.
                import plotly.express as px
                agg = df.groupby(x_col)[y_col].sum().reset_index().head(12)
                fig = px.pie(
                    agg, names=x_col, values=y_col,
                    title=f"{y_col} Share by {x_col}",
                    template="plotly_white",
                    hole=0.0,
                )
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict( size=11),
                    margin=dict(l=10, r=10, t=50, b=10),
                )
            st.plotly_chart(fig, width="stretch")
        except Exception as e:
            st.error(f"Chart error: {e}")

# ── TAB 2 — Correlations ───────────────────────────────────
with tab2:
    st.subheader("Correlation Matrix")

    if len(num_cols) < 2:
        st.info("Need at least 2 numeric columns.")
    else:
        sel = st.multiselect(
            "Select columns:",
            num_cols,
            default=num_cols[:min(8, len(num_cols))]
        )
        if sel and len(sel) >= 2:
            st.plotly_chart(make_heatmap(df[sel]), width="stretch")

            corr   = df[sel].corr()
            pairs  = []
            for i in range(len(sel)):
                for j in range(i + 1, len(sel)):
                    pairs.append({
                        "Column A":    sel[i],
                        "Column B":    sel[j],
                        "Correlation": round(corr.iloc[i, j], 3)
                    })
            pairs_df = pd.DataFrame(pairs).sort_values(
                "Correlation", key=abs, ascending=False
            )
            st.dataframe(pairs_df, width="stretch")

# ── TAB 3 — Anomaly detection ──────────────────────────────
with tab3:
    st.subheader("Outlier Detection")

    if not num_cols:
        st.info("No numeric columns found.")
    else:
        col = st.selectbox("Select column:", num_cols)
        s   = df[col].dropna()

        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr    = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr

        outliers    = df[(df[col] < lo) | (df[col] > hi)]
        outlier_pct = len(outliers) / max(len(df), 1) * 100

        ca, cb = st.columns(2)
        ca.metric("Outliers Found", f"{len(outliers):,}", f"{outlier_pct:.1f}% of rows")
        cb.metric("Normal Range",   f"{lo:,.2f} – {hi:,.2f}")

        fig = px.box(
            df, y=col,
            title=f"Box Plot: {col}",
            template="plotly_white",
            color_discrete_sequence=["#1565C0"],
            points="outliers"
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict( size=11),
            title_font=dict( size=14),
        )
        fig.update_xaxes(tickfont=dict(), title_font=dict())
        fig.update_yaxes(tickfont=dict(), title_font=dict())
        st.plotly_chart(fig, width="stretch")

        if not outliers.empty:
            st.markdown(f"**Outlier rows ({len(outliers):,}):**")
            st.dataframe(outliers.head(100), width="stretch")

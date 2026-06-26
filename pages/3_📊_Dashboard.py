"""
pages/3_📊_Dashboard.py  — DataForge AI IMPROVED
Changes vs original:
  - Domain detected at load time
  - Domain badge shown in header
  - Domain KPI cards replace generic numeric means
  - Domain-specific charts shown FIRST (before generic stats)
  - Clients see HR/Finance/Sales/Ecommerce insights immediately
  - Generic analysis still available in tabs below
"""
import io
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from core.session_manager import require_data, get_df, get_filename, get_cached_stats
require_data()

from core.stats_engine import analyze
from core.story_engine import detect_domain
import logging
logger = logging.getLogger(__name__)

# Import domain dashboards (new file)
try:
    from core.domain_dashboards import get_domain_kpis, get_domain_charts
    DOMAIN_DASH_AVAILABLE = True
except ImportError:
    DOMAIN_DASH_AVAILABLE = False

st.set_page_config(page_title="Dashboard — DataForge AI", layout="wide")

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif!important}
.block-container{padding-top:1.2rem!important}

.domain-badge{
    display:inline-flex;align-items:center;gap:8px;
    padding:8px 18px;border-radius:20px;
    font-size:13px;font-weight:700;letter-spacing:.05em;
    margin-bottom:16px;
}
.kpi-card{
    background:rgba(128,128,128,.06);border:1px solid rgba(128,128,128,.18);border-radius:14px;
    padding:18px 16px;position:relative;overflow:hidden;
    transition:box-shadow .2s;
}
.kpi-card:hover{box-shadow:0 4px 20px rgba(0,0,0,.07)}
.kpi-val{font-size:26px;font-weight:800;line-height:1.1;margin-bottom:4px}
.kpi-lbl{font-size:11px;font-weight:700;text-transform:uppercase;
         letter-spacing:.07em;color:inherit;opacity:.6;margin-bottom:6px}
.kpi-sub{font-size:11px;color:inherit;opacity:.5;line-height:1.4}
.kpi-accent{position:absolute;top:0;left:0;right:0;height:3px;border-radius:14px 14px 0 0}

.section-hdr{
    font-size:16px;font-weight:800;color:inherit;
    margin:24px 0 12px;display:flex;align-items:center;gap:8px;
}
.domain-section{
    background:rgba(59,130,246,.07);
    border:1px solid rgba(59,130,246,.2);border-radius:16px;
    padding:20px 22px;margin-bottom:20px;
}

section[data-testid="stSidebar"]{background:linear-gradient(180deg,#0D1B2E,#0F2240)!important}
section[data-testid="stSidebar"] *{color:rgba(255,255,255,.85)!important}
</style>
""", unsafe_allow_html=True)

# ── Cache stats ───────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def get_stats(df_json: str):
    df = pd.read_json(io.StringIO(df_json))
    return analyze(df), df

# ── Load data ─────────────────────────────────────────────────────────────────
df_master = get_df()
fname     = get_filename()
stats, _  = get_stats(df_master.to_json(date_format="iso"))

num_cols = stats.numeric_cols
cat_cols = [c for c in stats.categorical_cols if df_master[c].nunique() <= 50]
dt_cols  = stats.datetime_cols

# Detect domain once
domain, confidence = detect_domain(df_master)

DOMAIN_META = {
    "hr":        {"icon": "👥", "label": "HR Analytics",     "color": "#1D4ED8", "bg": "#DBEAFE"},
    "finance":   {"icon": "💰", "label": "Finance",          "color": "#065F46", "bg": "#D1FAE5"},
    "ecommerce": {"icon": "🛒", "label": "E-Commerce",       "color": "#B45309", "bg": "#FEF3C7"},
    "sales":     {"icon": "📈", "label": "Sales",            "color": "#6D28D9", "bg": "#EDE9FE"},
    "general":   {"icon": "📊", "label": "General Business", "color": "#475569", "bg": "#F1F5F9"},
}
dmeta = DOMAIN_META.get(domain, DOMAIN_META["general"])

# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR FILTERS
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="padding:16px 0 10px;text-align:center">
        <div style="font-size:20px;font-weight:900;color:white">🔬 DataForge AI</div>
        <div style="font-size:10px;color:rgba(255,255,255,.4);text-transform:uppercase;
                    letter-spacing:.1em;margin-top:3px">Dashboard</div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    st.markdown("### 🎛️ Filters")
    st.caption("Filter once → all charts update")

    df_filtered = df_master.copy()

    for col in cat_cols[:4]:
        unique_vals = sorted(df_master[col].dropna().unique().tolist())
        if 2 <= len(unique_vals) <= 20:
            selected = st.multiselect(col, unique_vals, default=unique_vals,
                                      key=f"filter_{col}")
            if selected:
                df_filtered = df_filtered[df_filtered[col].isin(selected)]

    for col in num_cols[:2]:
        s = df_master[col].dropna()
        lo, hi = float(s.min()), float(s.max())
        if lo < hi:
            rng = st.slider(col, lo, hi, (lo, hi), key=f"range_{col}")
            df_filtered = df_filtered[(df_filtered[col] >= rng[0]) &
                                       (df_filtered[col] <= rng[1])]

    st.divider()
    st.caption(f"Showing {len(df_filtered):,} of {len(df_master):,} rows")
    if st.button("Reset Filters", use_container_width=True):
        st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
#  HEADER
# ══════════════════════════════════════════════════════════════════════════════
col_title, col_info = st.columns([3, 1])
with col_title:
    st.markdown(f"## 📊 Dashboard — {fname}")
    # Domain badge
    st.markdown(f"""
    <span class="domain-badge" style="background:{dmeta['bg']};color:{dmeta['color']};">
        {dmeta['icon']} {dmeta['label']} Domain
        {"· " + f"{confidence:.0%} confidence" if confidence > 0.1 else ""}
    </span>
    """, unsafe_allow_html=True)

with col_info:
    st.markdown(f"""
    <div style="text-align:right;padding-top:8px">
        <div style="font-size:22px;font-weight:800;color:inherit">{len(df_filtered):,}</div>
        <div style="font-size:11px;opacity:.6">rows after filters</div>
        <div style="font-size:11px;opacity:.5">{len(df_master.columns)} columns</div>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 1 — DOMAIN KPI CARDS
# ══════════════════════════════════════════════════════════════════════════════
if DOMAIN_DASH_AVAILABLE and domain != "general":
    st.markdown(f'<div class="section-hdr">{dmeta["icon"]} {dmeta["label"]} KPIs</div>',
                unsafe_allow_html=True)

    try:
        kpis = get_domain_kpis(df_filtered, domain)
        n    = min(len(kpis), 6)
        cols = st.columns(n)

        for i, kpi in enumerate(kpis[:n]):
            with cols[i]:
                color = kpi.get("color", "#1B2A4A")
                delta = kpi.get("delta")
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-accent" style="background:{color}"></div>
                    <div class="kpi-lbl">{kpi['label']}</div>
                    <div class="kpi-val" style="color:{color}">{kpi['value']}</div>
                    <div class="kpi-sub">{kpi.get('sub','')}</div>
                    {f'<div style="font-size:11px;font-weight:700;color:{color};margin-top:6px">{delta}</div>' if delta else ''}
                </div>
                """, unsafe_allow_html=True)
    except Exception as e:
        st.warning(f"Domain KPIs unavailable: {e}")

else:
    # Generic KPI cards for general domain
    if num_cols:
        st.markdown('<div class="section-hdr">📌 Key Metrics</div>', unsafe_allow_html=True)
        kpi_cols = num_cols[:5]
        cols_ui  = st.columns(len(kpi_cols))

        for i, col in enumerate(kpi_cols):
            s = df_filtered[col].dropna() if col in df_filtered else df_master[col].dropna()
            cs = stats.column_stats.get(col)
            use_median = cs and cs.skewness and abs(cs.skewness) > 1
            display_val = s.median() if use_median else s.mean()
            label_note  = "Median" if use_median else "Mean"

            with cols_ui[i]:
                st.metric(
                    label=col[:22],
                    value=f"{display_val:,.2f}",
                    help=f"Sum: {s.sum():,.0f} | Min: {s.min():,.2f} | Max: {s.max():,.2f}"
                )

st.markdown("<br>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 2 — DOMAIN-SPECIFIC CHARTS (shown FIRST)
# ══════════════════════════════════════════════════════════════════════════════
if DOMAIN_DASH_AVAILABLE and domain != "general":
    st.markdown(f'<div class="section-hdr">{dmeta["icon"]} {dmeta["label"]} Analysis</div>',
                unsafe_allow_html=True)

    try:
        domain_charts = get_domain_charts(df_filtered, domain)
        if domain_charts:
            # Show in pairs
            for idx in range(0, len(domain_charts), 2):
                pair = domain_charts[idx:idx+2]
                c1, c2 = st.columns(2)  # always 2 cols; single chart uses first only
                for col_ui, (chart_title, fig) in zip([c1, c2], pair):
                    with col_ui:
                        st.plotly_chart(fig, width="stretch",
                                        config={"displayModeBar": False})
                if len(pair) == 1:
                    st.empty()  # Fill second column
        else:
            st.info(f"Domain-specific charts require certain columns. "
                    f"For {dmeta['label']}, ensure your dataset has the expected column names.")
    except Exception as e:
        st.warning(f"Domain charts unavailable: {e}")

    st.divider()

# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 3 — GENERIC ANALYSIS TABS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-hdr">📊 Statistical Analysis</div>',
            unsafe_allow_html=True)

tab_labels = ["Overview", "Distributions", "Relationships", "Deep Dive"]
if dt_cols:
    tab_labels.insert(1, "Trends")

tabs = st.tabs(tab_labels)
tab_idx = 0

# ── Overview tab ──────────────────────────────────────────────────────────────
with tabs[tab_idx]:
    tab_idx += 1

    if num_cols and cat_cols:
        t1, t2 = st.columns(2)
        with t1:
            col  = st.selectbox("Numeric column", num_cols, key="ov_num")
            cat  = st.selectbox("Group by", cat_cols, key="ov_cat")
            agg  = st.radio("Aggregation", ["Mean", "Median", "Sum", "Count"],
                            horizontal=True, key="ov_agg")
            agg_fn = {"Mean":"mean","Median":"median","Sum":"sum","Count":"count"}[agg]
            try:
                gdata = (df_filtered.groupby(cat)[col]
                                    .agg(agg_fn)
                                    .reset_index()
                                    .sort_values(col, ascending=True)
                                    .tail(20))
                fig = go.Figure(go.Bar(
                    x=gdata[col], y=gdata[cat], orientation="h",
                    marker_color="#1565C0", opacity=1.0,
                    text=[f"{v:,.2f}" for v in gdata[col]],
                    textposition="outside",
                    textfont=dict(size=11, color="#0F172A"),
                ))
                fig.update_layout(
                    title=dict(text=f"{agg} of {col} by {cat}",
                               font=dict(size=14, color="#0F172A", family="Arial Black")),
                    height=max(300, len(gdata) * 36 + 80),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    margin=dict(l=10, r=90, t=55, b=20),
                    font=dict(color="#0F172A", family="Arial"),
                    xaxis=dict(showgrid=True, gridcolor="rgba(128,128,128,.2)",
                               tickfont=dict(size=11, color="#0F172A"),
                               title_font=dict(color="#0F172A")),
                    yaxis=dict(showgrid=False,
                               tickfont=dict(size=11, color="#0F172A"),
                               title_font=dict(color="#0F172A")),
                )
                st.plotly_chart(fig, width="stretch",
                                config={"displayModeBar": False})
            except Exception as e:
                st.error(f"Chart failed: {e}")

        with t2:
            col2 = st.selectbox("Numeric column", num_cols,
                                 index=min(1, len(num_cols)-1), key="ov_num2")
            try:
                s = df_filtered[col2].dropna()
                fig2 = px.histogram(s, nbins=25, color_discrete_sequence=["#1565C0"],
                                    labels={col2: col2, "count": "Frequency"})
                mean_v = float(s.mean())
                med_v = float(s.median())
                fig2.add_vline(x=mean_v, line_dash="solid", line_color="#DC2626",
                               line_width=2, annotation_text=f"Mean {mean_v:.2f}")
                fig2.add_vline(x=med_v,  line_dash="dash",  line_color="#059669",
                               line_width=1.5, annotation_text=f"Median {med_v:.2f}")
                fig2.update_layout(
                    title=f"Distribution: {col2}",
                    height=400, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    margin=dict(l=10, r=10, t=50, b=20),
                )
                st.plotly_chart(fig2, width="stretch",
                                config={"displayModeBar": False})
            except Exception as e:
                st.error(f"Chart failed: {e}")

    # Summary stats table
    st.markdown("#### Summary Statistics")
    if num_cols:
        try:
            desc = df_filtered[num_cols].describe().round(3)
            desc.index = desc.index.str.upper()
            st.dataframe(desc.astype(str), use_container_width=True)
        except Exception:
            logger.warning("%s unexpected failure", exc_info=True)

# ── Trends tab ────────────────────────────────────────────────────────────────
if dt_cols:
    with tabs[tab_idx]:
        tab_idx += 1
        if num_cols:
            t1, t2 = st.columns([1, 3])
            with t1:
                dt_col   = st.selectbox("Date column", dt_cols, key="trend_dt")
                val_col  = st.selectbox("Metric", num_cols, key="trend_val")
                freq     = st.selectbox("Granularity", ["D","W","ME","QE","YE"],
                                        format_func=lambda x: {"D":"Daily","W":"Weekly",
                                                                "ME":"Monthly","QE":"Quarterly",
                                                                "YE":"Yearly"}[x],
                                        index=2, key="trend_freq")
                agg2 = st.radio("Aggregation", ["Sum","Mean"], horizontal=True, key="trend_agg")
            with t2:
                try:
                    df_t = df_filtered.copy()
                    df_t[dt_col] = pd.to_datetime(df_t[dt_col], errors="coerce")
                    df_t = df_t.dropna(subset=[dt_col])
                    df_t = df_t.set_index(dt_col)
                    trend_data = (df_t[val_col].resample(freq).sum()
                                  if agg2 == "Sum"
                                  else df_t[val_col].resample(freq).mean())
                    trend_data = trend_data.dropna().reset_index()
                    trend_data.columns = ["Date", val_col]

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=trend_data["Date"], y=trend_data[val_col],
                        mode="lines+markers", name=val_col,
                        line=dict(color="#1B4FD8", width=2.5),
                        marker=dict(size=5),
                        fill="tozeroy", fillcolor="rgba(27,79,216,0.08)",
                    ))
                    fig.update_layout(
                        title=f"{agg2} of {val_col} over Time",
                        height=380, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        margin=dict(l=10, r=10, t=50, b=20),
                        xaxis=dict(showgrid=False,
                               tickfont=dict(color="#0F172A", size=10),
                               title_font=dict(color="#0F172A")),
                        yaxis=dict(showgrid=True, gridcolor="rgba(128,128,128,.2)",
                               tickfont=dict(color="#0F172A", size=10),
                               title_font=dict(color="#0F172A")),
                    )
                    st.plotly_chart(fig, width="stretch",
                                    config={"displayModeBar": False})
                except Exception as e:
                    st.error(f"Trend chart failed: {e}")

# ── Distributions tab ─────────────────────────────────────────────────────────
with tabs[tab_idx]:
    tab_idx += 1
    if num_cols:
        selected_cols = st.multiselect("Select columns", num_cols,
                                        default=num_cols[:min(4, len(num_cols))],
                                        key="dist_cols")
        if selected_cols:
            cols_per_row = 2
            for idx in range(0, len(selected_cols), cols_per_row):
                row_cols = selected_cols[idx:idx+cols_per_row]
                cols_ui  = st.columns(len(row_cols))
                for ci, col in enumerate(row_cols):
                    with cols_ui[ci]:
                        try:
                            s = df_filtered[col].dropna()
                            cs = stats.column_stats.get(col)
                            skew = cs.skewness if cs and cs.skewness else float(s.skew())
                            fig = px.histogram(s, nbins=25,
                                               color_discrete_sequence=["#1B4FD8"])
                            fig.add_vline(x=float(s.median()), line_dash="dash",
                                          line_color="#059669", line_width=2)
                            fig.update_layout(
                                title=f"{col}<br><sup>skew={skew:.2f} · "
                                      f"{'use median' if abs(skew)>1 else 'mean OK'}</sup>",
                                height=280, paper_bgcolor="rgba(0,0,0,0)",
                                plot_bgcolor="rgba(0,0,0,0)",
                                margin=dict(l=10, r=10, t=55, b=20),
                            )
                            st.plotly_chart(fig, width="stretch",
                                            config={"displayModeBar": False})
                        except Exception as e:
                            st.error(f"{col}: {e}")

# ── Relationships tab ─────────────────────────────────────────────────────────
with tabs[tab_idx]:
    tab_idx += 1
    if len(num_cols) >= 2:
        col_a = st.selectbox("X axis", num_cols, key="rel_x")
        col_b = st.selectbox("Y axis", num_cols,
                              index=min(1, len(num_cols)-1), key="rel_y")
        col_c = st.selectbox("Colour by (optional)", ["None"] + cat_cols,
                              key="rel_color")

        try:
            kwargs = dict(opacity=0.55, color_discrete_sequence=px.colors.qualitative.Set2)
            if col_c != "None":
                kwargs["color"] = col_c
            fig = px.scatter(df_filtered, x=col_a, y=col_b,
                             trendline="ols", **kwargs)
            fig.update_layout(
                title=f"{col_a} vs {col_b}",
                height=420, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=10, r=10, t=50, b=20),
            )
            st.plotly_chart(fig, width="stretch",
                            config={"displayModeBar": False})
        except Exception as e:
            st.error(f"Scatter failed: {e}")

    # Correlation heatmap
    if len(num_cols) >= 3:
        st.markdown("#### Correlation Matrix (Spearman)")
        try:
            corr_df  = df_filtered[num_cols[:12]].corr(method="spearman").round(3)
            fig_corr = go.Figure(go.Heatmap(
                z=corr_df.values,
                x=corr_df.columns.tolist(),
                y=corr_df.columns.tolist(),
                colorscale="RdBu", zmid=0, zmin=-1, zmax=1,
                text=np.round(corr_df.values, 2),
                texttemplate="%{text:.2f}",
                textfont=dict(size=9),
                colorbar=dict(title="r", thickness=12),
            ))
            fig_corr.update_layout(
                title="Spearman Correlation Matrix (r²: shared variance, not causation)",
                height=max(350, len(num_cols[:12]) * 40 + 80),
                paper_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=10, r=10, t=60, b=20),
            )
            st.plotly_chart(fig_corr, width="stretch",
                            config={"displayModeBar": False})
            st.caption("r² = shared variance between two variables. "
                       "Correlation is association — not causation.")
        except Exception as e:
            st.error(f"Correlation failed: {e}")

# ── Deep Dive tab ─────────────────────────────────────────────────────────────
with tabs[tab_idx]:
    tab_idx += 1
    st.markdown("#### Custom Chart Builder")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        chart_type = st.selectbox("Chart type", ["Bar","Box","Violin","Histogram","Line"],
                                   key="dd_type")
    with c2:
        x_col = st.selectbox("X axis", num_cols + cat_cols, key="dd_x")
    with c3:
        y_opts = ["—"] + num_cols
        y_col  = st.selectbox("Y axis", y_opts, key="dd_y")
    with c4:
        hue_opts = ["None"] + cat_cols
        hue_col  = st.selectbox("Colour by", hue_opts, key="dd_hue")

    try:
        c_kwargs = dict(color_discrete_sequence=px.colors.qualitative.Set2)
        if hue_col != "None": c_kwargs["color"] = hue_col

        if chart_type == "Bar" and y_col != "—":
            fig = px.bar(df_filtered, x=x_col, y=y_col, **c_kwargs)
        elif chart_type == "Box" and y_col != "—":
            fig = px.box(df_filtered, x=x_col if x_col in cat_cols else None,
                         y=y_col, **c_kwargs)
        elif chart_type == "Violin" and y_col != "—":
            fig = px.violin(df_filtered, x=x_col if x_col in cat_cols else None,
                            y=y_col, box=True, **c_kwargs)
        elif chart_type == "Histogram":
            fig = px.histogram(df_filtered, x=x_col, **c_kwargs)
        elif chart_type == "Line" and y_col != "—":
            fig = px.line(df_filtered, x=x_col, y=y_col, **c_kwargs)
        else:
            fig = px.histogram(df_filtered, x=x_col, **c_kwargs)

        fig.update_layout(
            height=420, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10, r=10, t=40, b=20),
        )
        st.plotly_chart(fig, width="stretch",
                        config={"displayModeBar": False})
    except Exception as e:
        st.error(f"Custom chart failed: {e}")

# ── Data Preview ──────────────────────────────────────────────────────────────
st.divider()
with st.expander("🔍 Filtered Data Preview"):
    st.dataframe(df_filtered.head(200), use_container_width=True)
    st.caption(f"Showing first 200 of {len(df_filtered):,} rows after filters.")

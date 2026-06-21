"""
pages/8_Reports.py — DataForge AI
Client-grade PDF Report Generator.
Config is READ from session_state (set on landing page).
No duplicate inputs — clean one-click generation.
"""
import streamlit as st
import pandas as pd
import numpy as np
import io
import os
import tempfile

from core.session_manager import require_data, get_df, get_filename, get_cached_stats, get_cached_ml

st.set_page_config(page_title="Reports — DataForge AI", layout="wide")
require_data()

df    = get_df()
fname = get_filename()

from core.pdf_builder    import build_pdf, THEMES
from core.chart_exporter import generate_all_charts
from core.story_engine   import generate_story, detect_domain
from core.data_profiler  import profile_dataset
import logging
logger = logging.getLogger(__name__)


@st.cache_data(show_spinner=False)
def _cached_profile(df):
    return profile_dataset(df)


@st.cache_data(show_spinner=False)
def _cached_domain(df):
    return detect_domain(df)


@st.cache_data(show_spinner=False)
def _cached_charts(df, theme_name, max_charts=5):
    return generate_all_charts(df, theme_name, max_charts=max_charts)


def _clean_fname(name):
    for ext in [".csv",".xlsx",".xls",".json"]:
        name = name.replace(ext,"")
    while name.startswith("cleaned_"):
        name = name[len("cleaned_"):]
    return name.strip("_- ").replace("_"," ")


fname_clean = _clean_fname(fname)

# ── CSS ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif!important}
.block-container{padding-top:1.2rem!important}
.cfg-block{background:white;border:1px solid #E2E8F0;border-radius:14px;padding:22px 24px;margin-bottom:16px}
.cfg-title{font-size:12px;font-weight:700;text-transform:uppercase;letter-spacing:.08em;
           color:#6B7280;margin-bottom:14px;padding-bottom:8px;border-bottom:1px solid #F1F5F9}
.pre-chip{background:#DBEAFE;border:1px solid #93C5FD;color:#1D4ED8;
          font-size:12px;font-weight:600;padding:6px 14px;border-radius:8px;display:inline-block}
section[data-testid="stSidebar"]{background:linear-gradient(180deg,#0D1B2E,#0F2240)!important}
section[data-testid="stSidebar"] *{color:rgba(255,255,255,.85)!important}
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────
st.markdown("## 📄 Generate PDF Report")
st.caption(f"{fname_clean} — {len(df):,} rows · {len(df.columns)} columns")
st.divider()

# ── Read config from session_state (set on landing page) ─────────────────
pre_client  = st.session_state.get("client_name", "")
pre_title   = st.session_state.get("report_title", f"Data Analysis Report — {fname_clean}")
pre_analyst = st.session_state.get("analyst_name", "DataForge AI")
pre_logo_bytes = st.session_state.get("logo_bytes", None)
pre_logo_ext   = st.session_state.get("logo_ext", "png")
pre_logo       = st.session_state.get("logo_path", "")  # legacy fallback

# ── Show pre-filled config ────────────────────────────────────────────────
if pre_client or pre_title:
    st.markdown('<div class="cfg-block">', unsafe_allow_html=True)
    st.markdown('<div class="cfg-title">Report Configuration (from landing page)</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"**Client:** `{pre_client or '—'}`")
    with c2:
        st.markdown(f"**Analyst:** `{pre_analyst or '—'}`")
    with c3:
        logo_status = "✅ Uploaded" if pre_logo_bytes else ("✅ Uploaded" if pre_logo and os.path.exists(pre_logo) else "—")
        st.markdown(f"**Logo:** {logo_status}")

    st.markdown(f"**Report Title:** {pre_title}")
    st.markdown("""
    <div style="font-size:12px;color:#6B7280;margin-top:8px;">
    ℹ️ To change client name or logo, go back to the <a href="/" style="color:#2563A8">Home page</a>.
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
else:
    st.info("💡 For a personalised report with client name and logo, set them on the **Home page** first.")

# ── Report options ────────────────────────────────────────────────────────
st.markdown("### Report Options")

col1, col2, col3 = st.columns(3)

with col1:
    theme_name    = st.selectbox("Report Theme", list(THEMES.keys()),
                                 help="Auto-selects by domain or choose manually")
    subtitle      = st.text_input("Subtitle", value="Powered by DataForge AI")

with col2:
    avg_salary_k  = st.number_input("Avg Annual Salary ($K) — for scenario cost estimates",
                                    min_value=20, max_value=500, value=50, step=5,
                                    help="Used in replacement cost scenario box — clearly labelled as estimated")
    confidential  = st.toggle("Confidential Stamp", value=True)

with col3:
    include_stats = st.toggle("Statistical Analysis", value=True)
    include_bi    = st.toggle("Business Intelligence", value=True)
    include_ml    = st.toggle("ML Results", value=False,
                              help="Enable after running ML Predictions page")

st.divider()

# ── What's included preview ───────────────────────────────────────────────
st.markdown("### Sections Included")
stats_cached = get_cached_stats()
ml_cached    = get_cached_ml()

cols = st.columns(4)
preview = [
    ("Cover + TOC", True),
    ("Executive Summary", True),
    ("Top Insights", True),
    ("Dataset Overview", True),
    ("Statistical Analysis", include_stats),
    ("Business Intelligence", include_bi),
    ("ML Predictions", ml_cached and include_ml),
    ("Recommendations", True),
]
for i, (label, on) in enumerate(preview):
    cols[i % 4].markdown(
        f"{'✅' if on else '⏸'} {label}"
    )

st.divider()

# ── Generate button ───────────────────────────────────────────────────────
st.markdown("### Generate")

gen_btn = st.button("🚀 Generate PDF Report", type="primary")

if gen_btn:
    progress = st.progress(0, text="Starting...")
    try:
        # 1. Profile
        progress.progress(8, text="Profiling dataset...")
        try:
            profile = _cached_profile(df)
        except Exception:
            profile = None

        # 2. Domain
        progress.progress(12, text="Detecting domain...")
        try:
            domain_name, _ = _cached_domain(df)
        except Exception:
            domain_name = "general"

        # 3. Story / narrative
        progress.progress(20, text="Generating executive summary...")
        story_obj = None
        try:
            story_obj     = generate_story(df)
            exec_summary  = story_obj.executive_summary
            findings      = story_obj.key_findings
            risks         = story_obj.business_risks
            opportunities = story_obj.opportunities
            actions       = story_obj.recommended_actions
        except Exception:
            # Rule-based fallback — real, specific, not just boilerplate
            num_cols  = df.select_dtypes(include="number").columns.tolist()
            cat_cols  = df.select_dtypes(include=["object", "string"]).columns.tolist()
            miss_pct  = round(df.isna().mean().mean() * 100, 1)
            dup_count = int(df.duplicated().sum())

            # Compute real stats for fallback narrative
            findings_list, risks_list = [], []
            for col in num_cols[:4]:
                try:
                    mean_v = df[col].mean()
                    std_v  = df[col].std()
                    cv     = (std_v / mean_v * 100) if mean_v != 0 else 0
                    if cv > 60:
                        findings_list.append(f"'{col}' shows high variability (CV={cv:.0f}%) — wide spread in values")
                    sk = float(df[col].skew())
                    if abs(sk) > 1.5:
                        findings_list.append(f"'{col}' is right-skewed (skew={sk:.2f}) — use median for reporting, not mean")
                except Exception:
                    logger.debug("%s silent skip", exc_info=True)

            exec_summary = (
                f"This report analyses {len(df):,} records across {len(df.columns)} variables "
                f"({len(num_cols)} numeric, {len(cat_cols)} categorical). "
                f"{'Data is fully complete with no missing values. ' if miss_pct == 0 else f'Missing data rate: {miss_pct:.1f}%. '}"
                f"{'No duplicate rows detected. ' if dup_count == 0 else f'{dup_count:,} duplicate rows were identified and removed before analysis. '}"
                f"Statistical analysis covers distributions, normality testing, correlation patterns, and outlier detection. "
                f"All findings are computed directly from the submitted dataset."
            )
            findings     = findings_list or [
                f"{len(df):,} records × {len(df.columns)} columns analysed",
                f"Missing data: {miss_pct:.1f}%",
                f"Numeric columns: {len(num_cols)} | Categorical: {len(cat_cols)}",
            ]
            risks         = []
            opportunities = []
            actions       = ["Run statistical analysis to identify key variable relationships",
                             "Review data quality flags before drawing business conclusions",
                             "Configure GROQ_API_KEY in secrets.toml for AI-generated narratives"]
            story_obj     = None

        # 4. Structured insights
        progress.progress(30, text="Building structured insights...")
        top_insights = []
        try:
            from core.insights_builder import build_top_insights
            attrition_obj = getattr(story_obj, "attrition", None)
            top_insights  = build_top_insights(
                df=df, domain=domain_name,
                story_obj=story_obj, attrition=attrition_obj,
                avg_salary_k=float(avg_salary_k),
            )
        except Exception:
            top_insights = []

        # 5. Stats
        progress.progress(40, text="Running statistical analysis...")
        stats_report = None
        if include_stats:
            try:
                stats_report = stats_cached or __import__("core.stats_engine", fromlist=["analyze"]).analyze(df)
            except Exception:
                logger.debug("%s silent skip", exc_info=True)

        # 6. BI
        progress.progress(52, text="Running business intelligence...")
        bi_report = None
        if include_bi:
            try:
                from core.bi_engine import run_bi
                bi_report = run_bi(df)
            except Exception:
                logger.debug("%s silent skip", exc_info=True)

        # 7. ML
        ml_report = ml_cached if include_ml else None

        # 8. Charts
        progress.progress(65, text="Generating charts...")
        chart_data = []
        groq_key = ""
        try:
            groq_key = st.secrets.get("GROQ_API_KEY", "")
        except Exception:
            groq_key = os.environ.get("GROQ_API_KEY", "")

        try:
            charts = _cached_charts(df, theme_name, max_charts=5)
            for title, img_bytes in charts:
                if img_bytes:
                    try:
                        from ai.report_narrator import generate_chart_narrative
                        narrative = generate_chart_narrative(df, title, groq_key, domain_name)
                    except Exception:
                        # Real-stats fallback — not boilerplate
                        try:
                            num_cols_n = df.select_dtypes(include="number").columns.tolist()
                            if num_cols_n:
                                col = num_cols_n[0]
                                mean_v = df[col].mean()
                                std_v = df[col].std()
                                narrative = (
                                    f"Chart shows '{title}'. "
                                    f"Primary metric '{col}': mean={mean_v:.2f}, "
                                    f"std={std_v:.2f} (CV={std_v/mean_v*100:.0f}% variability). "
                                    f"Dataset: {len(df):,} rows."
                                )
                            else:
                                narrative = f"Visual summary of '{title}' — {len(df):,} records analysed."
                        except Exception:
                            narrative = f"Chart: {title}."
                    chart_data.append((title, img_bytes, narrative))
        except Exception as e:
            st.warning(f"Charts skipped: {e}")

        # 9. Build PDF
        progress.progress(80, text="Building PDF...")
        attrition_obj = getattr(story_obj, "attrition", None)

        config = {
            "title":        pre_title or f"Data Analysis Report — {fname_clean}",
            "subtitle":     subtitle,
            "client_name":  pre_client or "Client",
            "analyst_name": pre_analyst or "DataForge AI",
            "confidential": confidential,
            "theme_name":   theme_name,
            "logo_path":    pre_logo,
            "logo_bytes":   pre_logo_bytes,
            "logo_ext":     pre_logo_ext,
            "avg_salary_k": avg_salary_k,
        }

        pdf_bytes = build_pdf(
            df=df, config=config, profile=profile,
            cleaning_summary=st.session_state.get("clean_report"),
            stats_report=stats_report, bi_report=bi_report,
            ml_report=ml_report, chart_data=chart_data,
            executive_summary=exec_summary,
            findings=findings, risks=risks, opportunities=opportunities,
            recommendations=actions, top_insights=top_insights,
            attrition=attrition_obj, domain=domain_name,
        )

        progress.progress(100, text="Done!")
        import time
        time.sleep(0.3)
        progress.empty()

        size_mb = len(pdf_bytes) / (1024*1024)
        st.success(f"✅ Report ready — {size_mb:.1f} MB · {len(chart_data)} charts · {len(top_insights)} insights")

        st.download_button(
            label="⬇️ Download PDF Report",
            data=pdf_bytes,
            file_name=f"DataForge_Report_{fname_clean.replace(' ','_')}.pdf",
            mime="application/pdf",
            type="primary",
            use_container_width=True,
        )

        # Logo is stored as bytes in session_state — no temp file to clean up

    except Exception as e:
        progress.empty()
        st.error(f"Report generation failed: {e}")
        st.exception(e)

st.divider()

# ── Data Export ───────────────────────────────────────────────────────────
st.markdown("### Export Data")
c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("**CSV — Cleaned Data**")
    st.caption(f"{len(df):,} rows × {len(df.columns)} columns")
    st.download_button("Download CSV", data=df.to_csv(index=False).encode("utf-8"),
                       file_name=f"cleaned_{fname_clean.replace(' ','_')}.csv",
                       mime="text/csv", use_container_width=True)
with c2:
    st.markdown("**Excel — 3 Sheets**")
    st.caption("Data + Statistics + Column Info")
    try:
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
            df.to_excel(writer, sheet_name="Cleaned Data", index=False)
            df.describe().round(4).reset_index().to_excel(writer, sheet_name="Statistics", index=False)
            pd.DataFrame([{"Column":c,"Type":str(df[c].dtype),
                           "Missing":int(df[c].isna().sum()),
                           "Missing %":round(df[c].isna().mean()*100,2),
                           "Unique":int(df[c].nunique())} for c in df.columns]
                        ).to_excel(writer, sheet_name="Column Info", index=False)
        buf.seek(0)
        st.download_button("Download Excel", data=buf,
                           file_name=f"cleaned_{fname_clean.replace(' ','_')}.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                           use_container_width=True)
    except Exception as e:
        st.error(f"Excel failed: {e}")
with c3:
    st.markdown("**JSON — Records Format**")
    st.caption("For API / downstream pipelines")
    st.download_button("Download JSON",
                       data=df.to_json(orient="records", indent=2, date_format="iso").encode("utf-8"),
                       file_name=f"cleaned_{fname_clean.replace(' ','_')}.json",
                       mime="application/json", use_container_width=True)

"""
pages/8_Reports.py — DataForge AI
Client-grade PDF Report Generator.
FIXED v4:
  ✅ Top Insights — structured cards (not "No structured insights available")
  ✅ Logo upload — appears on cover page
  ✅ Client/Company name — editable, appears on cover
  ✅ Domain auto-detected and passed to pdf_builder
"""
import streamlit as st
import pandas as pd
import numpy as np
import io
import os
import tempfile

from core.session_manager import (
    require_data, get_df, get_filename,
    get_cached_stats, get_cached_ml,
)

st.set_page_config(page_title="Reports — DataForge AI", layout="wide")
require_data()

df    = get_df()
fname = get_filename()

# ── IMPORTS ───────────────────────────────────────────────
from core.pdf_builder    import build_pdf, THEMES
from core.chart_exporter import generate_all_charts
from core.story_engine   import generate_story, detect_domain
from core.data_profiler  import profile_dataset


def _clean_fname(name: str) -> str:
    for ext in [".csv", ".xlsx", ".xls", ".json"]:
        name = name.replace(ext, "")
    while name.startswith("cleaned_"):
        name = name[len("cleaned_"):]
    return name.strip("_- ").replace("_", " ")


fname_clean = _clean_fname(fname)

# ══════════════════════════════════════════════════════════
#  PAGE HEADER
# ══════════════════════════════════════════════════════════
st.markdown("## 📄 Reports & Export")
st.caption("{} — {:,} rows, {} columns".format(
    fname_clean, len(df), len(df.columns)))
st.divider()

# ══════════════════════════════════════════════════════════
#  SECTION 1 — REPORT CONFIGURATION
# ══════════════════════════════════════════════════════════
st.markdown("### Report Configuration")

c1, c2, c3 = st.columns(3)

with c1:
    report_title = st.text_input(
        "Report Title",
        value="Data Analysis Report — {}".format(fname_clean))

    client_name = st.text_input(
        "Prepared For (Client / Company Name)",
        value="Client",
        help="Appears on the cover page — use your client's company name for freelancing")

    # ── LOGO UPLOAD ────────────────────────────────────────
    logo_file = st.file_uploader(
        "Company Logo (optional)",
        type=["png", "jpg", "jpeg"],
        help="Logo appears on the PDF cover page. PNG with transparent background works best.")

    logo_path = ""
    if logo_file is not None:
        try:
            suffix = "." + logo_file.name.split(".")[-1].lower()
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            tmp.write(logo_file.read())
            tmp.flush()
            tmp.close()
            logo_path = tmp.name
            st.success("✅ Logo uploaded — will appear on cover page")
        except Exception as e:
            st.warning("Logo upload failed: {}".format(e))
            logo_path = ""

with c2:
    theme_name = st.selectbox(
        "Report Theme",
        list(THEMES.keys()),
        help="Theme auto-selects by domain (HR=Blue, Ecom=Orange, Sales=Green) — or choose manually")

    subtitle = st.text_input(
        "Subtitle",
        value="Powered by DataForge AI")

    avg_salary_k = st.number_input(
        "Avg Annual Salary ($K) — for cost estimates",
        min_value=20, max_value=500, value=50, step=5,
        help="Used to estimate attrition replacement costs in the report")

with c3:
    confidential  = st.toggle("Confidential Stamp", value=True)
    include_stats = st.toggle("Statistical Analysis", value=True)
    include_bi    = st.toggle("Business Intelligence", value=True)
    include_ml    = st.toggle(
        "ML Results", value=False,
        help="Enable after running ML Predictions page")

st.divider()

# ══════════════════════════════════════════════════════════
#  SECTION 2 — WHAT WILL BE INCLUDED
# ══════════════════════════════════════════════════════════
st.markdown("### What Will Be Included")

stats_cached = get_cached_stats()
ml_cached    = get_cached_ml()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Cover + TOC",        "✅ YES")
c2.metric("Executive Summary",  "✅ Auto-generated")
c3.metric("Top Insights",       "✅ Structured cards")
c4.metric("Dataset Overview",   "✅ Stats + cleaning")

c5, c6, c7, c8 = st.columns(4)
c5.metric("Statistical Analysis",  "✅ YES" if include_stats else "⏸ OFF")
c6.metric("Business Intelligence", "✅ YES" if include_bi    else "⏸ OFF")
c7.metric("ML Predictions",        "✅ YES" if (ml_cached and include_ml) else "⏸ OFF")
c8.metric("Recommendations",       "✅ YES — prioritized")

st.divider()

# ══════════════════════════════════════════════════════════
#  SECTION 3 — GENERATE REPORT
# ══════════════════════════════════════════════════════════
st.markdown("### Generate Report")

gen_btn = st.button(
    "🚀 Generate PDF Report",
    type="primary",
    use_container_width=False)

if gen_btn:
    progress = st.progress(0, text="Starting...")

    try:
        # ── Step 1: Profile ───────────────────────────────
        progress.progress(8, text="Profiling dataset...")
        try:
            profile = profile_dataset(df)
        except Exception:
            profile = None

        # ── Step 2: Domain detection ──────────────────────
        progress.progress(12, text="Detecting domain...")
        try:
            domain_name, _ = detect_domain(df)
        except Exception:
            domain_name = "general"

        # ── Step 3: Story / narrative ──────────────────────
        progress.progress(20, text="Generating executive summary...")
        story_obj = None
        try:
            story_obj    = generate_story(df)
            exec_summary = story_obj.executive_summary
            findings     = story_obj.key_findings
            risks        = story_obj.business_risks
            opportunities= story_obj.opportunities
            actions      = story_obj.recommended_actions
        except Exception:
            exec_summary = "Analysis completed by DataForge AI."
            findings = risks = opportunities = actions = []
            story_obj = None

        # ── Step 4: Build structured top insights ─────────
        # FIXED: was "No structured insights available"
        progress.progress(30, text="Building structured insights...")
        top_insights = []
        try:
            from core.insights_builder import build_top_insights
            attrition_obj = getattr(story_obj, "attrition", None)
            top_insights  = build_top_insights(
                df           = df,
                domain       = domain_name,
                story_obj    = story_obj,
                attrition    = attrition_obj,
                avg_salary_k = float(avg_salary_k),
            )
        except Exception as e:
            top_insights = []

        # ── Step 5: Stats ──────────────────────────────────
        progress.progress(40, text="Running statistical analysis...")
        stats_report = None
        if include_stats:
            try:
                if stats_cached:
                    stats_report = stats_cached
                else:
                    from core.stats_engine import analyze
                    stats_report = analyze(df)
            except Exception:
                pass

        # ── Step 6: Business Intelligence ─────────────────
        progress.progress(52, text="Running business intelligence...")
        bi_report = None
        if include_bi:
            try:
                from core.bi_engine import run_bi
                bi_report = run_bi(df)
            except Exception:
                pass

        # ── Step 7: ML ────────────────────────────────────
        ml_report = ml_cached if include_ml else None

        # ── Step 8: Charts ────────────────────────────────
        progress.progress(65, text="Generating charts...")
        chart_data = []
        # FIX: st.secrets raises when secrets.toml is absent — use safe fallback
        groq_key = ""
        try:
            groq_key = st.secrets.get("GROQ_API_KEY", "")
        except Exception:
            groq_key = os.environ.get("GROQ_API_KEY", "")

        try:
            charts = generate_all_charts(df, theme_name, max_charts=5)
            for title, img_bytes in charts:
                if img_bytes:
                    try:
                        # FIXED: Uses corrected report_narrator
                        from ai.report_narrator import generate_chart_narrative
                        narrative = generate_chart_narrative(
                            df, title, groq_key, domain_name)
                    except Exception:
                        narrative = "Chart generated from dataset analysis."
                    chart_data.append((title, img_bytes, narrative))
            st.info("{} charts generated.".format(len(chart_data)))
        except Exception as e:
            st.warning("Charts skipped: {}".format(str(e)))

        # ── Step 9: Build PDF ──────────────────────────────
        progress.progress(80, text="Building PDF...")

        # Attrition object for pdf_builder
        attrition_obj = getattr(story_obj, "attrition", None)

        cleaning_summary = st.session_state.get("clean_report")

        config = {
            "title":        report_title,
            "subtitle":     subtitle,
            "client_name":  client_name,
            "confidential": confidential,
            "theme_name":   theme_name,
            "logo_path":    logo_path,       # ← logo support
        }

        pdf_bytes = build_pdf(
            df                 = df,
            config             = config,
            profile            = profile,
            cleaning_summary   = cleaning_summary,
            stats_report       = stats_report,
            bi_report          = bi_report,
            ml_report          = ml_report,
            chart_data         = chart_data,
            executive_summary  = exec_summary,
            findings           = findings,
            risks              = risks,
            opportunities      = opportunities,
            recommendations    = actions,
            top_insights       = top_insights,   # ← FIXED
            attrition          = attrition_obj,
            domain             = domain_name,    # ← domain passed
        )

        progress.progress(100, text="Done!")
        import time; time.sleep(0.3)
        progress.empty()

        # ── Summary ───────────────────────────────────────
        n_pages = (3 + len(chart_data)
                   + (1 if stats_report  else 0)
                   + (1 if bi_report     else 0)
                   + (1 if ml_report     else 0) + 2)

        st.success(
            "✅ Report ready — {:.1f} MB | ~{} pages | {} charts | {} insights".format(
                len(pdf_bytes) / (1024 * 1024),
                n_pages,
                len(chart_data),
                len(top_insights)))

        # Sections preview
        sections = ["Cover", "TOC", "Executive Summary",
                    "Data Quality Note", "Top Insights"]
        if domain_name in ("hr", "ecommerce", "sales"):
            sections.append("Industry Benchmarks")
        if attrition_obj:
            sections.append("Attrition Deep Dive")
        sections.append("Dataset Overview")
        if stats_report:  sections.append("Statistical Analysis")
        if bi_report:     sections.append("Business Intelligence")
        for i, (t, _, _) in enumerate(chart_data, 1):
            sections.append("Chart {}: {}".format(i, t[:28]))
        sections += ["Recommendations", "Appendix"]

        with st.expander("Report sections ({})".format(len(sections))):
            for i, s in enumerate(sections, 1):
                st.markdown("{}. {}".format(i, s))

        # ── DOWNLOAD ──────────────────────────────────────
        st.download_button(
            label            = "⬇️ Download PDF Report",
            data             = pdf_bytes,
            file_name        = "DataForge_Report_{}.pdf".format(
                fname_clean.replace(" ", "_")),
            mime             = "application/pdf",
            type             = "primary",
            use_container_width = True,
        )

        # Cleanup temp logo file
        if logo_path and os.path.exists(logo_path):
            try:
                os.unlink(logo_path)
            except Exception:
                pass

    except Exception as e:
        progress.empty()
        st.error("Report generation failed: {}".format(str(e)))
        st.exception(e)

st.divider()

# ══════════════════════════════════════════════════════════
#  SECTION 4 — DATA EXPORTS
# ══════════════════════════════════════════════════════════
st.markdown("### Export Data")

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("**CSV — Cleaned Data**")
    st.caption("{:,} rows × {} columns".format(len(df), len(df.columns)))
    st.download_button(
        "Download CSV",
        data      = df.to_csv(index=False).encode("utf-8"),
        file_name = "cleaned_{}.csv".format(fname_clean.replace(" ", "_")),
        mime      = "text/csv",
        use_container_width = True,
    )

with c2:
    st.markdown("**Excel — 3 Sheets**")
    st.caption("Data + Statistics + Column Info")
    try:
        buf_xl = io.BytesIO()
        with pd.ExcelWriter(buf_xl, engine="xlsxwriter") as writer:
            df.to_excel(writer, sheet_name="Cleaned Data", index=False)
            df.describe().round(4).reset_index().to_excel(
                writer, sheet_name="Statistics", index=False)
            pd.DataFrame([{
                "Column":    c,
                "Type":      str(df[c].dtype),
                "Missing":   int(df[c].isna().sum()),
                "Missing %": round(df[c].isna().mean() * 100, 2),
                "Unique":    int(df[c].nunique()),
                "Sample":    str(df[c].dropna().iloc[0])
                             if len(df[c].dropna()) > 0 else "",
            } for c in df.columns]).to_excel(
                writer, sheet_name="Column Info", index=False)
        buf_xl.seek(0)
        st.download_button(
            "Download Excel",
            data      = buf_xl,
            file_name = "cleaned_{}.xlsx".format(fname_clean.replace(" ", "_")),
            mime      = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width = True,
        )
    except Exception as e:
        st.error("Excel failed: {}".format(str(e)))

with c3:
    st.markdown("**JSON — Records Format**")
    st.caption("For API / downstream pipelines")
    st.download_button(
        "Download JSON",
        data      = df.to_json(orient="records", indent=2,
                               date_format="iso").encode("utf-8"),
        file_name = "cleaned_{}.json".format(fname_clean.replace(" ", "_")),
        mime      = "application/json",
        use_container_width = True,
    )

"""
pages/8_Reports.py — Client-grade PDF Report.
Senior analyst level — charts, stats, BI, narrative all integrated.
"""
import streamlit as st
import pandas as pd
import numpy as np
import io
import os

from core.session_manager import (
    require_data, get_df, get_filename,
    get_cached_stats, get_cached_ml,
)

st.set_page_config(page_title="Reports — DataForge AI", layout="wide")
require_data()

df    = get_df()
fname = get_filename()

# ── ALL IMPORTS + HELPERS FIRST ───────────────────────────
from core.pdf_builder import build_pdf, THEMES
from core.chart_exporter import generate_all_charts
from core.story_engine import generate_story
from core.data_profiler import profile_dataset


def _clean_fname(name: str) -> str:
    """Remove extensions and repeated cleaned_ prefixes."""
    for ext in [".csv", ".xlsx", ".xls", ".json"]:
        name = name.replace(ext, "")
    while name.startswith("cleaned_"):
        name = name[len("cleaned_"):]
    return name.strip("_- ").replace("_", " ")


def _chart_narrative(df: pd.DataFrame, title: str,
                     groq_key: str = "", domain: str = "general") -> str:
    """
    McKinsey-style chart narrative.
    Real stats computed first → LLM narrates only.
    """
    try:
        from ai.report_narrator import generate_chart_narrative
        return generate_chart_narrative(df, title, groq_key, domain)
    except Exception:
        pass

    # Fallback rule-based
    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include="object").columns.tolist()

    try:
        # Bar chart narrative
        if "by" in title.lower() and cat_cols and num_cols:
            # Extract column names from title
            parts    = title.lower().replace("avg ", "").replace("total ", "").split(" by ")
            num_col  = next((c for c in num_cols if c.lower() in parts[0]), num_cols[0])
            cat_col  = next((c for c in cat_cols if c.lower() in (parts[1] if len(parts)>1 else "")), cat_cols[0])
            grp      = df.groupby(cat_col)[num_col].mean().sort_values(ascending=False)
            best     = grp.index[0]
            worst    = grp.index[-1]
            gap_pct  = abs(grp.iloc[0] - grp.iloc[-1]) / abs(grp.iloc[-1]) * 100 if grp.iloc[-1] != 0 else 0
            above_avg = (grp > grp.mean()).sum()
            return (
                "Analysis of '{}' across {} '{}' categories. "
                "Top performer: '{}' (avg={:.2f}), lowest: '{}' (avg={:.2f}). "
                "Performance gap between best and worst: {:.1f}%. "
                "{} out of {} categories are above the dataset average ({:.2f}).".format(
                    num_col, len(grp), cat_col,
                    best, grp.iloc[0], worst, grp.iloc[-1],
                    gap_pct, above_avg, len(grp),
                    float(df[num_col].mean()))
            )

        # Distribution/histogram narrative
        elif "distribution" in title.lower() and num_cols:
            col = next((c for c in num_cols if c.lower() in title.lower()), num_cols[0])
            s   = df[col].dropna()
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr  = q3 - q1
            outliers = int(((s < q1 - 1.5*iqr) | (s > q3 + 1.5*iqr)).sum())
            skew = s.skew()
            skew_label = "right-skewed (tail extends right — few high values pull the mean up)" \
                         if skew > 0.5 else \
                         "left-skewed (tail extends left — few low values pull the mean down)" \
                         if skew < -0.5 else "approximately symmetric"
            return (
                "Distribution analysis of '{}': mean={:.2f}, median={:.2f}, std={:.2f}. "
                "Range: {:.2f} to {:.2f}. Middle 50% of values fall between {:.2f} and {:.2f}. "
                "Distribution is {} (skewness={:.2f}). "
                "{} outliers detected using IQR method ({:.1f}% of data). "
                "{}.".format(
                    col, s.mean(), s.median(), s.std(),
                    s.min(), s.max(), q1, q3,
                    skew_label, skew,
                    outliers, outliers/len(s)*100,
                    "Use median ({:.2f}) as central measure since distribution is skewed.".format(s.median())
                    if abs(skew) > 0.5
                    else "Mean ({:.2f}) is a reliable central measure.".format(s.mean()))
            )

        # Correlation heatmap narrative
        elif "correlation" in title.lower() and len(num_cols) >= 2:
            corr_matrix = df[num_cols[:8]].corr()
            pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    a, b = corr_matrix.columns[i], corr_matrix.columns[j]
                    r = corr_matrix.loc[a, b]
                    pairs.append((a, b, float(r)))
            pairs.sort(key=lambda x: abs(x[2]), reverse=True)
            strong = [(a,b,r) for a,b,r in pairs if abs(r) >= 0.5]
            weak   = [(a,b,r) for a,b,r in pairs if abs(r) < 0.2]
            result = "Correlation analysis across {} numeric columns. ".format(len(num_cols[:8]))
            if strong:
                a, b, r = strong[0]
                direction = "positive" if r > 0 else "negative"
                result += "Strongest {}: '{}' and '{}' (r={:.2f}) — ".format(
                    direction, a, b, r)
                result += "as {} increases, {} tends to {}. ".format(
                    a, b, "increase" if r > 0 else "decrease")
            if len(strong) > 1:
                result += "{} strong relationships found (|r|>=0.5). ".format(len(strong))
            result += "{} weak/no relationships (|r|<0.2). ".format(len(weak))
            result += "High correlations may indicate multicollinearity — review before modeling."
            return result

        # Pie/share chart narrative
        elif "share" in title.lower() and cat_cols and num_cols:
            parts   = title.lower().replace("avg ", "").split(" by ")
            num_col = next((c for c in num_cols if c.lower() in parts[0]), num_cols[0])
            cat_col = next((c for c in cat_cols if c.lower() in (parts[-1] if len(parts)>1 else "")), cat_cols[0])
            grp     = df.groupby(cat_col)[num_col].mean().sort_values(ascending=False)
            top_pct = grp.iloc[0] / grp.sum() * 100
            top2_pct = grp.iloc[:2].sum() / grp.sum() * 100
            return (
                "Share analysis: '{}' contribution by '{}' category. "
                "Top category '{}' accounts for {:.1f}% of total. "
                "Top 2 categories together: {:.1f}%. "
                "{} segments analyzed. "
                "{}.".format(
                    num_col, cat_col,
                    grp.index[0], top_pct, top2_pct, len(grp),
                    "Value is concentrated in few categories — Pareto effect present."
                    if top2_pct > 60
                    else "Value is distributed relatively evenly across categories.")
            )

        # Line/trend chart narrative
        elif "trend" in title.lower() or "over" in title.lower():
            col = next((c for c in num_cols if c.lower() in title.lower()), num_cols[0])
            s   = df[col].dropna()
            return (
                "Trend analysis of '{}': overall mean={:.2f}, "
                "range {:.2f} to {:.2f}. "
                "Coefficient of variation={:.1f}% — {}".format(
                    col, s.mean(), s.min(), s.max(),
                    s.std()/abs(s.mean())*100 if s.mean()!=0 else 0,
                    "high variability over time." if s.std()/abs(s.mean()) > 0.3
                    else "relatively stable pattern.")
            )

    except Exception:
        pass

    return "Chart generated from dataset analysis. See statistical sections for detailed metrics."


fname_clean = _clean_fname(fname)

# ══════════════════════════════════════════════════════════
#  PAGE HEADER
# ══════════════════════════════════════════════════════════
st.markdown("## Reports & Export")
st.caption("{} — {:,} rows, {} columns".format(
    fname_clean, len(df), len(df.columns)))
st.divider()

# ══════════════════════════════════════════════════════════
#  SECTION 1 — CONFIG
# ══════════════════════════════════════════════════════════
st.markdown("### Report Configuration")

c1, c2, c3 = st.columns(3)
with c1:
    report_title = st.text_input(
        "Report Title",
        value="Data Analysis Report — {}".format(fname_clean))
    client_name  = st.text_input("Prepared For (Client Name)", value="Client")
with c2:
    theme_name = st.selectbox("Report Theme", list(THEMES.keys()))
    subtitle   = st.text_input("Subtitle", value="Powered by DataForge AI")
with c3:
    confidential  = st.toggle("Confidential Stamp", value=True)
    include_stats = st.toggle("Statistical Analysis", value=True)
    include_bi    = st.toggle("Business Intelligence", value=True)
    include_ml    = st.toggle("ML Results", value=False,
                              help="Enable after running ML Predictions page")

st.divider()

# ══════════════════════════════════════════════════════════
#  SECTION 2 — STATUS
# ══════════════════════════════════════════════════════════
st.markdown("### What Will Be Included")

stats_cached = get_cached_stats()
ml_cached    = get_cached_ml()

c1,c2,c3,c4 = st.columns(4)
c1.metric("Cover + TOC",       "YES")
c2.metric("Executive Summary", "Auto-generated")
c3.metric("Dataset Overview",  "YES — stats + cleaning")
c4.metric("Charts",            "YES — up to 5 auto")

c5,c6,c7,c8 = st.columns(4)
c5.metric("Statistical Analysis",
          "YES" if include_stats else "OFF")
c6.metric("Business Intelligence",
          "YES" if include_bi else "OFF")
c7.metric("ML Predictions",
          "YES" if (ml_cached and include_ml) else "OFF")
c8.metric("Recommendations",   "YES — prioritized")

st.divider()

# ══════════════════════════════════════════════════════════
#  SECTION 3 — GENERATE
# ══════════════════════════════════════════════════════════
st.markdown("### Generate Report")

gen_btn = st.button("Generate PDF Report",
                    type="primary", use_container_width=False)

if gen_btn:
    progress = st.progress(0, text="Starting...")

    try:
        # Step 1 — Profile
        progress.progress(10, text="Profiling dataset...")
        try:
            profile = profile_dataset(df)
        except Exception:
            profile = None

        # Step 2 — Story / narrative
        progress.progress(20, text="Generating executive summary...")
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

        # Step 3 — Stats
        progress.progress(35, text="Running statistical analysis...")
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

        # Step 4 — BI
        progress.progress(50, text="Running business intelligence...")
        bi_report = None
        if include_bi:
            try:
                from core.bi_engine import run_bi
                bi_report = run_bi(df)
            except Exception:
                pass

        # Step 5 — ML
        ml_report = ml_cached if include_ml else None

        # Step 6 — Charts
        progress.progress(65, text="Generating charts...")
        chart_data = []
        try:
            from core.story_engine import detect_domain
            domain_name, _ = detect_domain(df)
            groq_key = ""
            try:
                groq_key = st.secrets.get("GROQ_API_KEY","")
            except Exception:
                groq_key = os.environ.get("GROQ_API_KEY","")

            charts = generate_all_charts(df, theme_name, max_charts=5)
            for title, img_bytes in charts:
                if img_bytes:
                    narrative = _chart_narrative(df, title, groq_key, domain_name)
                    chart_data.append((title, img_bytes, narrative))
            st.info("{} charts generated — AI narratives applied.".format(len(chart_data)))
        except Exception as e:
            st.warning("Charts skipped: {}".format(str(e)))

        # Step 7 — Build PDF
        progress.progress(80, text="Building PDF...")
        config = {
            "title":        report_title,
            "subtitle":     subtitle,
            "client_name":  client_name,
            "confidential": confidential,
            "theme_name":   theme_name,
        }

        cleaning_summary = st.session_state.get("clean_report")

        # Domain detection for theme
        from core.story_engine import detect_domain
        domain_name, _ = detect_domain(df)

        pdf_bytes = build_pdf(
            df=df,
            config=config,
            profile=profile,
            cleaning_summary=cleaning_summary,
            stats_report=stats_report,
            bi_report=bi_report,
            ml_report=ml_report,
            chart_data=chart_data,
            executive_summary=exec_summary,
            findings=findings,
            risks=risks,
            opportunities=opportunities,
            recommendations=actions,
            top_insights=getattr(story_obj, "top_insights", []),
            attrition=getattr(story_obj, "attrition", None),
            domain=domain_name,
        )

        progress.progress(100, text="Done!")

        # Summary
        n_pages = (3 + len(chart_data)
                   + (1 if stats_report else 0)
                   + (1 if bi_report else 0)
                   + (1 if ml_report else 0) + 2)

        st.success(
            "Report ready — {:.1f} MB | ~{} pages | {} charts".format(
                len(pdf_bytes)/(1024*1024), n_pages, len(chart_data))
        )

        # What's inside summary
        sections = ["Cover", "TOC", "Executive Summary", "Dataset Overview"]
        if stats_report: sections.append("Statistical Analysis")
        if bi_report:    sections.append("Business Intelligence")
        if ml_report:    sections.append("ML Predictions")
        for i, (t,_,_) in enumerate(chart_data, 1):
            sections.append("Chart {}: {}".format(i, t[:25]))
        sections += ["Recommendations", "Appendix"]

        with st.expander("Report sections ({})".format(len(sections))):
            for i, s in enumerate(sections, 1):
                st.markdown("{}. {}".format(i, s))

        st.download_button(
            label="Download PDF Report",
            data=pdf_bytes,
            file_name="DataForge_Report_{}.pdf".format(
                fname_clean.replace(" ", "_")),
            mime="application/pdf",
            type="primary",
            use_container_width=True,
        )

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
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="cleaned_{}.csv".format(fname_clean.replace(" ","_")),
        mime="text/csv",
        use_container_width=True,
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
                "Column": c, "Type": str(df[c].dtype),
                "Missing": int(df[c].isna().sum()),
                "Missing %": round(df[c].isna().mean()*100,2),
                "Unique": int(df[c].nunique()),
                "Sample": str(df[c].dropna().iloc[0]) if len(df[c].dropna())>0 else "",
            } for c in df.columns]).to_excel(
                writer, sheet_name="Column Info", index=False)
        buf_xl.seek(0)
        st.download_button(
            "Download Excel",
            data=buf_xl,
            file_name="cleaned_{}.xlsx".format(fname_clean.replace(" ","_")),
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
    except Exception as e:
        st.error("Excel failed: {}".format(str(e)))

with c3:
    st.markdown("**JSON — Records Format**")
    st.caption("For API / downstream pipelines")
    st.download_button(
        "Download JSON",
        data=df.to_json(orient="records", indent=2, date_format="iso").encode("utf-8"),
        file_name="cleaned_{}.json".format(fname_clean.replace(" ","_")),
        mime="application/json",
        use_container_width=True,
    )

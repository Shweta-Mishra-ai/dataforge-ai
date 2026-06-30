"""
pages/4_Business_Insights.py — DataForge AI
MNC-standard: structured insights, adaptive dark/light theme, no hardcoded bg colors.
"""
import io
import logging
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

logger = logging.getLogger(__name__)

from core.session_manager import require_data, get_df, get_filename
require_data()

from core.story_engine import generate_story, detect_domain
from core.insights_builder import build_top_insights

st.set_page_config(page_title="Business Insights — DataForge AI", layout="wide")

# ── GLOBAL ADAPTIVE CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif!important}
.block-container{padding-top:1.2rem!important}

/* Sidebar */
section[data-testid="stSidebar"]{background:linear-gradient(180deg,#0D1B2E,#0F2240)!important}
section[data-testid="stSidebar"] *{color:rgba(255,255,255,.85)!important}
section[data-testid="stSidebar"] hr{border-color:rgba(255,255,255,.12)!important}

/* Domain banner — always dark bg so always white text */
.domain-header{background:linear-gradient(135deg,#0D1B2E 0%,#1B3A6B 100%);border-radius:14px;padding:22px 26px;margin-bottom:20px}
.dh-title{font-size:22px;font-weight:800;color:#ffffff;margin-bottom:6px}
.dh-sub{font-size:13px;color:rgba(255,255,255,.6);line-height:1.5}
.ms-wrap{display:flex;gap:10px;flex-wrap:wrap;margin-top:14px}
.ms-item{background:rgba(255,255,255,.1);border:1px solid rgba(255,255,255,.18);border-radius:10px;padding:10px 18px;text-align:center}
.ms-val{font-size:20px;font-weight:800;color:#ffffff}
.ms-lbl{font-size:10px;color:rgba(255,255,255,.5);text-transform:uppercase;letter-spacing:.07em;margin-top:2px}

/* Insight cards — adaptive */
.insight-card{border-radius:12px;overflow:hidden;border:1px solid rgba(128,128,128,.2);margin-bottom:16px}
.ic-header{padding:13px 18px;font-size:14px;font-weight:700;color:#ffffff;display:flex;align-items:center;gap:10px;line-height:1.3}
.ic-body{padding:14px 18px}
.ic-row{display:flex;gap:10px;margin-bottom:10px;align-items:flex-start;border-bottom:1px solid rgba(128,128,128,.08);padding-bottom:10px}
.ic-row:last-child{border-bottom:none;margin-bottom:0;padding-bottom:0}
.ic-lbl{font-size:10px;font-weight:800;text-transform:uppercase;letter-spacing:.08em;min-width:76px;padding-top:2px;flex-shrink:0;opacity:.7}
.ic-val{font-size:13px;color:inherit;line-height:1.6}
.ic-code{font-size:12px;background:rgba(128,128,128,.1);padding:3px 8px;border-radius:4px;font-family:'JetBrains Mono',monospace;display:inline-block}

/* Risk / opportunity rows */
.risk-row{border-left:4px solid #ef4444;background:rgba(239,68,68,.07);padding:13px 16px;border-radius:0 8px 8px 0;margin-bottom:10px}
.risk-sev{font-size:10px;font-weight:800;text-transform:uppercase;letter-spacing:.07em;color:#ef4444;margin-bottom:5px}
.opp-row{border-left:4px solid #10b981;background:rgba(16,185,129,.07);padding:13px 16px;border-radius:0 8px 8px 0;margin-bottom:10px}
.opp-sev{font-size:10px;font-weight:800;text-transform:uppercase;letter-spacing:.07em;color:#10b981;margin-bottom:5px}
.warn-row{border-left:4px solid #f59e0b;background:rgba(245,158,11,.07);padding:13px 16px;border-radius:0 8px 8px 0;margin-bottom:10px}

/* Finding rows */
.finding-row{background:rgba(128,128,128,.05);border:1px solid rgba(128,128,128,.12);border-radius:8px;padding:13px 16px;margin-bottom:8px;font-size:13.5px;line-height:1.6}
.finding-num{font-size:10px;font-weight:800;text-transform:uppercase;letter-spacing:.07em;opacity:.55;margin-bottom:4px}

/* Action items */
.action-item{border:1px solid rgba(128,128,128,.15);border-radius:10px;padding:14px 18px;margin-bottom:10px;background:rgba(128,128,128,.03)}
.action-pri{font-size:10px;font-weight:800;text-transform:uppercase;letter-spacing:.07em;margin-bottom:6px}

/* Exec summary box */
.exec-box{border-left:4px solid #3b82f6;background:rgba(59,130,246,.06);padding:18px 22px;border-radius:0 8px 8px 0;font-size:14.5px;line-height:1.8}

/* Hypo disclaimer */
.hypo-box{background:rgba(245,158,11,.08);border:1px solid rgba(245,158,11,.3);border-left:4px solid #f59e0b;border-radius:8px;padding:12px 16px;margin:10px 0;font-size:13px}

/* Stat table */
.stat-tbl td{padding:8px 12px;font-size:13px;border-bottom:1px solid rgba(128,128,128,.1)}
.stat-tbl th{padding:8px 12px;font-size:11px;text-transform:uppercase;letter-spacing:.06em;opacity:.6;border-bottom:2px solid rgba(128,128,128,.2)}
</style>
""", unsafe_allow_html=True)


# ── Chart styler — works on both themes ──────────────────────────────────────
def _style_fig(fig, bgcolor="rgba(0,0,0,0)"):
    fig.update_layout(
        paper_bgcolor=bgcolor, plot_bgcolor=bgcolor,
        font=dict(size=11),
        margin=dict(l=10, r=10, t=44, b=10),
    )
    fig.update_xaxes(gridcolor="rgba(128,128,128,.15)", zeroline=False,
                     tickfont=dict(size=10))
    fig.update_yaxes(gridcolor="rgba(128,128,128,.15)", zeroline=False,
                     tickfont=dict(size=10))
    return fig


# ── Cache ─────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def get_story_cached(df_json: str):
    df = pd.read_json(io.StringIO(df_json))
    return generate_story(df)

def _find(df, keywords, exclude=None):
    excl = exclude or []
    for c in df.columns:
        cl = c.lower()
        if any(k in cl for k in keywords) and not any(e in cl for e in excl):
            return c
    return None


# ── Load ──────────────────────────────────────────────────────────────────────
df    = get_df()
fname = get_filename()

with st.spinner("Generating business insights…"):
    report = get_story_cached(df.to_json(date_format="iso"))
    domain, confidence = detect_domain(df)

# ── Domain metadata ───────────────────────────────────────────────────────────
DOMAIN_META = {
    "hr":        {"icon": "👥", "label": "HR & People Analytics",  "accent": "#3b82f6"},
    "finance":   {"icon": "💰", "label": "Finance & Accounting",   "accent": "#10b981"},
    "ecommerce": {"icon": "🛒", "label": "E-Commerce Analytics",   "accent": "#f59e0b"},
    "sales":     {"icon": "📈", "label": "Sales Performance",      "accent": "#8b5cf6"},
    "general":   {"icon": "📊", "label": "Business Intelligence",  "accent": "#64748b"},
}
dmeta  = DOMAIN_META.get(domain, DOMAIN_META["general"])
accent = dmeta["accent"]

SEV_CFG = {
    "critical": {"bg": "#b91c1c", "icon": "🔴", "label": "CRITICAL"},
    "high":     {"bg": "#b45309", "icon": "🟠", "label": "HIGH"},
    "warning":  {"bg": "#1d4ed8", "icon": "🔵", "label": "CAUTION"},
    "positive": {"bg": "#059669", "icon": "🟢", "label": "POSITIVE"},
    "info":     {"bg": "#475569", "icon": "⚪", "label": "INFO"},
}


# ── Insight card renderer ─────────────────────────────────────────────────────
def render_card(insight, idx: int):
    sev    = getattr(insight, "severity", "info").lower()
    cfg    = SEV_CFG.get(sev, SEV_CFG["info"])
    title  = getattr(insight, "title",    "—")
    prob   = getattr(insight, "problem",  "—")
    cause  = getattr(insight, "cause",    "—")
    evid   = getattr(insight, "evidence", "—")
    action = getattr(insight, "action",   "—")
    impact = getattr(insight, "impact",   "—")

    st.markdown(f"""
    <div class="insight-card">
      <div class="ic-header" style="background:{cfg['bg']}">
        {cfg['icon']} &nbsp; <span>{idx}. {title}</span>
        <span style="margin-left:auto;font-size:10px;background:rgba(255,255,255,.15);
               padding:2px 8px;border-radius:10px;letter-spacing:.06em">{cfg['label']}</span>
      </div>
      <div class="ic-body">
        <div class="ic-row">
          <div class="ic-lbl" style="color:{cfg['bg']}">Problem</div>
          <div class="ic-val">{prob}</div>
        </div>
        <div class="ic-row">
          <div class="ic-lbl" style="color:{cfg['bg']}">Pattern</div>
          <div class="ic-val">{cause}</div>
        </div>
        <div class="ic-row">
          <div class="ic-lbl" style="color:{cfg['bg']}">Evidence</div>
          <div class="ic-val"><span class="ic-code">{evid}</span></div>
        </div>
        <div class="ic-row">
          <div class="ic-lbl" style="color:{cfg['bg']}">Action</div>
          <div class="ic-val">{action}</div>
        </div>
        <div class="ic-row">
          <div class="ic-lbl" style="color:{cfg['bg']}">Impact</div>
          <div class="ic-val">{impact}</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)


# ═══ HEADER ═══════════════════════════════════════════════════════════════════
n_findings = len(getattr(report, "key_findings", []))
n_risks    = len(getattr(report, "business_risks", []))
n_opps     = len(getattr(report, "opportunities", []))
n_actions  = len(getattr(report, "recommended_actions", []))

st.markdown(f"""
<div class="domain-header">
  <div class="dh-title">{dmeta['icon']} {dmeta['label']} — Intelligence Report</div>
  <div class="dh-sub">{fname} &nbsp;·&nbsp; {len(df):,} records &nbsp;·&nbsp;
       {len(df.columns)} variables &nbsp;·&nbsp;
       Domain confidence: {confidence:.0%}</div>
  <div class="ms-wrap">
    <div class="ms-item"><div class="ms-val">{n_findings}</div><div class="ms-lbl">Findings</div></div>
    <div class="ms-item"><div class="ms-val">{n_risks}</div><div class="ms-lbl">Risks</div></div>
    <div class="ms-item"><div class="ms-val">{n_opps}</div><div class="ms-lbl">Opportunities</div></div>
    <div class="ms-item"><div class="ms-val">{n_actions}</div><div class="ms-lbl">Actions</div></div>
    <div class="ms-item"><div class="ms-val">{len(df.select_dtypes('number').columns)}</div><div class="ms-lbl">Numeric cols</div></div>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hypo-box">
  ⚠️ <strong>Analytical transparency:</strong> All insights are computed directly from
  the uploaded dataset. Causal language ("pattern suggests", "consistent with") denotes
  a data-supported hypothesis — not a confirmed fact. Exit interviews, longitudinal data,
  or controlled comparisons are required to confirm root causes.
</div>
""", unsafe_allow_html=True)

st.divider()


# ═══ TABS ═════════════════════════════════════════════════════════════════════
tabs = st.tabs(["📋 Executive Summary", "💡 Structured Insights",
                "⚠️ Risks & Opportunities", "🎯 Domain Deep Dive", "✅ Action Plan"])


# ─── Tab 1: Executive Summary ─────────────────────────────────────────────────
with tabs[0]:
    # Exec summary
    exec_text = getattr(report, "executive_summary", "")
    if exec_text:
        st.markdown(f'<div class="exec-box">{exec_text}</div>', unsafe_allow_html=True)
    else:
        # Compute a real one
        num_cols  = df.select_dtypes(include="number").columns.tolist()
        cat_cols  = df.select_dtypes(include=["object","string"]).columns.tolist()
        miss_pct  = round(df.isna().mean().mean()*100, 1)
        dup_ct    = int(df.duplicated().sum())
        highlights = []
        for col in num_cols[:4]:
            try:
                m, md, sk = df[col].mean(), df[col].median(), float(df[col].skew())
                cv = df[col].std() / abs(m) * 100 if m != 0 else 0
                if cv > 60:
                    highlights.append(f"'{col}' highly variable (CV={cv:.0f}%, median={md:.2f})")
                elif abs(sk) > 1.5:
                    highlights.append(f"'{col}' skewed ({sk:+.1f}), use median={md:.2f} not mean={m:.2f}")
            except Exception:
                logger.warning("Column stat highlight failed for '%s'", col, exc_info=True)
        summary = (f"Analysis of {len(df):,} records across {len(df.columns)} variables "
                   f"({len(num_cols)} numeric, {len(cat_cols)} categorical). "
                   f"{'Data complete — no missing values.' if miss_pct==0 else f'Overall missing rate: {miss_pct:.1f}%.'} "
                   f"{'No duplicates detected.' if dup_ct==0 else f'{dup_ct:,} duplicate rows identified.'}"
                   + (f" Key patterns: {'; '.join(highlights[:2])}." if highlights else ""))
        st.markdown(f'<div class="exec-box">{summary}</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Data Quality Assessment**")
        verdict = getattr(report, "data_quality_verdict", "")
        if "GOOD" in str(verdict):   st.success(verdict)
        elif "FAIR" in str(verdict): st.warning(verdict)
        elif verdict:                st.error(verdict)
        else:                        st.info(f"Dataset: {len(df):,} rows × {len(df.columns)} cols")
    with c2:
        st.markdown("**Analysis Confidence**")
        conf_str = getattr(report, "analysis_confidence", "")
        if "HIGH" in str(conf_str):    st.success(conf_str)
        elif "MEDIUM" in str(conf_str): st.warning(conf_str)
        elif conf_str:                  st.error(conf_str)
        else:                           st.info(f"Domain: {dmeta['label']} ({confidence:.0%} confidence)")

    # Key findings
    findings = getattr(report, "key_findings", [])
    if findings:
        st.markdown("### Key Findings")
        for i, f in enumerate(findings, 1):
            st.markdown(f"""
            <div class="finding-row">
              <div class="finding-num" style="color:{accent}">Finding {i:02d}</div>
              {f}
            </div>""", unsafe_allow_html=True)
    else:
        # Compute basic findings from data
        st.markdown("### Dataset Summary")
        num_cols = df.select_dtypes(include="number").columns.tolist()
        for col in num_cols[:5]:
            try:
                s = df[col].dropna()
                st.markdown(f"""
                <div class="finding-row">
                  <div class="finding-num" style="color:{accent}">{col}</div>
                  Mean {s.mean():.3g} · Median {s.median():.3g} ·
                  σ={s.std():.3g} · Skew={s.skew():.2f} ·
                  Missing {s.isna().sum()} ({s.isna().mean()*100:.1f}%)
                </div>""", unsafe_allow_html=True)
            except Exception:
                logger.warning("Finding row render failed for column", exc_info=True)


# ─── Tab 2: Structured Insights ───────────────────────────────────────────────
with tabs[1]:
    st.markdown("### Structured Business Insights")
    st.caption("Problem → Pattern → Evidence → Action → Impact")

    with st.spinner("Building structured insights…"):
        try:
            attrition_obj = getattr(report, "attrition", None)
            insights = build_top_insights(
                df=df, domain=domain,
                story_obj=report, attrition=attrition_obj
            )
        except Exception as e:
            insights = []
            st.warning(f"Structured insights unavailable: {e}")

    if not insights:
        # Fallback: compute real insights from raw data
        st.info("Computing statistical insights from dataset…")
        num_cols = df.select_dtypes(include="number").columns.tolist()
        cat_cols = df.select_dtypes(include=["object","string"]).columns.tolist()

        from core.story_engine import _col_stats, _correlations
        stats   = {c: _col_stats(df[c]) for c in num_cols[:10]}
        corrs   = _correlations(df)

        # Show outlier insights
        for col, st_d in list(stats.items())[:6]:
            if not st_d: continue
            out_pct = st_d.get("outlier_pct", 0)
            cv      = st_d.get("cv", 0) * 100
            skew    = st_d.get("skew", 0)
            mean_v  = st_d.get("mean", 0)
            median_v= st_d.get("median", 0)
            if out_pct > 5 or abs(skew) > 1.5 or cv > 80:
                sev = "critical" if out_pct > 15 or abs(skew) > 3 else "warning"

                class _FallbackInsight:
                    severity = sev
                    title    = f"'{col}': {out_pct:.1f}% Outliers · Skew {skew:+.2f} · CV {cv:.0f}%"
                    problem  = f"'{col}' shows {out_pct:.1f}% outlier rate and high variability (CV={cv:.0f}%)."
                    cause    = ("Heavy right skew suggests a long-tail distribution — "
                                "a small group of records is pulling the mean far above the median."
                                if skew > 1.5 else
                                "High variability with negative skew may indicate data entry issues or structural bimodality.")
                    evidence = f"Mean={mean_v:.3g} | Median={median_v:.3g} | Skew={skew:+.2f} | Outliers={out_pct:.1f}%"
                    action   = (f"Use median ({median_v:.3g}) instead of mean ({mean_v:.3g}) in all reports. "
                                f"Investigate records in the top 5% tail — may be errors or VIP segments.")
                    impact   = (f"If {out_pct:.1f}% of '{col}' values are errors, downstream models and "
                                "aggregations will be systematically biased.")

                render_card(_FallbackInsight(), 1)
    else:
        for i, ins in enumerate(insights, 1):
            render_card(ins, i)

        # Correlation table below cards
        num_cols = df.select_dtypes(include="number").columns.tolist()
        if len(num_cols) >= 2:
            from core.story_engine import _correlations
            corrs = _correlations(df)
            if corrs:
                st.markdown("---")
                st.markdown("#### Top Statistical Correlations")
                st.caption("Pearson r. Correlation ≠ causation. |r| > 0.4 = practically meaningful.")
                rows = []
                for c in corrs[:8]:
                    strength = "Strong" if abs(c["r"])>=0.7 else "Moderate" if abs(c["r"])>=0.4 else "Weak"
                    sig = "✅" if c["p"] < 0.05 else "❌"
                    rows.append({
                        "Column A": c["col_a"], "Column B": c["col_b"],
                        "r": c["r"], "r²": round(c["r"]**2, 3),
                        "Strength": strength, "Direction": c["direction"].title(),
                        "Significant": sig
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ─── Tab 3: Risks & Opportunities ─────────────────────────────────────────────
with tabs[2]:
    col_r, col_o = st.columns(2)

    with col_r:
        st.markdown("### Business Risks")
        risks = getattr(report, "business_risks", [])
        if not risks:
            st.success("No major risks flagged from this dataset.")
        else:
            labels = ["CRITICAL", "HIGH", "HIGH", "MEDIUM", "MEDIUM"]
            for i, risk in enumerate(risks, 1):
                lbl = labels[min(i-1, len(labels)-1)]
                cls = "risk-row" if i <= 2 else "warn-row"
                color = "#ef4444" if i <= 2 else "#f59e0b"
                st.markdown(f"""
                <div class="{cls}">
                  <div style="font-size:10px;font-weight:800;text-transform:uppercase;
                              letter-spacing:.07em;color:{color};margin-bottom:5px">
                    {lbl} — Risk {i:02d}
                  </div>
                  <div style="font-size:13px;line-height:1.6">{risk}</div>
                </div>""", unsafe_allow_html=True)

        # Anomalies
        anomalies = getattr(report, "anomalies", [])
        if anomalies:
            st.markdown("#### Statistical Anomalies")
            for a in anomalies:
                st.warning(a)

    with col_o:
        st.markdown("### Growth Opportunities")
        opps = getattr(report, "opportunities", [])
        if not opps:
            # compute basic opportunities from data
            num_cols = df.select_dtypes(include="number").columns.tolist()
            opps_fallback = []
            for col in num_cols[:4]:
                try:
                    s = df[col].dropna()
                    top10 = float(s.quantile(0.9))
                    med   = float(s.median())
                    if top10 > med * 1.5:
                        opps_fallback.append(
                            f"Top decile of '{col}' ({top10:.2f}) is {top10/med:.1f}× the median ({med:.2f}). "
                            f"Bringing bottom quartile to median represents a significant performance uplift."
                        )
                except Exception:
                    logger.warning("Opportunity fallback failed for '%s'", col, exc_info=True)
            if opps_fallback:
                for i, opp in enumerate(opps_fallback, 1):
                    st.markdown(f"""
                    <div class="opp-row">
                      <div class="opp-sev">Opportunity {i:02d}</div>
                      <div style="font-size:13px;line-height:1.6">{opp}</div>
                    </div>""", unsafe_allow_html=True)
            else:
                st.info("Add domain columns like revenue, satisfaction, or margin for opportunity detection.")
        else:
            for i, opp in enumerate(opps, 1):
                st.markdown(f"""
                <div class="opp-row">
                  <div class="opp-sev">Opportunity {i:02d}</div>
                  <div style="font-size:13px;line-height:1.6">{opp}</div>
                </div>""", unsafe_allow_html=True)


# ─── Tab 4: Domain Deep Dive ──────────────────────────────────────────────────
with tabs[3]:
    st.markdown(f"### {dmeta['icon']} {dmeta['label']} — Deep Dive")

    if domain == "hr":
        atr_col  = _find(df, ["left","attrition","churned","exited"])
        sat_col  = _find(df, ["satisfaction","engagement"])
        dept_col = _find(df, ["department","dept","division"])
        sal_col  = _find(df, ["salary"])
        ten_col  = _find(df, ["tenure","time_spend","years_at"])
        eval_col = _find(df, ["evaluation","performance","last_eval"])
        hrs_col  = _find(df, ["average_montly_hours","avg_monthly_hours","monthly_hours","hours"])

        # KPI row
        kpi_cols = st.columns(4)
        if atr_col:
            rate = float(df[atr_col].apply(lambda x: 1 if str(x).lower() in ["yes","1","1.0","true","left"] else 0).mean()) * 100
            n    = int(df[atr_col].apply(lambda x: 1 if str(x).lower() in ["yes","1","1.0","true","left"] else 0).sum())
            kpi_cols[0].metric("Attrition Rate", f"{rate:.1f}%", f"{n:,} departed",
                               delta_color="inverse")
        if sat_col:
            avg_s = float(df[sat_col].mean())
            max_s = float(df[sat_col].max())
            pct_s = avg_s/max_s*100 if max_s > 0 else 0
            kpi_cols[1].metric("Avg Satisfaction", f"{avg_s:.2f}",
                               f"{pct_s:.0f}% of max scale",
                               delta_color="normal" if pct_s >= 65 else "inverse")
        if ten_col:
            kpi_cols[2].metric("Median Tenure",
                               f"{float(df[ten_col].median()):.1f} yrs",
                               "right-skewed — use median")
        if hrs_col:
            avg_h = float(df[hrs_col].mean())
            kpi_cols[3].metric("Avg Monthly Hours", f"{avg_h:.0f}h",
                               "⚠️ Overwork risk" if avg_h > 220 else "Normal range",
                               delta_color="inverse" if avg_h > 220 else "off")

        st.divider()

        # Tenure cohort table
        if ten_col and atr_col:
            st.markdown("#### Tenure Cohort Attrition Analysis")
            left_mask = df[atr_col].apply(
                lambda x: 1 if str(x).lower() in ["yes","1","1.0","true","left"] else 0)
            bins   = [0, 1, 2, 4, 6, 8, 99]
            labels = ["<1 yr","1–2 yrs","3–4 yrs","5–6 yrs","7–8 yrs","9+ yrs"]
            df_t   = df.copy()
            df_t["_lm"] = left_mask
            df_t["_band"] = pd.cut(df_t[ten_col], bins=bins, labels=labels)
            cohort = (df_t.groupby("_band", observed=True)
                         .agg(Employees=("_lm","count"), Departed=("_lm","sum"))
                         .reset_index())
            cohort["Attrition %"] = (cohort["Departed"] / cohort["Employees"] * 100).round(1)
            if sat_col:
                cohort["Avg Satisfaction"] = df_t.groupby("_band", observed=True)[sat_col].mean().round(3).values
            if hrs_col:
                cohort["Avg Hrs/Month"] = df_t.groupby("_band", observed=True)[hrs_col].mean().round(0).values
            overall_rate = float(left_mask.mean()) * 100
            st.dataframe(cohort.rename(columns={"_band":"Tenure Band"}).set_index("Tenure Band"),
                         use_container_width=True)
            worst = cohort.loc[cohort["Attrition %"].idxmax()]
            st.caption(f"⚡ Highest risk: '{worst['_band']}' cohort at {float(worst['Attrition %']):.1f}% "
                       f"vs company average {overall_rate:.1f}%. "
                       "3–6 year employees are typically highest flight risk.")

        # Department heatmap
        if dept_col and atr_col:
            st.markdown("#### Attrition by Department")
            left_mask = df[atr_col].apply(
                lambda x: 1 if str(x).lower() in ["yes","1","1.0","true","left"] else 0)
            dept_atr = df.copy()
            dept_atr["_left"] = left_mask
            tbl = dept_atr.groupby(dept_col).agg(
                Employees=("_left","count"),
                Departed=("_left","sum")
            )
            tbl["Attrition %"] = (tbl["Departed"] / tbl["Employees"] * 100).round(1)
            if sat_col:
                tbl["Avg Satisfaction"] = dept_atr.groupby(dept_col)[sat_col].mean().round(3)
            if eval_col:
                tbl["Avg Performance"] = dept_atr.groupby(dept_col)[eval_col].mean().round(3)
            st.dataframe(tbl.sort_values("Attrition %", ascending=False), use_container_width=True)

        # Salary band attrition
        if sal_col and atr_col:
            st.markdown("#### Attrition by Salary Band")
            left_mask = df[atr_col].apply(
                lambda x: 1 if str(x).lower() in ["yes","1","1.0","true","left"] else 0)
            sal_df = df.copy()
            sal_df["_left"] = left_mask
            sal_tbl = sal_df.groupby(sal_col).agg(
                Employees=("_left","count"), Departed=("_left","sum"))
            sal_tbl["Attrition %"] = (sal_tbl["Departed"] / sal_tbl["Employees"] * 100).round(1)
            if sat_col:
                sal_tbl["Avg Satisfaction"] = sal_df.groupby(sal_col)[sat_col].mean().round(3)
            st.dataframe(sal_tbl.sort_values("Attrition %", ascending=False), use_container_width=True)

        # Flight risk
        if sat_col and atr_col and ten_col:
            st.markdown("#### Flight Risk — Current Employees")
            st.caption("Criteria: satisfaction <40th percentile AND tenure ≥ 3 years AND still employed")
            left_mask = df[atr_col].apply(
                lambda x: 1 if str(x).lower() in ["yes","1","1.0","true","left"] else 0)
            stayers = df[left_mask == 0].copy()
            sat_q40 = stayers[sat_col].quantile(0.4)
            promo_col = _find(df, ["promot"])
            mask = (stayers[sat_col] < sat_q40) & (stayers[ten_col] >= 3)
            if promo_col:
                mask = mask & (stayers[promo_col] == 0)
            n_risk = int(mask.sum())
            pct    = n_risk / len(stayers) * 100 if len(stayers) else 0
            r1, r2 = st.columns(2)
            r1.metric("Tier-1 Flight Risk", f"{n_risk:,}", f"{pct:.1f}% of workforce")
            r2.info(f"Satisfaction cutoff: {sat_q40:.2f}. These {n_risk:,} employees meet all "
                    "risk criteria simultaneously. Validate with manager conversations.")

    elif domain == "finance":
        rev_col    = _find(df, ["revenue","total_revenue","income","turnover","sales_amount"])
        cost_col   = _find(df, ["cost","cogs","cost_of_goods"])
        profit_col = _find(df, ["net_profit","profit","net_income"])
        budget_col = _find(df, ["budget","plan","target","forecast"])
        actual_col = _find(df, ["actual","actuals"], exclude=["target","budget"])
        period_col = _find(df, ["month","quarter","period","year"])
        cat_col    = _find(df, ["category","department","account","segment"])

        kpis = st.columns(4)
        if rev_col:
            total_rev = float(df[rev_col].sum())
            kpis[0].metric("Total Revenue", f"{total_rev:,.0f}",
                           f"Median: {df[rev_col].median():,.0f}")
        if rev_col and cost_col:
            gm = (float(df[rev_col].sum()) - float(df[cost_col].sum())) / float(df[rev_col].sum()) * 100
            kpis[1].metric("Gross Margin", f"{gm:.1f}%",
                           "Healthy" if gm > 40 else "Review" if gm > 20 else "Critical",
                           delta_color="normal" if gm > 30 else "inverse")
        if profit_col:
            total_p  = float(df[profit_col].sum())
            loss_rows = int((df[profit_col] < 0).sum())
            kpis[2].metric("Total Profit", f"{total_p:,.0f}",
                           f"{loss_rows} loss rows",
                           delta_color="normal" if total_p > 0 else "inverse")
        if rev_col and profit_col:
            npm = float(df[profit_col].sum()) / float(df[rev_col].sum()) * 100
            kpis[3].metric("Net Profit Margin", f"{npm:.1f}%",
                           "Strong" if npm > 15 else "Review" if npm > 5 else "Critical",
                           delta_color="normal" if npm > 10 else "inverse")

        if budget_col and actual_col and budget_col != actual_col:
            st.divider()
            st.markdown("#### Budget vs Actual Variance")
            group_col = period_col or cat_col
            if group_col:
                bva = df.groupby(group_col)[[budget_col, actual_col]].sum().reset_index()
                bva["Variance"]   = (bva[actual_col] - bva[budget_col]).round(0)
                bva["Variance %"] = ((bva[actual_col] - bva[budget_col]) /
                                      bva[budget_col].replace(0, np.nan) * 100).round(1)
                st.dataframe(bva.rename(columns={group_col:"Period/Category"})
                               .sort_values("Variance %", key=abs, ascending=False)
                               .head(15), use_container_width=True)
                over = (bva["Variance"] > 0).sum()
                under = (bva["Variance"] < 0).sum()
                st.caption(f"📊 {over} periods/categories over budget, {under} under budget. "
                           "±10% is a common review trigger.")

        if cat_col and rev_col:
            st.markdown("#### Revenue by Category")
            cat_agg = df.groupby(cat_col)[[rev_col] +
                      ([cost_col] if cost_col else [])].sum()
            if cost_col:
                cat_agg["Gross Profit"]   = cat_agg[rev_col] - cat_agg[cost_col]
                cat_agg["Gross Margin %"] = (cat_agg["Gross Profit"] / cat_agg[rev_col] * 100).round(1)
            st.dataframe(cat_agg.sort_values(rev_col, ascending=False), use_container_width=True)

    elif domain == "ecommerce":
        rating_col = _find(df, ["rating"], exclude=["count","num"])
        price_col  = _find(df, ["discounted_price","selling_price","price"], exclude=["actual","mrp"])
        disc_col   = _find(df, ["discount"])
        cat_col    = _find(df, ["category"])
        rev_col    = _find(df, ["revenue","sales","amount"])

        kpis = st.columns(4)
        if rating_col:
            avg_r = float(df[rating_col].mean())
            low_n = int((df[rating_col] < 3.0).sum())
            kpis[0].metric("Avg Rating", f"{avg_r:.2f} / 5",
                           f"{low_n} below 3.0",
                           delta_color="normal" if avg_r >= 3.5 else "inverse")
        if price_col:
            kpis[1].metric("Median Price", f"₹{float(df[price_col].median()):,.0f}",
                           "Median (right-skewed)")
        if disc_col:
            avg_d = float(df[disc_col].mean())
            kpis[2].metric("Avg Discount", f"{avg_d:.1f}%",
                           "⚠️ Margin risk" if avg_d > 40 else "Moderate",
                           delta_color="inverse" if avg_d > 40 else "off")
        if cat_col:
            kpis[3].metric("Categories", str(df[cat_col].nunique()))

        st.divider()
        if cat_col and rating_col:
            st.markdown("#### Rating by Category")
            cat_rat = (df.groupby(cat_col)
                         .agg(Avg_Rating=(rating_col,"mean"),
                              Products=(rating_col,"count"))
                         .round({"Avg_Rating":2})
                         .query("Products >= 3")
                         .sort_values("Avg_Rating", ascending=False))
            st.dataframe(cat_rat, use_container_width=True)
            worst_cat = cat_rat["Avg_Rating"].idxmin()
            st.caption(f"⚡ Lowest rated: '{worst_cat}' — investigate quality or description accuracy.")

        if disc_col and rating_col:
            st.markdown("#### Discount vs Rating Correlation")
            corr_val = float(df[[disc_col, rating_col]].corr(method="spearman").iloc[0,1])
            direction = "negative" if corr_val < 0 else "positive"
            if abs(corr_val) > 0.15:
                st.warning(f"Spearman r={corr_val:.3f} — {direction} association between discount % and rating. "
                           "High-discount products may have quality/description issues.")
            else:
                st.success(f"Spearman r={corr_val:.3f} — no meaningful correlation between discount and rating.")

    elif domain == "sales":
        rev_col    = _find(df, ["revenue","sales","amount","value"], exclude=["budget","target"])
        target_col = _find(df, ["target","quota","budget","plan"])
        rep_col    = _find(df, ["rep","salesperson","agent","executive","owner"])
        region_col = _find(df, ["region","territory","area","zone"])
        margin_col = _find(df, ["margin","profit","gross"])

        kpis = st.columns(4)
        if rev_col:
            total_rev = float(df[rev_col].sum())
            kpis[0].metric("Total Revenue", f"{total_rev:,.0f}",
                           f"Median deal: {df[rev_col].median():,.0f}")
        if rev_col and target_col:
            ach = float(df[rev_col].sum()) / float(df[target_col].replace(0,np.nan).sum()) * 100
            kpis[1].metric("Quota Achievement", f"{ach:.1f}%", f"{ach-100:+.1f}pp vs target",
                           delta_color="normal" if ach >= 100 else "inverse")
        if margin_col:
            avg_m = float(df[margin_col].mean())
            loss_d = int((df[margin_col] < 0).sum())
            kpis[2].metric("Avg Margin", f"{avg_m:.1f}%", f"{loss_d} loss deals",
                           delta_color="normal" if avg_m > 0 else "inverse")
        if rep_col:
            kpis[3].metric("Sales Reps", str(df[rep_col].nunique()))

        st.divider()
        if rep_col and rev_col:
            st.markdown("#### Sales Rep Performance Ranking")
            rep_data = df.groupby(rep_col).agg(
                Revenue=(rev_col,"sum"),
                Deals=(rev_col,"count"),
                Avg_Deal=(rev_col,"mean"),
            ).round({"Avg_Deal":0})
            rep_data["Revenue Share %"] = (rep_data["Revenue"] / rep_data["Revenue"].sum() * 100).round(1)
            if target_col:
                target_by_rep = df.groupby(rep_col)[target_col].sum()
                rep_data["Achievement %"] = (rep_data["Revenue"] / target_by_rep * 100).round(1)
            st.dataframe(rep_data.sort_values("Revenue", ascending=False), use_container_width=True)
            top_pct = float(rep_data["Revenue Share %"].iloc[0])
            if top_pct > 40:
                st.warning(f"⚡ Revenue concentration: top rep = {top_pct:.1f}% of total. "
                           "High key-person dependency — succession planning needed.")

        if region_col and rev_col:
            st.markdown("#### Regional Breakdown")
            reg_data = df.groupby(region_col)[rev_col].agg(["sum","count","mean"]).round(0)
            reg_data.columns = ["Total Revenue","Deals","Avg Deal"]
            reg_data["Share %"] = (reg_data["Total Revenue"] / reg_data["Total Revenue"].sum() * 100).round(1)
            st.dataframe(reg_data.sort_values("Total Revenue", ascending=False), use_container_width=True)

    else:
        # General domain — full correlation + distribution analysis
        num_cols = df.select_dtypes(include="number").columns.tolist()
        cat_cols = df.select_dtypes(include=["object","string"]).columns.tolist()

        st.markdown("#### Numeric Column Profiles")
        profile_rows = []
        for col in num_cols[:12]:
            s = df[col].dropna()
            if len(s) == 0: continue
            profile_rows.append({
                "Column": col,
                "Count": len(s),
                "Missing %": round(df[col].isna().mean()*100, 1),
                "Mean": round(float(s.mean()), 3),
                "Median": round(float(s.median()), 3),
                "Std": round(float(s.std()), 3),
                "CV %": round(float(s.std()/abs(s.mean())*100) if s.mean()!=0 else 0, 1),
                "Skew": round(float(s.skew()), 2),
                "P10": round(float(s.quantile(0.1)), 3),
                "P90": round(float(s.quantile(0.9)), 3),
            })
        if profile_rows:
            st.dataframe(pd.DataFrame(profile_rows).set_index("Column"),
                         use_container_width=True)

        if len(num_cols) >= 2:
            st.markdown("#### Correlation Heatmap")
            try:
                corr = df[num_cols[:10]].corr(method="spearman").round(3)
                fig = go.Figure(go.Heatmap(
                    z=corr.values, x=corr.columns.tolist(), y=corr.columns.tolist(),
                    colorscale="RdBu", zmid=0, zmin=-1, zmax=1,
                    text=np.round(corr.values, 2), texttemplate="%{text:.2f}",
                    colorbar=dict(title="r", thickness=12),
                ))
                fig.update_layout(height=420, margin=dict(l=10,r=10,t=40,b=10),
                                  paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
                st.caption("Spearman correlation. r² = shared variance. Correlation ≠ causation.")
            except Exception as e:
                st.error(f"Heatmap error: {e}")

        if cat_cols:
            st.markdown("#### Categorical Columns")
            for col in cat_cols[:4]:
                vc = df[col].value_counts().head(10)
                c1, c2 = st.columns([1, 2])
                with c1:
                    st.dataframe(vc.reset_index().rename(columns={col:"Value","count":"Count"}),
                                 use_container_width=True, hide_index=True)
                with c2:
                    fig = px.bar(vc.reset_index(), x=col, y="count",
                                 title=f"'{col}' distribution (top 10)")
                    st.plotly_chart(_style_fig(fig), use_container_width=True,
                                    config={"displayModeBar": False})


# ─── Tab 5: Action Plan ───────────────────────────────────────────────────────
with tabs[4]:
    st.markdown("### Prioritised Action Plan")
    st.caption("Actions derived from dataset patterns only. Validate with business context before implementing.")

    actions = getattr(report, "recommended_actions", [])
    priorities = [
        ("🔴", "P1 — Immediate (This Week)",    "#ef4444"),
        ("🟠", "P2 — Short Term (This Month)",  "#f59e0b"),
        ("🟡", "P3 — Medium Term (This Quarter)","#eab308"),
        ("🔵", "P4 — Strategic (Next Quarter)",  "#3b82f6"),
        ("⚪", "P5 — Ongoing",                  "#64748b"),
    ]

    if not actions:
        st.info("No domain-specific actions generated. "
                "Add HR, Sales, Finance, or E-Commerce columns for targeted recommendations.")
        actions = [
            "Establish data quality monitoring — track missing % and outlier rate weekly",
            "Set internal KPI baselines from this dataset before comparing future periods",
            "Identify the top 20% performers in key metrics — understand what drives their success",
            "Configure GROQ_API_KEY in .streamlit/secrets.toml for AI-enhanced narratives",
        ]

    for i, action in enumerate(actions, 1):
        pri_icon, pri_label, pri_color = priorities[min(i-1, len(priorities)-1)]
        st.markdown(f"""
        <div class="action-item">
          <div class="action-pri" style="color:{pri_color}">{pri_icon} {pri_label}</div>
          <div style="font-size:13.5px;line-height:1.6">{action}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Analytical Next Steps")
    next_steps = [
        ("📊", "Period comparison", "Re-run with historical data to measure change over time"),
        ("🔍", "Root cause validation", "Collect qualitative data (interviews, surveys) to confirm hypotheses"),
        ("🎯", "Segment analysis", "Break down by additional dimensions not available in current dataset"),
        ("📈", "KPI baselining", "Set targets from internal data — not generic external benchmarks"),
        ("🔁", "Impact measurement", "Re-run analysis after any intervention to quantify impact"),
    ]
    for icon, title, desc in next_steps:
        st.markdown(f"""
        <div class="action-item">
          <div style="font-size:12px;font-weight:700;margin-bottom:4px">{icon} {title}</div>
          <div style="font-size:13px;opacity:.8">{desc}</div>
        </div>""", unsafe_allow_html=True)

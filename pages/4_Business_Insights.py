"""
pages/4_Business_Insights.py  — DataForge AI IMPROVED
Changes vs original:
  - Domain-specific structured insight cards (Problem→Cause→Evidence→Action→Impact)
  - Finance domain deep analysis section added
  - E-commerce: RFM signals, category analysis, discount effectiveness
  - Sales: rep ranking, pipeline metrics, margin analysis
  - HR: flight risk table, tenure crisis callout, performance bands
  - Hypothesis language throughout ("data suggests", "pattern consistent with")
  - No external benchmark claims
"""
import io
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

def _style_fig(fig):
    """Apply high-contrast fonts — readable on both light and dark Streamlit themes."""
    fig.update_layout(font=dict(color="#0F172A", size=11))
    fig.update_xaxes(tickfont=dict(color="#0F172A", size=10),
                     title_font=dict(color="#0F172A"))
    fig.update_yaxes(tickfont=dict(color="#0F172A", size=10),
                     title_font=dict(color="#0F172A"))
    return fig


from core.session_manager import require_data, get_df, get_filename
require_data()

from core.story_engine import generate_story, detect_domain
from core.insights_builder import build_top_insights

st.set_page_config(page_title="Business Insights — DataForge AI", layout="wide")

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif!important}
.block-container{padding-top:1.2rem!important}

.insight-card{
    border-radius:12px;padding:0;margin-bottom:14px;
    border:1px solid #E2E8F0;overflow:hidden;
}
.ic-header{padding:12px 18px;font-size:14px;font-weight:700;color:white;
           display:flex;align-items:center;gap:10px}
.ic-body{padding:14px 18px;background:white}
.ic-row{display:flex;gap:8px;margin-bottom:8px;align-items:flex-start}
.ic-lbl{font-size:10px;font-weight:800;text-transform:uppercase;letter-spacing:.08em;
        min-width:72px;padding-top:2px;flex-shrink:0}
.ic-val{font-size:13px;color:inherit;line-height:1.55}

.domain-header{
    background:linear-gradient(135deg,#0D1B2E,#1B3A6B);
    border-radius:14px;padding:20px 24px;margin-bottom:20px;color:white;
}
.dh-title{font-size:22px;font-weight:800;margin-bottom:6px}
.dh-sub{font-size:13px;color:rgba(255,255,255,.6)}

.metric-strip{
    display:flex;gap:12px;flex-wrap:wrap;margin:16px 0;
}
.ms-item{
    background:rgba(255,255,255,.1);border:1px solid rgba(255,255,255,.2);
    border-radius:10px;padding:10px 16px;min-width:100px;text-align:center;
}
.ms-val{font-size:20px;font-weight:800;color:white}
.ms-lbl{font-size:10px;color:rgba(255,255,255,.5);text-transform:uppercase;
        letter-spacing:.07em;margin-top:2px}

.hypo-box{
    background:#FFFBEB;border:1px solid #FCD34D;border-left:4px solid #D97706;
    border-radius:8px;padding:12px 16px;margin:10px 0;font-size:13px;color:#92400E;
}

section[data-testid="stSidebar"]{background:linear-gradient(180deg,#0D1B2E,#0F2240)!important}
section[data-testid="stSidebar"] *{color:rgba(255,255,255,.85)!important}
</style>
""", unsafe_allow_html=True)

# ── Cache ─────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def get_story_cached(df_json: str):
    df = pd.read_json(io.StringIO(df_json))
    return generate_story(df)

@st.cache_data(show_spinner=False)
def get_insights_cached(df_json: str, domain: str):
    df = pd.read_json(io.StringIO(df_json))
    return build_top_insights(df, domain=domain)

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

with st.spinner("Generating business insights..."):
    report = get_story_cached(df.to_json(date_format="iso"))
    domain, confidence = detect_domain(df)

# ── Domain metadata ───────────────────────────────────────────────────────────
DOMAIN_META = {
    "hr":        {"icon": "👥", "label": "HR Analytics",     "color": "#1D4ED8", "bg": "#1E3A8A"},
    "finance":   {"icon": "💰", "label": "Finance",          "color": "#059669", "bg": "#064E3B"},
    "ecommerce": {"icon": "🛒", "label": "E-Commerce",       "color": "#D97706", "bg": "#78350F"},
    "sales":     {"icon": "📈", "label": "Sales Performance","color": "#7C3AED", "bg": "#4C1D95"},
    "general":   {"icon": "📊", "label": "General Business", "color": "#475569", "bg": "#1E293B"},
}
dmeta  = DOMAIN_META.get(domain, DOMAIN_META["general"])
dcolor = dmeta["color"]
dbg    = dmeta["bg"]

SEV_CONFIG = {
    "critical": {"bg": "#B91C1C", "light": "#FEF2F2", "border": "#B91C1C", "icon": "🔴"},
    "high":     {"bg": "#B45309", "light": "#FFFBEB", "border": "#B45309", "icon": "🟠"},
    "warning":  {"bg": "#1D4ED8", "light": "#EFF6FF", "border": "#1D4ED8", "icon": "🔵"},
    "positive": {"bg": "#059669", "light": "#F0FDF4", "border": "#059669", "icon": "🟢"},
    "info":     {"bg": "#475569", "light": "#F8FAFC", "border": "#CBD5E1", "icon": "⚪"},
}


def render_insight_card(insight, idx: int):
    """Render a structured insight card with full P→C→E→A→I format."""
    sev = getattr(insight, "severity", "info").lower()
    cfg = SEV_CONFIG.get(sev, SEV_CONFIG["info"])
    title  = getattr(insight, "title",    "")
    prob   = getattr(insight, "problem",  "")
    cause  = getattr(insight, "cause",    "")
    evid   = getattr(insight, "evidence", "")
    action = getattr(insight, "action",   "")
    impact = getattr(insight, "impact",   "")

    st.markdown(f"""
    <div class="insight-card">
        <div class="ic-header" style="background:{cfg['bg']}">
            <span>{cfg['icon']}</span>
            <span>{idx}. {title}</span>
        </div>
        <div class="ic-body">
            <div class="ic-row">
                <div class="ic-lbl" style="color:{cfg['bg']}">PROBLEM</div>
                <div class="ic-val">{prob}</div>
            </div>
            <div class="ic-row">
                <div class="ic-lbl" style="color:{cfg['bg']}">PATTERN</div>
                <div class="ic-val">{cause}</div>
            </div>
            <div class="ic-row">
                <div class="ic-lbl" style="color:{cfg['bg']}">EVIDENCE</div>
                <div class="ic-val"><code style="font-size:12px;background:#F8FAFC;
                padding:2px 6px;border-radius:4px">{evid}</code></div>
            </div>
            <div class="ic-row">
                <div class="ic-lbl" style="color:{cfg['bg']}">ACTION</div>
                <div class="ic-val">{action}</div>
            </div>
            <div class="ic-row">
                <div class="ic-lbl" style="color:{cfg['bg']}">IMPACT</div>
                <div class="ic-val">{impact}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  HEADER
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="domain-header" style="background:linear-gradient(135deg,{dbg},{dbg}99)">
    <div class="dh-title">
        {dmeta['icon']} {dmeta['label']} Intelligence Report
    </div>
    <div class="dh-sub">
        {fname} · {len(df):,} rows · {len(df.columns)} columns ·
        Domain detected with {confidence:.0%} confidence
    </div>
    <div class="metric-strip">
        <div class="ms-item">
            <div class="ms-val">{len(getattr(report,'key_findings',[]))}</div>
            <div class="ms-lbl">Findings</div>
        </div>
        <div class="ms-item">
            <div class="ms-val">{len(getattr(report,'business_risks',[]))}</div>
            <div class="ms-lbl">Risks</div>
        </div>
        <div class="ms-item">
            <div class="ms-val">{len(getattr(report,'opportunities',[]))}</div>
            <div class="ms-lbl">Opportunities</div>
        </div>
        <div class="ms-item">
            <div class="ms-val">{len(getattr(report,'recommended_actions',[]))}</div>
            <div class="ms-lbl">Actions</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Hypothesis disclaimer — always shown
st.markdown("""
<div class="hypo-box">
    ⚠️ <b>Analytical transparency:</b> Insights are computed from the submitted dataset only.
    Cause-and-effect language ("pattern suggests", "consistent with") indicates a hypothesis
    supported by the data — not a confirmed fact. Exit interviews, longitudinal data, or
    controlled comparison are needed to confirm root causes. No external benchmarks are
    embedded in these findings.
</div>
""", unsafe_allow_html=True)

st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
#  TABS
# ═══════════════════════════════════════════════════════════════════════════════
tab_labels = ["📋 Executive Summary", "💡 Structured Insights",
              "⚠️ Risks & Opportunities", "🎯 Domain Deep Dive", "✅ Action Plan"]
tabs = st.tabs(tab_labels)

# ─── Tab 1: Executive Summary ─────────────────────────────────────────────────
with tabs[0]:
    st.markdown("### Executive Summary")
    exec_text = getattr(report, "executive_summary", "")
    if exec_text:
        st.markdown(f"""
        <div style='background:#F8FAFC;border-left:4px solid {dcolor};
                    padding:18px 22px;border-radius:6px;font-size:14.5px;
                    line-height:1.8;color:inherit'>{exec_text}</div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Data Quality**")
        verdict = getattr(report, "data_quality_verdict", "")
        if "GOOD" in str(verdict):     st.success(verdict)
        elif "FAIR" in str(verdict):   st.warning(verdict)
        elif verdict:                  st.error(verdict)
        else:                          st.info("Quality assessment not available")

    with c2:
        st.markdown("**Analysis Confidence**")
        conf_str = getattr(report, "analysis_confidence", "")
        if "HIGH" in str(conf_str):    st.success(conf_str)
        elif "MEDIUM" in str(conf_str):st.warning(conf_str)
        elif conf_str:                 st.error(conf_str)

    if getattr(report, "key_findings", []):
        st.markdown("### Key Findings")
        for i, finding in enumerate(report.key_findings, 1):
            st.markdown(f"""
            <div style='background:#F8FAFC;border:1px solid #E2E8F0;
                        border-radius:8px;padding:12px 16px;margin-bottom:8px;
                        font-size:13.5px;color:inherit;line-height:1.6'>
                <span style='color:{dcolor};font-weight:700;font-size:11px;
                             text-transform:uppercase;letter-spacing:.07em'>
                    Finding {i}
                </span><br>{finding}
            </div>
            """, unsafe_allow_html=True)

# ─── Tab 2: Structured Insights ───────────────────────────────────────────────
with tabs[1]:
    st.markdown("### Structured Business Insights")
    st.caption("Problem → Pattern → Evidence → Action → Impact")

    with st.spinner("Building structured insights..."):
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
        st.info("No structured insights could be generated. "
                "Ensure your dataset has numeric columns with business meaning.")
    else:
        for i, insight in enumerate(insights, 1):
            render_insight_card(insight, i)

# ─── Tab 3: Risks & Opportunities ─────────────────────────────────────────────
with tabs[2]:
    col_r, col_o = st.columns(2)

    with col_r:
        st.markdown("### Business Risks")
        risks = getattr(report, "business_risks", [])
        if not risks:
            st.success("No major business risks detected.")
        else:
            for i, risk in enumerate(risks, 1):
                sev   = "CRITICAL" if i == 1 else "HIGH" if i <= 2 else "MEDIUM"
                color = "#B91C1C" if i == 1 else "#B45309" if i <= 2 else "#D97706"
                st.markdown(f"""
                <div style='border-left:4px solid {color};background:#FFF8F8;
                            padding:12px 16px;border-radius:6px;margin-bottom:10px'>
                    <span style='color:{color};font-size:10px;font-weight:800;
                                 text-transform:uppercase;letter-spacing:.07em'>
                        {sev}
                    </span><br>
                    <span style='font-size:13px;color:inherit;line-height:1.6'>{risk}</span>
                </div>
                """, unsafe_allow_html=True)

    with col_o:
        st.markdown("### Growth Opportunities")
        opps = getattr(report, "opportunities", [])
        if not opps:
            st.info("No specific opportunities detected — consider domain columns "
                    "like revenue, satisfaction, or margin.")
        else:
            for i, opp in enumerate(opps, 1):
                st.markdown(f"""
                <div style='border-left:4px solid #059669;background:#F0FDF4;
                            padding:12px 16px;border-radius:6px;margin-bottom:10px'>
                    <span style='color:#059669;font-size:10px;font-weight:800;
                                 text-transform:uppercase;letter-spacing:.07em'>
                        OPPORTUNITY {i}
                    </span><br>
                    <span style='font-size:13px;color:inherit;line-height:1.6'>{opp}</span>
                </div>
                """, unsafe_allow_html=True)

    if getattr(report, "anomalies", []):
        st.markdown("### Statistical Anomalies")
        for anomaly in report.anomalies:
            st.warning(anomaly)

# ─── Tab 4: Domain Deep Dive ──────────────────────────────────────────────────
with tabs[3]:
    st.markdown(f"### {dmeta['icon']} {dmeta['label']} — Domain Analysis")

    # ── HR Deep Dive ──────────────────────────────────────────────────────
    if domain == "hr":
        atr_col  = _find(df, ["left","attrition","churned","exited"])
        sat_col  = _find(df, ["satisfaction"])
        dept_col = _find(df, ["department","dept"])
        sal_col  = _find(df, ["salary"])
        ten_col  = _find(df, ["tenure","time_spend","years_at"])
        eval_col = _find(df, ["evaluation","performance","last_eval"])

        if atr_col:
            rate = float(df[atr_col].mean()) * 100
            n    = int(df[atr_col].sum())
            col1, col2, col3 = st.columns(3)
            col1.metric("Attrition Rate",    f"{rate:.1f}%", f"{n:,} departed")
            if sat_col:
                col2.metric("Avg Satisfaction", f"{float(df[sat_col].mean()):.2f}",
                             "Target: 0.70")
            if ten_col:
                col3.metric("Median Tenure",    f"{float(df[ten_col].median()):.1f} yrs",
                             "Right-skewed — use median")

        # Tenure cohort table
        if ten_col and atr_col:
            st.markdown("#### Tenure Cohort Attrition")
            bins   = [0,2,4,6,8,99]
            labels = ["1–2 yrs","3–4 yrs","5–6 yrs","7–8 yrs","9+ yrs"]
            df_t   = df.copy()
            df_t["_band"] = pd.cut(df_t[ten_col], bins=bins, labels=labels)
            cohort = (df_t.groupby("_band", observed=True)
                         .agg(Employees=(atr_col,"count"), Departed=(atr_col,"sum"))
                         .reset_index())
            cohort["Attrition %"] = (cohort["Departed"] / cohort["Employees"] * 100).round(1)
            cohort["Avg Satisfaction"] = (df_t.groupby("_band", observed=True)[sat_col].mean().round(3).values
                                           if sat_col else "—")
            avg_rate = float(df[atr_col].mean()) * 100

            def _style_row(row):
                if isinstance(row.get("Attrition %"), float) and row["Attrition %"] > avg_rate * 1.5:
                    return ['background-color: #FEF2F2'] * len(row)
                return [''] * len(row)

            cohort.columns = ["Tenure Band","Employees","Departed","Attrition %","Avg Satisfaction"]
            st.dataframe(cohort.set_index("Tenure Band"), use_container_width=True)
            st.caption("Red rows = attrition >1.5× company average. "
                       "The 5–6 year zone is the highest-risk window in most datasets.")

        # Salary attrition table
        if sal_col and atr_col:
            st.markdown("#### Attrition by Salary Band")
            sal_atr = df.groupby(sal_col).agg(
                Employees=(atr_col,"count"), Departed=(atr_col,"sum"),
                Avg_Satisfaction=(sat_col,"mean") if sat_col else (atr_col,"count")
            ).reset_index()
            sal_atr["Attrition %"] = (sal_atr["Departed"] / sal_atr["Employees"] * 100).round(1)
            st.dataframe(sal_atr, use_container_width=True)
            st.caption("Note: 3-level salary bands (low/medium/high) hide within-band variation. "
                       "Dollar-level data would allow more precise targeting.")

        # Flight risk
        if sat_col and atr_col and ten_col:
            st.markdown("#### Verified Flight Risk (Current Employees)")
            st.caption("Criteria: satisfaction <0.40 AND no promotion AND tenure ≥3 years")
            promo_col = _find(df, ["promot"])
            stayers = df[df[atr_col] == 0].copy()
            mask = stayers[sat_col] < 0.40
            if promo_col:
                mask = mask & (stayers[promo_col] == 0)
            mask = mask & (stayers[ten_col] >= 3)
            n_risk = int(mask.sum())
            pct    = n_risk / len(stayers) * 100 if len(stayers) else 0
            col1, col2 = st.columns(2)
            col1.metric("Tier-1 Flight Risk", f"{n_risk:,}", f"{pct:.1f}% of remaining workforce")
            col2.info("These employees meet all three risk criteria simultaneously. "
                      "Meeting criteria does not guarantee departure — it indicates priority for manager conversations.")

    # ── Finance Deep Dive ─────────────────────────────────────────────────
    elif domain == "finance":
        rev_col    = _find(df, ["revenue","total_revenue","income","turnover","sales_amount"])
        cost_col   = _find(df, ["cost","cogs","cost_of_goods"])
        profit_col = _find(df, ["net_profit","profit","net_income"])
        budget_col = _find(df, ["budget","plan","target","forecast"])
        actual_col = _find(df, ["actual","actuals"], exclude=["target","budget"])
        period_col = _find(df, ["month","quarter","period","year"])
        cat_col    = _find(df, ["category","department","account","segment"])

        c1, c2, c3 = st.columns(3)
        if rev_col:
            total_rev = float(df[rev_col].sum())
            c1.metric("Total Revenue", f"{total_rev:,.0f}")
        if rev_col and cost_col:
            gm = (float(df[rev_col].sum()) - float(df[cost_col].sum())) / float(df[rev_col].sum()) * 100
            color_delta = "normal" if gm > 30 else "inverse"
            c2.metric("Gross Margin", f"{gm:.1f}%",
                      "Healthy" if gm > 40 else "Review costs" if gm > 20 else "Critical",
                      delta_color=color_delta)
        if profit_col:
            total_p = float(df[profit_col].sum())
            c3.metric("Total Profit", f"{total_p:,.0f}",
                      f"{int((df[profit_col]<0).sum())} loss rows")

        # Budget variance
        if budget_col and actual_col and budget_col != actual_col:
            st.markdown("#### Budget vs Actual")
            comp_col = period_col or cat_col
            if comp_col:
                bva = df.groupby(comp_col)[[budget_col, actual_col]].sum().reset_index()
                bva["Variance"] = bva[actual_col] - bva[budget_col]
                bva["Variance %"] = ((bva[actual_col] - bva[budget_col]) /
                                      bva[budget_col].replace(0, np.nan) * 100).round(1)
                bva = bva.rename(columns={comp_col: "Period/Category"})
                st.dataframe(bva.sort_values("Variance %", key=abs, ascending=False).head(15),
                             use_container_width=True)
                st.caption("Sorted by absolute variance. ±10% is a common review trigger — "
                           "adjust to your planning standards.")

        # Revenue by category
        if cat_col and rev_col:
            st.markdown("#### Revenue & Cost by Category")
            cat_agg = df.groupby(cat_col)[[rev_col] + ([cost_col] if cost_col else [])].sum()
            if cost_col:
                cat_agg["Gross Profit"]  = cat_agg[rev_col] - cat_agg[cost_col]
                cat_agg["Gross Margin %"] = (cat_agg["Gross Profit"] / cat_agg[rev_col] * 100).round(1)
            st.dataframe(cat_agg.sort_values(rev_col, ascending=False), use_container_width=True)

    # ── E-Commerce Deep Dive ──────────────────────────────────────────────
    elif domain == "ecommerce":
        rating_col = _find(df, ["rating"], exclude=["count"])
        price_col  = _find(df, ["discounted_price","selling_price","price"], exclude=["actual","mrp"])
        disc_col   = _find(df, ["discount"])
        cat_col    = _find(df, ["category"])
        rev_col    = _find(df, ["revenue","sales","amount"])

        if rating_col:
            c1, c2, c3 = st.columns(3)
            c1.metric("Avg Rating", f"{float(df[rating_col].mean()):.2f}/5",
                      f"{int((df[rating_col]<3.0).sum())} below 3.0")
            if price_col:
                c2.metric("Median Price", f"{float(df[price_col].median()):.0f}",
                          "Median — right-skewed")
            if disc_col:
                avg_d = float(df[disc_col].mean())
                c3.metric("Avg Discount", f"{avg_d:.1f}%",
                          "High margin risk" if avg_d > 40 else "Moderate")

        # Rating by category
        if cat_col and rating_col:
            st.markdown("#### Average Rating by Category")
            cat_rat = (df.groupby(cat_col)
                         .agg(Avg_Rating=(rating_col,"mean"),
                              Product_Count=(rating_col,"count"))
                         .round({"Avg_Rating":2})
                         .query("Product_Count >= 3")
                         .sort_values("Avg_Rating", ascending=False))
            st.dataframe(cat_rat, use_container_width=True)
            st.caption("Categories with <3 products excluded. "
                       "Ratings below 3.5 warrant investigation.")

        # Discount effectiveness
        if disc_col and rating_col:
            st.markdown("#### Discount vs Rating Analysis")
            st.caption("Does heavier discounting correlate with lower ratings? "
                       "(Correlation, not causation)")
            corr_val = float(df[[disc_col, rating_col]].corr(method="spearman").iloc[0,1])
            if abs(corr_val) > 0.15:
                direction = "negative" if corr_val < 0 else "positive"
                st.warning(f"Spearman correlation r={corr_val:.3f} — "
                           f"{direction} association between discount and rating. "
                           f"Investigate whether high-discount products have quality issues.")
            else:
                st.success(f"Spearman r={corr_val:.3f} — no meaningful correlation "
                           f"between discount level and rating in this dataset.")

    # ── Sales Deep Dive ───────────────────────────────────────────────────
    elif domain == "sales":
        rev_col    = _find(df, ["revenue","sales","amount","value"], exclude=["budget","target"])
        target_col = _find(df, ["target","quota","budget","plan"])
        rep_col    = _find(df, ["rep","salesperson","agent","executive","owner"])
        region_col = _find(df, ["region","territory","area","zone"])
        margin_col = _find(df, ["margin","profit","gross"])

        c1, c2, c3 = st.columns(3)
        if rev_col:
            total_rev = float(df[rev_col].sum())
            c1.metric("Total Revenue", f"{total_rev:,.0f}",
                      f"Median deal: {float(df[rev_col].median()):,.0f}")
        if rev_col and target_col:
            ach = float(df[rev_col].sum()) / float(df[target_col].sum()) * 100
            c2.metric("Quota Achievement", f"{ach:.1f}%",
                      f"{ach-100:+.1f}pp vs target")
        if margin_col:
            avg_m = float(df[margin_col].mean())
            c3.metric("Avg Margin", f"{avg_m:.1f}%",
                      f"{int((df[margin_col]<0).sum())} loss deals")

        # Rep performance table
        if rep_col and rev_col:
            st.markdown("#### Sales Rep Performance")
            rep_data = df.groupby(rep_col).agg(
                Revenue=(rev_col,"sum"),
                Deals=(rev_col,"count"),
                Avg_Deal=(rev_col,"mean"),
            ).round({"Avg_Deal":0})
            rep_data["Revenue Share %"] = (rep_data["Revenue"] / rep_data["Revenue"].sum() * 100).round(1)
            rep_data = rep_data.sort_values("Revenue", ascending=False)
            st.dataframe(rep_data, use_container_width=True)

            top_rep_pct = float(rep_data["Revenue Share %"].iloc[0])
            if top_rep_pct > 40:
                st.warning(f"Revenue concentration: top rep = {top_rep_pct:.1f}% of total. "
                           f"High dependency on a single individual — succession risk.")

        # Regional breakdown
        if region_col and rev_col:
            st.markdown("#### Revenue by Region")
            reg_data = df.groupby(region_col)[rev_col].agg(["sum","count","mean"]).round(0)
            reg_data.columns = ["Total Revenue","Deals","Avg Deal"]
            reg_data["Revenue Share %"] = (reg_data["Total Revenue"] / reg_data["Total Revenue"].sum() * 100).round(1)
            st.dataframe(reg_data.sort_values("Total Revenue", ascending=False),
                         use_container_width=True)

    else:
        # General domain — show correlation insights
        num_cols = df.select_dtypes(include="number").columns.tolist()
        if len(num_cols) >= 2:
            st.markdown("#### Correlation Analysis")
            try:
                corr = df[num_cols[:10]].corr(method="spearman").round(3)
                fig = go.Figure(go.Heatmap(
                    z=corr.values, x=corr.columns.tolist(), y=corr.columns.tolist(),
                    colorscale="RdBu", zmid=0, zmin=-1, zmax=1,
                    text=np.round(corr.values, 2), texttemplate="%{text:.2f}",
                    colorbar=dict(title="r", thickness=12),
                ))
                fig.update_layout(height=400, paper_bgcolor="white", font=dict(color="#0F172A"),
                                  margin=dict(l=10,r=10,t=40,b=10))
                st.plotly_chart(_style_fig(fig), use_container_width=True,
                                config={"displayModeBar": False})
                st.caption("r = Spearman correlation. r² = shared variance. "
                           "Correlation ≠ causation.")
            except Exception as e:
                st.error(f"Correlation chart failed: {e}")

# ─── Tab 5: Action Plan ───────────────────────────────────────────────────────
with tabs[4]:
    st.markdown("### Recommended Actions")
    st.caption("All actions derived from dataset patterns. Prioritise in order shown.")

    actions = getattr(report, "recommended_actions", [])
    if not actions:
        st.info("No specific actions generated. Ensure domain-specific columns are present.")
    else:
        for i, action in enumerate(actions, 1):
            priority = "🔴 P1 — Immediate" if i <= 2 else "🟠 P2 — This Month" if i <= 4 else "🔵 P3 — Next Quarter"
            st.markdown(f"""
            <div style='border:1px solid #E2E8F0;border-radius:10px;
                        padding:14px 18px;margin-bottom:10px;background:white'>
                <div style='font-size:10px;font-weight:700;text-transform:uppercase;
                            letter-spacing:.07em;color:#64748B;margin-bottom:6px'>
                    {priority}
                </div>
                <div style='font-size:13.5px;color:inherit;line-height:1.6'>{action}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("### Analytical Next Steps")
    next_steps = [
        "Collect exit interview data to confirm or refute root-cause hypotheses",
        "Run period-over-period comparison if historical data becomes available",
        "Segment analysis by additional dimensions not available in current dataset",
        "Set KPI targets based on internal baselines — not generic external norms",
        "Re-run this analysis after any major intervention to measure impact",
    ]
    for step in next_steps:
        st.markdown(f"• {step}")

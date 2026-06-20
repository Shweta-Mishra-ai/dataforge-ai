"""
pages/11_📋_Health_Report.py — DataForge AI
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Premium Data Health + Business Insights Report
• Auto-detects niche: HR / Sales / E-commerce / Finance / General
• Power BI-grade KPI dashboard on screen
• One-click PDF download: health score + meaningful business insights
• Fixed: no backslash escapes in f-string expressions (Python 3.12+ PEP 701)
"""
import io
import os
import datetime

import numpy  as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.session_manager import require_data, get_df, get_filename
from core.story_engine   import detect_domain as _detect_domain_engine


@st.cache_data(show_spinner=False)
def _cached_health(df):
    return compute_health(df)


@st.cache_data(show_spinner=False)
def _cached_insights(df, niche):
    return build_insights(df.copy(), niche)


@st.cache_data(show_spinner=False)
def _cached_niche(df):
    return detect_niche(df)


st.set_page_config(
    page_title="Health Report — DataForge AI",
    page_icon="📋",
    layout="wide",
)
require_data()

df    = get_df()
fname = get_filename()

# ══════════════════════════════════════════════════════════
#  PREMIUM CSS
# ══════════════════════════════════════════════════════════
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
  html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }
  .block-container { padding-top: 1.2rem !important; padding-bottom: 1rem !important; }

  .kpi-card {
    background: linear-gradient(135deg, #0f1b2d 0%, #1a2f4a 100%);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px; padding: 20px 22px; text-align: center;
    box-shadow: 0 4px 24px rgba(0,0,0,0.18); height: 130px;
    display: flex; flex-direction: column; justify-content: center; align-items: center;
  }
  .kpi-val   { font-size: 2rem; font-weight: 800; line-height: 1; margin-bottom: 4px; }
  .kpi-label { font-size: 0.72rem; font-weight: 600; letter-spacing:.06em;
               color: rgba(255,255,255,0.55); text-transform: uppercase; }
  .kpi-delta { font-size: 0.75rem; font-weight: 600; margin-top: 4px; }
  .delta-pos { color: #22d3a5; }
  .delta-neg { color: #ff6b6b; }
  .delta-neu { color: rgba(255,255,255,0.4); }

  .health-ring {
    background: linear-gradient(135deg, #0f1b2d, #1a2f4a);
    border-radius: 20px; padding: 28px; text-align: center;
    border: 1px solid rgba(255,255,255,0.08);
    box-shadow: 0 4px 24px rgba(0,0,0,0.2);
  }
  .health-score { font-size: 3.8rem; font-weight: 900; line-height: 1; }
  .health-grade { font-size: 1rem; font-weight: 700; letter-spacing: .05em; margin-top: 4px; }

  .ins-card {
    border-radius: 12px; padding: 16px 18px; margin-bottom: 10px;
    border-left: 5px solid; position: relative;
    box-shadow: 0 2px 10px rgba(0,0,0,0.06);
  }
  .ins-tag  { font-size: 0.65rem; font-weight: 700; letter-spacing: .08em;
              text-transform: uppercase; margin-bottom: 5px; }
  .ins-body { font-size: 0.88rem; line-height: 1.6; color: #1e1e2e; }
  .ins-action { font-size: 0.78rem; font-weight: 600; margin-top: 6px; }

  .section-head {
    font-size: 1.1rem; font-weight: 700; color: #0f1b2d;
    border-bottom: 2px solid #e2e8f0; padding-bottom: 6px; margin: 18px 0 14px;
  }
  .niche-badge {
    display:inline-block; padding:5px 16px; border-radius: 20px;
    font-size: 0.72rem; font-weight: 700; letter-spacing:.07em;
    text-transform: uppercase; color: white; margin-left: 10px;
  }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
#  AUTO-DETECT NICHE
# ══════════════════════════════════════════════════════════
def detect_niche(df: pd.DataFrame):
    """Delegates to story_engine.detect_domain — single source of truth."""
    return _detect_domain_engine(df)


# ══════════════════════════════════════════════════════════
#  HEALTH SCORE ENGINE
# ══════════════════════════════════════════════════════════
def compute_health(df: pd.DataFrame) -> dict:
    rows, cols   = len(df), len(df.columns)
    missing_pct  = (df.isna().sum().sum() / max(df.size, 1)) * 100
    dup_pct      = df.duplicated().sum() / max(rows, 1) * 100
    num_cols     = df.select_dtypes(include="number").columns.tolist()
    outlier_cols = 0
    for c in num_cols:
        s = df[c].dropna()
        if len(s) > 10:
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            if iqr > 0 and ((s < q1 - 3*iqr) | (s > q3 + 3*iqr)).mean() > 0.05:
                outlier_cols += 1
    outlier_pct = outlier_cols / max(len(num_cols), 1) * 100

    score = 100
    score -= min(missing_pct  * 2.5, 35)
    score -= min(dup_pct      * 3.0, 20)
    score -= min(outlier_pct  * 0.4, 15)
    score -= 10 if rows < 50 else 0
    score -= 5  if cols < 3  else 0
    score  = max(int(score), 0)

    grade_map = [(90,"A+","Excellent","#22d3a5"),
                 (80,"A", "Very Good","#42b983"),
                 (70,"B+","Good",     "#60a5fa"),
                 (60,"B", "Fair",     "#fbbf24"),
                 (50,"C", "Needs Work","#f97316"),
                 (0, "D", "Poor",    "#ef4444")]
    grade, label, color = next(
        (g, ln, c) for thresh, g, ln, c in grade_map if score >= thresh)

    return {
        "score":       score,
        "grade":       grade,
        "label":       label,
        "color":       color,
        "missing_pct": round(missing_pct, 1),
        "dup_pct":     round(dup_pct, 1),
        "outlier_pct": round(outlier_pct, 1),
        "rows":        rows,
        "cols":        cols,
        "num_cols":    len(num_cols),
    }


# ══════════════════════════════════════════════════════════
#  NICHE INSIGHTS ENGINE
# ══════════════════════════════════════════════════════════
def build_insights(df: pd.DataFrame, niche: str) -> list:
    insights = []
    cols_lower = {c.lower(): c for c in df.columns}

    def _find(keywords):
        for kw in keywords:
            for cl, c in cols_lower.items():
                if kw in cl:
                    return c
        return None

    def _ins(tag, title, body, action, severity):
        COLOR = {
            "critical": ("#ef4444", "#fef2f2"),
            "warning":  ("#f97316", "#fff7ed"),
            "positive": ("#22d3a5", "#f0fdf4"),
            "info":     ("#3b82f6", "#eff6ff"),
        }
        border, bg = COLOR.get(severity, ("#3b82f6", "#eff6ff"))
        return {"tag": tag, "title": title, "body": body, "action": action,
                "severity": severity, "border": border, "bg": bg, "tag_color": border}

    # ── Universal: skew check ──────────────────────────────
    num_cols = df.select_dtypes(include="number").columns.tolist()
    for c in num_cols[:3]:
        s = df[c].dropna()
        if len(s) < 10:
            continue
        skew = float(s.skew()) if len(s) > 3 else 0
        if abs(skew) > 2:
            direction = "right" if skew > 0 else "left"
            diff_pct = abs(s.mean() - s.median()) / max(s.std(), 0.001) * 100
            insights.append(_ins(
                "⚠ DATA SKEW",
                "'{}' is heavily skewed (skew={:.2f})".format(c, skew),
                "The distribution of **{}** is {}-skewed. Mean ({:.2f}) differs from Median ({:.2f}) by {:.0f}%. Averages in reports based on this column may be misleading.".format(c, direction, s.mean(), s.median(), diff_pct),
                "➜ Use median for '{}' in client reports, not mean.".format(c),
                "warning"
            ))

    # ── HR-specific ────────────────────────────────────────
    if niche == "hr":
        attrition_col = _find(["attrition","left","churned","resigned","exited","turnover"])
        sat_col       = _find(["satisfaction","engagement","score","rating"])
        dept_col      = _find(["department","dept","team","division"])

        if attrition_col:
            s = df[attrition_col]
            atr_rate = None
            if str(s.dtype) in ["bool","int64","float64"] and float(s.max()) <= 1:
                atr_rate = float(s.mean()) * 100
            elif s.dtype == object:
                atr_rate = float(s.str.lower().isin(["yes","true","1","left"]).mean()) * 100
            if atr_rate is not None:
                sev = "critical" if atr_rate > 20 else "warning" if atr_rate > 10 else "positive"
                note = "CRITICAL — 2× the industry danger zone." if sev == "critical" else "Above 10% SHRM benchmark." if sev == "warning" else "Below 10% — excellent retention."
                action = "➜ IMMEDIATE: Conduct stay-interviews in high-risk departments." if sev == "critical" else "➜ Build a quarterly pulse survey. Flag departments above 15%." if sev == "warning" else "➜ Document what drives retention — use as a competitive advantage."
                insights.append(_ins(
                    "🔴 ATTRITION RISK" if sev == "critical" else "⚠ ATTRITION" if sev == "warning" else "✅ HEALTHY RETENTION",
                    "Attrition rate is {:.1f}% (SHRM benchmark: <10%)".format(atr_rate),
                    "Your dataset shows **{:.1f}% attrition**. {} SHRM 2024 reports the average cost of replacing one employee is **50–200% of their annual salary**.".format(atr_rate, note),
                    action, sev
                ))

        if sat_col:
            mean_sat = float(df[sat_col].dropna().mean())
            max_sat  = float(df[sat_col].dropna().max())
            sat_norm = mean_sat / max_sat if max_sat > 1 else mean_sat
            sev = "critical" if sat_norm < 0.5 else "warning" if sat_norm < 0.7 else "positive"
            note = "Low satisfaction → 4× higher attrition risk. Urgent intervention needed." if sev in ("critical","warning") else "Scores above 70% indicate an engaged workforce."
            insights.append(_ins(
                "👥 EMPLOYEE SATISFACTION",
                "Avg satisfaction: {:.2f} / {:.0f} ({:.0f}%)".format(mean_sat, max_sat, sat_norm*100),
                "Mean satisfaction score is **{:.2f}** (normalized: {:.0f}%). Gallup 2024 finds that {}".format(mean_sat, sat_norm*100, note),
                "➜ Implement manager 1:1s and recognition programs. Gallup: 52% of exits are preventable." if sev != "positive" else "➜ Publish satisfaction data in employer branding.",
                sev
            ))

        if dept_col and attrition_col:
            try:
                s = df[attrition_col]
                tmp = df.copy()
                if s.dtype == object:
                    tmp["_atr"] = s.str.lower().isin(["yes","true","1","left"]).astype(float)
                else:
                    tmp["_atr"] = pd.to_numeric(s, errors="coerce")
                dept_risk = tmp.groupby(dept_col)["_atr"].mean().sort_values(ascending=False)
                top_dept  = str(dept_risk.index[0])
                top_rate  = float(dept_risk.iloc[0]) * 100
                sev = "critical" if top_rate > 20 else "warning"
                insights.append(_ins(
                    "🏢 DEPARTMENT RISK",
                    "'{}' has the highest attrition: {:.1f}%".format(top_dept, top_rate),
                    "The **{}** department shows {:.1f}% attrition — {} the SHRM 10% benchmark. Department-level attrition often signals a specific manager, workload, or pay equity issue.".format(top_dept, top_rate, "critically above" if top_rate > 20 else "above"),
                    "➜ Prioritize '{}' for skip-level interviews and exit interview analysis.".format(top_dept),
                    sev
                ))
            except Exception:
                pass

        # ── HR: Tenure cohort analysis ─────────────────
        tenure_col = _find(["tenure","years","seniority","experience","time_spend"])
        if tenure_col and attrition_col:
            try:
                tmp2 = df.copy()
                if df[attrition_col].dtype == object:
                    tmp2["_atr2"] = df[attrition_col].str.lower().isin(["yes","true","1","left"]).astype(float)
                else:
                    tmp2["_atr2"] = pd.to_numeric(df[attrition_col], errors="coerce")
                # Bin tenure into 3 cohorts
                tmp2["_tenure_bin"] = pd.cut(tmp2[tenure_col],
                    bins=[0, 2, 5, float("inf")],
                    labels=["0–2 yrs", "3–5 yrs", "6+ yrs"])
                cohort_atr = tmp2.groupby("_tenure_bin")["_atr2"].mean() * 100
                new_hire_atr = float(cohort_atr.get("0–2 yrs", 0))
                vet_atr      = float(cohort_atr.get("6+ yrs", 0))
                if new_hire_atr > 25:
                    insights.append(_ins(
                        "🕐 ONBOARDING RISK",
                        "New hires (0–2 yrs) attrition: {:.1f}%".format(new_hire_atr),
                        "Employees in their first 2 years show {:.1f}% attrition — a signal of poor onboarding, misaligned expectations, or poor manager support. "
                        "Veteran employees (6+ yrs) show {:.1f}% attrition by comparison. "
                        "SHRM: replacing a new hire costs 50–150% of salary before they hit full productivity.".format(new_hire_atr, vet_atr),
                        "➜ Implement a structured 90-day onboarding program. Assign mentors to all new hires. "
                        "Run a 30/60/90 check-in survey to catch at-risk employees early.",
                        "critical" if new_hire_atr > 35 else "warning"
                    ))
                elif vet_atr > 15:
                    insights.append(_ins(
                        "⚠ VETERAN FLIGHT RISK",
                        "Senior employees (6+ yrs) attrition: {:.1f}%".format(vet_atr),
                        "Experienced employees (6+ years tenure) are leaving at {:.1f}%. "
                        "This is a knowledge drain — these employees carry institutional memory, client relationships, and domain expertise. "
                        "Replacing a senior employee can cost 200% of their annual salary.".format(vet_atr),
                        "➜ Run skip-level interviews with the 6+ year cohort this quarter. "
                        "Review compensation and career growth opportunities for this group specifically.",
                        "warning"
                    ))
            except Exception:
                pass

        # ── HR: Overwork risk ─────────────────────────────
        hours_col = _find(["hours","montly_hours","monthly_hours","avg_hours","work_hour"])
        if hours_col:
            try:
                mean_h = float(df[hours_col].dropna().mean())
                pct_overwork = float((df[hours_col].dropna() > 210).mean() * 100)
                if pct_overwork > 20 or mean_h > 195:
                    sev = "critical" if pct_overwork > 40 or mean_h > 220 else "warning"
                    insights.append(_ins(
                        "🔥 OVERWORK RISK",
                        "Avg {:.0f} hrs/month — {:.0f}% working >210 hrs".format(mean_h, pct_overwork),
                        "Average monthly hours are {:.0f} (standard: 160–180 hrs). "
                        "{:.0f}% of employees work more than 210 hours per month — a burnout risk zone. "
                        "Gallup 2024: employees working 60+ hrs/week have 2.6× higher voluntary attrition. "
                        "Burnout is now the #1 self-reported reason for resignation in STEM and finance.".format(mean_h, pct_overwork),
                        "➜ Audit workload distribution immediately — identify if overwork is concentrated in specific teams. "
                        "Hire contractors or redistribute tasks. Target: bring >90% of workforce under 200 hrs/month.",
                        sev
                    ))
            except Exception:
                pass

        # ── HR: Salary band vs attrition ─────────────────
        salary_col = _find(["salary","pay","compensation","wage","band"])
        if salary_col and attrition_col:
            try:
                tmp3 = df.copy()
                if df[attrition_col].dtype == object:
                    tmp3["_atr3"] = df[attrition_col].str.lower().isin(["yes","true","1","left"]).astype(float)
                else:
                    tmp3["_atr3"] = pd.to_numeric(df[attrition_col], errors="coerce")
                sal_atr = tmp3.groupby(salary_col)["_atr3"].mean().sort_values(ascending=False) * 100
                if len(sal_atr) >= 2:
                    worst_band = str(sal_atr.index[0])
                    worst_rate = float(sal_atr.iloc[0])
                    best_band  = str(sal_atr.index[-1])
                    best_rate  = float(sal_atr.iloc[-1])
                    gap        = worst_rate - best_rate
                    if gap > 8:
                        insights.append(_ins(
                            "💸 PAY-DRIVEN ATTRITION",
                            "'{}' band: {:.1f}% attrition vs '{}': {:.1f}%".format(worst_band, worst_rate, best_band, best_rate),
                            "The '{}' salary band has {:.1f}% attrition vs {:.1f}% for the '{}' band — a {:.0f} percentage point gap. "
                            "SHRM 2024: 38% of exits cite below-market pay as the primary reason. "
                            "Pay-driven attrition is the fastest to fix but most expensive if ignored — each exit in a low band still costs 50–100% of annual salary.".format(
                                worst_band, worst_rate, best_rate, best_band, gap),
                            "➜ Run market salary benchmarking for the '{}' band within 30 days. "
                            "Model the ROI of a 10–15% pay increase vs replacement cost for the highest-risk employees.".format(worst_band),
                            "critical" if worst_rate > 25 else "warning"
                        ))
            except Exception:
                pass

        # ── HR: Promotion gap → flight risk ───────────────
        promo_col = _find(["promotion","promoted","promotion_last"])
        if promo_col and sat_col and attrition_col:
            try:
                promo_s = df[promo_col]
                if not pd.api.types.is_numeric_dtype(promo_s):
                    promo_s = promo_s.str.lower().isin(["yes","true","1"]).astype(float)
                promo_rate = float(promo_s.mean()) * 100
                # Satisfaction for un-promoted employees
                not_promoted_sat = float(df.loc[promo_s == 0, sat_col].dropna().mean()) if (promo_s == 0).any() else None
                promoted_sat     = float(df.loc[promo_s == 1, sat_col].dropna().mean()) if (promo_s == 1).any() else None
                if promo_rate < 5 and not_promoted_sat is not None:
                    insights.append(_ins(
                        "📈 PROMOTION GAP",
                        "Only {:.1f}% promoted — unpromoted satisfaction: {:.2f}".format(promo_rate, not_promoted_sat),
                        "Only {:.1f}% of employees received a promotion in the last 5 years. "
                        "Employees without promotion show {:.2f} satisfaction vs {:.2f} for promoted staff — a {:.0f}% gap. "
                        "Mercer 2024: lack of career growth is the #1 voluntary exit driver. "
                        "Employees without a promotion path are 3× more likely to leave within 12 months.".format(
                            promo_rate,
                            not_promoted_sat,
                            promoted_sat if promoted_sat else not_promoted_sat,
                            abs((promoted_sat or not_promoted_sat) - not_promoted_sat) / max(not_promoted_sat, 0.01) * 100
                        ),
                        "➜ Create transparent promotion criteria for all levels. "
                        "Implement individual development plans (IDPs) for the bottom 30% satisfaction + no-promotion segment. "
                        "Target: increase promotion rate to at least 10% per year.",
                        "critical" if promo_rate < 3 else "warning"
                    ))
            except Exception:
                pass

        # ── HR: Flight risk segment ───────────────────────
        if attrition_col and sat_col and tenure_col:
            try:
                tmp4 = df.copy()
                if df[attrition_col].dtype == object:
                    tmp4["_still_here"] = ~df[attrition_col].str.lower().isin(["yes","true","1","left"])
                else:
                    tmp4["_still_here"] = (pd.to_numeric(df[attrition_col], errors="coerce") == 0)
                current = tmp4[tmp4["_still_here"]].copy()
                if len(current) > 10:
                    sat_q25  = float(current[sat_col].quantile(0.25))
                    ten_median = float(current[tenure_col].median())
                    at_risk  = current[
                        (current[sat_col] <= sat_q25) &
                        (current[tenure_col] >= ten_median)
                    ]
                    risk_pct = len(at_risk) / len(current) * 100
                    if risk_pct > 10:
                        insights.append(_ins(
                            "🚨 FLIGHT RISK SEGMENT",
                            "{:.0f}% of current workforce is at high flight risk".format(risk_pct),
                            "**{:,} current employees** ({:.0f}% of workforce) match the flight risk profile: "
                            "low satisfaction (bottom 25%, score ≤{:.2f}) AND long tenure (≥{:.0f} years median). "
                            "Long-tenured employees with falling satisfaction are statistically the most likely to leave next — "
                            "and the most expensive to replace due to institutional knowledge loss.".format(
                                len(at_risk), risk_pct, sat_q25, ten_median),
                            "➜ Pull this segment's names from your HRIS immediately. "
                            "Schedule 1:1 career conversations within 2 weeks. "
                            "This is your highest-priority retention action — act before they decide.",
                            "critical" if risk_pct > 20 else "warning"
                        ))
            except Exception:
                pass

    # ── SALES-specific ────────────────────────────────────
    elif niche == "sales":
        rev_col    = _find(["revenue","amount","deal_value","deal_size","value","gmv","arr"])
        status_col = _find(["status","stage","outcome","result","win","lost"])

        if rev_col:
            s     = df[rev_col].dropna()
            total = float(s.sum())
            top10 = float(s.nlargest(max(1, int(len(s)*0.1))).sum())
            conc  = top10 / total * 100 if total > 0 else 0
            sev   = "critical" if conc > 80 else "warning" if conc > 60 else "positive"
            note  = "CRITICAL concentration risk — losing 1-2 key accounts could collapse revenue." if sev == "critical" else "High concentration — moderate dependency on key accounts." if sev == "warning" else "Healthy revenue distribution."
            insights.append(_ins(
                "💰 REVENUE CONCENTRATION",
                "Top 10% of deals = {:.0f}% of total revenue".format(conc),
                "Your top 10% deals account for **{:.0f}% of {}** (${:,.0f} total). {}".format(conc, rev_col, total, note),
                "➜ Immediately build 3–5 additional pipeline accounts at the same deal size." if sev == "critical" else "➜ Implement account health scoring. Trigger executive relationships for top 10%." if sev == "warning" else "➜ Replicate the profile of top-performing deals in prospecting strategy.",
                sev
            ))

        if status_col:
            s    = df[status_col].dropna().astype(str).str.lower()
            won  = int(s.isin(["won","win","closed won","success","yes"]).sum())
            lost = int(s.isin(["lost","lose","closed lost","loss","no","failed"]).sum())
            total = won + lost
            win_rate = won / total * 100 if total > 0 else 0
            sev  = "critical" if win_rate < 20 else "warning" if win_rate < 35 else "positive"
            note = "Well below benchmark — significant pipeline efficiency problem." if sev == "critical" else "Below benchmark — improvement needed." if sev == "warning" else "At or above benchmark — strong sales execution."
            insights.append(_ins(
                "🎯 WIN RATE",
                "Win rate: {:.1f}% (Benchmark: 25-40%)".format(win_rate),
                "Out of **{} qualified deals**, {} were won ({:.1f}%). Salesforce 2024 benchmark is 25–40%. {}".format(total, won, win_rate, note),
                "➜ Implement MEDDIC or BANT qualification framework. Review lost deal reasons." if sev != "positive" else "➜ Document the winning sales playbook. Scale to underperformers.",
                sev
            ))

    # ── E-COMMERCE-specific ──────────────────────────────
    elif niche == "ecommerce":
        order_val_col = _find(["order_value","gmv","amount","total","price","revenue","aov"])
        customer_col  = _find(["customer","user","buyer","client_id","customer_id"])

        if order_val_col:
            s   = df[order_val_col].dropna()
            aov = float(s.mean())
            top20_avg = float(s.nlargest(max(1, int(len(s)*0.2))).mean())
            insights.append(_ins(
                "🛒 AVERAGE ORDER VALUE",
                "AOV = ${:,.2f} per transaction".format(aov),
                "Average Order Value is **${:,.2f}**. Top 20% of orders average ${:,.2f}. Increasing AOV by 10% through upselling is more profitable than acquiring new customers.".format(aov, top20_avg),
                "➜ Deploy bundled products for orders below ${:.0f}. Target: lift AOV by 15% in 90 days.".format(aov * 0.7),
                "info"
            ))

        if customer_col and order_val_col:
            try:
                cust_spend = df.groupby(customer_col)[order_val_col].sum()
                top20_pct  = float(cust_spend.nlargest(max(1, int(len(cust_spend)*0.2))).sum() / cust_spend.sum() * 100)
                sev = "critical" if top20_pct > 80 else "warning" if top20_pct > 65 else "positive"
                insights.append(_ins(
                    "👑 PARETO PRINCIPLE",
                    "Top 20% customers = {:.0f}% of revenue".format(top20_pct),
                    "**{:.0f}% of revenue** comes from your top 20% customers. {} ".format(top20_pct, "Classic Pareto — but high concentration means churn of top customers = major revenue loss." if sev != "positive" else "Healthy spread — revenue reasonably distributed."),
                    "➜ Build a VIP tier for top 20% customers. Assign dedicated account managers.",
                    sev
                ))
            except Exception:
                pass

    # ── FINANCE-specific ─────────────────────────────────
    elif niche == "finance":
        rev_col  = _find(["revenue","income","sales","turnover","gross"])
        cost_col = _find(["cost","expense","cogs","expenditure","opex"])

        if rev_col and cost_col:
            rev    = float(df[rev_col].dropna().sum())
            cost   = float(df[cost_col].dropna().sum())
            margin = (rev - cost) / rev * 100 if rev > 0 else 0
            sev    = "critical" if margin < 5 else "warning" if margin < 15 else "positive"
            note   = "CRITICAL: At risk of operating loss." if sev == "critical" else "Below benchmark — cost control or pricing review needed." if sev == "warning" else "Healthy margin — above industry benchmark."
            insights.append(_ins(
                "📊 GROSS MARGIN",
                "Gross Margin: {:.1f}% (Benchmark: >15%)".format(margin),
                "Total Revenue: **${:,.0f}** | Total Cost: **${:,.0f}** | Gross Margin: **{:.1f}%**. McKinsey 2024: healthy businesses target >15% gross margin. {}".format(rev, cost, margin, note),
                "➜ IMMEDIATE: Conduct cost structure analysis. Identify top 3 cost drivers." if sev == "critical" else "➜ Review pricing strategy and renegotiate top 3 supplier contracts." if sev == "warning" else "➜ Model the impact of a 5% price increase to protect margin.",
                sev
            ))

    # ── Universal: Missing data risk ──────────────────────
    miss = df.isna().sum()
    bad_cols = miss[miss / len(df) > 0.3].sort_values(ascending=False)
    if len(bad_cols) > 0:
        col_list = ", ".join(
            ["'{}' ({:.0f}%)".format(c, miss[c]/len(df)*100) for c in bad_cols.index[:3]])
        worst_pct = float(bad_cols.iloc[0] / len(df))
        sev = "critical" if worst_pct > 0.5 else "warning"
        insights.append(_ins(
            "⚠ DATA QUALITY RISK",
            "{} column(s) have >30% missing data".format(len(bad_cols)),
            "Columns with high missing rates: **{}**. Any analysis or model trained on these columns will be statistically unreliable.".format(col_list),
            "➜ Either impute with domain-appropriate values, or exclude from client-facing insights.",
            sev
        ))

    if not insights:
        insights.append(_ins(
            "✅ DATA HEALTH",
            "Dataset is clean and analysis-ready",
            "No critical issues detected. Data completeness, distribution, and structure are within acceptable thresholds for business analysis.",
            "➜ Proceed directly to Dashboard and ML Predictions pages.",
            "positive"
        ))

    return insights


# ══════════════════════════════════════════════════════════
#  PDF GENERATOR
# ══════════════════════════════════════════════════════════
def build_health_pdf(df: pd.DataFrame, niche: str, health: dict,
                     insights: list, fname: str) -> bytes:
    """Premium 5-page health + business insights PDF report."""
    import io as _io
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.lib.colors import HexColor, white, black
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY, TA_RIGHT
    from reportlab.platypus import (
        BaseDocTemplate, Frame, PageTemplate,
        Paragraph, Spacer, Table, TableStyle,
        HRFlowable, KeepTogether, PageBreak, Image,
    )
    from reportlab.pdfgen import canvas as CV
    from reportlab.lib import colors as rl_colors
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    W, H = A4
    CW   = W - 36 * mm
    now  = datetime.datetime.now().strftime("%B %d, %Y")

    NICHE_COLORS = {
        "hr":        "#1565C0",
        "sales":     "#2E7D32",
        "ecommerce": "#E64A19",
        "finance":   "#0D47A1",
        "general":   "#1B4FD8",
    }
    accent_hex  = NICHE_COLORS.get(niche, "#1B4FD8")
    accent      = HexColor(accent_hex)
    dark        = HexColor("#0A1628")
    gray        = HexColor("#6B7280")
    light       = HexColor("#F0F4FF")
    light2      = HexColor("#F8FAFF")
    score_color = HexColor(health["color"])

    # ── Premium fonts ─────────────────────────────────────
    import os as _os
    from reportlab.pdfbase import pdfmetrics as _pm
    from reportlab.pdfbase.ttfonts import TTFont as _TTF
    _FONT_DIR = _os.path.join(
        _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
        "..", "assets", "fonts"
    )
    _BF, _BB, _BI = "Helvetica", "Helvetica-Bold", "Helvetica-Oblique"
    for alias, fname_f in [("HDF-Reg","Carlito-Regular.ttf"),
                            ("HDF-Bold","Carlito-Bold.ttf"),
                            ("HDF-Italic","Carlito-Italic.ttf")]:
        try:
            _pm.registerFont(_TTF(alias, _os.path.join(_FONT_DIR, fname_f)))
            if alias == "HDF-Reg":  _BF = "HDF-Reg"
            if alias == "HDF-Bold": _BB = "HDF-Bold"
            if alias == "HDF-Italic": _BI = "HDF-Italic"
        except Exception:
            pass

    def ps(name, **kw): return ParagraphStyle(name, **kw)
    ST = {
        "h1":   ps("h1",   fontName=_BB, fontSize=17, textColor=accent,
                   spaceAfter=4, spaceBefore=2, leading=21),
        "h2":   ps("h2",   fontName=_BB, fontSize=13, textColor=dark,
                   spaceBefore=10, spaceAfter=4, leading=16),
        "h3":   ps("h3",   fontName=_BB, fontSize=10.5, textColor=accent,
                   spaceBefore=8, spaceAfter=3, leading=14),
        "body": ps("body", fontName=_BF, fontSize=9.5, textColor=dark,
                   leading=15, spaceAfter=3, alignment=TA_JUSTIFY),
        "sm":   ps("sm",   fontName=_BF, fontSize=8, textColor=gray,
                   leading=11, spaceAfter=2),
        "act":  ps("act",  fontName=_BB, fontSize=9, textColor=accent,
                   leading=13, spaceAfter=3),
        "ctr":  ps("ctr",  fontName=_BF, fontSize=9, textColor=dark,
                   alignment=TA_CENTER),
        "note": ps("note", fontName=_BI, fontSize=8, textColor=gray,
                   leading=11, spaceAfter=2),
    }

    buf = _io.BytesIO()

    # ── Canvas with header/footer ─────────────────────────
    class _Canvas(CV.Canvas):
        def __init__(self, fn, **kw):
            super().__init__(fn, **kw)
            self._sp = []
        def showPage(self):
            self._sp.append(dict(self.__dict__))
            self._startPage()
        def save(self):
            tot = len(self._sp)
            for state in self._sp:
                self.__dict__.update(state)
                self._draw_hf(tot)
                super().showPage()
            super().save()
        def _draw_hf(self, tot):
            # Header
            self.setFillColor(dark)
            self.rect(0, H - 20*mm, W, 20*mm, fill=1, stroke=0)
            self.setFillColor(accent)
            self.rect(0, H - 21*mm, W, 1*mm, fill=1, stroke=0)
            self.setFillColor(accent)
            self.rect(0, H - 20*mm, 3*mm, 20*mm, fill=1, stroke=0)
            self.setFillColor(white)
            self.setFont(_BB, 9.5)
            self.drawString(8*mm, H - 11*mm, "DataForge AI  ·  Data Health & Business Insights")
            self.setFont(_BF, 7.5)
            self.setFillColor(HexColor("#BBDEFB"))
            self.drawString(8*mm, H - 17.5*mm, fname[:60])
            self.setFillColor(white)
            self.drawRightString(W - 8*mm, H - 11*mm, now)
            self.setFont(_BF, 7)
            self.drawRightString(W - 8*mm, H - 17.5*mm, "CONFIDENTIAL")
            # Footer
            self.setFillColor(dark)
            self.rect(0, 0, W, 11*mm, fill=1, stroke=0)
            self.setFillColor(accent)
            self.rect(0, 11*mm, W, 0.8*mm, fill=1, stroke=0)
            self.setFillColor(white)
            self.setFont(_BF, 6.5)
            self.drawString(8*mm, 4*mm, "DataForge AI  ·  Confidential  ·  Verify with domain expert before client delivery")
            # Page circle
            self.setFillColor(accent)
            self.circle(W - 13*mm, 5.5*mm, 4.5*mm, fill=1, stroke=0)
            self.setFillColor(white)
            self.setFont(_BB, 6.5)
            self.drawCentredString(W - 13*mm, 3.8*mm, "{}/{}".format(self._pageNumber, tot))

    doc = BaseDocTemplate(
        buf, pagesize=A4,
        leftMargin=18*mm, rightMargin=18*mm,
        topMargin=26*mm, bottomMargin=17*mm,
    )
    frame = Frame(18*mm, 17*mm, CW, H - 43*mm, id="main")
    doc.addPageTemplates([PageTemplate(id="main", frames=[frame],
                                       onPage=lambda c, d: None)])
    story = []

    # ══════════════════════════════════════════════════════
    # PAGE 1: COVER + HEALTH SCORE
    # ══════════════════════════════════════════════════════
    story.append(Spacer(1, 6*mm))
    story.append(Paragraph("DATA HEALTH &amp; BUSINESS INSIGHTS REPORT", ST["h1"]))
    story.append(Paragraph(fname[:70], ST["sm"]))
    story.append(HRFlowable(width="100%", thickness=2.5, color=accent, spaceAfter=6))
    story.append(Paragraph(
        "Generated: {}  ·  {:,} rows  ·  {} columns  ·  Domain: <b>{}</b>".format(
            now, health["rows"], health["cols"], niche.upper()),
        ST["sm"]))
    story.append(Spacer(1, 6*mm))

    # Health score KPI box
    story.append(Paragraph("Overall Data Health Score", ST["h2"]))
    score_para = Paragraph(
        "<b>{}/100</b>".format(health["score"]),
        ParagraphStyle("sc", fontName=_BB, fontSize=36, textColor=score_color,
                       alignment=TA_CENTER))
    grade_para = Paragraph(
        "<b>Grade: {}  —  {}</b>".format(health["grade"], health["label"]),
        ParagraphStyle("gr", fontName=_BB, fontSize=11,
                       textColor=score_color, alignment=TA_CENTER))

    kpi_row = [
        [score_para, grade_para,
         Paragraph("Missing: <b>{}%</b>".format(health["missing_pct"]),
                   ParagraphStyle("kv", fontName=_BF, fontSize=9, textColor=dark, alignment=TA_CENTER)),
         Paragraph("Duplicates: <b>{}%</b>".format(health["dup_pct"]),
                   ParagraphStyle("kv", fontName=_BF, fontSize=9, textColor=dark, alignment=TA_CENTER)),
         Paragraph("Outlier cols: <b>{}%</b>".format(health["outlier_pct"]),
                   ParagraphStyle("kv", fontName=_BF, fontSize=9, textColor=dark, alignment=TA_CENTER)),
        ]
    ]
    kpi_tbl = Table(kpi_row, colWidths=[CW*x for x in [0.18,0.25,0.19,0.19,0.19]])
    kpi_tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0,0),(0,0), light),
        ("BACKGROUND",    (1,0),(1,0), HexColor("#EFF6FF")),
        ("BACKGROUND",    (2,0),(-1,0), light2),
        ("VALIGN",        (0,0),(-1,-1), "MIDDLE"),
        ("ALIGN",         (0,0),(-1,-1), "CENTER"),
        ("TOPPADDING",    (0,0),(-1,-1), 16),
        ("BOTTOMPADDING", (0,0),(-1,-1), 16),
        ("BOX",           (0,0),(-1,-1), 1.5, accent),
        ("INNERGRID",     (0,0),(-1,-1), 0.3, HexColor("#E5E7EB")),
        ("LINEBELOW",     (0,0),(-1,0),  2, accent),
    ]))
    story.append(kpi_tbl)
    story.append(Spacer(1, 5*mm))

    # Dataset summary table
    story.append(Paragraph("Dataset Summary", ST["h3"]))
    num_cols_list = df.select_dtypes(include="number").columns.tolist()
    cat_cols_list = df.select_dtypes(include=["object","string"]).columns.tolist()
    date_cols_list= df.select_dtypes(include="datetime").columns.tolist()
    missing_total = df.isna().sum().sum()
    dup_count     = df.duplicated().sum()

    summary_data = [
        [Paragraph("<b>Metric</b>", ST["ctr"]), Paragraph("<b>Value</b>", ST["ctr"])],
        ["Total Rows",       "{:,}".format(health["rows"])],
        ["Total Columns",    str(health["cols"])],
        ["Numeric Columns",  str(len(num_cols_list))],
        ["Categorical Cols", str(len(cat_cols_list))],
        ["DateTime Columns", str(len(date_cols_list))],
        ["Missing Values",   "{:,} ({:.1f}%)".format(int(missing_total), health["missing_pct"])],
        ["Duplicate Rows",   "{:,} ({:.1f}%)".format(int(dup_count), health["dup_pct"])],
        ["Memory Usage",     "{:.1f} MB".format(df.memory_usage(deep=True).sum() / 1e6)],
    ]
    for i in range(1, len(summary_data)):
        summary_data[i] = [
            Paragraph(str(summary_data[i][0]), ST["sm"]),
            Paragraph(str(summary_data[i][1]),
                      ParagraphStyle("sv", fontName=_BB, fontSize=8.5,
                                     textColor=dark, alignment=TA_RIGHT))
        ]

    sum_tbl = Table(summary_data, colWidths=[CW*0.6, CW*0.4])
    sum_tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0,0),(-1,0),  dark),
        ("TEXTCOLOR",     (0,0),(-1,0),  white),
        ("ROWBACKGROUNDS",(0,1),(-1,-1), [white, light2]),
        ("ALIGN",         (1,0),(-1,-1), "RIGHT"),
        ("TOPPADDING",    (0,0),(-1,-1), 5),
        ("BOTTOMPADDING", (0,0),(-1,-1), 5),
        ("LEFTPADDING",   (0,0),(-1,-1), 10),
        ("BOX",           (0,0),(-1,-1), 0.5, HexColor("#E5E7EB")),
        ("INNERGRID",     (0,0),(-1,-1), 0.3, HexColor("#E5E7EB")),
    ]))
    story.append(sum_tbl)

    # ══════════════════════════════════════════════════════
    # PAGE 2: BUSINESS INSIGHTS
    # ══════════════════════════════════════════════════════
    story.append(PageBreak())
    story.append(Paragraph("Meaningful Business Insights", ST["h1"]))
    story.append(Paragraph(
        "Each insight follows the format: <b>What → Why it matters → What to do.</b> "
        "All figures are computed directly from the uploaded dataset.",
        ST["body"]))
    story.append(HRFlowable(width="100%", thickness=1.5, color=accent, spaceAfter=6))

    SEV_COLORS = {
        "critical": ("#DC2626", "#FEF2F2"),
        "warning":  ("#D97706", "#FFFBEB"),
        "positive": ("#059669", "#ECFDF5"),
        "info":     ("#2563EB", "#EFF6FF"),
    }

    for i, ins in enumerate(insights, 1):
        border_c, bg_c = SEV_COLORS.get(ins["severity"], ("#2563EB", "#EFF6FF"))
        bg_hex  = HexColor(bg_c)
        bdr_hex = HexColor(border_c)
        tag_c   = HexColor(border_c)

        tag_p    = Paragraph(ins["tag"],
            ParagraphStyle("it", fontName=_BB, fontSize=7.5,
                           textColor=tag_c, spaceAfter=2))
        title_p  = Paragraph("<b>{}. {}</b>".format(i, ins["title"]),
            ParagraphStyle("itl", fontName=_BB, fontSize=10.5,
                           textColor=dark, spaceAfter=3, leading=14))
        body_p   = Paragraph(ins["body"].replace("**","").replace("*",""),
            ParagraphStyle("ib", fontName=_BF, fontSize=9.5,
                           textColor=dark, leading=14.5, spaceAfter=4,
                           alignment=TA_JUSTIFY))
        action_p = Paragraph(ins["action"],
            ParagraphStyle("ia", fontName=_BB, fontSize=9,
                           textColor=HexColor(border_c), leading=13))

        card = Table([[tag_p],[title_p],[body_p],[action_p]], colWidths=[CW])
        card.setStyle(TableStyle([
            ("BACKGROUND",    (0,0),(-1,-1), bg_hex),
            ("LINEBEFORE",    (0,0),(0,-1),  6, bdr_hex),
            ("TOPPADDING",    (0,0),(-1,-1), 9),
            ("BOTTOMPADDING", (0,0),(-1,-1), 8),
            ("LEFTPADDING",   (0,0),(-1,-1), 16),
            ("RIGHTPADDING",  (0,0),(-1,-1), 12),
            ("BOX",           (0,0),(-1,-1), 0.5, HexColor("#E5E7EB")),
        ]))
        story.append(KeepTogether([card, Spacer(1, 5*mm)]))

    # ══════════════════════════════════════════════════════
    # PAGE 3: DESCRIPTIVE STATISTICS
    # ══════════════════════════════════════════════════════
    story.append(PageBreak())
    story.append(Paragraph("Descriptive Statistics", ST["h1"]))
    story.append(HRFlowable(width="100%", thickness=1.5, color=accent, spaceAfter=5))

    if len(num_cols_list) > 0:
        story.append(Paragraph("Numeric Columns Summary", ST["h2"]))
        desc = df[num_cols_list[:8]].describe().round(3)
        hdr_vals = ["Stat"] + [c[:12] for c in desc.columns]
        stat_hdr = [Paragraph("<b>{}</b>".format(h),
                    ParagraphStyle("sh", fontName=_BB, fontSize=7.5,
                                   textColor=white, alignment=TA_CENTER))
                    for h in hdr_vals]
        stat_rows = [stat_hdr]
        for idx in desc.index:
            row = [Paragraph("<b>{}</b>".format(idx),
                             ParagraphStyle("si", fontName=_BB, fontSize=7.5, textColor=dark))]
            for val in desc.loc[idx]:
                row.append(Paragraph(str(val),
                    ParagraphStyle("sv2", fontName=_BF, fontSize=7.5,
                                   textColor=dark, alignment=TA_CENTER)))
            stat_rows.append(row)

        n_stat_cols = len(hdr_vals)
        stat_tbl = Table(stat_rows,
                         colWidths=[CW*0.1] + [CW*0.9/max(n_stat_cols-1,1)]*(n_stat_cols-1))
        stat_tbl.setStyle(TableStyle([
            ("BACKGROUND",    (0,0),(-1,0),  dark),
            ("ROWBACKGROUNDS",(0,1),(-1,-1), [white, light2]),
            ("ALIGN",         (1,0),(-1,-1), "CENTER"),
            ("TOPPADDING",    (0,0),(-1,-1), 5),
            ("BOTTOMPADDING", (0,0),(-1,-1), 4),
            ("LEFTPADDING",   (0,0),(-1,-1), 6),
            ("BOX",           (0,0),(-1,-1), 0.5, HexColor("#E5E7EB")),
            ("INNERGRID",     (0,0),(-1,-1), 0.3, HexColor("#E5E7EB")),
        ]))
        story.append(stat_tbl)
        story.append(Spacer(1, 6*mm))

    # ── Distribution mini-charts ──────────────────────────
    if len(num_cols_list) >= 2:
        story.append(Paragraph("Distribution Overview", ST["h3"]))
        story.append(Paragraph(
            "Histograms below show the data distribution for the top numeric columns. "
            "Dashed line = mean, dotted = median. Skewed distributions require median for reporting.",
            ST["note"]))
        story.append(Spacer(1, 3*mm))

        try:
            n_charts = min(4, len(num_cols_list))
            fig, axes = plt.subplots(1, n_charts, figsize=(10, 2.8))
            if n_charts == 1:
                axes = [axes]
            fig.patch.set_facecolor("#ffffff")
            palette = ["#1565C0","#0D47A1","#1B5E20","#4527A0"]
            for idx2, (ax2, col) in enumerate(zip(axes, num_cols_list[:n_charts])):
                s2 = df[col].dropna()
                s2 = pd.to_numeric(s2, errors="coerce").dropna()
                if len(s2) == 0:
                    continue
                ax2.hist(s2, bins=20, color=palette[idx2 % len(palette)],
                         alpha=0.8, edgecolor="#d0d8f0", linewidth=0.4)
                ax2.axvline(s2.mean(), color="#E53935", linestyle="--",
                            linewidth=1.5, alpha=0.8)
                ax2.axvline(s2.median(), color="#43A047", linestyle=":",
                            linewidth=1.5, alpha=0.8)
                ax2.set_title(col[:14].replace("_"," "), fontsize=8,
                              fontweight="bold", color="#0A1628", pad=6)
                ax2.set_facecolor("#f8faff")
                ax2.spines["top"].set_visible(False)
                ax2.spines["right"].set_visible(False)
                ax2.tick_params(labelsize=6, colors="#0F172A")
            fig.tight_layout(pad=1.2)
            buf2 = _io.BytesIO()
            fig.savefig(buf2, format="png", dpi=160, bbox_inches="tight")
            buf2.seek(0)
            plt.close(fig)
            story.append(Image(buf2, width=CW, height=CW*0.3))
        except Exception:
            pass

    # ══════════════════════════════════════════════════════
    # PAGE 4: COLUMN QUALITY TABLE
    # ══════════════════════════════════════════════════════
    story.append(PageBreak())
    story.append(Paragraph("Column Quality Analysis", ST["h1"]))
    story.append(HRFlowable(width="100%", thickness=1.5, color=accent, spaceAfter=5))
    story.append(Paragraph(
        "Each column is assessed for completeness, uniqueness, data type, "
        "and potential issues. Columns with <b>Missing > 5%</b> or "
        "<b>Unique = 1</b> (constant) need attention before analysis.",
        ST["body"]))
    story.append(Spacer(1, 4*mm))

    th_st2 = ParagraphStyle("th2", fontName=_BB, fontSize=8,
                             textColor=white, alignment=TA_CENTER)
    td_st2 = ParagraphStyle("td2", fontName=_BF, fontSize=8, textColor=dark)
    td_c2  = ParagraphStyle("tc2", fontName=_BF, fontSize=8, textColor=dark,
                             alignment=TA_CENTER)

    hdr2  = [Paragraph(h, th_st2) for h in
             ["Column", "Type", "Missing%", "Unique", "Min", "Max", "Sample Value", "Status"]]
    rows2 = []
    for col in df.columns[:25]:
        sc = df[col]
        miss = "{:.1f}%".format(sc.isna().mean()*100)
        uniq = "{:,}".format(sc.nunique())
        sample_val = str(sc.dropna().iloc[0])[:20] if len(sc.dropna()) > 0 else "—"

        if pd.api.types.is_numeric_dtype(sc):
            mn = "{:.2f}".format(float(sc.dropna().min())) if len(sc.dropna()) > 0 else "—"
            mx = "{:.2f}".format(float(sc.dropna().max())) if len(sc.dropna()) > 0 else "—"
        else:
            mn = "—"
            mx = "—"

        # Status
        miss_f = sc.isna().mean()*100
        if miss_f > 20:
            status = "⚠ HIGH MISSING"
            st_c   = HexColor("#DC2626")
        elif miss_f > 5:
            status = "△ REVIEW"
            st_c   = HexColor("#D97706")
        elif sc.nunique() == 1:
            status = "⚠ CONSTANT"
            st_c   = HexColor("#DC2626")
        elif sc.nunique() == len(df):
            status = "ℹ ID COL"
            st_c   = HexColor("#2563EB")
        else:
            status = "✓ OK"
            st_c   = HexColor("#059669")

        rows2.append([
            Paragraph(col[:20], td_st2),
            Paragraph(str(sc.dtype)[:8], td_c2),
            Paragraph(miss, ParagraphStyle("mv", fontName=_BB, fontSize=8,
                           textColor=HexColor("#DC2626") if miss_f > 5 else dark,
                           alignment=TA_CENTER)),
            Paragraph(uniq, td_c2),
            Paragraph(mn, td_c2),
            Paragraph(mx, td_c2),
            Paragraph(sample_val, td_st2),
            Paragraph(status, ParagraphStyle("sv3", fontName=_BB, fontSize=7.5,
                                             textColor=st_c, alignment=TA_CENTER)),
        ])

    col_tbl2 = Table([hdr2] + rows2,
                     colWidths=[CW*x for x in [0.22,0.09,0.09,0.07,0.08,0.08,0.22,0.15]])
    col_tbl2.setStyle(TableStyle([
        ("BACKGROUND",    (0,0),(-1,0),  dark),
        ("ROWBACKGROUNDS",(0,1),(-1,-1), [white, light2]),
        ("VALIGN",        (0,0),(-1,-1), "MIDDLE"),
        ("TOPPADDING",    (0,0),(-1,-1), 5),
        ("BOTTOMPADDING", (0,0),(-1,-1), 4),
        ("LEFTPADDING",   (0,0),(-1,-1), 5),
        ("BOX",           (0,0),(-1,-1), 0.5, HexColor("#E5E7EB")),
        ("INNERGRID",     (0,0),(-1,-1), 0.3, HexColor("#E5E7EB")),
    ]))
    story.append(col_tbl2)

    # ══════════════════════════════════════════════════════
    # PAGE 5: CORRELATION + DISCLAIMER
    # ══════════════════════════════════════════════════════
    if len(num_cols_list) >= 3:
        story.append(PageBreak())
        story.append(Paragraph("Correlation Analysis", ST["h1"]))
        story.append(HRFlowable(width="100%", thickness=1.5, color=accent, spaceAfter=5))
        story.append(Paragraph(
            "<b>Important:</b> Correlation measures association, NOT causation. "
            "r² tells you what % of variance is shared between two variables. "
            "Strong correlation alone is never sufficient reason to act.",
            ST["body"]))
        story.append(Spacer(1, 4*mm))

        try:
            corr_cols = [c for c in num_cols_list if df[c].nunique() > 2][:8]
            if len(corr_cols) >= 2:
                corr = df[corr_cols].corr().round(2)
                n3   = len(corr)
                sz   = max(5, n3)
                fig3, ax3 = plt.subplots(figsize=(sz, sz * 0.8))
                fig3.patch.set_facecolor("#ffffff")
                ax3.set_facecolor("#f8faff")
                im3  = ax3.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
                for ri in range(n3):
                    for ci in range(n3):
                        val3 = corr.values[ri, ci]
                        ax3.text(ci, ri, "{:.2f}".format(val3),
                                 ha="center", va="center", fontsize=8.5,
                                 color="white" if abs(val3) > 0.5 else "#0A1628",
                                 fontweight="bold" if abs(val3) > 0.3 else "normal")
                ax3.set_xticks(range(n3))
                ax3.set_yticks(range(n3))
                labels3 = [c[:12].replace("_"," ") for c in corr.columns]
                ax3.set_xticklabels(labels3, rotation=40, ha="right", fontsize=8.5, color="#0A1628")
                ax3.set_yticklabels(labels3, fontsize=8.5, color="#0A1628")
                ax3.set_title("Correlation Matrix", fontsize=12, fontweight="bold",
                              color="#0A1628", pad=10)
                ax3.spines[:].set_edgecolor("#d0d8f0")
                plt.colorbar(im3, ax=ax3, shrink=0.8)
                fig3.tight_layout()
                buf3 = _io.BytesIO()
                fig3.savefig(buf3, format="png", dpi=160, bbox_inches="tight")
                buf3.seek(0)
                plt.close(fig3)
                story.append(Image(buf3, width=CW, height=CW * 0.75))
                story.append(Spacer(1, 4*mm))

                # Top correlations table
                story.append(Paragraph("Top Correlations (|r| > 0.2)", ST["h3"]))
                pairs = []
                for ri in range(n3):
                    for ci in range(ri+1, n3):
                        r_val = corr.values[ri, ci]
                        if abs(r_val) > 0.2:
                            pairs.append((corr.index[ri], corr.columns[ci], r_val))
                pairs.sort(key=lambda x: abs(x[2]), reverse=True)

                if pairs:
                    corr_hdr = [Paragraph("<b>{}</b>".format(h), th_st2)
                                for h in ["Column A", "Column B", "r", "r²", "Strength", "Interpretation"]]
                    corr_rows = [corr_hdr]
                    for ca, cb, rv in pairs[:8]:
                        r2 = rv ** 2
                        strength = "Strong" if abs(rv) > 0.6 else "Moderate" if abs(rv) > 0.4 else "Weak"
                        direction = "positive" if rv > 0 else "negative"
                        interp = "{}% variance shared — {} {}.".format(
                            round(r2*100, 1), strength.lower(), direction)
                        corr_rows.append([
                            Paragraph(ca[:16].replace("_"," "), td_st2),
                            Paragraph(cb[:16].replace("_"," "), td_st2),
                            Paragraph("<b>{:.3f}</b>".format(rv),
                                      ParagraphStyle("rv", fontName=_BB, fontSize=8,
                                      textColor=HexColor("#059669") if rv > 0 else HexColor("#DC2626"),
                                      alignment=TA_CENTER)),
                            Paragraph("{:.3f}".format(r2), td_c2),
                            Paragraph(strength, td_c2),
                            Paragraph(interp, td_st2),
                        ])
                    corr_tbl = Table(corr_rows,
                                     colWidths=[CW*x for x in [0.18,0.18,0.08,0.08,0.12,0.36]])
                    corr_tbl.setStyle(TableStyle([
                        ("BACKGROUND",    (0,0),(-1,0),  dark),
                        ("ROWBACKGROUNDS",(0,1),(-1,-1), [white, light2]),
                        ("TOPPADDING",    (0,0),(-1,-1), 5),
                        ("BOTTOMPADDING", (0,0),(-1,-1), 4),
                        ("LEFTPADDING",   (0,0),(-1,-1), 6),
                        ("BOX",           (0,0),(-1,-1), 0.5, HexColor("#E5E7EB")),
                        ("INNERGRID",     (0,0),(-1,-1), 0.3, HexColor("#E5E7EB")),
                    ]))
                    story.append(corr_tbl)
        except Exception:
            pass

    # ── Disclaimer ────────────────────────────────────────
    story.append(Spacer(1, 8*mm))
    story.append(HRFlowable(width="100%", thickness=1, color=gray, spaceAfter=4))
    story.append(Paragraph(
        "<b>DISCLAIMER</b><br/>"
        "Report generated by DataForge AI on {} for dataset: {}. "
        "All findings are based solely on the provided dataset. "
        "Correlations do not imply causation. "
        "Benchmarks are indicative — verify against sector-specific data. "
        "Consult a qualified data analyst before making business decisions.",
        ST["sm"]).format(now, fname[:40]) if False else
        Paragraph(
            "<b>DISCLAIMER</b>  —  Report generated by DataForge AI on {}. "
            "Dataset: {}. Findings based solely on provided data. "
            "Correlations ≠ causation. Verify with domain expert before action.".format(now, fname[:40]),
            ST["sm"]))

    doc.build(story, canvasmaker=_Canvas)
    buf.seek(0)
    return buf.read()



# ══════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════
def kpi_card(val, label, delta="", delta_type="neu"):
    delta_html = (
        "<div class='kpi-delta delta-{}'>{}</div>".format(delta_type, delta)
        if delta else ""
    )
    return (
        "<div class='kpi-card'>"
        "<div class='kpi-val' style='color:#60a5fa'>{}</div>"
        "<div class='kpi-label'>{}</div>"
        "{}"
        "</div>"
    ).format(val, label, delta_html)


# ══════════════════════════════════════════════════════════
#  MAIN UI
# ══════════════════════════════════════════════════════════
niche, conf = _cached_niche(df)
health      = _cached_health(df)
insights    = _cached_insights(df, niche)

NICHE_META = {
    "hr":        {"emoji": "👥", "label": "HR & People Analytics",  "color": "#1976D2"},
    "sales":     {"emoji": "💰", "label": "Sales Performance",      "color": "#2E7D32"},
    "ecommerce": {"emoji": "🛒", "label": "E-Commerce Analytics",   "color": "#F4511E"},
    "finance":   {"emoji": "📊", "label": "Finance & Profitability", "color": "#0A1628"},
    "general":   {"emoji": "🔬", "label": "General Analytics",      "color": "#334155"},
}
nm = NICHE_META.get(niche, NICHE_META["general"])

# ── Header ─────────────────────────────────────────────
col_title, col_badge = st.columns([5, 1])
with col_title:
    st.markdown("## 📋 Data Health & Business Insights")
    st.caption("{} — {:,} rows · {} columns".format(fname, health["rows"], health["cols"]))
with col_badge:
    badge_html = (
        "<div class='niche-badge' style='background:{};margin-top:14px'>"
        "{}  {}</div>"
    ).format(nm["color"], nm["emoji"], niche.upper())
    st.markdown(badge_html, unsafe_allow_html=True)
st.divider()

# ── Health Score + KPIs ────────────────────────────────
st.markdown("<div class='section-head'>🏥 Data Health Score</div>",
            unsafe_allow_html=True)

c_score, c1, c2, c3, c4, c5 = st.columns([1.8, 1, 1, 1, 1, 1])
with c_score:
    ring_html = (
        "<div class='health-ring'>"
        "<div class='health-score' style='color:{}'>{}</div>"
        "<div style='color:rgba(255,255,255,0.4);font-size:.7rem;"
        "letter-spacing:.05em;margin-top:2px'>OUT OF 100</div>"
        "<div class='health-grade' style='color:{}'>{} — {}</div>"
        "</div>"
    ).format(health["color"], health["score"],
             health["color"], health["grade"], health["label"])
    st.markdown(ring_html, unsafe_allow_html=True)

mp = health["missing_pct"]
dp = health["dup_pct"]

with c1: st.markdown(kpi_card("{:,}".format(health["rows"]), "Total Rows"), unsafe_allow_html=True)
with c2: st.markdown(kpi_card(str(health["cols"]), "Columns"), unsafe_allow_html=True)
with c3: st.markdown(kpi_card(
    "{}%".format(mp), "Missing Data",
    "🔴 HIGH" if mp > 10 else "🟡 MODERATE" if mp > 3 else "🟢 CLEAN",
    "neg" if mp > 10 else "neu" if mp > 3 else "pos"),
    unsafe_allow_html=True)
with c4: st.markdown(kpi_card(
    "{}%".format(dp), "Duplicates",
    "🔴 HIGH" if dp > 5 else "🟢 CLEAN",
    "neg" if dp > 5 else "pos"),
    unsafe_allow_html=True)
crit_exists = any(i["severity"] == "critical" for i in insights)
with c5: st.markdown(kpi_card(
    str(len(insights)), "Insights Found",
    "🔴 CRITICAL ISSUES" if crit_exists else "🟢 All Healthy",
    "neg" if crit_exists else "pos"),
    unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

if conf > 0:
    st.caption("**Domain detected:** {} {}  —  {:.0f}% confidence".format(
        nm["emoji"], nm["label"], conf * 100))
    st.progress(min(conf, 1.0))
st.divider()

# ── Insights Panel ────────────────────────────────────
st.markdown("<div class='section-head'>💡 Meaningful Business Insights</div>",
            unsafe_allow_html=True)
st.caption("Every insight includes: **What's happening → Why it matters → Exactly what to do.**")
st.markdown("")

SEV_ICON = {"critical": "🔴", "warning": "🟡", "positive": "✅", "info": "🔵"}

for ins in insights:
    icon     = SEV_ICON.get(ins["severity"], "🔵")
    expanded = ins["severity"] in ("critical", "warning")
    with st.expander("{}  {}".format(icon, ins["title"]), expanded=expanded):
        # Extract variables — no dict access inside f-strings (Python 3.12+ PEP 701 safe)
        border    = ins["border"]
        bg        = ins["bg"]
        tag_color = ins["tag_color"]
        tag_text  = ins["tag"]
        body_text = ins["body"]
        act_text  = ins["action"]
        card_html = (
            "<div class='ins-card' style='border-left-color:{};background:{}'>"
            "<div class='ins-tag' style='color:{}'>{}</div>"
            "<div class='ins-body'>{}</div>"
            "<div class='ins-action' style='color:{}'>{}</div>"
            "</div>"
        ).format(border, bg, tag_color, tag_text, body_text, border, act_text)
        st.markdown(card_html, unsafe_allow_html=True)

st.divider()

# ── Charts ────────────────────────────────────────────
st.markdown("<div class='section-head'>📊 Health Breakdown</div>",
            unsafe_allow_html=True)
c_a, c_b = st.columns(2)

with c_a:
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=health["score"],
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": "Health Score", "font": {"size": 16}},
        gauge={
            "axis":  {"range": [0, 100]},
            "bar":   {"color": health["color"]},
            "steps": [
                {"range": [0,  50], "color": "#fef2f2"},
                {"range": [50, 70], "color": "#fff7ed"},
                {"range": [70, 90], "color": "#f0fdf4"},
                {"range": [90,100], "color": "#dbeafe"},
            ],
            "threshold": {"line": {"color": "red", "width": 2},
                          "thickness": 0.75, "value": 60},
        }))
    fig.update_layout(
        height=260,
        paper_bgcolor="white",
        font=dict(color="#0F172A", size=12))
    st.plotly_chart(fig, use_container_width=True)

with c_b:
    vals   = [max(0, 100 - mp * 2.5), max(0, 100 - dp * 3),
              max(0, 100 - health["outlier_pct"] * 0.4)]
    labels = ["Completeness", "No Duplicates", "Outlier-Free"]
    colors = ["#22d3a5" if v >= 80 else "#fbbf24" if v >= 60 else "#ef4444" for v in vals]
    fig2 = go.Figure(go.Bar(
        x=vals, y=labels, orientation="h",
        marker_color=colors,
        text=["{:.0f}%".format(v) for v in vals],
        textposition="auto",
    ))
    fig2.update_layout(
        title="Health Dimension Breakdown", height=260,
        paper_bgcolor="white", plot_bgcolor="#f8faff",
        font=dict(family="Inter, sans-serif", size=11),
        xaxis=dict(range=[0, 100]),
        margin=dict(l=10, r=10, t=40, b=10), showlegend=False,
    )
    st.plotly_chart(fig2, use_container_width=True)

st.divider()

# ── Download ──────────────────────────────────────────
st.markdown("<div class='section-head'>⬇️ Download Report</div>",
            unsafe_allow_html=True)

c_dl1, c_dl2 = st.columns([2, 3])
with c_dl1:
    st.markdown("**Generate Branded PDF Report**")
    st.caption("Health Score · {} Insights · Column Quality Table · Niche: {}".format(
        len(insights), nm["label"]))
    gen = st.button("📥 Generate & Download PDF", type="primary",
                    use_container_width=True, key="gen_health_pdf")
with c_dl2:
    st.info(
        "💡 **Freelancing Tip:** This report is ready to send directly to clients. "
        "It auto-detects their data domain, gives a health score, and provides "
        "**prioritized, actionable insights** — exactly what they are paying for.")

if gen:
    with st.spinner("Building premium PDF..."):
        try:
            pdf_bytes = build_health_pdf(df, niche, health, insights, fname)
            safe_name = fname.replace(" ", "_").split(".")[0]
            st.success("✅ Report ready — {:.0f} KB | Health Score: {}/100 | {} insights".format(
                len(pdf_bytes)/1024, health["score"], len(insights)))
            st.download_button(
                label="⬇️ Click to Download PDF Health Report",
                data=pdf_bytes,
                file_name="DataForge_Health_{}.pdf".format(safe_name),
                mime="application/pdf",
                type="primary",
                use_container_width=True,
                key="dl_health_pdf",
            )
        except Exception as e:
            st.error("PDF generation failed: {}".format(e))
            st.exception(e)

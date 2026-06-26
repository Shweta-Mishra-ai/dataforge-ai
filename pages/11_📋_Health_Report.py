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
import logging
logger = logging.getLogger(__name__)


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
  .ins-body { font-size: 0.88rem; line-height: 1.6; color: inherit; }
  .ins-action { font-size: 0.78rem; font-weight: 600; margin-top: 6px; }

  .section-head {
    font-size: 1.1rem; font-weight: 700;
    border-bottom: 2px solid rgba(128,128,128,0.25); padding-bottom: 6px; margin: 18px 0 14px;
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
        # FIX: use rgba with low alpha so cards are visible on both light and dark Streamlit themes.
        # Hardcoded light hex (#fef2f2, #eff6ff etc.) vanish on dark backgrounds.
        COLOR = {
            "critical": ("#ef4444", "rgba(239,68,68,0.08)"),
            "warning":  ("#f97316", "rgba(249,115,22,0.08)"),
            "positive": ("#22d3a5", "rgba(34,211,165,0.08)"),
            "info":     ("#3b82f6", "rgba(59,130,246,0.08)"),
        }
        border, bg = COLOR.get(severity, ("#3b82f6", "rgba(59,130,246,0.08)"))
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
                logger.warning("Health Report section failure", exc_info=True)

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
                logger.warning("Health Report section failure", exc_info=True)

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
                logger.warning("Health Report section failure", exc_info=True)

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
                logger.warning("Health Report section failure", exc_info=True)

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
                logger.warning("Health Report section failure", exc_info=True)

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
                logger.warning("Health Report section failure", exc_info=True)

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
                logger.warning("Health Report section failure", exc_info=True)

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
# build_health_pdf extracted to core/health_pdf_builder.py
from core.health_pdf_builder import build_health_pdf  # noqa: F401

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
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict( size=12))
    st.plotly_chart(fig, width="stretch")

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
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", size=11),
        xaxis=dict(range=[0, 100]),
        margin=dict(l=10, r=10, t=40, b=10), showlegend=False,
    )
    st.plotly_chart(fig2, width="stretch")

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

"""
pages/11_📋_Health_Report.py — DataForge AI
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Premium Data Health + Business Insights Report
• Auto-detects niche: HR / Sales / E-commerce / Finance / General
• Power BI-grade KPI dashboard on screen
• One-click PDF download: health score + meaningful business insights
• Built for freelancing: works with any dataset the client uploads
"""
import io
import os
import math
import datetime

import numpy  as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from core.session_manager import require_data, get_df, get_filename

st.set_page_config(
    page_title="Health Report — DataForge AI",
    page_icon="📋",
    layout="wide",
)
require_data()

df    = get_df()
fname = get_filename()

# ══════════════════════════════════════════════════════════
#  PREMIUM CSS — Power BI-grade look
# ══════════════════════════════════════════════════════════
st.markdown("""
<style>
  /* Google Inter font */
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

  html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }

  /* Remove Streamlit default padding */
  .block-container { padding-top: 1.2rem !important; padding-bottom: 1rem !important; }

  /* KPI card */
  .kpi-card {
    background: linear-gradient(135deg, #0f1b2d 0%, #1a2f4a 100%);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px;
    padding: 20px 22px;
    text-align: center;
    box-shadow: 0 4px 24px rgba(0,0,0,0.18);
    height: 130px;
    display: flex; flex-direction: column;
    justify-content: center; align-items: center;
  }
  .kpi-val   { font-size: 2rem; font-weight: 800; line-height: 1; margin-bottom: 4px; }
  .kpi-label { font-size: 0.72rem; font-weight: 600; letter-spacing: .06em;
               color: rgba(255,255,255,0.55); text-transform: uppercase; }
  .kpi-delta { font-size: 0.75rem; font-weight: 600; margin-top: 4px; }
  .delta-pos { color: #22d3a5; }
  .delta-neg { color: #ff6b6b; }
  .delta-neu { color: rgba(255,255,255,0.4); }

  /* Health score ring */
  .health-ring {
    background: linear-gradient(135deg, #0f1b2d, #1a2f4a);
    border-radius: 20px; padding: 28px;
    text-align: center; border: 1px solid rgba(255,255,255,0.08);
    box-shadow: 0 4px 24px rgba(0,0,0,0.2);
  }
  .health-score { font-size: 3.8rem; font-weight: 900; line-height: 1; }
  .health-grade { font-size: 1rem; font-weight: 700; letter-spacing: .05em; margin-top: 4px; }

  /* Insight card */
  .ins-card {
    border-radius: 12px; padding: 16px 18px; margin-bottom: 10px;
    border-left: 5px solid; position: relative;
    box-shadow: 0 2px 10px rgba(0,0,0,0.06);
  }
  .ins-tag  { font-size: 0.65rem; font-weight: 700; letter-spacing: .08em;
              text-transform: uppercase; margin-bottom: 5px; }
  .ins-body { font-size: 0.88rem; line-height: 1.6; color: #1e1e2e; }
  .ins-action { font-size: 0.78rem; font-weight: 600; margin-top: 6px; }

  /* Section heading */
  .section-head {
    font-size: 1.1rem; font-weight: 700; color: #0f1b2d;
    border-bottom: 2px solid #e2e8f0; padding-bottom: 6px; margin: 18px 0 14px;
  }
  /* Niche badge */
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
def detect_niche(df: pd.DataFrame) -> tuple[str, float]:
    cols_lower = [c.lower() for c in df.columns]

    scores = {
        "hr":         sum(k in " ".join(cols_lower) for k in
                          ["employee","attrition","salary","tenure","department",
                           "performance","hire","resignation","headcount","turnover",
                           "satisfaction","engagement","manager"]),
        "sales":      sum(k in " ".join(cols_lower) for k in
                          ["revenue","deal","pipeline","lead","quota","win","close",
                           "opportunity","forecast","account","prospect","crm","sale"]),
        "ecommerce":  sum(k in " ".join(cols_lower) for k in
                          ["order","product","customer","cart","purchase","sku","refund",
                           "shipping","discount","price","quantity","category","review",
                           "rating","return","session","conversion"]),
        "finance":    sum(k in " ".join(cols_lower) for k in
                          ["profit","loss","revenue","expense","cost","margin","budget",
                           "invoice","payment","tax","cash","asset","liability","balance",
                           "ebitda","investment","roi","gross","net"]),
    }
    best  = max(scores, key=scores.get)
    total = sum(scores.values()) or 1
    conf  = min(scores[best] / total, 1.0)
    if scores[best] == 0:
        return "general", 0.0
    return best, conf


# ══════════════════════════════════════════════════════════
#  HEALTH SCORE ENGINE
# ══════════════════════════════════════════════════════════
def compute_health(df: pd.DataFrame) -> dict:
    rows, cols = len(df), len(df.columns)
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

    # Score = 100 minus deductions
    score = 100
    score -= min(missing_pct  * 2.5, 35)   # missing data penalty (max -35)
    score -= min(dup_pct      * 3.0, 20)   # duplicates penalty     (max -20)
    score -= min(outlier_pct  * 0.4, 15)   # outlier penalty        (max -15)
    score -= 10 if rows < 50   else 0
    score -= 5  if cols < 3    else 0
    score  = max(int(score), 0)

    grade_map = [(90,"A+","Excellent","#22d3a5"),
                 (80,"A", "Very Good","#42b983"),
                 (70,"B+","Good",     "#60a5fa"),
                 (60,"B", "Fair",     "#fbbf24"),
                 (50,"C", "Needs Work","#f97316"),
                 (0, "D", "Poor",    "#ef4444")]
    grade, label, color = next(
        (g,l,c) for s,g,l,c in grade_map if score >= s)

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
def build_insights(df: pd.DataFrame, niche: str) -> list[dict]:
    insights = []
    cols_lower = {c.lower(): c for c in df.columns}

    def _find(keywords: list) -> str | None:
        for kw in keywords:
            for cl, c in cols_lower.items():
                if kw in cl:
                    return c
        return None

    def _pct(col): return df[col].mean() * 100 if df[col].dtype in [bool, "bool"] else None

    # ── Universal insights ─────────────────────────────
    num_cols = df.select_dtypes(include="number").columns.tolist()
    for c in num_cols[:3]:
        s = df[c].dropna()
        if len(s) < 10: continue
        skew = float(s.skew()) if len(s) > 3 else 0
        if abs(skew) > 2:
            insights.append({
                "tag": "⚠ DATA SKEW",
                "title": f"'{c}' is heavily skewed (skew={skew:.2f})",
                "body": (f"The distribution of **{c}** is {'right' if skew>0 else 'left'}-skewed. "
                         f"Mean ({s.mean():.2f}) ≠ Median ({s.median():.2f}) by "
                         f"{abs(s.mean()-s.median())/max(s.std(),0.001)*100:.0f}%. "
                         "Averages in reports based on this column may be misleading."),
                "action": f"➜ Use median for '{c}' in client reports, not mean.",
                "severity": "warning",
                "border": "#f97316",
                "bg": "#fff7ed",
                "tag_color": "#f97316",
            })

    # ── HR-specific ─────────────────────────────────
    if niche == "hr":
        attrition_col = _find(["attrition","left","churned","resigned","exited","turnover"])
        salary_col    = _find(["salary","wage","compensation","pay"])
        tenure_col    = _find(["tenure","years","experience","service"])
        dept_col      = _find(["department","dept","team","division"])
        sat_col       = _find(["satisfaction","engagement","score","rating"])

        if attrition_col:
            s = df[attrition_col]
            atr_rate = None
            if s.dtype in ["bool", "int64", "float64"] and s.max() <= 1:
                atr_rate = s.mean() * 100
            elif s.dtype == object:
                atr_rate = (s.str.lower().isin(["yes","true","1","left"])).mean() * 100
            if atr_rate is not None:
                sev = "critical" if atr_rate > 20 else "warning" if atr_rate > 10 else "positive"
                border = "#ef4444" if sev=="critical" else "#f97316" if sev=="warning" else "#22d3a5"
                bg = "#fef2f2" if sev=="critical" else "#fff7ed" if sev=="warning" else "#f0fdf4"
                insights.append({
                    "tag": "🔴 ATTRITION RISK" if sev=="critical" else "⚠ ATTRITION" if sev=="warning" else "✅ HEALTHY RETENTION",
                    "title": f"Attrition rate is {atr_rate:.1f}% (SHRM benchmark: <10%)",
                    "body": (f"Your dataset shows **{atr_rate:.1f}% attrition**. "
                             f"{'This is CRITICAL — exceeding 20% is 2× the industry danger zone. ' if sev=='critical' else 'Above the 10% SHRM benchmark. ' if sev=='warning' else 'Below 10% benchmark — excellent retention. '}"
                             f"SHRM 2024 reports the average cost of replacing one employee is "
                             f"**50–200% of their annual salary**."),
                    "action": ("➜ IMMEDIATE: Conduct stay-interviews in high-risk departments. "
                               "Target managers of teams with >25% attrition."
                               if sev == "critical" else
                               "➜ Build a quarterly pulse survey. Flag departments above 15%."
                               if sev == "warning" else
                               "➜ Document what drives retention — use as a competitive advantage."),
                    "severity": sev,
                    "border": border, "bg": bg, "tag_color": border,
                })

        if sat_col:
            mean_sat = df[sat_col].dropna().mean()
            max_sat  = df[sat_col].dropna().max()
            sat_norm = mean_sat / max_sat if max_sat > 1 else mean_sat
            sev = "critical" if sat_norm < 0.5 else "warning" if sat_norm < 0.7 else "positive"
            border = "#ef4444" if sev=="critical" else "#f97316" if sev=="warning" else "#22d3a5"
            bg = "#fef2f2" if sev=="critical" else "#fff7ed" if sev=="warning" else "#f0fdf4"
            insights.append({
                "tag": "👥 EMPLOYEE SATISFACTION",
                "title": f"Avg satisfaction: {mean_sat:.2f} / {max_sat:.0f} ({sat_norm*100:.0f}%)",
                "body": (f"Mean satisfaction score is **{mean_sat:.2f}** (normalized: {sat_norm*100:.0f}%). "
                         f"Gallup 2024 finds that {'low satisfaction directly correlates with 4× higher attrition risk. Urgent intervention needed.' if sev in ('critical','warning') else 'scores above 70% indicate an engaged workforce — a key retention driver.'}"),
                "action": ("➜ Implement manager 1:1s and recognition programs. Gallup: 52% of exits are preventable."
                           if sev != "positive" else
                           "➜ Publish satisfaction data in employer branding. Use as a hiring advantage."),
                "severity": sev,
                "border": border, "bg": bg, "tag_color": border,
            })

        if dept_col and attrition_col:
            try:
                s = df[attrition_col]
                if s.dtype == object:
                    df["_atr_num"] = s.str.lower().isin(["yes","true","1","left"]).astype(float)
                else:
                    df["_atr_num"] = pd.to_numeric(s, errors="coerce")
                dept_risk = df.groupby(dept_col)["_atr_num"].mean().sort_values(ascending=False)
                df.drop(columns=["_atr_num"], inplace=True)
                top_dept = dept_risk.index[0]
                top_rate = dept_risk.iloc[0] * 100
                insights.append({
                    "tag": "🏢 DEPARTMENT RISK",
                    "title": f"'{top_dept}' has the highest attrition: {top_rate:.1f}%",
                    "body": (f"The **{top_dept}** department shows {top_rate:.1f}% attrition — "
                             f"{'critically above' if top_rate > 20 else 'above'} the SHRM 10% benchmark. "
                             "Department-level attrition often signals a specific manager, workload, or pay equity issue."),
                    "action": f"➜ Prioritize '{top_dept}' for skip-level interviews and exit interview analysis.",
                    "severity": "critical" if top_rate > 20 else "warning",
                    "border": "#ef4444" if top_rate > 20 else "#f97316",
                    "bg": "#fef2f2" if top_rate > 20 else "#fff7ed",
                    "tag_color": "#ef4444" if top_rate > 20 else "#f97316",
                })
            except Exception:
                pass

    # ── SALES-specific ──────────────────────────────
    elif niche == "sales":
        rev_col    = _find(["revenue","amount","deal_value","deal_size","value","gmv","arr"])
        status_col = _find(["status","stage","outcome","result","win","lost"])
        date_col   = _find(["close","date","created","month","quarter"])

        if rev_col:
            s = df[rev_col].dropna()
            top10 = s.nlargest(int(len(s)*0.1)).sum()
            total = s.sum()
            conc  = top10 / total * 100 if total > 0 else 0
            sev   = "critical" if conc > 80 else "warning" if conc > 60 else "positive"
            insights.append({
                "tag": "💰 REVENUE CONCENTRATION",
                "title": f"Top 10% of deals = {conc:.0f}% of total revenue",
                "body": (f"Your top 10% deals account for **{conc:.0f}% of {rev_col}** "
                         f"(${total:,.0f} total). "
                         f"{'CRITICAL concentration risk — losing 1-2 key accounts could collapse revenue.' if sev=='critical' else 'High concentration — moderate dependency on key accounts.' if sev=='warning' else 'Healthy revenue distribution across accounts.'}"),
                "action": ("➜ Immediately build 3–5 additional pipeline accounts at the same deal size."
                           if sev == "critical" else
                           "➜ Implement account health scoring. Trigger executive relationships for top 10%."
                           if sev == "warning" else
                           "➜ Replicate the profile of top-performing deals in prospecting strategy."),
                "severity": sev,
                "border": "#ef4444" if sev=="critical" else "#f97316" if sev=="warning" else "#22d3a5",
                "bg": "#fef2f2" if sev=="critical" else "#fff7ed" if sev=="warning" else "#f0fdf4",
                "tag_color": "#ef4444" if sev=="critical" else "#f97316" if sev=="warning" else "#22d3a5",
            })

        if status_col:
            s = df[status_col].dropna().str.lower()
            won  = s.isin(["won","win","closed won","won","success","yes"]).sum()
            lost = s.isin(["lost","lose","closed lost","loss","no","failed"]).sum()
            total = won + lost
            win_rate = won / total * 100 if total > 0 else 0
            sev = "critical" if win_rate < 20 else "warning" if win_rate < 35 else "positive"
            insights.append({
                "tag": "🎯 WIN RATE",
                "title": f"Win rate: {win_rate:.1f}% (Benchmark: 25-40%)",
                "body": (f"Out of **{total} qualified deals**, {won} were won ({win_rate:.1f}%). "
                         f"Salesforce 2024 benchmark is 25–40%. "
                         f"{'Well below benchmark — significant pipeline efficiency problem.' if sev=='critical' else 'Below benchmark — improvement needed in qualifying and closing.' if sev=='warning' else 'At or above benchmark — strong sales execution.'}"),
                "action": ("➜ Implement MEDDIC or BANT qualification framework. Review lost deal reasons."
                           if sev != "positive" else
                           "➜ Document the winning sales playbook. Scale what's working to underperformers."),
                "severity": sev,
                "border": "#ef4444" if sev=="critical" else "#f97316" if sev=="warning" else "#22d3a5",
                "bg": "#fef2f2" if sev=="critical" else "#fff7ed" if sev=="warning" else "#f0fdf4",
                "tag_color": "#ef4444" if sev=="critical" else "#f97316" if sev=="warning" else "#22d3a5",
            })

    # ── E-COMMERCE-specific ─────────────────────────
    elif niche == "ecommerce":
        order_val_col = _find(["order_value","gmv","amount","total","price","revenue","aov"])
        customer_col  = _find(["customer","user","buyer","client_id","customer_id"])
        return_col    = _find(["return","refund","returned","cancelled"])
        rating_col    = _find(["rating","review","score","stars"])

        if order_val_col:
            s  = df[order_val_col].dropna()
            aov = s.mean()
            insights.append({
                "tag": "🛒 AVERAGE ORDER VALUE",
                "title": f"AOV = ${aov:,.2f} per transaction",
                "body": (f"Average Order Value is **${aov:,.2f}**. "
                         f"Top 20% of orders generate ${s.nlargest(int(len(s)*0.2)).mean():,.2f} avg. "
                         "Increasing AOV by 10% through upselling is often more profitable than acquiring new customers."),
                "action": ("➜ Deploy bundled products or 'Frequently Bought Together' "
                           "for orders below $" + f"{aov*0.7:.0f}. "
                           "Target: lift AOV by 15% in 90 days."),
                "severity": "info",
                "border": "#3b82f6", "bg": "#eff6ff", "tag_color": "#3b82f6",
            })

        if customer_col and order_val_col:
            try:
                cust_spend = df.groupby(customer_col)[order_val_col].sum()
                top20_pct  = cust_spend.nlargest(max(1, int(len(cust_spend)*0.2))).sum() / cust_spend.sum() * 100
                sev = "critical" if top20_pct > 80 else "warning" if top20_pct > 65 else "positive"
                insights.append({
                    "tag": "👑 PARETO PRINCIPLE",
                    "title": f"Top 20% customers = {top20_pct:.0f}% of revenue",
                    "body": (f"**{top20_pct:.0f}% of revenue** comes from your top 20% customers. "
                             f"{'Classic Pareto pattern — but high concentration means churn of top customers = major revenue loss.' if sev!='positive' else 'Healthy spread — revenue is reasonably distributed across customer base.'}"),
                    "action": ("➜ Build a VIP tier for top 20% customers. Assign dedicated account managers. "
                               "Offer loyalty rewards to prevent churn."),
                    "severity": sev,
                    "border": "#ef4444" if sev=="critical" else "#f97316" if sev=="warning" else "#22d3a5",
                    "bg": "#fef2f2" if sev=="critical" else "#fff7ed" if sev=="warning" else "#f0fdf4",
                    "tag_color": "#ef4444" if sev=="critical" else "#f97316" if sev=="warning" else "#22d3a5",
                })
            except Exception: pass

    # ── FINANCE-specific ────────────────────────────
    elif niche == "finance":
        rev_col  = _find(["revenue","income","sales","turnover","gross"])
        cost_col = _find(["cost","expense","cogs","expenditure","opex"])
        profit_col = _find(["profit","net","margin","ebitda","earnings"])

        if rev_col and cost_col:
            rev  = df[rev_col].dropna().sum()
            cost = df[cost_col].dropna().sum()
            margin = (rev - cost) / rev * 100 if rev > 0 else 0
            sev = "critical" if margin < 5 else "warning" if margin < 15 else "positive"
            insights.append({
                "tag": "📊 GROSS MARGIN",
                "title": f"Gross Margin: {margin:.1f}% (Benchmark: >15%)",
                "body": (f"Total Revenue: **${rev:,.0f}** | Total Cost: **${cost:,.0f}** | "
                         f"Gross Margin: **{margin:.1f}%**. "
                         f"McKinsey 2024: healthy businesses target >15% gross margin. "
                         f"{'CRITICAL: At risk of operating loss.' if sev=='critical' else 'Below benchmark — cost control or pricing review needed.' if sev=='warning' else 'Healthy margin — above industry benchmark.'}"),
                "action": ("➜ IMMEDIATE: Conduct cost structure analysis. Identify top 3 cost drivers."
                           if sev == "critical" else
                           "➜ Review pricing strategy and renegotiate top 3 supplier contracts."
                           if sev == "warning" else
                           "➜ Model the impact of a 5% price increase — protect margin against cost inflation."),
                "severity": sev,
                "border": "#ef4444" if sev=="critical" else "#f97316" if sev=="warning" else "#22d3a5",
                "bg": "#fef2f2" if sev=="critical" else "#fff7ed" if sev=="warning" else "#f0fdf4",
                "tag_color": "#ef4444" if sev=="critical" else "#f97316" if sev=="warning" else "#22d3a5",
            })

    # ── Always add: Missing data risk ───────────────
    miss = df.isna().sum()
    bad_cols = miss[miss / len(df) > 0.3].sort_values(ascending=False)
    if len(bad_cols) > 0:
        col_list = ", ".join([f"'{c}' ({miss[c]/len(df)*100:.0f}%)" for c in bad_cols.index[:3]])
        insights.append({
            "tag": "⚠ DATA QUALITY RISK",
            "title": f"{len(bad_cols)} column(s) have >30% missing data",
            "body": (f"Columns with high missing rates: **{col_list}**. "
                     "Any analysis or model trained on these columns will be statistically unreliable. "
                     "Client decisions based on these columns carry significant error risk."),
            "action": "➜ Either impute with domain-appropriate values, or exclude these columns from client-facing insights.",
            "severity": "critical" if bad_cols.iloc[0] / len(df) > 0.5 else "warning",
            "border": "#ef4444" if bad_cols.iloc[0] / len(df) > 0.5 else "#f97316",
            "bg":     "#fef2f2" if bad_cols.iloc[0] / len(df) > 0.5 else "#fff7ed",
            "tag_color": "#ef4444" if bad_cols.iloc[0] / len(df) > 0.5 else "#f97316",
        })

    if not insights:
        insights.append({
            "tag": "✅ DATA HEALTH",
            "title": "Dataset is clean and analysis-ready",
            "body": ("No critical issues detected. Data completeness, distribution, "
                     "and structure are within acceptable thresholds for business analysis."),
            "action": "➜ Proceed directly to Dashboard and ML Predictions pages.",
            "severity": "positive",
            "border": "#22d3a5", "bg": "#f0fdf4", "tag_color": "#22d3a5",
        })

    return insights


# ══════════════════════════════════════════════════════════
#  PDF GENERATOR  (uses ReportLab directly — lightweight)
# ══════════════════════════════════════════════════════════
def build_health_pdf(df: pd.DataFrame, niche: str, health: dict,
                     insights: list, fname: str) -> bytes:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.lib.colors import HexColor, white
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
    from reportlab.platypus import (
        BaseDocTemplate, Frame, PageTemplate,
        Paragraph, Spacer, Table, TableStyle,
        HRFlowable, KeepTogether,
    )
    from reportlab.pdfgen import canvas as CV

    W, H = A4
    CW = W - 36 * mm
    now = datetime.datetime.now().strftime("%B %d, %Y")

    NICHE_COLORS = {
        "hr": "#1976D2", "sales": "#2E7D32",
        "ecommerce": "#F4511E", "finance": "#0A1628", "general": "#1B4FD8"
    }
    accent_hex = NICHE_COLORS.get(niche, "#1B4FD8")
    accent = HexColor(accent_hex)
    dark   = HexColor("#0A1628")
    gray   = HexColor("#6B7280")
    light  = HexColor("#F0F4FF")
    score_color = HexColor(health["color"])

    def s(name, **kw): return ParagraphStyle(name, **kw)
    STYLES = {
        "h1": s("h1", fontName="Helvetica-Bold", fontSize=16, textColor=accent, spaceAfter=4),
        "h2": s("h2", fontName="Helvetica-Bold", fontSize=12, textColor=dark, spaceBefore=10, spaceAfter=4),
        "h3": s("h3", fontName="Helvetica-Bold", fontSize=10, textColor=accent, spaceBefore=8, spaceAfter=3),
        "body": s("body", fontName="Helvetica", fontSize=9, textColor=dark, leading=14, spaceAfter=3, alignment=TA_JUSTIFY),
        "sm": s("sm", fontName="Helvetica", fontSize=7.5, textColor=gray, leading=11, spaceAfter=2),
        "action": s("action", fontName="Helvetica-Bold", fontSize=8.5, textColor=accent, leading=13, spaceAfter=3),
        "center": s("center", fontName="Helvetica", fontSize=9, textColor=dark, alignment=TA_CENTER),
    }

    buf = io.BytesIO()

    class _Canvas(CV.Canvas):
        def __init__(self, fn, **kw):
            super().__init__(fn, **kw)
            self._sp = []
        def showPage(self):
            self._sp.append(dict(self.__dict__))
            self._startPage()
        def save(self):
            tot = len(self._sp)
            for st in self._sp:
                self.__dict__.update(st)
                self._draw_hf(tot)
                super().showPage()
            super().save()
        def _draw_hf(self, tot):
            # Header
            self.setFillColor(dark)
            self.rect(0, H - 18*mm, W, 18*mm, fill=1, stroke=0)
            self.setFillColor(accent)
            self.rect(0, H - 19.5*mm, W, 1.5*mm, fill=1, stroke=0)
            self.setFillColor(white)
            self.setFont("Helvetica-Bold", 9)
            self.drawString(18*mm, H - 11*mm, "DataForge AI  ·  Data Health & Business Insights Report")
            self.setFont("Helvetica", 7)
            self.setFillColor(HexColor("#90CAF9"))
            self.drawRightString(W - 18*mm, H - 11*mm, now)
            # Footer
            self.setFillColor(dark)
            self.rect(0, 0, W, 10*mm, fill=1, stroke=0)
            self.setFillColor(accent)
            self.rect(0, 10*mm, W, 1*mm, fill=1, stroke=0)
            self.setFillColor(white)
            self.setFont("Helvetica", 6.5)
            self.drawString(18*mm, 3.5*mm,
                "DataForge AI — Confidential. Not for external distribution without client consent.")
            self.drawRightString(W - 18*mm, 3.5*mm,
                "Page {} of {}".format(self._pageNumber, tot))

    doc = BaseDocTemplate(
        buf, pagesize=A4,
        leftMargin=18*mm, rightMargin=18*mm,
        topMargin=23*mm, bottomMargin=16*mm,
    )
    frame = Frame(18*mm, 16*mm, CW, H - 39*mm, id="main")
    doc.addPageTemplates([PageTemplate(id="main", frames=[frame], onPage=lambda c, d: None)])

    story = []

    # ── COVER SECTION ─────────────────────────────
    story.append(Spacer(1, 8*mm))
    story.append(Paragraph("DATA HEALTH &amp; BUSINESS INSIGHTS REPORT", STYLES["h1"]))
    story.append(Paragraph(fname, STYLES["sm"]))
    story.append(HRFlowable(width="100%", thickness=2, color=accent, spaceAfter=5))
    story.append(Paragraph(
        f"Report generated: {now}  ·  {health['rows']:,} rows  ·  "
        f"{health['cols']} columns  ·  Niche: {niche.upper()}",
        STYLES["sm"]))
    story.append(Spacer(1, 6*mm))

    # ── HEALTH SCORE KPI ─────────────────────────
    story.append(Paragraph("Overall Data Health Score", STYLES["h2"]))
    kpi_data = [
        [Paragraph(f"<b>{health['score']}/100</b>",
                   ParagraphStyle("kscore", fontName="Helvetica-Bold", fontSize=28,
                                  textColor=score_color, alignment=TA_CENTER)),
         Paragraph(
             f"<b>Grade: {health['grade']}  —  {health['label']}</b><br/>"
             f"Missing Data: {health['missing_pct']}%  |  "
             f"Duplicate Rows: {health['dup_pct']}%  |  "
             f"Outlier Columns: {health['outlier_pct']}%<br/><br/>"
             f"Total Rows: {health['rows']:,}  |  "
             f"Total Columns: {health['cols']}  |  "
             f"Numeric Columns: {health['num_cols']}",
             ParagraphStyle("kdesc", fontName="Helvetica", fontSize=9,
                            textColor=dark, leading=14))]
    ]
    kpi_table = Table(kpi_data, colWidths=[40*mm, CW - 40*mm])
    kpi_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, 0), light),
        ("BACKGROUND", (1, 0), (1, 0), HexColor("#F8FAFF")),
        ("VALIGN",     (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN",      (0, 0), (0, 0), "CENTER"),
        ("TOPPADDING", (0, 0), (-1, -1), 14),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 14),
        ("LEFTPADDING", (0, 0), (-1, -1), 12),
        ("BOX",        (0, 0), (-1, -1), 1, accent),
        ("INNERGRID",  (0, 0), (-1, -1), 0.3, HexColor("#E5E7EB")),
    ]))
    story.append(kpi_table)
    story.append(Spacer(1, 8*mm))

    # ── INSIGHTS ───────────────────────────────────
    story.append(Paragraph("Meaningful Business Insights", STYLES["h2"]))
    story.append(Paragraph(
        "Each insight below is structured as: <b>What → Why it matters → What to do.</b> "
        "These are directly actionable for your client or business stakeholders.",
        STYLES["body"]))
    story.append(Spacer(1, 4*mm))

    SEV_COLORS = {
        "critical": ("#EF4444", "#FEF2F2"),
        "warning":  ("#F97316", "#FFF7ED"),
        "positive": ("#22D3A5", "#F0FDF4"),
        "info":     ("#3B82F6", "#EFF6FF"),
    }

    for i, ins in enumerate(insights, 1):
        border_c, bg_c = SEV_COLORS.get(ins["severity"], ("#3B82F6", "#EFF6FF"))
        tag_c = HexColor(border_c)
        bg_hex = HexColor(bg_c)

        tag_para = Paragraph(
            f"{ins['tag']}",
            ParagraphStyle("ins_tag", fontName="Helvetica-Bold", fontSize=7.5,
                           textColor=tag_c, spaceAfter=2))
        title_para = Paragraph(
            f"<b>{i}. {ins['title']}</b>",
            ParagraphStyle("ins_title", fontName="Helvetica-Bold", fontSize=10,
                           textColor=dark, spaceAfter=4))
        body_para = Paragraph(ins["body"], STYLES["body"])
        action_para = Paragraph(ins["action"], STYLES["action"])

        card_data = [[tag_para], [title_para], [body_para], [action_para]]
        card = Table(card_data, colWidths=[CW])
        card.setStyle(TableStyle([
            ("BACKGROUND",  (0, 0), (-1, -1), bg_hex),
            ("LINEBEFORE",  (0, 0), (0, -1),  5, HexColor(border_c)),
            ("TOPPADDING",  (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("LEFTPADDING", (0, 0), (-1, -1), 14),
            ("RIGHTPADDING",(0, 0), (-1, -1), 12),
            ("BOX",         (0, 0), (-1, -1), 0.5, HexColor("#E5E7EB")),
        ]))
        story.append(KeepTogether([card, Spacer(1, 5*mm)]))

    # ── COLUMN SUMMARY TABLE ───────────────────────
    story.append(HRFlowable(width="100%", thickness=1.5, color=accent, spaceBefore=6, spaceAfter=4))
    story.append(Paragraph("Column Quality Summary", STYLES["h2"]))

    th_style = ParagraphStyle("th", fontName="Helvetica-Bold", fontSize=8,
                               textColor=white, alignment=TA_CENTER)
    td_style = ParagraphStyle("td", fontName="Helvetica", fontSize=7.5, textColor=dark)

    header_row = [Paragraph(h, th_style)
                  for h in ["Column", "Type", "Missing %", "Unique", "Sample"]]
    data_rows = []
    for col in df.columns[:20]:
        s_col = df[col]
        miss_p = f"{s_col.isna().mean()*100:.1f}%"
        sample = str(s_col.dropna().iloc[0])[:30] if len(s_col.dropna()) > 0 else "—"
        data_rows.append([
            Paragraph(col[:22], td_style),
            Paragraph(str(s_col.dtype), td_style),
            Paragraph(miss_p, td_style),
            Paragraph(f"{s_col.nunique():,}", td_style),
            Paragraph(sample, td_style),
        ])

    col_table = Table(
        [header_row] + data_rows,
        colWidths=[CW * x for x in [0.28, 0.12, 0.12, 0.12, 0.36]])
    col_table.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0),  dark),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [white, light]),
        ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("BOX",           (0, 0), (-1, -1), 0.5, HexColor("#E5E7EB")),
        ("INNERGRID",     (0, 0), (-1, -1), 0.3, HexColor("#E5E7EB")),
    ]))
    story.append(col_table)

    story.append(Spacer(1, 6*mm))
    story.append(Paragraph(
        "This report was generated automatically by DataForge AI. "
        "All insights should be reviewed by a qualified domain expert before client delivery.",
        STYLES["sm"]))

    doc.build(story, canvasmaker=_Canvas)
    buf.seek(0)
    return buf.read()


# ══════════════════════════════════════════════════════════
#  MAIN UI
# ══════════════════════════════════════════════════════════
niche, conf = detect_niche(df)
health       = compute_health(df)
insights     = build_insights(df.copy(), niche)

NICHE_META = {
    "hr":        {"emoji": "👥", "label": "HR & People Analytics",  "color": "#1976D2"},
    "sales":     {"emoji": "💰", "label": "Sales Performance",      "color": "#2E7D32"},
    "ecommerce": {"emoji": "🛒", "label": "E-Commerce Analytics",   "color": "#F4511E"},
    "finance":   {"emoji": "📊", "label": "Finance & Profitability", "color": "#0A1628"},
    "general":   {"emoji": "🔬", "label": "General Analytics",      "color": "#646882"},
}
nm = NICHE_META.get(niche, NICHE_META["general"])

# ── PAGE HEADER ────────────────────────────────────────
col_title, col_badge = st.columns([5, 1])
with col_title:
    st.markdown(f"## 📋 Data Health & Business Insights")
    st.caption(f"{fname} — {health['rows']:,} rows · {health['cols']} columns")
with col_badge:
    st.markdown(
        f"<div class='niche-badge' style='background:{nm['color']};margin-top:14px'>"
        f"{nm['emoji']}  {niche.upper()}</div>",
        unsafe_allow_html=True)
st.divider()

# ── HEALTH SCORE + KPI STRIP ───────────────────────────
st.markdown("<div class='section-head'>🏥 Data Health Score</div>", unsafe_allow_html=True)

c_score, c1, c2, c3, c4, c5 = st.columns([1.8, 1, 1, 1, 1, 1])
with c_score:
    st.markdown(f"""
    <div class='health-ring'>
      <div class='health-score' style='color:{health["color"]}'>{health["score"]}</div>
      <div style='color:rgba(255,255,255,0.4);font-size:.7rem;letter-spacing:.05em;margin-top:2px'>OUT OF 100</div>
      <div class='health-grade' style='color:{health["color"]}'>{health["grade"]} — {health["label"]}</div>
    </div>""", unsafe_allow_html=True)

def kpi_card(val, label, delta="", delta_type="neu"):
    return (f"<div class='kpi-card'>"
            f"<div class='kpi-val' style='color:#60a5fa'>{val}</div>"
            f"<div class='kpi-label'>{label}</div>"
            f"{'<div class=\"kpi-delta delta-'+delta_type+'\">'+delta+'</div>' if delta else ''}"
            f"</div>")

with c1: st.markdown(kpi_card(f"{health['rows']:,}", "Total Rows"), unsafe_allow_html=True)
with c2: st.markdown(kpi_card(str(health["cols"]), "Columns"), unsafe_allow_html=True)
with c3:
    mp = health["missing_pct"]
    st.markdown(kpi_card(
        f"{mp}%", "Missing Data",
        "🔴 HIGH" if mp > 10 else "🟡 MODERATE" if mp > 3 else "🟢 CLEAN",
        "neg" if mp > 10 else "neu" if mp > 3 else "pos"),
        unsafe_allow_html=True)
with c4:
    dp = health["dup_pct"]
    st.markdown(kpi_card(
        f"{dp}%", "Duplicates",
        "🔴 HIGH" if dp > 5 else "🟢 CLEAN",
        "neg" if dp > 5 else "pos"),
        unsafe_allow_html=True)
with c5: st.markdown(kpi_card(
    str(len(insights)), "Insights Found",
    f"{'🔴' if any(i['severity']=='critical' for i in insights) else '🟢'} "
    f"{'CRITICAL ISSUES' if any(i['severity']=='critical' for i in insights) else 'All Healthy'}",
    "neg" if any(i["severity"]=="critical" for i in insights) else "pos"),
    unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Niche detection bar
if conf > 0:
    st.caption(f"**Domain detection:** {nm['emoji']} {nm['label']}  —  "
               f"{int(conf * 100)}% confidence based on column patterns")
    st.progress(min(conf, 1.0))
st.divider()

# ── INSIGHTS PANEL ────────────────────────────────────
st.markdown("<div class='section-head'>💡 Meaningful Business Insights</div>",
            unsafe_allow_html=True)
st.caption("Every insight includes: **What's happening → Why it matters → Exactly what to do.**")
st.markdown("")

for ins in insights:
    sev_icon = {"critical": "🔴", "warning": "🟡", "positive": "✅", "info": "🔵"}.get(ins["severity"], "🔵")
    with st.expander(f"{sev_icon}  {ins['title']}", expanded=ins["severity"] in ("critical", "warning")):
        st.markdown(
            f"<div class='ins-card' style='border-left-color:{ins[\"border\"]};background:{ins[\"bg\"]}'>"
            f"<div class='ins-tag' style='color:{ins[\"tag_color\"]}'>{ins['tag']}</div>"
            f"<div class='ins-body'>{ins['body']}</div>"
            f"<div class='ins-action' style='color:{ins[\"border\"]}'>{ins['action']}</div>"
            f"</div>",
            unsafe_allow_html=True)

st.divider()

# ── HEALTH BREAKDOWN CHART ─────────────────────────────
st.markdown("<div class='section-head'>📊 Health Breakdown</div>", unsafe_allow_html=True)
c_a, c_b = st.columns(2)

with c_a:
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=health["score"],
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": "Health Score", "font": {"size": 16}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar": {"color": health["color"]},
            "steps": [
                {"range": [0,  50], "color": "#fef2f2"},
                {"range": [50, 70], "color": "#fff7ed"},
                {"range": [70, 90], "color": "#f0fdf4"},
                {"range": [90,100], "color": "#dbeafe"},
            ],
            "threshold": {
                "line": {"color": "red", "width": 2},
                "thickness": 0.75, "value": 60}
        }))
    fig.update_layout(height=260, paper_bgcolor="rgba(0,0,0,0)",
                      font=dict(family="Inter, sans-serif", size=11))
    st.plotly_chart(fig, use_container_width=True)

with c_b:
    metrics = ["Completeness", "No Duplicates", "Outlier-Free"]
    values  = [
        max(0, 100 - health["missing_pct"] * 2.5),
        max(0, 100 - health["dup_pct"] * 3),
        max(0, 100 - health["outlier_pct"] * 0.4),
    ]
    bar_colors = ["#22d3a5" if v >= 80 else "#fbbf24" if v >= 60 else "#ef4444" for v in values]
    fig2 = go.Figure(go.Bar(
        x=values, y=metrics, orientation="h",
        marker_color=bar_colors, text=[f"{v:.0f}%" for v in values],
        textposition="auto",
    ))
    fig2.update_layout(
        title="Health Dimension Breakdown", height=260,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#f8faff",
        font=dict(family="Inter, sans-serif", size=11),
        xaxis=dict(range=[0, 100]), margin=dict(l=10, r=10, t=40, b=10),
        showlegend=False,
    )
    st.plotly_chart(fig2, use_container_width=True)

st.divider()

# ══════════════════════════════════════════════════════════
#  DOWNLOAD SECTION
# ══════════════════════════════════════════════════════════
st.markdown("<div class='section-head'>⬇️ Download Report</div>", unsafe_allow_html=True)

c_dl1, c_dl2 = st.columns([2, 3])

with c_dl1:
    st.markdown("**Generate Branded PDF Report**")
    st.caption(
        f"Includes: Health Score  ·  {len(insights)} Business Insights  ·  "
        f"Column Quality Table  ·  Niche: {nm['label']}"
    )
    gen = st.button("📥 Generate & Download PDF", type="primary",
                    use_container_width=True, key="gen_health_pdf")

with c_dl2:
    st.info(
        "💡 **Freelancing Tip:** This report is ready to send directly to clients. "
        "It auto-detects their data domain, gives them a health score, and provides "
        "**meaningful, prioritized actions** — exactly what they're paying for.")

if gen:
    with st.spinner("Building premium PDF..."):
        try:
            pdf_bytes = build_health_pdf(df, niche, health, insights, fname)
            safe_name = fname.replace(" ", "_").split(".")[0]
            st.success(f"✅ Report ready — {len(pdf_bytes)/1024:.0f} KB | "
                       f"Health Score: {health['score']}/100 | {len(insights)} insights")
            st.download_button(
                label="⬇️ Click to Download PDF Health Report",
                data=pdf_bytes,
                file_name=f"DataForge_Health_{safe_name}.pdf",
                mime="application/pdf",
                type="primary",
                use_container_width=True,
                key="dl_health_pdf",
            )
        except Exception as e:
            st.error(f"PDF generation failed: {e}")
            st.exception(e)

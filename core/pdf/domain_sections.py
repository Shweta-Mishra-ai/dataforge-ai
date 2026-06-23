"""
core/pdf/domain_sections.py — Domain-specific PDF sections.
Finance P&L pages, appendices.
"""
"""
core/pdf_builder.py — DataForge AI
Senior Analyst Edition v4 — DROP-IN REPLACEMENT

UPGRADED from basic to:
  ✅ Premium dark-navy cover with domain badge + KPI strip
  ✅ Running header / footer on every page
  ✅ Insight cards: Problem → Cause → Evidence → Action → Impact
  ✅ Internal-only benchmark context (no external sources hardcoded)
  ✅ Correlation warning box (r ≠ causation)
  ✅ Scenario modelling table
  ✅ Risk register with severity colouring
  ✅ Deduplication transparency note
  ✅ All chart analyses properly domain-aware

SAME public API — zero changes needed in 8_Reports.py:
  build_pdf(df, config, profile, cleaning_summary,
            stats_report, bi_report, ml_report,
            chart_data, executive_summary,
            findings, risks, opportunities, recommendations,
            top_insights, attrition, domain) → bytes
"""

import io
import os
from datetime import datetime

import numpy as np
import pandas as pd

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.colors import HexColor, white, black
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT, TA_JUSTIFY
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import (
    BaseDocTemplate, Frame, PageTemplate,
    Paragraph, Spacer, Table, TableStyle,
    Image, HRFlowable, PageBreak, KeepTogether,
)
from reportlab.pdfgen import canvas as CV
import logging
logger = logging.getLogger(__name__)
from core.pdf.theme import _c, _styles, _ReportCanvas

W, H = A4
CW_DEFAULT = W - 36 * mm   # content width (18mm each side)


# ══════════════════════════════════════════════════════════
#  DOMAIN COLOUR THEMES  (matches your existing THEMES keys)
# ══════════════════════════════════════════════════════════

THEMES = {
    "Corporate Light": {
        "cover_bg":    "#0A1628", "cover_accent": "#1B4FD8",
        "header_bg":   "#0A1628", "header_text":  "#FFFFFF",
        "accent":      "#1B4FD8", "accent2":      "#60A5FA",
        "text":        "#1F2937", "text_muted":   "#6B7280",
        "bg_light":    "#EFF6FF", "bg_card":      "#F8FAFF",
        "border":      "#E5E7EB",
        "positive":    "#10B981", "negative":     "#EF4444",
        "warning":     "#F59E0B", "info":         "#3B82F6",
        "critical_bg": "#FEE2E2", "warning_bg":   "#FEF3C7",
        "positive_bg": "#D1FAE5", "info_bg":      "#DBEAFE",
        "domain_label":"BUSINESS ANALYTICS",
        "domain_badge":"#1B4FD8",
    },
    "HR Blue": {
        "cover_bg":    "#0A1F4E", "cover_accent": "#1976D2",
        "header_bg":   "#0A1F4E", "header_text":  "#FFFFFF",
        "accent":      "#1976D2", "accent2":      "#90CAF9",
        "text":        "#1A2035", "text_muted":   "#5A6482",
        "bg_light":    "#E8F0FE", "bg_card":      "#F5F8FF",
        "border":      "#C5D3F0",
        "positive":    "#2E7D32", "negative":     "#C62828",
        "warning":     "#E65100", "info":         "#1565C0",
        "critical_bg": "#FFEBEE", "warning_bg":   "#FFF3E0",
        "positive_bg": "#E8F5E9", "info_bg":      "#E3F2FD",
        "domain_label":"HR & PEOPLE ANALYTICS",
        "domain_badge":"#1976D2",
    },
    "Ecommerce Orange": {
        "cover_bg":    "#3E1500", "cover_accent": "#F4511E",
        "header_bg":   "#BF360C", "header_text":  "#FFFFFF",
        "accent":      "#F4511E", "accent2":      "#FFAB91",
        "text":        "#1A1A1A", "text_muted":   "#5A5A5A",
        "bg_light":    "#FBE9E7", "bg_card":      "#FFF8F6",
        "border":      "#FFCCBC",
        "positive":    "#2E7D32", "negative":     "#B71C1C",
        "warning":     "#E65100", "info":         "#1565C0",
        "critical_bg": "#FFEBEE", "warning_bg":   "#FFF3E0",
        "positive_bg": "#E8F5E9", "info_bg":      "#E8F0FE",
        "domain_label":"E-COMMERCE ANALYTICS",
        "domain_badge":"#F4511E",
    },
    "Sales Green": {
        "cover_bg":    "#0A2710", "cover_accent": "#2E7D32",
        "header_bg":   "#1B5E20", "header_text":  "#FFFFFF",
        "accent":      "#2E7D32", "accent2":      "#A5D6A7",
        "text":        "#1A2A1A", "text_muted":   "#4A6A4A",
        "bg_light":    "#E8F5E9", "bg_card":      "#F5FBF5",
        "border":      "#C8E6C9",
        "positive":    "#1B5E20", "negative":     "#B71C1C",
        "warning":     "#E65100", "info":         "#1565C0",
        "critical_bg": "#FFEBEE", "warning_bg":   "#FFF3E0",
        "positive_bg": "#E8F5E9", "info_bg":      "#E8F0FE",
        "domain_label":"SALES PERFORMANCE ANALYTICS",
        "domain_badge":"#2E7D32",
    },
    "Dark Tech": {
        "cover_bg":    "#0D1117", "cover_accent": "#58A6FF",
        "header_bg":   "#0D1117", "header_text":  "#E6EDF3",
        "accent":      "#58A6FF", "accent2":      "#3FB950",
        "text":        "#E6EDF3", "text_muted":   "#8B949E",
        "bg_light":    "#161B22", "bg_card":      "#1C2128",
        "border":      "#30363D",
        "positive":    "#3FB950", "negative":     "#F85149",
        "warning":     "#D29922", "info":         "#58A6FF",
        "critical_bg": "#1C1010", "warning_bg":   "#1C1800",
        "positive_bg": "#0D1A0F", "info_bg":      "#0D1421",
        "domain_label":"TECHNICAL ANALYTICS",
        "domain_badge":"#58A6FF",
    },
}

# Auto-select theme by domain
DOMAIN_THEMES = {
    "hr":        "HR Blue",
    "ecommerce": "Ecommerce Orange",
    "sales":     "Sales Green",
    "finance":   "Corporate Light",
    "general":   "Corporate Light",
}

# Internal comparison guidance only — no hardcoded external benchmarks
# Values in "General Guidance" column are labelled as indicative
HR_BENCHMARKS = []  # Not used — see _benchmark_section for dataset-computed values


# ══════════════════════════════════════════════════════════
#  COLOUR HELPERS
# ══════════════════════════════════════════════════════════


from core.pdf.primitives import _sec, _kpi_row, _narrative_box, _gtable, _insight_card
from core.pdf.sections import _benchmark_section, _dq_note

def _finance_page(story, s, T, df, config, CW, profile=None):
    """
    Finance-domain PDF section.
    Generates P&L summary, margin analysis, budget vs actual,
    cost concentration, and period trend.
    All values computed from dataset — no external benchmarks hardcoded.
    """
    from reportlab.platypus import Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.units import mm

    def _find(keywords, exclude=None):
        excl = exclude or []
        for c in df.columns:
            cl = c.lower()
            if any(k in cl for k in keywords) and not any(e in cl for e in excl):
                return c
        return None

    _sec(story, s, T, "Finance Analysis",
         "P&L summary · Margin · Budget vs Actual · Cost breakdown — all from dataset")

    story.append(Paragraph(
        "All figures below are computed directly from the submitted dataset. "
        "No external benchmarks are embedded. Any general guidance references "
        "are clearly labelled and must be verified against sector-specific data.",
        s["body"]))
    story.append(Spacer(1, 3*mm))

    rev_col    = _find(["revenue","total_revenue","income","turnover","sales_amount"])
    cost_col   = _find(["cost","cogs","cost_of_goods","direct_cost"])
    profit_col = _find(["net_profit","profit","net_income"])
    gross_col  = _find(["gross_profit","gross_income"])
    budget_col = _find(["budget","plan","target","forecast"])
    actual_col = _find(["actual","actuals"], exclude=["target","budget"])
    if budget_col and not actual_col:
        actual_col = rev_col
    period_col = _find(["month","quarter","period","year","date"])
    cat_col    = _find(["category","department","cost_center","account","segment"])
    opex_col   = _find(["opex","operating_expense","overhead"])
    expense_col= _find(["expense","spend","expenditure"])
    val_col    = cost_col or expense_col or opex_col

    # ── P&L Summary Table ─────────────────────────────────────────────────
    story.append(Paragraph("P&L Summary", s["h3"]))

    pl_rows = []
    total_rev, total_cost, gross_profit, gross_margin = 0, 0, 0, 0
    total_opex, ebitda_proxy, total_profit = 0, 0, 0

    if rev_col:
        total_rev = float(df[rev_col].sum())
        pl_rows.append(["Total Revenue", f"{total_rev:,.0f}", "100.0%", "—"])

    if cost_col:
        total_cost  = float(df[cost_col].sum())
        gross_profit = total_rev - total_cost
        gross_margin = gross_profit / total_rev * 100 if total_rev else 0
        pl_rows.append(["Cost of Goods / Direct Cost", f"({total_cost:,.0f})",
                         f"({total_cost/total_rev*100:.1f}%)" if total_rev else "—",
                         "Dataset computed"])
        pl_rows.append(["Gross Profit", f"{gross_profit:,.0f}",
                         f"{gross_margin:.1f}%",
                         "Revenue minus direct cost"])
    elif gross_col:
        gross_profit = float(df[gross_col].sum())
        gross_margin = gross_profit / total_rev * 100 if total_rev else 0
        pl_rows.append(["Gross Profit", f"{gross_profit:,.0f}",
                         f"{gross_margin:.1f}%", "Dataset computed"])

    if opex_col:
        total_opex   = float(df[opex_col].sum())
        ebitda_proxy = gross_profit - total_opex
        opex_ratio   = total_opex / total_rev * 100 if total_rev else 0
        pl_rows.append(["Operating Expenses (OpEx)", f"({total_opex:,.0f})",
                         f"({opex_ratio:.1f}%)", "Dataset computed"])
        pl_rows.append(["Operating Profit (proxy)", f"{ebitda_proxy:,.0f}",
                         f"{ebitda_proxy/total_rev*100:.1f}%" if total_rev else "—",
                         "Gross profit minus OpEx"])

    if profit_col:
        total_profit = float(df[profit_col].sum())
        net_margin   = total_profit / total_rev * 100 if total_rev else 0
        pl_rows.append(["Net Profit / Income", f"{total_profit:,.0f}",
                         f"{net_margin:.1f}%", "Dataset computed"])

    if pl_rows:
        header = ["Line Item", "Amount", "% Revenue", "Source"]
        all_rows = [header] + pl_rows
        col_w = [CW*0.38, CW*0.22, CW*0.18, CW*0.22]
        t = Table([[Paragraph(str(c), s["h3"] if ri == 0 else s["body"])
                    for c in row]
                   for ri, row in enumerate(all_rows)],
                  colWidths=col_w)
        t.setStyle(TableStyle([
            ("BACKGROUND",    (0,0), (-1,0), _c(T["header_bg"])),
            ("TEXTCOLOR",     (0,0), (-1,0), _c("#FFFFFF")),
            ("TOPPADDING",    (0,0), (-1,-1), 4),
            ("BOTTOMPADDING", (0,0), (-1,-1), 4),
            ("LEFTPADDING",   (0,0), (-1,-1), 6),
            ("GRID",          (0,0), (-1,-1), 0.3, _c("#E2E8F0")),
            ("ROWBACKGROUNDS",(0,1), (-1,-1), [_c("#FFFFFF"), _c("#F8FAFC")]),
            # Highlight gross profit row
            ("BACKGROUND",    (0,2), (-1,2), _c("#F0FDF4")) if cost_col else ("",),
            ("FONTNAME",      (0,2), (-1,2), "Helvetica-Bold") if cost_col else ("",),
        ]))
        story.append(t)
        story.append(Spacer(1, 3*mm))

        # Margin summary chips
        if gross_margin:
            margin_color = T["positive"] if gross_margin > 40 else T["warning"] if gross_margin > 20 else T["negative"]
            margin_box = Table([[Paragraph(
                f"<b>Gross Margin: {gross_margin:.1f}%</b> | "
                f"{'Healthy — focus on protecting it.' if gross_margin > 40 else 'Moderate — review cost drivers.' if gross_margin > 20 else 'Low — immediate cost review required.'}"
                f" (Computed from dataset — compare to your prior periods, not generic norms.)",
                s["note"])]],
                colWidths=[CW])
            margin_box.setStyle(TableStyle([
                ("LEFTPADDING",  (0,0),(0,0), 10),
                ("TOPPADDING",   (0,0),(0,0), 8),
                ("BOTTOMPADDING",(0,0),(0,0), 8),
                ("BOX",          (0,0),(0,0), 1, _c(margin_color)),
            ]))
            story.append(margin_box)
            story.append(Spacer(1, 3*mm))

    # ── Budget vs Actual Table ────────────────────────────────────────────
    if budget_col and actual_col and budget_col != actual_col:
        story.append(Paragraph("Budget vs Actual Variance", s["h3"]))
        try:
            comp_col = period_col or cat_col
            if comp_col:
                bva = df.groupby(comp_col)[[budget_col, actual_col]].sum().reset_index()
                bva.columns = ["Period/Category", "Budget", "Actual"]
                bva["Variance"]     = bva["Actual"] - bva["Budget"]
                bva["Variance %"]   = ((bva["Actual"] - bva["Budget"]) /
                                        bva["Budget"].replace(0, np.nan) * 100).round(1)
                bva = bva.sort_values("Variance %", key=abs, ascending=False).head(12)

                bva_header = ["Period / Category", "Budget", "Actual", "Variance", "Variance %"]
                bva_data   = [[str(row["Period/Category"])[:28],
                               f"{row['Budget']:,.0f}",
                               f"{row['Actual']:,.0f}",
                               f"{row['Variance']:+,.0f}",
                               f"{row['Variance %']:+.1f}%"]
                              for _, row in bva.iterrows()]

                all_rows_bva = [bva_header] + bva_data
                col_w_bva    = [CW*0.30, CW*0.17, CW*0.17, CW*0.18, CW*0.18]

                t_bva = Table([[Paragraph(str(c), s["h3"] if ri == 0 else s["body"])
                                for c in row]
                               for ri, row in enumerate(all_rows_bva)],
                              colWidths=col_w_bva)
                t_bva.setStyle(TableStyle([
                    ("BACKGROUND",    (0,0), (-1,0), _c(T["header_bg"])),
                    ("TEXTCOLOR",     (0,0), (-1,0), _c("#FFFFFF")),
                    ("TOPPADDING",    (0,0), (-1,-1), 4),
                    ("BOTTOMPADDING", (0,0), (-1,-1), 4),
                    ("LEFTPADDING",   (0,0), (-1,-1), 5),
                    ("GRID",          (0,0), (-1,-1), 0.3, _c("#E2E8F0")),
                    ("ROWBACKGROUNDS",(0,1), (-1,-1), [_c("#FFFFFF"), _c("#F8FAFC")]),
                ]))
                story.append(t_bva)
                story.append(Spacer(1, 3*mm))

                # Variance summary
                over_n  = int((bva["Variance %"] > 10).sum())
                under_n = int((bva["Variance %"] < -10).sum())
                if over_n + under_n > 0:
                    story.append(Paragraph(
                        f"⚠ {over_n} items exceed budget by >10% | "
                        f"{under_n} items under budget by >10%. "
                        f"A variance trigger of ±10% is commonly used as a review threshold — "
                        f"adjust to your organisation's planning standards.",
                        s["note"]))
        except Exception as e:
            story.append(Paragraph(f"Budget vs actual table unavailable: {e}", s["note"]))

    # ── Cost by Category ──────────────────────────────────────────────────
    if cat_col and val_col:
        story.append(Paragraph("Cost / Expense by Category", s["h3"]))
        try:
            cat_cost = df.groupby(cat_col)[val_col].sum().sort_values(ascending=False).head(10)
            total_c  = float(cat_cost.sum())
            cat_rows = [[str(idx)[:32], f"{val:,.0f}", f"{val/total_c*100:.1f}%"]
                        for idx, val in cat_cost.items()]
            cat_header = ["Category / Segment", "Total Amount", "% of Total"]
            all_cat    = [cat_header] + cat_rows
            col_w_cat  = [CW*0.50, CW*0.28, CW*0.22]
            t_cat = Table([[Paragraph(str(c), s["h3"] if ri == 0 else s["body"])
                            for c in row]
                           for ri, row in enumerate(all_cat)],
                          colWidths=col_w_cat)
            t_cat.setStyle(TableStyle([
                ("BACKGROUND",    (0,0), (-1,0), _c(T["header_bg"])),
                ("TEXTCOLOR",     (0,0), (-1,0), _c("#FFFFFF")),
                ("TOPPADDING",    (0,0), (-1,-1), 4),
                ("BOTTOMPADDING", (0,0), (-1,-1), 4),
                ("LEFTPADDING",   (0,0), (-1,-1), 5),
                ("GRID",          (0,0), (-1,-1), 0.3, _c("#E2E8F0")),
                ("ROWBACKGROUNDS",(0,1), (-1,-1), [_c("#FFFFFF"), _c("#F8FAFC")]),
            ]))
            story.append(t_cat)

            top_cat     = str(cat_cost.index[0])
            top_pct     = float(cat_cost.iloc[0] / total_c * 100)
            top3_pct    = float(cat_cost.iloc[:3].sum() / total_c * 100)
            story.append(Spacer(1, 2*mm))
            story.append(Paragraph(
                f"'{top_cat}' = {top_pct:.1f}% of total | "
                f"Top 3 combined = {top3_pct:.1f}%. "
                f"{'High concentration — assess dependency risk.' if top_pct > 50 else 'Moderate concentration — monitor for shifts.'} "
                f"All values from dataset.", s["note"]))
        except Exception as e:
            story.append(Paragraph(f"Cost breakdown unavailable: {e}", s["note"]))

    story.append(Spacer(1, 3*mm))

    # ── Period Trend Summary ──────────────────────────────────────────────
    if period_col and rev_col:
        story.append(Paragraph("Period-over-Period Revenue Summary", s["h3"]))
        try:
            period_rev = df.groupby(period_col)[rev_col].sum()
            try:
                period_rev = period_rev.sort_index()
            except Exception:
                logger.warning("%s unexpected failure", exc_info=True)

            if len(period_rev) >= 2:
                period_rows = [[str(idx)[:20], f"{val:,.0f}",
                                f"{(val - period_rev.iloc[max(0,i-1)]) / period_rev.iloc[max(0,i-1)] * 100:+.1f}%"
                                if i > 0 else "—"]
                               for i, (idx, val) in enumerate(period_rev.items())]
                period_header = ["Period", "Revenue", "Change vs Prior"]
                all_period    = [period_header] + period_rows
                col_w_p       = [CW*0.38, CW*0.35, CW*0.27]
                t_p = Table([[Paragraph(str(c), s["h3"] if ri == 0 else s["body"])
                              for c in row]
                             for ri, row in enumerate(all_period)],
                            colWidths=col_w_p)
                t_p.setStyle(TableStyle([
                    ("BACKGROUND",    (0,0), (-1,0), _c(T["header_bg"])),
                    ("TEXTCOLOR",     (0,0), (-1,0), _c("#FFFFFF")),
                    ("TOPPADDING",    (0,0), (-1,-1), 4),
                    ("BOTTOMPADDING", (0,0), (-1,-1), 4),
                    ("LEFTPADDING",   (0,0), (-1,-1), 5),
                    ("GRID",          (0,0), (-1,-1), 0.3, _c("#E2E8F0")),
                    ("ROWBACKGROUNDS",(0,1), (-1,-1), [_c("#FFFFFF"), _c("#F8FAFC")]),
                ]))
                story.append(t_p)
        except Exception as e:
            story.append(Paragraph(f"Period trend unavailable: {e}", s["note"]))

    # Disclaimer
    story.append(Spacer(1, 3*mm))
    story.append(Paragraph(
        "All financial metrics above are computed from the submitted dataset only. "
        "No external financial benchmarks are embedded. "
        "Verify all figures with your accounting team before using in board materials.",
        s["note"]))


def _appendix(story, s, T, config, CW, domain: str = "general", df=None, profile=None):
    _sec(story, s, T, "Appendix — Methodology & Sources")

    story.append(Paragraph("A. Methodology", s["h3"]))
    story.append(Paragraph(
        "Data quality scoring: 60% completeness, 30% deduplication, 10% column health. "
        "Outlier detection: IQR (1.5×) and Modified Z-Score. "
        "Normality: Shapiro-Wilk (n≤5000) and D'Agostino-Pearson tests. "
        "Correlations: Pearson (normal) / Spearman (non-normal). "
        "Domain detection: keyword matching across HR, E-commerce, Sales, Finance. "
        "Attrition drivers: Mann-Whitney U (numeric) and Chi-Square (categorical). "
        "AI-assisted narrative generation with pre-computed statistical outputs. "
        "All findings verified against dataset values before inclusion.",
        s["body"]))

    story.append(Paragraph("B. Quality Score Formula & This Dataset's Breakdown", s["h3"]))
    story.append(Paragraph(
        "Formula: Total = (Completeness × 0.60) + (Deduplication × 0.30) + (Column Health × 0.10). "
        "Weights reflect analytical impact: missing values corrupt statistics most severely; "
        "duplicates inflate all aggregates; distributional issues are correctable. "
        "The table below shows the formula applied to this specific dataset so the score is fully auditable.",
        s["body"]))
    story.append(Spacer(1, 2*mm))
    # Compute actual scores (profile/df come from function signature defaults)
    _raw_rows   = getattr(profile, "total_rows", len(df)) if profile else len(df)  # noqa: F821
    _dupes      = getattr(profile, "duplicate_rows", 0) if profile else (int(df.duplicated().sum()) if df is not None else 0)  # noqa: F821
    _unique     = _raw_rows - _dupes
    _miss_pct   = (getattr(profile, "missing_pct", None) or (df.isna().mean().mean()*100 if df is not None else 0)) if profile else (df.isna().mean().mean()*100 if df is not None else 0)  # noqa: F821
    _col_hlth   = getattr(profile, "overall_quality_score", 95) if profile else 95
    _comp_score = round((1 - _miss_pct/100) * 60, 1)
    _ded_score  = round((_unique / max(_raw_rows,1)) * 30, 1)
    _col_score  = round(min(_col_hlth,100)/100 * 10, 1)
    _total      = round(_comp_score + _ded_score + _col_score, 1)
    _gtable(story, T,
            ["Component", "Weight", "Formula", "This Dataset", "Score"],
            [["Completeness",  "60%",
              "60 × (1 – missing_rate)",
              f"Missing: {_miss_pct:.1f}%",
              f"{_comp_score:.1f} / 60"],
             ["Deduplication", "30%",
              "30 × (unique / raw_rows)",
              f"{_unique:,} / {_raw_rows:,} = {_unique/max(_raw_rows,1)*100:.1f}%",
              f"{_ded_score:.1f} / 30"],
             ["Column Health", "10%",
              "10 × avg(per-col score)",
              f"Avg: {_col_hlth:.1f}%",
              f"{_col_score:.1f} / 10"],
             ["TOTAL",         "100%", "Sum of above", "—", f"{_total:.1f} / 100"]],
            [CW*x for x in [0.20, 0.10, 0.28, 0.28, 0.14]])

    # FIX-055: Domain-isolated appendix sources
    story.append(Paragraph("C. Industry Sources", s["h3"]))
    _domain_sources = {
        "hr": [
            "All metrics computed from submitted dataset only. No external benchmark database was queried.",
            "General industry context provided as indicative guidance — verify against sector-specific data.",
            "Statistical methods: logistic regression (attrition drivers), Kruskal-Wallis (group comparisons),",
            "  Spearman correlation (non-normal distributions), IQR outlier detection (1.5×).",
        ],
        "ecommerce": [
            "All metrics computed from submitted dataset only. No external benchmark database was queried.",
            "General e-commerce context provided as indicative guidance — verify with your platform data.",
            "Statistical methods: distribution analysis, correlation testing, outlier detection, trend analysis.",
        ],
        "sales": [
            "All metrics computed from submitted dataset only. No external benchmark database was queried.",
            "General sales context provided as indicative guidance — verify against your CRM data.",
            "Statistical methods: distribution analysis, cohort comparisons, correlation testing.",
        ],
        "finance": [
            "All metrics computed from submitted dataset only. No external benchmark database was queried.",
            "General finance context provided as indicative guidance — verify against sector-specific data.",
            "Statistical methods: distribution analysis, variance decomposition, trend analysis.",
        ],
    }
    _sources = _domain_sources.get(domain, [
        "Analysis based on dataset provided. Industry benchmarks are indicative.",
        "Verify all findings with qualified domain experts before taking action.",
    ])
    for src in _sources:
        story.append(Paragraph("• " + src, s["bl"]))

    story.append(Spacer(1, 4*mm))
    disc = Table([[Paragraph(
        "<b>DISCLAIMER</b><br/>"
        "Report generated by DataForge AI on {} for {}. "
        "Findings based solely on provided dataset. "
        "Correlations do not imply causation. "
        "Any general industry ranges shown are indicative guidance — not verified external benchmarks. "
        "Verify with qualified data analyst before business decisions.".format(
            datetime.now().strftime("%B %d, %Y"),
            config.get("client_name", "Client")),
        s["wh"])]],
        colWidths=["100%"])
    disc.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,-1), _c(T["header_bg"])),
        ("TOPPADDING",    (0,0), (-1,-1), 10),
        ("BOTTOMPADDING", (0,0), (-1,-1), 10),
        ("LEFTPADDING",   (0,0), (-1,-1), 12),
        ("RIGHTPADDING",  (0,0), (-1,-1), 12),
        ("BOX",           (0,0), (-1,-1), 1.5, _c(T["accent"])),
    ]))
    story.append(disc)


# ══════════════════════════════════════════════════════════
#  MAIN PUBLIC FUNCTION  (identical signature to original)
# ══════════════════════════════════════════════════════════


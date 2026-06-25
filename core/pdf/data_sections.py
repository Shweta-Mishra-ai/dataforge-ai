"""
core/pdf/data_sections.py
Data sections: dataset overview, stats, BI analysis, charts, recommendations.
"""
"""
core/pdf/sections.py — Report sections: exec summary, insights, stats, charts.
Each function appends to story[]. No global state.
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


from core.pdf.primitives import _sec, _kpi_row, _narrative_box, _gtable, _insight_card, _toc


def _dataset_overview(story, s, T, df, profile, CW):
    _sec(story, s, T, "Dataset Overview & Descriptive Statistics",
         "Column breakdown, statistical summary, correlations, and distribution flags")

    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
    dt_cols  = df.select_dtypes(include="datetime").columns.tolist()

    # ── Dataset composition strip ──────────────────────────
    _gtable(story, T,
            ["Type", "Count", "Sample Columns"],
            [["Numeric",     len(num_cols), ", ".join(num_cols[:6]) or "None"],
             ["Categorical", len(cat_cols), ", ".join(cat_cols[:6]) or "None"],
             ["DateTime",    len(dt_cols),  ", ".join(dt_cols[:4]) or "None"]],
            [CW*0.20, CW*0.12, CW*0.68])

    # ── Descriptive statistics table ───────────────────────
    if num_cols:
        story.append(Paragraph("Descriptive Statistics", s["h3"]))
        import re as _re
        _ID_KW = {"index","idx","id","rowid","row_id","empid","emp_id",
                  "order_id","orderid","product_id","customer_id","user_id"}
        def _is_id_col(col, series):
            cl = col.lower().strip()
            if cl in _ID_KW or _re.search(r'\bid\b|\bindex\b', cl):
                return True
            if len(series.dropna()) > 10:
                try:
                    diffs = series.dropna().sort_values().diff().dropna()
                    if (diffs == 1).mean() > 0.90:
                        return True
                except Exception:
                    logger.warning("id-col check failed", exc_info=True)
            return False

        filtered_num = [c for c in num_cols if not _is_id_col(c, df[c])]
        show = filtered_num[:6] if filtered_num else num_cols[:6]
        desc = df[show].describe().round(3)

        hrow = ["Stat"] + [c[:9] for c in show]
        rows = [hrow] + [
            [stat] + [str(desc.loc[stat, c]) for c in show]
            for stat in ["mean", "std", "min", "25%", "50%", "75%", "max"]
            if stat in desc.index
        ]
        cw_s = CW / (len(show) + 1)
        tbl = Table(rows, colWidths=[cw_s] * (len(show)+1), repeatRows=1)
        tbl.setStyle(TableStyle([
            ("FONTNAME",      (0,0), (-1,0),  "Helvetica-Bold"),
            ("FONTNAME",      (0,1), (-1,-1), "Helvetica"),
            ("FONTSIZE",      (0,0), (-1,-1), 8),
            ("TEXTCOLOR",     (0,0), (-1,0),  HexColor("#FFFFFF")),
            ("BACKGROUND",    (0,0), (-1,0),  _c(T["header_bg"])),
            ("ROWBACKGROUNDS",(0,1),(-1,-1),  [HexColor("#FFFFFF"), _c(T["bg_light"])]),
            ("GRID",          (0,0), (-1,-1), 0.3, _c(T["border"])),
            ("ALIGN",         (0,0), (-1,-1), "CENTER"),
            ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
            ("TOPPADDING",    (0,0), (-1,-1), 4),
            ("BOTTOMPADDING", (0,0), (-1,-1), 4),
        ]))
        story.append(KeepTogether([tbl]))
        story.append(Spacer(1, 3*mm))

        # ── Per-column insight flags ───────────────────────
        story.append(Paragraph("Column-Level Analytical Flags", s["h3"]))
        flag_rows = [["Column", "Mean", "Median", "CV %", "Skew", "Outliers %", "Flag"]]
        for col in filtered_num[:10]:
            try:
                s_col = df[col].dropna()
                if len(s_col) < 3:
                    continue
                mean_v   = float(s_col.mean())
                med_v    = float(s_col.median())
                std_v    = float(s_col.std())
                cv       = std_v / abs(mean_v) * 100 if mean_v != 0 else 0
                skew_v   = float(s_col.skew())
                q1, q3   = float(s_col.quantile(0.25)), float(s_col.quantile(0.75))
                iqr      = q3 - q1
                out_pct  = float(((s_col < q1-1.5*iqr) | (s_col > q3+1.5*iqr)).mean() * 100)
                if out_pct > 15 or abs(skew_v) > 3:
                    flag = "⚠ Heavy outliers"
                elif cv > 80:
                    flag = "⚠ High variability"
                elif abs(skew_v) > 1.5:
                    flag = "Use median"
                elif out_pct > 5:
                    flag = "Review tail"
                else:
                    flag = "✓ Normal"
                flag_rows.append([
                    col[:12],
                    f"{mean_v:.3g}", f"{med_v:.3g}",
                    f"{cv:.0f}%", f"{skew_v:+.2f}",
                    f"{out_pct:.1f}%", flag,
                ])
            except Exception:
                logger.warning("flag row failed", exc_info=True)
        if len(flag_rows) > 1:
            cws = [CW*x for x in [0.18, 0.1, 0.1, 0.09, 0.09, 0.12, 0.32]]
            ft = Table(flag_rows, colWidths=cws, repeatRows=1)
            ft.setStyle(TableStyle([
                ("FONTNAME",      (0,0), (-1,0),  "Helvetica-Bold"),
                ("FONTNAME",      (0,1), (-1,-1), "Helvetica"),
                ("FONTSIZE",      (0,0), (-1,-1), 8),
                ("TEXTCOLOR",     (0,0), (-1,0),  HexColor("#FFFFFF")),
                ("BACKGROUND",    (0,0), (-1,0),  _c(T["header_bg"])),
                ("ROWBACKGROUNDS",(0,1),(-1,-1),  [HexColor("#FFFFFF"), _c(T["bg_light"])]),
                ("GRID",          (0,0), (-1,-1), 0.3, _c(T["border"])),
                ("ALIGN",         (0,0), (-1,-1), "CENTER"),
                ("ALIGN",         (0,0), (0,-1),  "LEFT"),
                ("ALIGN",         (-1,0),(-1,-1), "LEFT"),
                ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
                ("TOPPADDING",    (0,0), (-1,-1), 3),
                ("BOTTOMPADDING", (0,0), (-1,-1), 3),
            ]))
            story.append(KeepTogether([ft]))
            story.append(Spacer(1, 3*mm))

        # ── Top correlations ──────────────────────────────
        if len(filtered_num) >= 2:
            story.append(Paragraph("Top Correlations (Pearson r)", s["h3"]))
            try:
                from scipy import stats as _scipy_stats
                corr_rows = [["Column A", "Column B", "r", "r²", "Strength", "Sig?"]]
                cols_for_corr = filtered_num[:8]
                for i in range(len(cols_for_corr)):
                    for j in range(i+1, len(cols_for_corr)):
                        a, b = cols_for_corr[i], cols_for_corr[j]
                        try:
                            common = df[[a,b]].dropna()
                            if len(common) < 10:
                                continue
                            r, p = _scipy_stats.spearmanr(common[a], common[b])
                            if abs(r) >= 0.25:
                                strength = "Strong" if abs(r)>=0.7 else "Moderate" if abs(r)>=0.4 else "Weak"
                                sig = "Yes" if p < 0.05 else "No"
                                corr_rows.append([a[:12], b[:12], f"{r:+.3f}",
                                                  f"{r**2:.3f}", strength, sig])
                        except Exception:
                            logger.debug("corr pair skip", exc_info=True)
                corr_header = corr_rows[0]
                corr_data   = corr_rows[1:]
                corr_data.sort(key=lambda x: abs(float(x[2])), reverse=True)
                top_corr = [corr_header] + corr_data[:6]
                if len(top_corr) > 1:
                    cws2 = [CW*x for x in [0.20, 0.20, 0.10, 0.10, 0.18, 0.10]]
                    ctbl = Table(top_corr, colWidths=cws2, repeatRows=1)
                    ctbl.setStyle(TableStyle([
                        ("FONTNAME",      (0,0), (-1,0),  "Helvetica-Bold"),
                        ("FONTNAME",      (0,1), (-1,-1), "Helvetica"),
                        ("FONTSIZE",      (0,0), (-1,-1), 8),
                        ("TEXTCOLOR",     (0,0), (-1,0),  HexColor("#FFFFFF")),
                        ("BACKGROUND",    (0,0), (-1,0),  _c(T["header_bg"])),
                        ("ROWBACKGROUNDS",(0,1),(-1,-1),  [HexColor("#FFFFFF"), _c(T["bg_light"])]),
                        ("GRID",          (0,0), (-1,-1), 0.3, _c(T["border"])),
                        ("ALIGN",         (0,0), (-1,-1), "CENTER"),
                        ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
                        ("TOPPADDING",    (0,0), (-1,-1), 3),
                        ("BOTTOMPADDING", (0,0), (-1,-1), 3),
                    ]))
                    story.append(KeepTogether([ctbl]))
                    story.append(Paragraph(
                        "Correlation ≠ causation. r² = shared variance. "
                        "Significant at p<0.05 only. Strong correlations (|r|≥0.7) warrant "
                        "causal investigation via controlled comparison or regression.",
                        s["note"]))
            except Exception as _e:
                logger.debug("Correlation table failed: %s", _e)


# ══════════════════════════════════════════════════════════
#  STATISTICAL ANALYSIS
# ══════════════════════════════════════════════════════════

def _stats_section(story, s, T, stats_report, CW):
    if stats_report is None: return
    _sec(story, s, T, "Statistical Analysis",
         "Distribution, normality, correlations")

    # Correlation honest-warning box
    warn_t = Table([[Paragraph(
        "<b>⚠ Analyst Note:</b> "
        "Correlation r does NOT mean 'Variable A changes Variable B by r%.' "
        "That is a dangerous misread. r = -0.35 means the two variables share "
        "12.3% of their variance (r² = 0.123). Association only — "
        "NOT causation, NOT magnitude of effect.",
        s["warn"])]],
        colWidths=["100%"])
    warn_t.setStyle(TableStyle([
        ("LEFTPADDING",  (0,0), (-1,-1), 10),
        ("RIGHTPADDING", (0,0), (-1,-1), 10),
        ("TOPPADDING",   (0,0), (-1,-1), 8),
        ("BOTTOMPADDING",(0,0), (-1,-1), 8),
        ("BOX",          (0,0), (-1,-1), 1, _c(T["warning"])),
    ]))
    story.append(warn_t)
    story.append(Spacer(1, 3*mm))

    # Distribution summary
    col_stats = getattr(stats_report, "column_stats", {})
    if col_stats:
        story.append(Paragraph("Distribution Summary", s["h3"]))
        for col, cs in list(col_stats.items())[:8]:
            if getattr(cs, "mean", None) is None: continue
            normal = "Normal" if getattr(cs, "is_normal", False) else "Non-normal"
            sk_lbl = getattr(cs, "skew_label", "") or ""
            outs   = getattr(cs, "outlier_count_iqr", 0)
            story.append(Paragraph(
                "• '{}': {} | {} | Outliers: {}".format(col, normal, sk_lbl, outs),
                s["bl"]))

    # Correlations
    corrs = getattr(stats_report, "correlations", [])
    sig   = [c for c in corrs
             if getattr(c, "is_significant", False)
             and abs(getattr(c, "pearson_r", 0)) >= 0.15]
    if sig:
        story.append(Spacer(1, 2*mm))
        story.append(Paragraph("Significant Correlations (Correct Interpretation)", s["h3"]))
        rows = [[c.col_a, c.col_b,
                 str(round(c.pearson_r, 4)),
                 str(round(getattr(c, "p_value", 0), 4)),
                 c.strength.title(),
                 "r²={:.3f} — {:.1f}% variance shared. Association only.".format(
                     c.pearson_r**2, c.pearson_r**2 * 100)]
                for c in sig[:6]]
        _gtable(story, T,
                ["Col A", "Col B", "r", "p", "Strength", "Interpretation"],
                rows, [CW*x for x in [0.17, 0.17, 0.08, 0.08, 0.12, 0.38]])


# ══════════════════════════════════════════════════════════
#  BUSINESS INTELLIGENCE
# ══════════════════════════════════════════════════════════

def _bi_section(story, s, T, bi_report, CW):
    if bi_report is None: return
    _sec(story, s, T, "Business Intelligence",
         "Benchmarking, cohort analysis, segment performance")

    brief = getattr(bi_report, "executive_brief", "")
    if brief:
        _narrative_box(story, s, T, brief)

    # Benchmarks
    bms = getattr(bi_report, "benchmarks", [])
    if bms:
        story.append(Paragraph("Benchmarking Summary", s["h3"]))
        rows = [[bm.column, str(bm.mean), str(bm.median),
                 str(bm.top_10_pct), str(bm.bottom_10_pct),
                 bm.benchmark_label.split("—")[0].strip()[:15]]
                for bm in bms[:4]]
        _gtable(story, T,
                ["Metric", "Mean", "Median", "Top 10%", "Bottom 10%", "Variation"],
                rows, [CW*x for x in [0.22, 0.12, 0.12, 0.12, 0.13, 0.29]])

    # Segment performance
    segs = getattr(bi_report, "segments", [])
    if segs:
        story.append(Spacer(1, 2*mm))
        story.append(Paragraph("Segment Performance", s["h3"]))
        seg_rows = []
        for seg in segs[:8]:
            strengths_str = ", ".join(seg.strengths[:2]) if seg.strengths else "—"
            weakness_str  = ", ".join(seg.weaknesses[:2]) if seg.weaknesses else "—"
            seg_rows.append([
                str(seg.segment_name)[:18],
                str(seg.n),
                "{:.0f}".format(seg.health_score),
                strengths_str[:30],
                weakness_str[:30],
            ])
        _gtable(story, T,
                ["Segment", "N", "Score", "Strengths", "Weaknesses"],
                seg_rows,
                [CW*x for x in [0.20, 0.07, 0.08, 0.32, 0.33]])

        # Opportunities callout
        opps = [seg for seg in segs if seg.weaknesses]
        if opps:
            story.append(Spacer(1, 2*mm))
            story.append(Paragraph("Segment Opportunities", s["h3"]))
            for seg in opps[:3]:
                story.append(Paragraph(
                    "• {}: {}".format(seg.segment_name, seg.opportunity),
                    s["bl"]))

    # Cohorts
    sig_c = [c for c in getattr(bi_report, "cohorts", [])
             if c.is_significant]
    if sig_c:
        story.append(Paragraph("Significant Cohort Differences", s["h3"]))
        for c in sig_c[:3]:
            story.append(Paragraph("• " + c.interpretation, s["bl"]))

    # Key insights
    ki = getattr(bi_report, "key_insights", [])
    if ki:
        story.append(Spacer(1, 2*mm))
        story.append(Paragraph("Key Business Insights", s["h3"]))
        for ins in ki[:5]:
            story.append(Paragraph("• " + str(ins), s["bl"]))


# ══════════════════════════════════════════════════════════
#  CHART PAGE
# ══════════════════════════════════════════════════════════

def _chart_page(story, s, T, img_bytes, title, narrative, num, CW):
    _sec(story, s, T, "Chart {}: {}".format(num, title))
    if img_bytes:
        try:
            img = Image(io.BytesIO(img_bytes),
                        width=CW, height=CW * 0.48)
            story.append(KeepTogether([img, Spacer(1, 3*mm)]))
        except Exception:
            logger.warning("%s unexpected failure", exc_info=True)
    if narrative:
        story.append(Paragraph("Analysis", s["h3"]))
        _narrative_box(story, s, T, narrative)


# ══════════════════════════════════════════════════════════
#  RECOMMENDATIONS
# ══════════════════════════════════════════════════════════

def _recommendations(story, s, T, actions, CW):
    _sec(story, s, T, "Recommendations & Action Plan",
         "Prioritised by urgency — act on CRITICAL items first")

    pri_map = {
        "CRITICAL":   (T["negative"],  T["critical_bg"]),
        "SHORT TERM": (T["warning"],   T["warning_bg"]),
        "LONG TERM":  (T["info"],      T["info_bg"]),
    }

    for action in actions[:9]:
        action_str = str(action)
        priority   = "LONG TERM"
        for p in ("CRITICAL", "SHORT TERM"):
            if p in action_str.upper():
                priority = p
                break

        col, bg = pri_map.get(priority, (T["accent"], T["bg_light"]))
        text    = (action_str
                   .replace("[CRITICAL] ", "")
                   .replace("[SHORT TERM] ", "")
                   .replace("[LONG TERM] ", "")
                   .strip())

        card = Table([[
            Paragraph(priority, ParagraphStyle(
                "pri", fontName="Helvetica-Bold", fontSize=7,
                textColor=HexColor(col), alignment=TA_CENTER)),
            Paragraph(text, s["body"]),
        ]], colWidths=[CW*0.14, CW*0.86])
        card.setStyle(TableStyle([
            ("BACKGROUND",    (0,0), (-1,-1), HexColor(bg)),
            ("LINEBEFORE",    (0,0), (0,-1),  3, HexColor(col)),
            ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
            ("TOPPADDING",    (0,0), (-1,-1), 5),
            ("BOTTOMPADDING", (0,0), (-1,-1), 5),
            ("LEFTPADDING",   (0,0), (-1,-1), 6),
            ("RIGHTPADDING",  (0,0), (-1,-1), 6),
        ]))
        story.append(KeepTogether([card, Spacer(1, 2*mm)]))

    story.append(Spacer(1, 3*mm))
    story.append(Paragraph(
        "All recommendations are based solely on the provided dataset. "
        "Verify with domain experts before implementation.",
        s["sm"]))


# ══════════════════════════════════════════════════════════
#  APPENDIX
# ══════════════════════════════════════════════════════════

